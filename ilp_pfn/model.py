"""Decoder-only transformer with rotary position embeddings and a KV cache for sampling."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import N_ROLES, N_SECTIONS, VOCAB, token_classes


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    d_ff: int = 1536
    max_len: int = 2048       # RoPE table size (training may use shorter sequences)
    rope_base: float = 10000.0
    role_emb: bool = False    # add section + atom-role embeddings to the token embeddings
    query_head: bool = False  # yes/no head for query atoms (auxiliary loss)


class KVCache:
    def __init__(self):
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    @property
    def length(self) -> int:
        return 0 if self.k is None else self.k.shape[2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, T, hd); cos/sin: (T, hd/2). Rotates interleaved pairs."""
    x1, x2 = x[..., ::2].float(), x[..., 1::2].float()
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.h, self.hd = cfg.n_heads, cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, cache: KVCache | None = None):
        B, T, D = x.shape
        q, k, v = (t.view(B, T, self.h, self.hd).transpose(1, 2)
                   for t in self.qkv(x).split(D, dim=2))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            if cache.k is not None:
                assert T == 1, "incremental decoding feeds one token at a time"
                k, v = torch.cat([cache.k, k], 2), torch.cat([cache.v, v], 2)
            cache.k, cache.v = k, v
        # causal mask while queries and keys are aligned; a single new query sees everything
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(cfg.d_model), nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff, bias=False), nn.GELU(),
                                 nn.Linear(cfg.d_ff, cfg.d_model, bias=False))

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.ln1(x), cos, sin, cache)
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight  # tied
        hd = cfg.d_model // cfg.n_heads
        inv = 1.0 / cfg.rope_base ** (torch.arange(0, hd, 2).float() / hd)
        freqs = torch.outer(torch.arange(cfg.max_len).float(), inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        if cfg.role_emb:
            assert cfg.vocab_size == len(VOCAB)
            self.sec_emb = nn.Embedding(N_SECTIONS, cfg.d_model)
            self.role_emb = nn.Embedding(N_ROLES, cfg.d_model)
            is_pred, is_term, section = token_classes(VOCAB)
            self.register_buffer("tok_is_pred", torch.tensor(is_pred), persistent=False)
            self.register_buffer("tok_is_term", torch.tensor(is_term), persistent=False)
            self.register_buffer("tok_section", torch.tensor(section), persistent=False)
        if cfg.query_head:
            self.qhead = nn.Linear(cfg.d_model, 1)
        self.apply(self._init)
        for blk in self.blocks:  # scaled init of residual projections
            for w in (blk.attn.proj.weight, blk.mlp[2].weight):
                nn.init.normal_(w, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def new_caches(self) -> list[KVCache]:
        return [KVCache() for _ in self.blocks]

    def roles(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Section id and atom role of every token, from the token ids alone (prefix scans).

        ``idx`` must start at the beginning of the sequence; a token's ids depend only on the
        tokens before it, so this is compatible with causal decoding.
        """
        pos = torch.arange(idx.shape[1], device=idx.device).expand_as(idx)
        none = torch.full_like(idx, -1)
        is_pred, is_term = self.tok_is_pred[idx], self.tok_is_term[idx]
        section = torch.cummax(self.tok_section[idx], dim=1).values
        last_pred = torch.cummax(torch.where(is_pred, pos, none), dim=1).values
        last_break = torch.cummax(torch.where(~is_pred & ~is_term, pos, none), dim=1).values
        in_atom = is_term & (last_pred > last_break)  # a term directly following a predicate
        arg_role = (1 + pos - last_pred).clamp(max=N_ROLES - 1)
        role = torch.where(is_pred, torch.ones_like(idx),
                           torch.where(in_atom, arg_role, torch.zeros_like(idx)))
        return section, role

    def forward(self, idx: torch.Tensor, caches: list[KVCache] | None = None,
                roles: tuple[torch.Tensor, torch.Tensor] | None = None) -> torch.Tensor:
        """Returns final hidden states (B, T, D); apply ``logits`` to the positions you need.

        ``roles``: (section, role) ids for ``idx``; computed from ``idx`` when omitted, which is
        only correct if ``idx`` is the whole sequence (training, prefill).
        """
        T = idx.shape[1]
        off = caches[0].length if caches is not None else 0
        assert off + T <= self.cfg.max_len, "sequence longer than the RoPE table"
        cos, sin = self.cos[off:off + T], self.sin[off:off + T]
        x = self.emb(idx)
        if self.cfg.role_emb:
            section, role = roles if roles is not None else self.roles(idx)
            x = x + self.sec_emb(section) + self.role_emb(role)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, caches[i] if caches is not None else None)
        return self.ln_f(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.head(h)

    def query_logit(self, h: torch.Tensor) -> torch.Tensor:
        """Log-odds that the query atom ending at each given position is true."""
        return self.qhead(h).squeeze(-1)


def load_state_compat(model: GPT, sd: dict) -> bool:
    """Load a state dict, also from older checkpoints. Returns True if anything was adapted.

    Older checkpoints may have fewer embedding rows (tokens/sections appended later; new rows
    get a fresh init) and no query head (left at its init).
    """
    own, sd, adapted = model.state_dict(), dict(sd), False
    for k in ("emb.weight", "sec_emb.weight"):
        if k in sd and k in own and sd[k].shape[0] < own[k].shape[0]:
            extra = torch.randn(own[k].shape[0] - sd[k].shape[0], own[k].shape[1]) * 0.02
            sd[k] = torch.cat([sd[k], extra.to(sd[k].dtype)])
            adapted = True
    if "head.weight" in sd:
        sd["head.weight"] = sd["emb.weight"]  # tied
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if set(missing) - {"qhead.weight", "qhead.bias"} or unexpected:
        raise RuntimeError(f"incompatible checkpoint: missing {missing}, unexpected {unexpected}")
    return adapted or bool(missing)


@torch.no_grad()
def generate(model: GPT, prompt: list[int], n: int, max_new: int, temperature: float,
             eos_id: int, device) -> list[list[int]]:
    """Sample ``n`` continuations of one prompt (greedy if temperature <= 0).

    The prompt is encoded once and its KV cache shared across the n samples.
    Each returned sequence ends with eos unless it was truncated.
    """
    caches = model.new_caches()
    prompt_t = torch.tensor([prompt], device=device)
    h = model(prompt_t, caches)
    for c in caches:
        c.k, c.v = c.k.expand(n, -1, -1, -1), c.v.expand(n, -1, -1, -1)
    logits = model.logits(h[:, -1]).float().expand(n, -1)
    out = torch.zeros(n, 0, dtype=torch.long, device=device)

    def step_roles():  # role ids of the newest token, derived from the full history
        if not model.cfg.role_emb:
            return None
        section, role = model.roles(torch.cat([prompt_t.expand(n, -1), out], 1))
        return section[:, -1:], role[:, -1:]

    done = torch.zeros(n, dtype=torch.bool, device=device)
    for _ in range(min(max_new, model.cfg.max_len - len(prompt))):
        if temperature <= 0:
            tok = logits.argmax(-1)
        else:
            tok = torch.multinomial(F.softmax(logits / temperature, -1), 1).squeeze(1)
        tok = torch.where(done, torch.full_like(tok, eos_id), tok)
        out = torch.cat([out, tok[:, None]], 1)
        done |= tok == eos_id
        if bool(done.all()) or caches[0].length + 1 > model.cfg.max_len:
            break
        logits = model.logits(model(tok[:, None], caches, step_roles())[:, -1]).float()
    seqs = []
    for row in out.tolist():
        seqs.append(row[:row.index(eos_id) + 1] if eos_id in row else row)
    return seqs
