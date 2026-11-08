"""Train the ILP-PFN with teacher forcing on rules sampled from the prior.

    python -m ilp_pfn.train --out_dir runs/pilot --steps 2000 --batch_size 32     # local smoke test
    python -m ilp_pfn.train --out_dir runs/base --compile 1                       # A40 defaults
    python -m ilp_pfn.train --out_dir runs/cur --curriculum default --final_steps 50000

With ``--curriculum`` training moves through increasingly hard priors, advancing when the greedy
consistency on the current level's validation set passes that level's threshold (see
``curriculum.py``). The learning rate stays constant until the last level is reached and is then
cosine-decayed over ``--final_steps``. Resumes automatically from ``<out_dir>/ckpt.pt``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .curriculum import CurriculumState, Level, load_curriculum, mixture
from .data import TaskStream, collate
from .evaluate import autocast, evaluate_items, pick_device, teacher_forced
from .model import GPT, ModelConfig, load_state_compat
from .prior import PriorConfig
from .splits import make_items, split_seed
from .tokenizer import VOCAB


@dataclass
class TrainConfig:
    out_dir: str = "runs/debug"
    seed: int = 0
    device: str = "auto"
    # model
    d_model: int = 384
    n_layers: int = 8
    n_heads: int = 6
    role_emb: int = 1            # section + atom-role embeddings (0 = plain token embeddings)
    # data
    prior: str = ""              # JSON dict of PriorConfig overrides, e.g. '{"allow_negation": true}'
    max_len: int = 1024
    num_workers: int = 8
    pad_multiple: int = 64       # fewer distinct shapes for torch.compile
    # curriculum
    curriculum: str = "none"     # "none", "default", or path to a JSON list of levels
    cur_metric: str = "greedy_consistent"
    cur_patience: int = 2        # consecutive evaluations above threshold needed to advance
    cur_min_steps: int = 1000    # minimum steps per level
    cur_max_steps: int = 30000   # advance anyway after this many steps on a level
    replay_frac: float = 0.2     # fraction of each batch drawn from earlier levels
    final_steps: int = 0         # steps on the last level (LR decay); 0 = until --steps
    # optimisation
    steps: int = 100_000         # hard cap on total steps
    batch_size: int = 256
    lr: float = 6e-4
    min_lr_frac: float = 0.1
    warmup: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    compile: int = 0
    # logging / eval
    log_every: int = 50
    eval_every: int = 2000
    ckpt_every: int = 2000
    n_val: int = 512             # teacher-forced validation tasks (per level)
    n_val_decode: int = 200      # greedy-decoded validation tasks (per level)
    eval_final: int = 1          # during a curriculum, also decode the final level's val set


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(TrainConfig):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return TrainConfig(**vars(ap.parse_args()))


def end_step(cfg: TrainConfig, final_start: int | None) -> int:
    if final_start is not None and cfg.final_steps > 0:
        return min(cfg.steps, final_start + cfg.final_steps)
    return cfg.steps


def lr_at(step: int, cfg: TrainConfig, final_start: int | None) -> float:
    """Warmup, constant until the last level starts, then cosine decay to min_lr_frac * lr."""
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    if final_start is None:
        return cfg.lr
    start = max(final_start, cfg.warmup)
    p = min(1.0, max(0.0, (step - start) / max(1, end_step(cfg, final_start) - start)))
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def make_optimizer(model: GPT, cfg: TrainConfig, device) -> torch.optim.Optimizer:
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [dict(params=decay, weight_decay=cfg.weight_decay),
              dict(params=no_decay, weight_decay=0.0)]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, 0.95),
                             fused=device.type == "cuda")


def save(path: Path, **state):
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a killed job never leaves a corrupt checkpoint


def main():
    cfg = parse_args()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    prior = PriorConfig.from_dict(json.loads(cfg.prior)) if cfg.prior else PriorConfig()
    levels = load_curriculum(cfg.curriculum, prior)
    gate_metrics = {"greedy_valid", "greedy_consistent", "greedy_semantic", "greedy_exact",
                    "greedy_query_acc", "tf_token_acc", "tf_rule_acc"}
    if cfg.cur_metric not in gate_metrics:
        raise ValueError(f"--cur_metric must be one of {sorted(gate_metrics)}")
    mcfg = ModelConfig(vocab_size=len(VOCAB), d_model=cfg.d_model, n_layers=cfg.n_layers,
                       n_heads=cfg.n_heads, d_ff=4 * cfg.d_model, max_len=max(2048, cfg.max_len),
                       role_emb=bool(cfg.role_emb))
    model = GPT(mcfg).to(device)
    opt = make_optimizer(model, cfg, device)

    step, ckpt, state = 0, out / "ckpt.pt", CurriculumState()
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        if load_state_compat(model, ck["model"]):
            print("checkpoint from an older model version: adapted; optimizer state reset")
        else:
            opt.load_state_dict(ck["opt"])
        step = ck["step"]
        if "curriculum_state" in ck:
            state = CurriculumState(**ck["curriculum_state"])
        print(f"resumed from step {step} (level {state.level})")
    state.start(levels)
    (out / "config.json").write_text(json.dumps(dict(
        train=dataclasses.asdict(cfg), model=dataclasses.asdict(mcfg), prior=prior.to_dict(),
        curriculum=[dict(name=lv.name, threshold=lv.threshold, prior=lv.prior.to_dict())
                    for lv in levels]), indent=1))
    print(f"device {device}  params {model.n_params() / 1e6:.2f}M  "
          f"levels {[lv.name for lv in levels]}")

    val_cache: dict[int, list] = {}

    def val_items(k: int):
        if k not in val_cache:
            lv: Level = levels[k]
            name = "iid" if lv.prior == prior else f"curriculum/{lv.name}"
            val_cache[k] = make_items(lv.prior, max(cfg.n_val, cfg.n_val_decode),
                                      split_seed(name), max_len=cfg.max_len)
        return val_cache[k]

    def make_batches():
        priors, weights = mixture(levels, state.level, cfg.replay_frac)
        # fresh stream per (re)start / level change, so tasks are never repeated
        stream = TaskStream(priors, cfg.max_len, seed=(cfg.seed, step), weights=weights)
        loader = DataLoader(stream,
                            batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                            collate_fn=partial(collate, multiple=cfg.pad_multiple),
                            pin_memory=device.type == "cuda",
                            persistent_workers=cfg.num_workers > 0,
                            prefetch_factor=4 if cfg.num_workers > 0 else None)
        return iter(loader)

    def checkpoint():
        save(ckpt, model=model.state_dict(), opt=opt.state_dict(), step=step,
             model_cfg=dataclasses.asdict(mcfg), prior_cfg=prior.to_dict(),
             train_cfg=dataclasses.asdict(cfg), curriculum_state=dataclasses.asdict(state))

    fwd = torch.compile(model, dynamic=False) if cfg.compile else model
    log = open(out / "metrics.jsonl", "a")
    batches = make_batches()
    t_last, data_time, tokens, losses = time.time(), 0.0, 0, []

    while step < end_step(cfg, state.final_start):
        model.train()
        lr = lr_at(step, cfg, state.final_start)
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        x, y, mask = (t.to(device, non_blocking=True) for t in next(batches))
        data_time += time.time() - t0
        with autocast(device):
            h = fwd(x)
            loss = F.cross_entropy(model.logits(h[mask]).float(), y[mask])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        losses.append(loss.item())
        tokens += x.numel()

        if step % cfg.log_every == 0:
            dt = time.time() - t_last
            rec = dict(step=step, level=state.level, loss=sum(losses) / len(losses), lr=lr,
                       grad_norm=float(gnorm), tok_per_s=tokens / dt,
                       tasks_per_s=cfg.log_every * cfg.batch_size / dt,
                       data_wait_frac=data_time / dt)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in rec.items()), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t_last, data_time, tokens, losses = time.time(), 0.0, 0, []

        last = len(levels) - 1
        if step % cfg.eval_every == 0 or step == end_step(cfg, state.final_start):
            t0 = time.time()
            items = val_items(state.level)
            ev = dict(step=step, level=state.level, level_name=levels[state.level].name)
            ev.update(teacher_forced(model, items[:cfg.n_val], device))
            summary, _ = evaluate_items(model, items[:cfg.n_val_decode], device, k=0)
            ev.update({k: v for k, v in summary.items() if k.startswith(("greedy", "majority"))})
            if state.level < last and cfg.eval_final:
                fin, _ = evaluate_items(model, val_items(last)[:cfg.n_val_decode], device, k=0)
                ev["final_level"] = {k: fin[k] for k in
                                     ("greedy_valid", "greedy_consistent", "greedy_query_acc")}
            ev["eval_seconds"] = time.time() - t0
            print("EVAL " + json.dumps(ev), flush=True)
            log.write(json.dumps(dict(eval=ev)) + "\n")
            event = state.observe(ev.get(cfg.cur_metric), step, levels, cfg.cur_min_steps,
                                  cfg.cur_max_steps, cfg.cur_patience)
            if event:
                h = state.history[-1]
                msg = dict(advance=dict(h, to_level=state.level, to_name=levels[state.level].name))
                print("ADVANCE " + json.dumps(msg["advance"]), flush=True)
                log.write(json.dumps(msg) + "\n")
                del batches  # shuts down the old workers
                batches = make_batches()
                checkpoint()
            log.flush()
            t_last = time.time()

        if step % cfg.ckpt_every == 0 or step == end_step(cfg, state.final_start):
            checkpoint()
    log.close()
    print("curriculum history:", json.dumps(state.history))


if __name__ == "__main__":
    main()
