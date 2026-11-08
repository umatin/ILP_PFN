"""Train the ILP-PFN with teacher forcing on rules sampled from the prior.

    python -m ilp_pfn.train --out_dir runs/pilot --steps 2000 --batch_size 32   # local smoke test
    python -m ilp_pfn.train --out_dir runs/base --compile 1                     # A40 defaults

Resumes automatically from ``<out_dir>/ckpt.pt``.
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

from .data import TaskStream, collate
from .evaluate import autocast, evaluate_items, pick_device, teacher_forced
from .model import GPT, ModelConfig
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
    # data
    prior: str = ""              # JSON dict of PriorConfig overrides, e.g. '{"allow_negation": true}'
    max_len: int = 1024
    num_workers: int = 8
    pad_multiple: int = 64       # fewer distinct shapes for torch.compile
    # optimisation
    steps: int = 100_000
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
    n_val: int = 512             # teacher-forced validation tasks
    n_val_decode: int = 200      # greedy-decoded validation tasks


def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    for f in dataclasses.fields(TrainConfig):
        ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return TrainConfig(**vars(ap.parse_args()))


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
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
    mcfg = ModelConfig(vocab_size=len(VOCAB), d_model=cfg.d_model, n_layers=cfg.n_layers,
                       n_heads=cfg.n_heads, d_ff=4 * cfg.d_model, max_len=max(2048, cfg.max_len))
    model = GPT(mcfg).to(device)
    opt = make_optimizer(model, cfg, device)

    step, ckpt = 0, out / "ckpt.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step = ck["step"]
        print(f"resumed from step {step}")
    (out / "config.json").write_text(json.dumps(dict(train=dataclasses.asdict(cfg),
                                                     model=dataclasses.asdict(mcfg),
                                                     prior=prior.to_dict()), indent=1))
    print(f"device {device}  params {model.n_params() / 1e6:.2f}M")

    val_items = make_items(prior, cfg.n_val, split_seed("iid"), max_len=cfg.max_len)
    # new data stream after every (re)start, so a resumed run never repeats tasks
    loader = DataLoader(TaskStream(prior, cfg.max_len, seed=(cfg.seed, step)),
                        batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                        collate_fn=partial(collate, multiple=cfg.pad_multiple),
                        pin_memory=device.type == "cuda",
                        persistent_workers=cfg.num_workers > 0,
                        prefetch_factor=4 if cfg.num_workers > 0 else None)
    fwd = torch.compile(model, dynamic=False) if cfg.compile else model
    log = open(out / "metrics.jsonl", "a")
    batches = iter(loader)
    t_last, data_time, tokens, losses = time.time(), 0.0, 0, []

    while step < cfg.steps:
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
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
            rec = dict(step=step, loss=sum(losses) / len(losses), lr=lr_at(step, cfg),
                       grad_norm=float(gnorm), tok_per_s=tokens / dt,
                       tasks_per_s=cfg.log_every * cfg.batch_size / dt,
                       data_wait_frac=data_time / dt)
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in rec.items()), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t_last, data_time, tokens, losses = time.time(), 0.0, 0, []

        if step % cfg.eval_every == 0 or step == cfg.steps:
            t0 = time.time()
            ev = teacher_forced(model, val_items, device)
            summary, _ = evaluate_items(model, val_items[:cfg.n_val_decode], device, k=0)
            ev.update({k: v for k, v in summary.items() if k.startswith(("greedy", "majority"))})
            ev.update(step=step, eval_seconds=time.time() - t0)
            print("EVAL " + json.dumps(ev), flush=True)
            log.write(json.dumps(dict(eval=ev)) + "\n")
            log.flush()
            t_last = time.time()

        if step % cfg.ckpt_every == 0 or step == cfg.steps:
            save(ckpt, model=model.state_dict(), opt=opt.state_dict(), step=step,
                 model_cfg=dataclasses.asdict(mcfg), prior_cfg=prior.to_dict(),
                 train_cfg=dataclasses.asdict(cfg))
    log.close()


if __name__ == "__main__":
    main()
