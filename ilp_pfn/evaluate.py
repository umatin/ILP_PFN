"""Evaluate a checkpoint: greedy and sampled rules, checked against each task.

    python -m ilp_pfn.evaluate --ckpt runs/x/ckpt.pt --splits iid,more_constants --n 500 --k 16
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import collate
from .logic import format_clause
from .model import GPT, ModelConfig, generate, load_state_compat
from .prior import PriorConfig, Task
from .splits import SPLITS, make_items, split_config, split_seed
from .tokenizer import VOCAB, Encoded, encode_queries
from .verify import Check, check_rule, majority_query_acc


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def teacher_forced(model: GPT, items, device, batch_size: int = 64) -> dict:
    """Loss / token accuracy on the rule tokens, and fraction of rules reproduced exactly."""
    model.eval()
    tot_loss = tot_tok = tot_correct = seq_ok = 0
    for i in range(0, len(items), batch_size):
        batch = [(enc.ids, enc.rule_start) for _, enc in items[i:i + batch_size]]
        x, y, mask = (t.to(device) for t in collate(batch)[:3])
        with autocast(device):
            logits = model.logits(model(x)).float()
        loss = F.cross_entropy(logits[mask], y[mask], reduction="sum")
        correct = (logits.argmax(-1) == y) & mask
        tot_loss += loss.item()
        tot_tok += int(mask.sum())
        tot_correct += int(correct.sum())
        seq_ok += int(((correct | ~mask).all(1)).sum())
    return dict(tf_loss=tot_loss / tot_tok, tf_token_acc=tot_correct / tot_tok,
                tf_rule_acc=seq_ok / len(items))


@torch.no_grad()
def query_accuracy(model: GPT, items, device, n_queries: int = 16, batch_size: int = 64,
                   seed: int = 0) -> dict:
    """Accuracy of the yes/no query head per query kind (target / base / derived facts)."""
    model.eval()
    rng = np.random.default_rng(seed)  # same queries at every evaluation
    hits, totals = Counter(), Counter()
    for i in range(0, len(items), batch_size):
        qs = [encode_queries(task, enc, rng, n_queries) for task, enc in items[i:i + batch_size]]
        x, _, _, qmask, qlab = collate([(q.ids, len(q.ids), q.positions, q.labels) for q in qs])
        x, qmask = x.to(device), qmask.to(device)
        with autocast(device):
            pred = (model.query_logit(model(x)[qmask]).float() > 0).cpu()
        correct = pred == qlab[qmask.cpu()].bool()
        kinds = [k for q in qs for k in q.kinds]  # same order as the boolean mask (row-major)
        for k, c in zip(kinds, correct.tolist()):
            hits[k] += c
            totals[k] += 1
    return {f"q_acc_{k}": hits[k] / totals[k] for k in sorted(totals)}


def _select(checks: list[Check]) -> Check | None:
    """Pick the shortest rule consistent with the shown examples (ties: first sampled)."""
    ok = [c for c in checks if c.valid and c.consistent]
    return min(ok, key=lambda c: c.n_literals) if ok else None


@torch.no_grad()
def evaluate_items(model: GPT, items: list[tuple[Task, Encoded]], device, k: int = 0,
                   temperature: float = 1.0, max_new: int = 128) -> tuple[dict, list[dict]]:
    """Greedy decoding always; with k > 0 also k samples, reported at k' in {1, 4, 16, ..} <= k."""
    model.eval()
    ks = [x for x in (1, 4, 16, 64) if x <= k]
    records = []
    for task, enc in items:
        prompt = enc.ids[:enc.rule_start]
        with autocast(device):
            greedy = generate(model, prompt, 1, max_new, 0.0, VOCAB.eos_id, device)[0]
            samples = generate(model, prompt, k, max_new, temperature, VOCAB.eos_id,
                               device) if k else []
        g = check_rule(task, enc, VOCAB.toks(greedy))
        rec = dict(meta=task.meta, greedy=g.__dict__, greedy_tokens=" ".join(VOCAB.toks(greedy)),
                   majority_query_acc=majority_query_acc(task))
        checks = [check_rule(task, enc, VOCAB.toks(s)) for s in samples]
        for kk in ks:
            sel = _select(checks[:kk])
            rec[f"any_consistent@{kk}"] = sel is not None
            rec[f"selected_semantic@{kk}"] = bool(sel and sel.semantic)
            rec[f"selected_query_acc@{kk}"] = sel.query_acc if sel else None
        records.append(rec)
    return summarize(records, ks), records


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize(records: list[dict], ks: list[int] = ()) -> dict:
    g = [r["greedy"] for r in records]
    # held-out accuracy: invalid rules fall back to the majority-class guess
    qa = [x["query_acc"] if x["valid"] else r["majority_query_acc"] for x, r in zip(g, records)]
    s = dict(n=len(records),
             greedy_valid=_mean([x["valid"] for x in g]),
             greedy_consistent=_mean([x["consistent"] for x in g]),
             greedy_semantic=_mean([x["semantic"] for x in g]),
             greedy_exact=_mean([x["exact"] for x in g]),
             greedy_query_acc=_mean(qa),
             majority_query_acc=_mean([r["majority_query_acc"] for r in records]),
             greedy_errors=dict(Counter(x["error"] for x in g if not x["valid"])))
    for kk in ks:
        for key in ("any_consistent", "selected_semantic"):
            s[f"{key}@{kk}"] = _mean([r[f"{key}@{kk}"] for r in records])
        s[f"selected_query_acc@{kk}"] = _mean(
            [r[f"selected_query_acc@{kk}"] if r[f"selected_query_acc@{kk}"] is not None
             else r["majority_query_acc"] for r in records])
    # breakdowns of greedy consistency / semantic correctness
    for key in ("n_clauses", "target_arity", "uses_derived", "reasoning_depth", "n_literals"):
        groups = defaultdict(list)
        for r in records:
            groups[r["meta"][key]].append(r["greedy"])
        s[f"by_{key}"] = {str(v): dict(n=len(xs), consistent=_mean([x["consistent"] for x in xs]),
                                       semantic=_mean([x["semantic"] for x in xs]))
                          for v, xs in sorted(groups.items())}
    return s


def show_example(task: Task, enc: Encoded, greedy_tokens: str) -> str:
    name = lambda p: enc.pred_tok[p]  # noqa: E731
    lines = [f"  n={task.n} pos={len(task.pos)} neg={len(task.neg)}"]
    lines += [f"  bk:    {format_clause(c, name)}" for c in task.bk]
    lines += [f"  true:  {format_clause(c, name)}" for c in task.target_clauses]
    lines += [f"  model: {greedy_tokens}"]
    return "\n".join(lines)


def load_checkpoint(path, device) -> tuple[GPT, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["model_cfg"], vocab_size=len(VOCAB))  # vocab may have grown since
    model = GPT(ModelConfig(**cfg))
    load_state_compat(model, ck["model"])
    return model.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    model, ck = load_checkpoint(args.ckpt, device)
    prior = PriorConfig.from_dict(ck["prior_cfg"])
    out = Path(args.out or Path(args.ckpt).parent / "eval")
    out.mkdir(parents=True, exist_ok=True)
    print(f"step {ck['step']}  params {model.n_params() / 1e6:.1f}M  device {device}")
    for split in args.splits.split(","):
        t0 = time.time()
        items = make_items(split_config(split, prior), args.n, split_seed(split),
                           max_len=model.cfg.max_len - 64)
        summary, records = evaluate_items(model, items, device, k=args.k,
                                          temperature=args.temperature)
        summary.update(teacher_forced(model, items, device))
        if model.cfg.query_head:
            summary.update(query_accuracy(model, items, device))
        summary["seconds"] = time.time() - t0
        (out / f"{split}.json").write_text(json.dumps(summary, indent=1))
        with open(out / f"{split}_records.jsonl", "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        keys = ["greedy_valid", "greedy_consistent", "greedy_semantic", "greedy_exact",
                "greedy_query_acc", "majority_query_acc"] + [k for k in summary if "@" in k]
        keys += [k for k in summary if k.startswith("q_acc_")]
        print(f"\n== {split} ({summary['seconds']:.0f}s)")
        print("  " + "  ".join(f"{k}={summary[k]:.3f}" for k in keys if summary[k] is not None))
        print(f"  errors: {summary['greedy_errors']}")
        for (task, enc), r in list(zip(items, records))[:2]:
            print(show_example(task, enc, r["greedy_tokens"]))


if __name__ == "__main__":
    main()
