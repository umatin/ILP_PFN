"""Evaluation splits: the training prior plus controlled distribution shifts."""
from __future__ import annotations

import zlib

import numpy as np

from .prior import PriorConfig, Task, sample_task
from .tokenizer import Encoded, encode

SPLITS: dict[str, dict] = {
    "iid": {},
    "more_constants": dict(n_const=(12, 16)),
    "longer_bodies": dict(body_len=(4, 4)),
    "more_clauses": dict(n_clauses=(3, 3)),
    "deeper_bk": dict(n_bk=(4, 5), p_require_derived=1.0),
    "denser_facts": dict(density=(0.4, 0.6)),
}


def split_config(name: str, base: PriorConfig = PriorConfig()) -> PriorConfig:
    return base.replace(**SPLITS[name])


def split_seed(name: str, seed: int = 0) -> list[int]:
    # disjoint from training streams, which are seeded with (seed, step, worker) triples
    return [10**9 + seed, zlib.crc32(name.encode())]


def make_items(cfg: PriorConfig, n: int, seed, max_len: int | None = None
               ) -> list[tuple[Task, Encoded]]:
    """Deterministic set of (task, encoding) pairs."""
    rng = np.random.default_rng(seed)
    items = []
    while len(items) < n:
        task = sample_task(rng, cfg)
        enc = encode(task, rng)
        if max_len is None or len(enc.ids) <= max_len:
            items.append((task, enc))
    return items
