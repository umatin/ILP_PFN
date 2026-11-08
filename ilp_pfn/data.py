"""Infinite stream of freshly sampled tasks (one new task per training example) and batching."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .prior import PriorConfig, sample_task
from .tokenizer import VOCAB, encode


class TaskStream(IterableDataset):
    """Tasks from one prior, or from a weighted mixture of priors (curriculum replay)."""

    def __init__(self, priors: PriorConfig | list[PriorConfig], max_len: int,
                 seed: tuple[int, ...], weights: list[float] | None = None):
        self.priors = [priors] if isinstance(priors, PriorConfig) else list(priors)
        w = np.asarray(weights if weights is not None else [1.0] * len(self.priors), float)
        self.weights = w / w.sum()
        self.max_len, self.seed = max_len, tuple(seed)

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([*self.seed, info.id if info else 0])
        while True:
            k = int(rng.choice(len(self.priors), p=self.weights)) if len(self.priors) > 1 else 0
            task = sample_task(rng, self.priors[k])
            enc = encode(task, rng)
            if len(enc.ids) <= self.max_len:
                yield enc.ids, enc.rule_start


def collate(batch: list[tuple[list[int], int]], multiple: int = 1):
    """Right-pad; returns inputs x, targets y and a mask selecting rule tokens (incl. <eos>)."""
    L = max(len(ids) for ids, _ in batch) - 1
    L = -(-L // multiple) * multiple
    x = torch.full((len(batch), L), VOCAB.pad_id, dtype=torch.long)
    y = torch.full((len(batch), L), VOCAB.pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.bool)
    for b, (ids, rule_start) in enumerate(batch):
        t = torch.tensor(ids, dtype=torch.long)
        n = len(ids) - 1
        x[b, :n], y[b, :n] = t[:-1], t[1:]
        mask[b, rule_start - 1:n] = True  # y[j] = ids[j+1] is a rule token iff j+1 >= rule_start
    return x, y, mask
