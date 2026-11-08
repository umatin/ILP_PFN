"""Infinite stream of freshly sampled tasks (one new task per training example) and batching.

Two kinds of training sequences:
  * rule sequences:  prompt <rule> canonical rule <eos>    -> next-token loss on the rule
  * query sequences: prompt <query> atom ; atom ; ... <eos> -> yes/no loss at each query atom
    (only when ``query_frac`` > 0; see ``tokenizer.encode_queries``)
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .prior import PriorConfig, sample_task
from .tokenizer import VOCAB, encode, encode_queries


class TaskStream(IterableDataset):
    """Tasks from one prior, or from a weighted mixture of priors (curriculum replay)."""

    def __init__(self, priors: PriorConfig | list[PriorConfig], max_len: int,
                 seed: tuple[int, ...], weights: list[float] | None = None,
                 query_frac: float = 0.0, n_queries: int = 16):
        self.priors = [priors] if isinstance(priors, PriorConfig) else list(priors)
        w = np.asarray(weights if weights is not None else [1.0] * len(self.priors), float)
        self.weights = w / w.sum()
        self.max_len, self.seed = max_len, tuple(seed)
        self.query_frac, self.n_queries = query_frac, n_queries

    def __iter__(self):
        info = get_worker_info()
        rng = np.random.default_rng([*self.seed, info.id if info else 0])
        while True:
            k = int(rng.choice(len(self.priors), p=self.weights)) if len(self.priors) > 1 else 0
            task = sample_task(rng, self.priors[k])
            enc = encode(task, rng)
            if self.query_frac > 0 and rng.random() < self.query_frac:
                q = encode_queries(task, enc, rng, self.n_queries)
                if len(q.ids) <= self.max_len:
                    yield q.ids, len(q.ids), q.positions, q.labels  # no rule tokens
            elif len(enc.ids) <= self.max_len:
                yield enc.ids, enc.rule_start, [], []


def collate(batch, multiple: int = 1):
    """Right-pad a batch of (ids, rule_start[, query_positions, query_labels]).

    Returns inputs x, targets y, ``mask`` selecting rule tokens (incl. <eos>) in y, and
    ``qmask`` / ``qlab`` marking query positions in x and their yes/no labels.
    """
    L = max(len(item[0]) for item in batch) - 1
    L = -(-L // multiple) * multiple
    x = torch.full((len(batch), L), VOCAB.pad_id, dtype=torch.long)
    y = torch.full((len(batch), L), VOCAB.pad_id, dtype=torch.long)
    mask = torch.zeros((len(batch), L), dtype=torch.bool)
    qmask = torch.zeros((len(batch), L), dtype=torch.bool)
    qlab = torch.zeros((len(batch), L), dtype=torch.float)
    for b, item in enumerate(batch):
        ids, rule_start = item[0], item[1]
        t = torch.tensor(ids, dtype=torch.long)
        n = len(ids) - 1
        x[b, :n], y[b, :n] = t[:-1], t[1:]
        mask[b, rule_start - 1:n] = True  # y[j] = ids[j+1] is a rule token iff j+1 >= rule_start
        if len(item) > 2 and item[2]:
            pos = torch.tensor(item[2], dtype=torch.long)
            qmask[b, pos] = True
            qlab[b, pos] = torch.tensor(item[3], dtype=torch.float)
    return x, y, mask, qmask, qlab
