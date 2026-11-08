"""Performance-gated curriculum over priors.

A curriculum is a list of levels. Each level is a ``PriorConfig`` (overrides on top of the base
prior) with an advancement threshold. Training stays on a level until the gating metric, measured
on a fixed validation set of *that level*, reaches the threshold on ``patience`` consecutive
evaluations (and at least ``min_steps`` were spent there). After ``max_steps`` on a level it
advances anyway; the history marks this as forced. The last level has no threshold: it is the
final training distribution, on which the learning rate is decayed.

Custom curricula: a JSON list of level dicts, e.g.
    [{"name": "tiny", "threshold": 0.9, "n_const": [4, 5], "body_len": [1, 1]}, ..., {"name": "full"}]
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .prior import PriorConfig

DEFAULT_CURRICULUM: list[dict] = [
    # one literal, no BK: learn the format, arities, and to read single facts
    dict(name="L0_single_literal", threshold=0.9, n_const=(5, 6), n_unary=(1, 1),
         n_binary=(1, 1), n_bk=(0, 0), p_tc=0.0, n_clauses=(1, 1), body_len=(1, 1),
         max_extra_vars=0),
    # conjunctions and one existential variable, still no BK
    dict(name="L1_conjunctions", threshold=0.85, n_const=(5, 7), n_unary=(1, 2),
         n_binary=(1, 2), n_bk=(0, 0), p_tc=0.0, n_clauses=(1, 1), body_len=(1, 2)),
    # background rules (incl. rare transitive closure) and disjunctions
    dict(name="L2_bk_disjunction", threshold=0.8, n_const=(5, 8), n_unary=(1, 2),
         n_binary=(1, 2), n_bk=(0, 1), p_tc=0.1, n_clauses=(1, 2), body_len=(1, 2)),
    # nearly the full prior
    dict(name="L3_near_full", threshold=0.75, n_const=(4, 9), n_bk=(0, 2), p_tc=0.2,
         body_len=(1, 3)),
    # the full (base) prior
    dict(name="L4_full"),
]


@dataclass(frozen=True)
class Level:
    name: str
    prior: PriorConfig
    threshold: float | None


def load_curriculum(spec: str, base: PriorConfig) -> list[Level]:
    """``spec``: "none" (single level = base prior), "default", or a path to a JSON file."""
    if spec in ("", "none"):
        return [Level("full", base, None)]
    raw = DEFAULT_CURRICULUM if spec == "default" else json.loads(Path(spec).read_text())
    levels = []
    for i, d in enumerate(raw):
        d = dict(d)
        name = d.pop("name", f"level{i}")
        threshold = d.pop("threshold", None)
        overrides = {k: tuple(v) if isinstance(v, list) else v for k, v in d.items()}
        levels.append(Level(name, base.replace(**overrides), threshold))
    if levels[-1].threshold is not None:
        levels[-1] = Level(levels[-1].name, levels[-1].prior, None)  # last level never advances
    return levels


def mixture(levels: list[Level], k: int, replay_frac: float):
    """Training distribution at level k: the current level plus replay of earlier levels."""
    if k == 0 or replay_frac <= 0:
        return [levels[k].prior], [1.0]
    return ([lv.prior for lv in levels[:k + 1]],
            [replay_frac / k] * k + [1.0 - replay_frac])


@dataclass
class CurriculumState:
    level: int = 0
    level_start: int = 0               # step at which the current level began
    final_start: int | None = None     # step at which the last level began
    streak: int = 0                    # consecutive evaluations above threshold
    history: list[dict] = field(default_factory=list)

    def start(self, levels: list[Level]) -> None:
        if len(levels) == 1 and self.final_start is None:
            self.final_start = 0

    def observe(self, value: float | None, step: int, levels: list[Level], min_steps: int,
                max_steps: int, patience: int) -> str | None:
        """Record one evaluation of the gating metric. Returns "passed"/"forced" on advancement."""
        if self.level >= len(levels) - 1:
            return None
        threshold = levels[self.level].threshold
        passed = value is not None and threshold is not None and value >= threshold
        self.streak = self.streak + 1 if passed else 0
        spent = step - self.level_start
        if spent < min_steps:
            return None
        if self.streak >= patience:
            event = "passed"
        elif spent >= max_steps:
            event = "forced"
        else:
            return None
        self.history.append(dict(level=self.level, name=levels[self.level].name,
                                 start=self.level_start, end=step, steps=spent,
                                 metric=value, threshold=threshold, event=event))
        self.level += 1
        self.level_start, self.streak = step, 0
        if self.level == len(levels) - 1:
            self.final_start = step
        return event
