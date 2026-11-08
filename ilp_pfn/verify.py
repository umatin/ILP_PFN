"""Checking a generated rule against a task: syntax, consistency, semantics, held-out accuracy."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .logic import NotStratifiable, evaluate_program
from .prior import Task
from .tokenizer import VOCAB, Encoded, RuleError, parse_rule, program_tokens


@dataclass
class Check:
    valid: bool                    # parses, well-formed, stratifiable
    error: str | None = None       # reason when not valid
    consistent: bool = False       # covers all shown positives and no shown negatives
    semantic: bool = False         # same extension as the true target in this world
    exact: bool = False            # same canonical program as the true target
    query_acc: float | None = None # accuracy on held-out examples (None if there are none)
    n_literals: int = 0


def check_rule(task: Task, enc: Encoded, rule_tokens: list[str]) -> Check:
    """``rule_tokens``: generated tokens after <rule>, normally ending with <eos>."""
    if "<eos>" not in rule_tokens:
        return Check(False, "no_eos")
    toks = rule_tokens[:rule_tokens.index("<eos>")]
    try:
        clauses = parse_rule(toks, task, enc)
    except RuleError as e:
        return Check(False, e.kind)
    try:
        ext = evaluate_program(task.bk_ext, clauses, task.arities, task.n)[task.target]
    except NotStratifiable:
        return Check(False, "not_stratifiable")
    consistent = all(ext[t] for t in task.pos) and not any(ext[t] for t in task.neg)
    truth = VOCAB.toks(enc.ids[enc.rule_start:-1])
    queries = [(t, True) for t in task.query_pos] + [(t, False) for t in task.query_neg]
    qa = float(np.mean([bool(ext[t]) == lab for t, lab in queries])) if queries else None
    return Check(True, None, bool(consistent), bool(np.array_equal(ext, task.target_ext)),
                 program_tokens(clauses, enc.pred_tok) == truth, qa,
                 sum(len(c.body) for c in clauses))


def majority_query_acc(task: Task) -> float | None:
    """Baseline: predict the majority label among the shown examples for every query."""
    queries = len(task.query_pos) + len(task.query_neg)
    if not queries:
        return None
    guess_pos = len(task.pos) >= len(task.neg)
    return (len(task.query_pos) if guess_pos else len(task.query_neg)) / queries
