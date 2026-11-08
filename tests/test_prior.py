import numpy as np
import pytest

from ilp_pfn.logic import Clause, clause_extension, clause_problems, evaluate_program, is_connected
from ilp_pfn.prior import PriorConfig, sample_task
from ilp_pfn.splits import SPLITS, split_config


def _check_task(t, cfg):
    # BK extensions are exactly what the BK rules entail
    ref = evaluate_program(t.base_ext, t.bk, t.arities, t.n) if t.bk else dict(t.base_ext)
    for p in t.derived_preds:
        assert np.array_equal(ref[p], t.bk_ext[p])
    # target clauses: well-formed, non-recursive, minimal, each with unique coverage
    for c in t.bk:
        assert is_connected(c, cfg.linked_head)
    exts = []
    for c in t.target_clauses:
        assert not clause_problems(c, t.arities) and is_connected(c, cfg.linked_head)
        assert all(l.pred != t.target for l in c.body)
        assert cfg.body_len[0] <= len(c.body) <= cfg.body_len[1]
        e = clause_extension(c, t.bk_ext, t.n)
        assert e.any()
        for i in range(len(c.body)):
            shorter = Clause(c.head, c.head_args, c.body[:i] + c.body[i + 1:])
            if not clause_problems(shorter, t.arities) and is_connected(shorter, cfg.linked_head):
                assert not np.array_equal(clause_extension(shorter, t.bk_ext, t.n), e)
        exts.append(e)
    assert np.array_equal(np.logical_or.reduce(exts), t.target_ext)
    for i in range(len(exts)):
        others = [exts[j] for j in range(len(exts)) if j != i]
        if others:
            assert (exts[i] & ~np.logical_or.reduce(others)).any()
    assert cfg.pos_rate[0] <= t.target_ext.mean() <= cfg.pos_rate[1]
    # examples: correctly labelled, shown and held-out disjoint, together cover everything
    assert all(t.target_ext[x] for x in t.pos + t.query_pos)
    assert not any(t.target_ext[x] for x in t.neg + t.query_neg)
    shown, held = set(t.pos + t.neg), set(t.query_pos + t.query_neg)
    assert not shown & held
    assert len(shown | held) == t.n ** t.target_arity
    assert cfg.min_pos <= len(t.pos) <= cfg.max_pos and cfg.min_neg <= len(t.neg) <= cfg.max_neg
    assert t.target_arity == 2 or t.n >= cfg.min_n_unary_target
    if t.meta["uses_derived"]:
        assert any(l.pred in t.derived_preds for c in t.target_clauses for l in c.body)


@pytest.mark.parametrize("negation", [False, True])
def test_prior_invariants(negation):
    cfg = PriorConfig(allow_negation=negation)
    rng = np.random.default_rng(0)
    for _ in range(200):
        _check_task(sample_task(rng, cfg), cfg)


@pytest.mark.parametrize("split", sorted(SPLITS))
def test_split_invariants(split):
    cfg = split_config(split)
    rng = np.random.default_rng(1)
    for _ in range(20):
        _check_task(sample_task(rng, cfg), cfg)


def test_determinism():
    a = sample_task(np.random.default_rng(5))
    b = sample_task(np.random.default_rng(5))
    assert a.target_clauses == b.target_clauses and a.pos == b.pos


def test_prior_uses_derived_predicates():
    rng = np.random.default_rng(3)
    tasks = [sample_task(rng) for _ in range(300)]
    frac = np.mean([t.meta["uses_derived"] for t in tasks])
    assert 0.3 < frac < 0.95
    assert any(t.meta["has_tc"] for t in tasks)
    assert any(t.meta["n_clauses"] == 2 for t in tasks)
