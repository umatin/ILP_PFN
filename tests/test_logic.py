import itertools

import numpy as np
import pytest

from ilp_pfn.logic import (Clause, Literal, NotStratifiable, clause_extension, clause_problems,
                           evaluate_program, is_connected, stratify)
from ilp_pfn.prior import PriorConfig, sample_task


# ---------------------------------------------------------------- independent reference evaluator
def brute_clause(c: Clause, ext, n):
    vs = c.variables()
    out = np.zeros((n,) * len(c.head_args), dtype=bool)
    for vals in itertools.product(range(n), repeat=len(vs)):
        a = dict(zip(vs, vals))
        if all(bool(ext[l.pred][tuple(a[x] for x in l.args)]) != l.neg for l in c.body):
            out[tuple(a[x] for x in c.head_args)] = True
    return out


def brute_program(ext_in, clauses, arities, n):
    ext = dict(ext_in)
    for c in clauses:
        ext.setdefault(c.head, np.zeros((n,) * arities[c.head], dtype=bool))
    for stratum in stratify(clauses):
        changed = True
        while changed:
            changed = False
            for c in clauses:
                if c.head in stratum:
                    new = ext[c.head] | brute_clause(c, ext, n)
                    changed |= not np.array_equal(new, ext[c.head])
                    ext[c.head] = new
    return ext


def random_world(rng, n, arities):
    return {p: rng.random((n,) * a) < 0.4 for p, a in arities.items()}


# ---------------------------------------------------------------- hand-written cases
def test_grandparent():
    n = 4
    parent = np.zeros((n, n), bool)
    parent[0, 1] = parent[1, 2] = parent[2, 3] = True
    c = Clause(9, (0, 1), (Literal(0, (0, 2)), Literal(0, (2, 1))))
    gp = clause_extension(c, {0: parent}, n)
    assert {tuple(t) for t in np.argwhere(gp)} == {(0, 2), (1, 3)}


def test_head_argument_order():
    n = 3
    e = np.zeros((n, n), bool)
    e[0, 2] = True
    inv = Clause(9, (1, 0), (Literal(0, (0, 1)),))  # t(Y, X) :- p(X, Y)
    assert clause_extension(inv, {0: e}, n)[2, 0]
    assert clause_extension(inv, {0: e}, n).sum() == 1


def test_transitive_closure_matches_reachability():
    rng = np.random.default_rng(0)
    for _ in range(20):
        n = int(rng.integers(3, 8))
        edge = rng.random((n, n)) < 0.25
        tc = [Clause(1, (0, 1), (Literal(0, (0, 1)),)),
              Clause(1, (0, 1), (Literal(0, (0, 2)), Literal(1, (2, 1))))]
        got = evaluate_program({0: edge}, tc, {0: 2, 1: 2}, n)[1]
        reach = edge.copy()
        for k in range(n):  # Warshall
            reach |= reach[:, [k]] & reach[[k], :]
        assert np.array_equal(got, reach)


def test_stratified_negation():
    n = 4
    node = np.ones(n, bool)
    edge = np.zeros((n, n), bool)
    edge[0, 1] = edge[1, 2] = True
    arities = {0: 1, 1: 2, 2: 1, 3: 1}
    prog = [Clause(2, (0,), (Literal(1, (0, 1)),)),                    # has_out(X) :- edge(X,Y)
            Clause(3, (0,), (Literal(0, (0,)), Literal(2, (0,), True)))]  # sink(X) :- node, not has_out
    ext = evaluate_program({0: node, 1: edge}, prog, arities, n)
    assert ext[3].tolist() == [False, False, True, True]


def test_not_stratifiable():
    prog = [Clause(1, (0,), (Literal(0, (0,)), Literal(1, (0,), True)))]
    with pytest.raises(NotStratifiable):
        stratify(prog)


def test_static_checks():
    ar = {0: 1, 1: 2, 9: 2}
    assert clause_problems(Clause(9, (0, 1), (Literal(1, (0, 1)),)), ar) == []
    assert "unsafe" in clause_problems(Clause(9, (0, 1), (Literal(0, (0,)),)), ar)
    assert "unsafe" in clause_problems(
        Clause(9, (0, 1), (Literal(1, (0, 1)), Literal(0, (2,), True))), ar)
    assert "arity" in clause_problems(Clause(9, (0, 1), (Literal(1, (0,)),)), ar)
    assert "unknown_pred" in clause_problems(Clause(9, (0, 1), (Literal(5, (0, 1)),)), ar)
    assert not is_connected(Clause(9, (0,), (Literal(0, (0,)), Literal(0, (2,)))))
    cartesian = Clause(9, (0, 1), (Literal(0, (0,)), Literal(0, (1,))))
    chain = Clause(9, (0, 1), (Literal(1, (0, 2)), Literal(1, (2, 1))))
    assert is_connected(cartesian) and not is_connected(cartesian, linked_head=True)
    assert is_connected(chain, linked_head=True)


# ---------------------------------------------------------------- randomised comparison
def test_clause_extension_matches_brute_force():
    rng = np.random.default_rng(1)
    arities = {0: 1, 1: 1, 2: 2, 3: 2}
    for _ in range(300):
        n = int(rng.integers(2, 6))
        ext = random_world(rng, n, arities)
        head_arity = int(rng.integers(1, 3))
        n_vars = head_arity + int(rng.integers(0, 3))
        body = []
        for _ in range(int(rng.integers(1, 5))):
            p = int(rng.integers(4))
            body.append(Literal(p, tuple(int(a) for a in rng.integers(0, n_vars, arities[p])),
                                bool(rng.random() < 0.3)))
        c = Clause(9, tuple(rng.permutation(n_vars)[:head_arity].tolist()), tuple(body))
        assert np.array_equal(clause_extension(c, ext, n), brute_clause(c, ext, n)), c


@pytest.mark.parametrize("negation", [False, True])
def test_prior_programs_match_brute_force(negation):
    rng = np.random.default_rng(2)
    cfg = PriorConfig(n_const=(3, 5), allow_negation=negation)
    for _ in range(40):
        t = sample_task(rng, cfg)
        ref = brute_program(t.base_ext, t.bk + t.target_clauses, t.arities, t.n)
        got = evaluate_program(t.base_ext, t.bk + t.target_clauses, t.arities, t.n)
        for p in t.derived_preds + [t.target]:
            assert np.array_equal(got[p], ref[p])
        assert np.array_equal(got[t.target], t.target_ext)
