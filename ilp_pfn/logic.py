"""Function-free Datalog over a finite, closed domain, evaluated bottom-up with dense tensors.

Conventions
-----------
* Constants are the integers ``0..n-1``.
* Predicates are integers with arity 1 or 2.  The *extension* of a predicate is a numpy
  bool array of shape ``(n,)`` or ``(n, n)``.
* Variables inside a clause are small non-negative integers.
* Negation is negation-as-failure under stratified semantics.  Because the domain is finite
  and closed, ``not p(X)`` is just the complement of p's final extension.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Literal:
    pred: int
    args: tuple[int, ...]
    neg: bool = False


@dataclass(frozen=True)
class Clause:
    head: int
    head_args: tuple[int, ...]
    body: tuple[Literal, ...]

    def variables(self) -> list[int]:
        vs = set(self.head_args)
        for lit in self.body:
            vs.update(lit.args)
        return sorted(vs)


Extensions = dict[int, np.ndarray]


class NotStratifiable(ValueError):
    pass


# --------------------------------------------------------------------------- static checks
def clause_problems(clause: Clause, arities: dict[int, int]) -> list[str]:
    """Well-formedness checks. Returns problem names; an empty list means the clause is fine."""
    problems = []
    if not clause.body:
        problems.append("empty_body")
    if len(set(clause.head_args)) != len(clause.head_args):
        problems.append("repeated_head_var")
    if arities.get(clause.head) != len(clause.head_args):
        problems.append("head_arity")
    for lit in clause.body:
        if lit.pred not in arities:
            problems.append("unknown_pred")
        elif arities[lit.pred] != len(lit.args):
            problems.append("arity")
    # Safety: head variables and variables of negated literals occur in a positive literal.
    pos_vars = {a for lit in clause.body if not lit.neg for a in lit.args}
    needed = set(clause.head_args) | {a for lit in clause.body if lit.neg for a in lit.args}
    if not needed <= pos_vars:
        problems.append("unsafe")
    keys = [(lit.pred, lit.args) for lit in clause.body]
    if len(set(keys)) != len(keys):
        problems.append("duplicate_literal")
    return problems


def is_connected(clause: Clause, linked_head: bool = False) -> bool:
    """Every variable is linked through shared body literals to some head variable.

    With ``linked_head`` the head variables must also be linked to each other, which excludes
    "cartesian product" clauses such as ``t(X,Y) :- p(X), q(Y)``.
    """
    parent = {v: v for v in clause.variables()}

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for lit in clause.body:
        for a, b in zip(lit.args, lit.args[1:]):
            parent[find(a)] = find(b)
    head_roots = {find(v) for v in clause.head_args}
    if linked_head and len(head_roots) > 1:
        return False
    return all(find(v) in head_roots for v in parent)


# --------------------------------------------------------------------------- evaluation
def _literal_array(e: np.ndarray, axes: list[int], k: int) -> np.ndarray:
    """Place a unary/binary extension on the given variable axes of a k-dim grid (broadcastable)."""
    shape = [1] * k
    if len(axes) == 1:
        shape[axes[0]] = e.shape[0]
        return e.reshape(shape)
    a, b = axes
    if a == b:  # p(X, X)
        shape[a] = e.shape[0]
        return np.diagonal(e).reshape(shape)
    if a > b:
        e, a, b = e.T, b, a
    shape[a], shape[b] = e.shape
    return e.reshape(shape)


def clause_extension(clause: Clause, ext: Extensions, n: int) -> np.ndarray:
    """Set of head tuples derivable by one application of ``clause`` given extensions ``ext``."""
    vars_ = clause.variables()
    idx = {v: i for i, v in enumerate(vars_)}
    k = len(vars_)
    acc = np.ones((n,) * k, dtype=bool)
    for lit in clause.body:
        arr = _literal_array(ext[lit.pred], [idx[a] for a in lit.args], k)
        acc &= ~arr if lit.neg else arr
    head_axes = [idx[v] for v in clause.head_args]
    other = tuple(i for i in range(k) if i not in head_axes)
    if other:
        acc = acc.any(axis=other)
    remaining = sorted(head_axes)
    return np.transpose(acc, [remaining.index(a) for a in head_axes])


def stratify(clauses: list[Clause]) -> list[list[int]]:
    """Group head predicates into strata (lowest first). Raises NotStratifiable."""
    heads = sorted({c.head for c in clauses})
    s = {p: 0 for p in heads}
    while True:
        changed = False
        for c in clauses:
            for lit in c.body:
                if lit.pred in s:
                    need = s[lit.pred] + (1 if lit.neg else 0)
                    if s[c.head] < need:
                        s[c.head] = need
                        changed = True
        if any(v > len(heads) for v in s.values()):
            raise NotStratifiable("negation through recursion")
        if not changed:
            break
    return [[p for p in heads if s[p] == level] for level in sorted(set(s.values()))]


def evaluate_program(ext_in: Extensions, clauses: list[Clause], arities: dict[int, int],
                     n: int) -> Extensions:
    """Least (stratified) model of ``clauses`` on top of the given input extensions."""
    ext = dict(ext_in)
    heads = {c.head for c in clauses}
    clash = heads & set(ext_in)
    if clash:
        raise ValueError(f"predicates {clash} are both given and defined by rules")
    for p in heads:
        ext[p] = np.zeros((n,) * arities[p], dtype=bool)
    for stratum in stratify(clauses):
        members = set(stratum)
        rules = [c for c in clauses if c.head in members]
        changed = True
        while changed:  # naive fixpoint; monotone within a stratum
            changed = False
            for c in rules:
                new = ext[c.head] | clause_extension(c, ext, n)
                if not np.array_equal(new, ext[c.head]):
                    ext[c.head] = new
                    changed = True
    return ext


def true_tuples(e: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(int(i) for i in t) for t in np.argwhere(e)]


# --------------------------------------------------------------------------- pretty printing
VAR_NAMES = "XYZWUVST"


def format_clause(c: Clause, name=lambda p: f"p{p}") -> str:
    v = lambda a: VAR_NAMES[a] if a < len(VAR_NAMES) else f"V{a}"  # noqa: E731
    atom = lambda p, args: f"{name(p)}({','.join(v(a) for a in args)})"  # noqa: E731
    body = ", ".join(("not " if l.neg else "") + atom(l.pred, l.args) for l in c.body)
    return f"{atom(c.head, c.head_args)} :- {body}."
