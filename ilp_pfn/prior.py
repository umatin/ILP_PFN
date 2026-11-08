"""The prior over ILP tasks: a random world, random background rules, a random target rule.

One draw produces a ``Task``:
  1. a finite world: constants and base (extensional) facts,
  2. background knowledge (BK): rules defining derived predicates, possibly recursive
     (transitive closure), each defined from base / earlier derived predicates,
  3. a target program (1..k non-recursive clauses) over base + derived predicates,
  4. labelled examples of the target predicate (closed world), split into shown and held-out.

The model only sees base facts, BK *rules*, and shown examples; it must output the target rule.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from .logic import (Clause, Extensions, Literal, clause_extension, clause_problems,
                    evaluate_program, is_connected, true_tuples)


@dataclass(frozen=True)
class PriorConfig:
    n_const: tuple[int, int] = (4, 10)          # inclusive ranges everywhere
    n_unary: tuple[int, int] = (1, 3)           # base unary predicates
    n_binary: tuple[int, int] = (1, 3)          # base binary predicates
    density: tuple[float, float] = (0.1, 0.4)   # per-predicate fact probability
    n_bk: tuple[int, int] = (0, 3)              # non-recursive derived predicates
    bk_body_len: tuple[int, int] = (1, 3)
    p_tc: float = 0.3                           # prob. of an extra transitive-closure predicate
    target_arity: tuple[int, int] = (1, 2)
    n_clauses: tuple[int, int] = (1, 2)
    body_len: tuple[int, int] = (1, 3)
    max_extra_vars: int = 1                     # existential variables per clause
    allow_negation: bool = False
    p_neg: float = 0.25
    p_require_derived: float = 0.5              # prob. the target must use a derived predicate
    pos_rate: tuple[float, float] = (0.1, 0.9)  # fraction of target tuples that are true
    linked_head: bool = True                    # head vars must be connected through the body
    min_n_unary_target: int = 6                 # unary targets only in worlds this large
    min_pos: int = 2                            # minimum shown positives / negatives
    min_neg: int = 2
    max_pos: int = 10
    max_neg: int = 10
    show_frac: float = 0.7                      # at most this fraction of each class is shown
    max_tries: int = 2000

    @classmethod
    def from_dict(cls, d: dict) -> "PriorConfig":
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def replace(self, **kw) -> "PriorConfig":
        return dataclasses.replace(self, **kw)


@dataclass
class Task:
    n: int
    arities: dict[int, int]            # all predicates: base, derived, target
    base_preds: list[int]
    derived_preds: list[int]
    target: int
    base_ext: Extensions
    bk: list[Clause]
    bk_ext: Extensions                 # base + derived extensions
    target_clauses: list[Clause]
    target_ext: np.ndarray
    pos: list[tuple[int, ...]]         # shown examples
    neg: list[tuple[int, ...]]
    query_pos: list[tuple[int, ...]]   # held-out examples (disjoint from shown)
    query_neg: list[tuple[int, ...]]
    meta: dict = field(default_factory=dict)

    @property
    def target_arity(self) -> int:
        return self.arities[self.target]


def _ri(rng: np.random.Generator, lo_hi: tuple[int, int]) -> int:
    return int(rng.integers(lo_hi[0], lo_hi[1] + 1))


def _sample_body(rng, avail, head_arity, body_len, max_extra, allow_neg, p_neg):
    n_vars = head_arity + int(rng.integers(0, max_extra + 1))
    lits = []
    for _ in range(body_len):
        p, ar = avail[int(rng.integers(len(avail)))]
        args = tuple(int(a) for a in rng.integers(0, n_vars, size=ar))
        lits.append(Literal(p, args, bool(allow_neg and rng.random() < p_neg)))
    return tuple(lits)


def _well_formed(c: Clause, arities, linked_head: bool) -> bool:
    return not clause_problems(c, arities) and is_connected(c, linked_head)


def _is_minimal(c: Clause, e: np.ndarray, ext: Extensions, arities, n: int,
                linked_head: bool) -> bool:
    """No body literal can be dropped without changing the clause's extension in this world."""
    for i in range(len(c.body)):
        shorter = Clause(c.head, c.head_args, c.body[:i] + c.body[i + 1:])
        if not _well_formed(shorter, arities, linked_head):
            continue  # literal is needed for safety / connectivity
        if np.array_equal(clause_extension(shorter, ext, n), e):
            return False
    return True


def _sample_target_clause(rng, cfg, target, arity, avail, ext, arities, n, tries=30):
    for _ in range(tries):
        body = _sample_body(rng, avail, arity, _ri(rng, cfg.body_len), cfg.max_extra_vars,
                            cfg.allow_negation, cfg.p_neg)
        c = Clause(target, tuple(range(arity)), body)
        if not _well_formed(c, arities, cfg.linked_head):
            continue
        e = clause_extension(c, ext, n)
        if e.any() and _is_minimal(c, e, ext, arities, n, cfg.linked_head):
            return c, e
    return None


def _try_sample_task(rng: np.random.Generator, cfg: PriorConfig) -> Task | None:
    n = _ri(rng, cfg.n_const)
    arities: dict[int, int] = {}
    base_ext: Extensions = {}
    pid = 0
    for arity, count in ((1, _ri(rng, cfg.n_unary)), (2, _ri(rng, cfg.n_binary))):
        for _ in range(count):
            e = rng.random((n,) * arity) < rng.uniform(*cfg.density)
            if not e.any():
                e[tuple(rng.integers(0, n, size=arity))] = True
            arities[pid], base_ext[pid] = arity, e
            pid += 1
    base_preds = list(base_ext)

    # ---- background knowledge
    ext: Extensions = dict(base_ext)
    depth = {p: 0 for p in base_preds}
    bk: list[Clause] = []
    derived: list[int] = []
    plan = ["rule"] * _ri(rng, cfg.n_bk) + (["tc"] if rng.random() < cfg.p_tc else [])
    rng.shuffle(plan)
    for kind in plan:
        new = pid
        if kind == "tc":
            binaries = [p for p in ext if arities[p] == 2]
            b = binaries[int(rng.integers(len(binaries)))]
            arities[new] = 2
            clauses = [Clause(new, (0, 1), (Literal(b, (0, 1)),)),
                       Clause(new, (0, 1), (Literal(b, (0, 2)), Literal(new, (2, 1))))]
            e = evaluate_program(ext, clauses, arities, n)[new]
            if np.array_equal(e, ext[b]):  # b already transitive: tc(b) would be an alias
                del arities[new]
                continue
        else:
            arity = int(rng.integers(1, 3))
            arities[new] = arity
            avail = [(p, arities[p]) for p in ext]
            found = None
            for _ in range(30):
                body = _sample_body(rng, avail, arity, _ri(rng, cfg.bk_body_len),
                                    cfg.max_extra_vars, cfg.allow_negation, cfg.p_neg)
                c = Clause(new, tuple(range(arity)), body)
                if not _well_formed(c, arities, cfg.linked_head):
                    continue
                e = clause_extension(c, ext, n)
                if not e.any() or e.all():
                    continue
                if any(arities[p] == arity and np.array_equal(e, ext[p]) for p in ext):
                    continue  # alias of an existing predicate
                found = (c, e)
                break
            if found is None:
                del arities[new]
                continue
            clauses, e = [found[0]], found[1]
        ext[new] = e
        bk.extend(clauses)
        derived.append(new)
        depth[new] = 1 + max(depth[l.pred] for c in clauses for l in c.body if l.pred != new)
        pid += 1

    # ---- target program
    arity_choices = [a for a in range(cfg.target_arity[0], cfg.target_arity[1] + 1)
                     if a == 2 or n >= cfg.min_n_unary_target]
    if not arity_choices:
        return None
    target, ta = pid, arity_choices[int(rng.integers(len(arity_choices)))]
    arities[target] = ta
    avail = [(p, arities[p]) for p in ext]
    require_derived = bool(derived) and rng.random() < cfg.p_require_derived
    m = _ri(rng, cfg.n_clauses)
    for _ in range(50):
        picked = [_sample_target_clause(rng, cfg, target, ta, avail, ext, arities, n)
                  for _ in range(m)]
        if any(p is None for p in picked):
            continue
        clauses = [p[0] for p in picked]
        exts = [p[1] for p in picked]
        if require_derived and not any(l.pred in derived for c in clauses for l in c.body):
            continue
        target_ext = np.logical_or.reduce(exts)
        if not cfg.pos_rate[0] <= target_ext.mean() <= cfg.pos_rate[1]:
            continue
        n_true = int(target_ext.sum())
        if n_true < cfg.min_pos or target_ext.size - n_true < cfg.min_neg:
            continue  # not enough evidence can be shown
        # every clause must cover some positive that no other clause covers
        if m > 1 and not all(
                (exts[i] & ~np.logical_or.reduce([exts[j] for j in range(m) if j != i])).any()
                for i in range(m)):
            continue
        break
    else:
        return None

    # ---- examples
    P, N = true_tuples(target_ext), true_tuples(~target_ext)
    rng.shuffle(P)
    rng.shuffle(N)
    kp = min(cfg.max_pos, max(cfg.min_pos, int(cfg.show_frac * len(P))))
    kn = min(cfg.max_neg, max(cfg.min_neg, int(cfg.show_frac * len(N))))

    used = {l.pred for c in clauses for l in c.body}
    meta = dict(n=n, n_base=len(base_preds), n_derived=len(derived),
                has_tc=any(len(c.body) == 2 and c.body[1].pred == c.head for c in bk),
                target_arity=ta, n_clauses=m, body_lens=[len(c.body) for c in clauses],
                n_literals=sum(len(c.body) for c in clauses),
                uses_derived=bool(used & set(derived)),
                reasoning_depth=max(depth[p] for p in used),
                pos_rate=float(target_ext.mean()))
    return Task(n=n, arities=arities, base_preds=base_preds, derived_preds=derived,
                target=target, base_ext=base_ext, bk=bk, bk_ext=ext, target_clauses=clauses,
                target_ext=target_ext, pos=P[:kp], neg=N[:kn], query_pos=P[kp:],
                query_neg=N[kn:], meta=meta)


def sample_task(rng: np.random.Generator, cfg: PriorConfig = PriorConfig()) -> Task:
    for _ in range(cfg.max_tries):
        task = _try_sample_task(rng, cfg)
        if task is not None:
            return task
    raise RuntimeError("prior rejection sampling failed; the config is too restrictive")
