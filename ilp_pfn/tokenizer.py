"""Serialisation of tasks into token sequences, canonical rule form, and rule parsing.

Sequence layout (atoms are written without brackets, e.g. ``p3 c5 c1``)::

    <bos> <facts> p3 c5 c1 ; p7 c4 ; ... <bk> p1 V0 V1 :- p3 V0 V2 ; p3 V2 V1 . ...
          <pos> T c1 c2 ; ... <neg> T c4 c4 ; ... <rule> T V0 V1 :- p1 V0 V2 ; p7 V2 . <eos>

* Constants and predicates get *fresh random token ids in every episode* (anonymisation);
  the target predicate is always ``T``.
* Facts, BK clauses and examples are shuffled; the target program is written in canonical form
  (head vars V0.., literals sorted, extra vars numbered by first occurrence, clauses sorted).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .logic import Clause, Literal, clause_problems, true_tuples
from .prior import Task

SPECIALS = ["<pad>", "<bos>", "<eos>", "<facts>", "<bk>", "<pos>", "<neg>", "<rule>",
            ":-", ";", ".", "not", "T"]


class Vocab:
    def __init__(self, n_const: int = 32, n_pred: int = 16, n_var: int = 6):
        self.n_const, self.n_pred, self.n_var = n_const, n_pred, n_var
        # tokens added later are appended at the end so existing ids stay stable
        self.tokens = (SPECIALS + [f"c{i}" for i in range(n_const)]
                       + [f"p{i}" for i in range(n_pred)] + [f"V{i}" for i in range(n_var)]
                       + ["<query>"])
        self.index = {t: i for i, t in enumerate(self.tokens)}
        self.pad_id, self.eos_id = self.index["<pad>"], self.index["<eos>"]

    def __len__(self) -> int:
        return len(self.tokens)

    def ids(self, tokens: list[str]) -> list[int]:
        return [self.index[t] for t in tokens]

    def toks(self, ids) -> list[str]:
        return [self.tokens[int(i)] for i in ids]


VOCAB = Vocab()

# Section ids (0 = before <facts>); sections always appear in this order in a sequence.
SECTION_OF = {"<facts>": 1, "<bk>": 2, "<pos>": 3, "<neg>": 4, "<rule>": 5, "<query>": 6}
N_SECTIONS = 7
# Atom roles: 0 = structural token, 1 = predicate, 2 = first argument, 3 = second argument,
# 4 = any further argument (only in malformed generations).
N_ROLES = 5


def token_classes(vocab: Vocab = VOCAB) -> tuple[list[bool], list[bool], list[int]]:
    """Per-token-id tables used to derive section / atom-role ids from a token sequence."""
    is_pred = [t == "T" or (t[0] == "p" and t[1:].isdigit()) for t in vocab.tokens]
    is_term = [t[0] in "cV" and t[1:].isdigit() for t in vocab.tokens]
    section = [SECTION_OF.get(t, 0) for t in vocab.tokens]
    return is_pred, is_term, section


@dataclass
class Encoded:
    ids: list[int]
    rule_start: int            # index of the first rule token (right after <rule>)
    pred_tok: dict[int, str]   # internal predicate id -> token ("T" for the target)
    const_tok: dict[int, str]  # internal constant -> token


class RuleError(ValueError):
    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


# --------------------------------------------------------------------------- canonical form
def _pkey(tok: str) -> int:
    return -1 if tok == "T" else int(tok[1:])


def canonical_clause(c: Clause, pred_tok: dict[int, str]) -> Clause:
    """Head vars -> 0..a-1, literals sorted, extra vars renamed by first occurrence.

    Exact (unique) for clauses with at most one extra variable; deterministic otherwise.
    """
    ren = {v: i for i, v in enumerate(c.head_args)}
    body = list(dict.fromkeys(c.body))  # dedupe, keep order
    body.sort(key=lambda l: (_pkey(pred_tok[l.pred]), l.neg,
                             tuple(ren.get(a, 99) for a in l.args)))
    for lit in body:
        for a in lit.args:
            if a not in ren:
                ren[a] = len(ren)
    body = [Literal(l.pred, tuple(ren[a] for a in l.args), l.neg) for l in body]
    body.sort(key=lambda l: (_pkey(pred_tok[l.pred]), l.neg, l.args))
    return Clause(c.head, tuple(range(len(c.head_args))), tuple(body))


def canonical_program(clauses: list[Clause], pred_tok: dict[int, str]) -> list[Clause]:
    key = lambda c: [(_pkey(pred_tok[l.pred]), l.neg, l.args) for l in c.body]  # noqa: E731
    out = {tuple(clause_tokens(c, pred_tok)): c
           for c in (canonical_clause(c, pred_tok) for c in clauses)}
    return sorted(out.values(), key=key)


# --------------------------------------------------------------------------- serialisation
def _join(atoms: list[list[str]], sep: str = ";") -> list[str]:
    out: list[str] = []
    for i, a in enumerate(atoms):
        if i:
            out.append(sep)
        out.extend(a)
    return out


def clause_tokens(c: Clause, pred_tok: dict[int, str]) -> list[str]:
    lits = [(["not"] if l.neg else []) + [pred_tok[l.pred]] + [f"V{a}" for a in l.args]
            for l in c.body]
    return [pred_tok[c.head]] + [f"V{a}" for a in c.head_args] + [":-"] + _join(lits) + ["."]


def program_tokens(clauses: list[Clause], pred_tok: dict[int, str]) -> list[str]:
    return [t for c in canonical_program(clauses, pred_tok) for t in clause_tokens(c, pred_tok)]


def encode(task: Task, rng: np.random.Generator, vocab: Vocab = VOCAB) -> Encoded:
    if task.n > vocab.n_const:
        raise ValueError(f"{task.n} constants > vocab capacity {vocab.n_const}")
    others = [p for p in task.arities if p != task.target]
    if len(others) > vocab.n_pred:
        raise ValueError(f"{len(others)} predicates > vocab capacity {vocab.n_pred}")
    const_tok = {i: f"c{int(c)}" for i, c in
                 enumerate(rng.choice(vocab.n_const, size=task.n, replace=False))}
    pred_tok = {p: f"p{int(t)}" for p, t in
                zip(others, rng.choice(vocab.n_pred, size=len(others), replace=False))}
    pred_tok[task.target] = "T"

    def atoms(pred, tuples):
        return [[pred_tok[pred]] + [const_tok[a] for a in t] for t in tuples]

    def shuffled(xs):
        return [xs[i] for i in rng.permutation(len(xs))]

    facts = [a for p in task.base_preds for a in atoms(p, true_tuples(task.base_ext[p]))]
    toks = ["<bos>", "<facts>"] + _join(shuffled(facts)) + ["<bk>"]
    for c in shuffled([canonical_clause(c, pred_tok) for c in task.bk]):
        toks += clause_tokens(c, pred_tok)
    toks += ["<pos>"] + _join(shuffled(atoms(task.target, task.pos)))
    toks += ["<neg>"] + _join(shuffled(atoms(task.target, task.neg)))
    toks += ["<rule>"]
    rule_start = len(toks)
    toks += program_tokens(task.target_clauses, pred_tok) + ["<eos>"]
    return Encoded(vocab.ids(toks), rule_start, pred_tok, const_tok)


# --------------------------------------------------------------------------- query sequences
QUERY_KINDS = ("target", "base", "derived")
QUERY_MIX = (0.5, 0.25, 0.25)  # derived falls back to base when the task has no BK


@dataclass
class QueryEncoded:
    ids: list[int]
    positions: list[int]   # index (into ids) of the last token of each query atom
    labels: list[float]    # 1.0 if the queried atom is true in the task's world
    kinds: list[str]       # "target" / "base" / "derived"


def encode_queries(task: Task, enc: Encoded, rng: np.random.Generator, n_queries: int = 16,
                   vocab: Vocab = VOCAB) -> QueryEncoded:
    """Prompt (same as ``enc`` up to <rule>, no rule) followed by unlabelled query atoms.

        ... <neg> T c4 c9 ; ... <query> T c4 c9 ; p7 c3 c11 ; p9 c2 ; ... <eos>

    Answers are never part of the sequence: a separate yes/no head reads the hidden state at
    the last token of each atom, so no query can see another query's answer or the true rule.
    Each query is true with probability 1/2 when both labels are possible.
    """
    toks = vocab.toks(enc.ids[:enc.rule_start - 1]) + ["<query>"]
    positions, labels, kinds = [], [], []
    for i in range(n_queries):
        kind = QUERY_KINDS[int(rng.choice(3, p=QUERY_MIX))]
        if kind == "derived" and not task.derived_preds:
            kind = "base"
        if kind == "target":
            pred, ext = task.target, task.target_ext
        else:
            pool = task.base_preds if kind == "base" else task.derived_preds
            pred = pool[int(rng.integers(len(pool)))]
            ext = task.bk_ext[pred]
        want = bool(rng.random() < 0.5)
        cands = np.argwhere(ext == want)
        if len(cands) == 0:
            cands = np.argwhere(ext != want)
        t = tuple(int(a) for a in cands[int(rng.integers(len(cands)))])
        if i:
            toks.append(";")
        toks += [enc.pred_tok[pred]] + [enc.const_tok[a] for a in t]
        positions.append(len(toks) - 1)
        labels.append(float(ext[t]))
        kinds.append(kind)
    toks.append("<eos>")  # keeps the last query token inside the model input (ids[:-1])
    return QueryEncoded(vocab.ids(toks), positions, labels, kinds)


# --------------------------------------------------------------------------- parsing
def _is_var(t: str) -> bool:
    return len(t) > 1 and t[0] == "V" and t[1:].isdigit()


def parse_rule(tokens: list[str], task: Task, enc: Encoded) -> list[Clause]:
    """Parse rule tokens (without <eos>) into target clauses. Raises RuleError(kind)."""
    inv = {t: p for p, t in enc.pred_tok.items()}
    clauses, i, L = [], 0, len(tokens)
    if L == 0:
        raise RuleError("empty")
    while i < L:
        if tokens[i] != "T":
            raise RuleError("syntax")
        i += 1
        head = []
        while i < L and _is_var(tokens[i]):
            head.append(int(tokens[i][1:]))
            i += 1
        if i >= L or tokens[i] != ":-":
            raise RuleError("syntax")
        i += 1
        body = []
        while True:
            neg = i < L and tokens[i] == "not"
            i += neg
            if i >= L:
                raise RuleError("syntax")
            if tokens[i] not in inv:
                is_pred = tokens[i][0] == "p" and tokens[i][1:].isdigit()
                raise RuleError("unknown_pred" if is_pred else "syntax")
            pred = inv[tokens[i]]
            i += 1
            args = []
            while i < L and _is_var(tokens[i]):
                args.append(int(tokens[i][1:]))
                i += 1
            body.append(Literal(pred, tuple(args), neg))
            if i >= L:
                raise RuleError("syntax")
            if tokens[i] == ";":
                i += 1
            elif tokens[i] == ".":
                i += 1
                break
            else:
                raise RuleError("syntax")
        clause = Clause(task.target, tuple(head), tuple(body))
        # duplicate literals are harmless redundancy; canonical_program removes them
        problems = [p for p in clause_problems(clause, task.arities) if p != "duplicate_literal"]
        if problems:
            raise RuleError(problems[0])
        clauses.append(clause)
    return clauses
