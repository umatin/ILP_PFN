import numpy as np
import pytest

from ilp_pfn.logic import Clause, Literal
from ilp_pfn.prior import PriorConfig, sample_task
from ilp_pfn.tokenizer import (VOCAB, RuleError, canonical_clause, encode, parse_rule,
                               program_tokens)
from ilp_pfn.verify import check_rule


def _sample(seed, **kw):
    rng = np.random.default_rng(seed)
    t = sample_task(rng, PriorConfig(**kw))
    return t, encode(t, rng)


@pytest.mark.parametrize("negation", [False, True])
def test_ground_truth_passes_all_checks(negation):
    for seed in range(200):
        t, enc = _sample(seed, allow_negation=negation)
        assert all(0 <= i < len(VOCAB) for i in enc.ids)
        assert VOCAB.toks(enc.ids)[enc.rule_start - 1] == "<rule>"
        assert enc.ids[-1] == VOCAB.eos_id
        chk = check_rule(t, enc, VOCAB.toks(enc.ids[enc.rule_start:]))
        assert chk.valid and chk.consistent and chk.semantic and chk.exact, chk
        assert chk.query_acc in (None, 1.0)


def test_prompt_contains_no_derived_facts_and_no_rule():
    t, enc = _sample(3)
    prompt = VOCAB.toks(enc.ids[:enc.rule_start])
    n_facts = sum(int(e.sum()) for e in t.base_ext.values())
    facts = prompt[prompt.index("<facts>") + 1:prompt.index("<bk>")]
    assert sum(t.startswith("p") for t in facts) == n_facts  # exactly the base facts
    assert len(facts) == sum((1 + t.arities[p]) * int(e.sum()) for p, e in t.base_ext.items())
    assert "T" not in prompt[:prompt.index("<bk>")]
    # every example is preceded by its own label, and labels match the task
    first = min(i for i, x in enumerate(prompt) if x in ("<pos>", "<neg>"))
    ex = prompt[first:]
    labels = [x for x in ex if x in ("<pos>", "<neg>")]
    assert labels.count("<pos>") == len(t.pos) and labels.count("<neg>") == len(t.neg)
    assert ex.count("T") == len(t.pos) + len(t.neg) and ";" not in ex


def test_canonical_form_is_invariant():
    rng = np.random.default_rng(0)
    pred_tok = {0: "p3", 1: "p1", 2: "p7", 9: "T"}
    for _ in range(200):
        body = [Literal(int(rng.integers(3)), (0, 2) if rng.random() < .5 else (2, 1))
                for _ in range(3)] + [Literal(0, (0, 1))]
        c = Clause(9, (0, 1), tuple(body))
        perm = tuple(body[i] for i in rng.permutation(len(body)))
        renamed = Clause(9, (0, 1), tuple(Literal(l.pred, tuple(5 if a == 2 else a for a in l.args))
                                          for l in perm))
        assert canonical_clause(c, pred_tok) == canonical_clause(renamed, pred_tok)
        assert program_tokens([c], pred_tok) == program_tokens([renamed, c], pred_tok)


def test_parser_rejects_malformed_rules():
    t, enc = _sample(7, target_arity=(2, 2))
    binary = next(tok for p, tok in enc.pred_tok.items() if t.arities[p] == 2 and tok != "T")
    unused = next(f"p{i}" for i in range(VOCAB.n_pred) if f"p{i}" not in enc.pred_tok.values())
    good = ["T", "V0", "V1", ":-", binary, "V0", "V1", "."]
    assert parse_rule(good, t, enc)
    cases = {
        "syntax": ["T", "V0", "V1", ":-", binary, "V0", "V1"],             # missing '.'
        "head_arity": ["T", "V0", ":-", binary, "V0", "V1", "."],
        "arity": ["T", "V0", "V1", ":-", binary, "V0", ";", binary, "V0", "V1", "."],
        "unknown_pred": ["T", "V0", "V1", ":-", unused, "V0", "V1", "."],
        "unsafe": ["T", "V0", "V1", ":-", binary, "V0", "V0", "."],
        "empty": [],
    }
    for kind, toks in cases.items():
        with pytest.raises(RuleError) as e:
            parse_rule(toks, t, enc)
        assert e.value.kind == kind, (kind, e.value.kind)
    assert check_rule(t, enc, good).error == "no_eos"
    assert check_rule(t, enc, good + ["<eos>"]).valid


def test_wrong_rule_is_detected():
    for seed in range(50):
        t, enc = _sample(seed, target_arity=(2, 2))
        p = next(tok for q, tok in enc.pred_tok.items() if t.arities[q] == 2 and tok != "T")
        chk = check_rule(t, enc, ["T", "V0", "V1", ":-", p, "V0", "V1", ".", "<eos>"])
        assert chk.valid
        truth_ext = t.target_ext
        q = [x for x in t.pos + t.neg]
        cand = t.bk_ext[next(k for k, v in enc.pred_tok.items() if v == p)]
        expected = all(cand[x] == truth_ext[x] for x in q)
        assert chk.consistent == expected
        assert chk.semantic == bool(np.array_equal(cand, truth_ext))
