import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ilp_pfn.data import collate
from ilp_pfn.model import GPT, ModelConfig, generate, load_state_compat
from ilp_pfn.prior import sample_task
from ilp_pfn.tokenizer import VOCAB, encode, encode_queries
from ilp_pfn.verify import check_rule


def tiny(**kw):
    torch.manual_seed(0)
    return GPT(ModelConfig(vocab_size=len(VOCAB), d_model=64, n_layers=2, n_heads=4, d_ff=128,
                           max_len=1024, **kw))


def _items(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        t = sample_task(rng)
        out.append((t, encode(t, rng)))
    return out


@pytest.mark.parametrize("role_emb", [False, True])
def test_kv_cache_matches_full_forward(role_emb):
    model = tiny(role_emb=role_emb).eval()
    _, enc = _items(1)[0]
    ids = torch.tensor([enc.ids, enc.ids])
    T, split = ids.shape[1], enc.rule_start
    with torch.no_grad():
        full = model.logits(model(ids))
        caches = model.new_caches()
        step = [model.logits(model(ids[:, :split], caches))]
        for t in range(split, T):
            roles = None
            if role_emb:  # roles of the newest token from the full history, as in generate()
                sec, role = model.roles(ids[:, :t + 1])
                roles = (sec[:, -1:], role[:, -1:])
            step.append(model.logits(model(ids[:, t:t + 1], caches, roles)))
    assert torch.allclose(full, torch.cat(step, 1), atol=1e-4)


def test_roles_of_an_encoded_sequence():
    model = tiny(role_emb=True)
    toks = ["<bos>", "<facts>", "p2", "c11", ";", "p7", "c11", "c3", "<bk>",
            "p9", "V0", "V1", ":-", "not", "p2", "V0", ";", "p7", "V1", "V0", ".",
            "<pos>", "T", "c3", "c11", "<neg>", "T", "c11", "c3", "<rule>",
            "T", "V0", "V1", ":-", "p7", "V1", "V0", ".", "<eos>"]
    sec, role = model.roles(torch.tensor([VOCAB.ids(toks)]))
    expect_role = [0, 0, 1, 2, 0, 1, 2, 3, 0,
                   1, 2, 3, 0, 0, 1, 2, 0, 1, 2, 3, 0,
                   0, 1, 2, 3, 0, 1, 2, 3, 0,
                   1, 2, 3, 0, 1, 2, 3, 0, 0]
    expect_sec = [0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                  3, 3, 3, 3, 4, 4, 4, 4] + [5] * 10
    assert role[0].tolist() == expect_role
    assert sec[0].tolist() == expect_sec


def test_causality():
    model = tiny().eval()
    a = torch.randint(0, len(VOCAB), (1, 30))
    b = a.clone()
    b[0, 20:] = (b[0, 20:] + 1) % len(VOCAB)
    with torch.no_grad():
        assert torch.allclose(model(a)[:, :20], model(b)[:, :20], atol=1e-5)


def test_mask_selects_exactly_rule_tokens():
    items = _items(8)
    x, y, mask, qmask, _ = collate([(e.ids, e.rule_start) for _, e in items], multiple=64)
    assert x.shape[1] % 64 == 0 and not qmask.any()
    for b, (_, e) in enumerate(items):
        assert y[b][mask[b]].tolist() == e.ids[e.rule_start:]
        assert x[b, :len(e.ids) - 1].tolist() == e.ids[:-1]


def test_overfits_and_generates_valid_rules():
    """Sanity: a tiny model memorises a few tasks and decoding reproduces the exact rules."""
    items = _items(4, seed=1)
    model = tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x, y, mask = collate([(e.ids, e.rule_start) for _, e in items])[:3]
    for _ in range(300):
        loss = F.cross_entropy(model.logits(model(x)[mask]), y[mask])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.05
    model.eval()
    for task, enc in items:
        out = generate(model, enc.ids[:enc.rule_start], 3, 64, 0.0, VOCAB.eos_id, "cpu")
        assert out[0] == out[1] == enc.ids[enc.rule_start:]
        chk = check_rule(task, enc, VOCAB.toks(out[0]))
        assert chk.valid and chk.exact and chk.semantic


# ---------------------------------------------------------------- auxiliary query sequences
def test_query_encoding():
    rng = np.random.default_rng(0)
    kinds = set()
    for task, enc in _items(30, seed=2):
        q = encode_queries(task, enc, rng, n_queries=16)
        toks = VOCAB.toks(q.ids)
        # same prompt as the rule sequence, then queries instead of the rule
        assert q.ids[:enc.rule_start - 1] == enc.ids[:enc.rule_start - 1]
        assert "<rule>" not in toks and toks[enc.rule_start - 1] == "<query>"
        assert toks[-1] == "<eos>" and max(q.positions) < len(q.ids) - 1
        inv_c = {t: c for c, t in enc.const_tok.items()}
        inv_p = {t: p for p, t in enc.pred_tok.items()}
        for pos, lab, kind in zip(q.positions, q.labels, q.kinds):
            start = pos - task.arities[inv_p[toks[pos - 1]]] if toks[pos - 1] in inv_p else pos - 2
            atom = toks[start:pos + 1]
            pred = inv_p[atom[0]]
            args = tuple(inv_c[a] for a in atom[1:])
            ext = task.target_ext if pred == task.target else task.bk_ext[pred]
            assert len(args) == task.arities[pred] and bool(ext[args]) == bool(lab)
            assert (kind == "target") == (pred == task.target)
            assert (kind == "derived") == (pred in task.derived_preds)
            kinds.add(kind)
    assert kinds == {"target", "base", "derived"}


def test_collate_mixed_batch():
    rng = np.random.default_rng(1)
    (t0, e0), (t1, e1) = _items(2, seed=3)
    q = encode_queries(t1, e1, rng, n_queries=5)
    x, y, mask, qmask, qlab = collate([(e0.ids, e0.rule_start, [], []),
                                       (q.ids, len(q.ids), q.positions, q.labels)])
    assert y[0][mask[0]].tolist() == e0.ids[e0.rule_start:] and not mask[1].any()
    assert not qmask[0].any() and qmask[1].nonzero().squeeze(1).tolist() == q.positions
    assert qlab[1][qmask[1]].tolist() == q.labels
    assert (x[1][qmask[1]] == torch.tensor([q.ids[p] for p in q.positions])).all()


def test_loads_older_checkpoints():
    new = tiny(role_emb=True, query_head=True)
    old_sd = {k: v.clone() for k, v in new.state_dict().items() if not k.startswith("qhead")}
    old_sd["emb.weight"] = old_sd["emb.weight"][:-1]      # vocab without <query>
    old_sd["head.weight"] = old_sd["emb.weight"]              # older models had a tied head
    old_sd["sec_emb.weight"] = old_sd["sec_emb.weight"][:-1]  # sections without <query>
    model = tiny(role_emb=True, query_head=True)
    assert load_state_compat(model, old_sd)
    assert torch.equal(model.emb.weight[:-1], old_sd["emb.weight"])
    assert torch.equal(model.head.weight[:-1], old_sd["emb.weight"])
    assert not load_state_compat(model, new.state_dict())  # same version: nothing adapted
