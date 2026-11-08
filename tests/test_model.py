import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ilp_pfn.data import collate
from ilp_pfn.model import GPT, ModelConfig, generate, load_state_compat
from ilp_pfn.prior import sample_task
from ilp_pfn.tokenizer import VOCAB, encode
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
    x, y, mask = collate([(e.ids, e.rule_start) for _, e in items], multiple=64)
    assert x.shape[1] % 64 == 0
    for b, (_, e) in enumerate(items):
        assert y[b][mask[b]].tolist() == e.ids[e.rule_start:]
        assert x[b, :len(e.ids) - 1].tolist() == e.ids[:-1]


def test_overfits_and_generates_valid_rules():
    """Sanity: a tiny model memorises a few tasks and decoding reproduces the exact rules."""
    items = _items(4, seed=1)
    model = tiny()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    x, y, mask = collate([(e.ids, e.rule_start) for _, e in items])
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


def test_loads_checkpoints_from_the_query_version():
    """Checkpoints with an extra <query> row, an extra section and a query head still load."""
    new = tiny(role_emb=True)
    sd = {k: v.clone() for k, v in new.state_dict().items()}
    for k in ("emb.weight", "head.weight", "sec_emb.weight"):
        sd[k] = torch.cat([sd[k], torch.randn(1, sd[k].shape[1])])
    sd["qhead.weight"], sd["qhead.bias"] = torch.randn(1, 64), torch.zeros(1)
    model = tiny(role_emb=True)
    assert load_state_compat(model, sd)
    for k in ("emb.weight", "head.weight", "sec_emb.weight"):
        assert torch.equal(model.state_dict()[k], new.state_dict()[k])
    assert not load_state_compat(model, new.state_dict())  # current version: nothing adapted
