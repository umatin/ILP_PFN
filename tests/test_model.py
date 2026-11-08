import numpy as np
import torch
import torch.nn.functional as F

from ilp_pfn.data import collate
from ilp_pfn.model import GPT, ModelConfig, generate
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


def test_kv_cache_matches_full_forward():
    model = tiny().eval()
    ids = torch.randint(0, len(VOCAB), (2, 40))
    with torch.no_grad():
        full = model.logits(model(ids))
        caches = model.new_caches()
        step = [model.logits(model(ids[:, :25], caches))]
        for t in range(25, 40):
            step.append(model.logits(model(ids[:, t:t + 1], caches)))
    assert torch.allclose(full, torch.cat(step, 1), atol=1e-4)


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
