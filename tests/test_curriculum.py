import json
from collections import Counter

import numpy as np
import pytest

from ilp_pfn.curriculum import CurriculumState, load_curriculum, mixture
from ilp_pfn.data import TaskStream
from ilp_pfn.prior import PriorConfig, sample_task
from ilp_pfn.tokenizer import encode
from test_prior import _check_task


@pytest.mark.parametrize("k", range(5))
def test_default_levels_sample_valid_tasks(k):
    levels = load_curriculum("default", PriorConfig())
    rng = np.random.default_rng(k)
    for _ in range(40):
        t = sample_task(rng, levels[k].prior)
        _check_task(t, levels[k].prior)
        assert len(encode(t, rng).ids) <= 1024


def test_levels_get_harder():
    levels = load_curriculum("default", PriorConfig())
    assert levels[-1].prior == PriorConfig() and levels[-1].threshold is None
    rng = np.random.default_rng(0)
    stats = []
    for lv in levels:
        tasks = [sample_task(rng, lv.prior) for _ in range(300)]
        stats.append({k: np.mean([t.meta[k] for t in tasks])
                      for k in ("n_literals", "n_derived", "reasoning_depth", "n")})
    # rule size grows over the first levels; later levels grow the BK and the world instead
    assert stats[0]["n_literals"] == 1 < stats[1]["n_literals"] < stats[2]["n_literals"]
    for key in ("n_derived", "reasoning_depth"):
        assert [s[key] for s in stats] == sorted(s[key] for s in stats)
    assert stats[-1]["n"] > stats[0]["n"]


def test_custom_curriculum_file(tmp_path):
    spec = [{"name": "a", "threshold": 0.5, "n_const": [4, 5], "body_len": [1, 1]},
            {"name": "b", "threshold": 0.9}]
    path = tmp_path / "cur.json"
    path.write_text(json.dumps(spec))
    levels = load_curriculum(str(path), PriorConfig())
    assert [lv.name for lv in levels] == ["a", "b"]
    assert levels[0].prior.n_const == (4, 5) and levels[-1].threshold is None
    assert load_curriculum("none", PriorConfig())[0].prior == PriorConfig()


def test_mixture_and_stream_weights():
    levels = load_curriculum("default", PriorConfig())
    assert mixture(levels, 0, 0.2) == ([levels[0].prior], [1.0])
    priors, w = mixture(levels, 2, 0.2)
    assert priors == [lv.prior for lv in levels[:3]] and np.isclose(sum(w), 1)
    assert np.allclose(w, [0.1, 0.1, 0.8])
    stream = iter(TaskStream(priors, 1024, seed=(0, 0), weights=w))
    # level-0 tasks are single-literal; count them in the stream
    from ilp_pfn.tokenizer import VOCAB
    n_single = 0
    for _ in range(400):
        ids, rs, *_ = next(stream)
        rule = VOCAB.toks(ids[rs:])
        n_single += rule.count(":-") == 1 and ";" not in rule
    assert n_single < 400  # mixture actually contains harder levels


def test_advancement_logic():
    levels = load_curriculum("default", PriorConfig())
    s = CurriculumState()
    s.start(levels)
    obs = lambda v, step: s.observe(v, step, levels, min_steps=100, max_steps=1000, patience=2)  # noqa: E731
    assert obs(0.95, 50) is None and s.streak == 1      # passed, but too early
    assert obs(0.95, 100) == "passed" and s.level == 1  # patience reached after min_steps
    assert obs(0.95, 200) is None                       # streak restarts on a new level
    assert obs(0.10, 300) is None and s.streak == 0     # failure resets the streak
    assert obs(0.10, 1100) == "forced" and s.level == 2
    assert s.history[-1]["event"] == "forced" and s.history[-1]["steps"] == 1000
    for step in (1200, 1300, 1400, 1500, 1600, 1700):
        obs(1.0, step)
    assert s.level == len(levels) - 1 and s.final_start is not None
    assert obs(1.0, 9999) is None  # the last level never advances
    assert Counter(h["event"] for h in s.history) == {"passed": 3, "forced": 1}


def test_single_level_starts_final():
    s = CurriculumState()
    s.start(load_curriculum("none", PriorConfig()))
    assert s.final_start == 0
