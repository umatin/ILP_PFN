# ILP-PFN

A prior-fitted network for inductive logic programming. A transformer is trained, with teacher
forcing, to write down a Datalog rule that explains labelled examples. It is given base facts and
background-knowledge (BK) rules, and every training task is freshly sampled from a prior over
worlds, BK programs and target rules. Minimising cross-entropy on the sampled rule makes the model
approximate the posterior over rules under that prior.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # numpy, torch, pytest
pytest                           # ~10 s
```

## Usage

```bash
# local smoke test (MPS / CPU)
python -m ilp_pfn.train --out_dir runs/pilot --d_model 256 --n_layers 4 --n_heads 4 \
    --steps 4000 --batch_size 32 --lr 1e-3 --warmup 200 --num_workers 4

# A40 (edit the #SBATCH lines first); resubmitting resumes from runs/base/ckpt.pt
sbatch scripts/train_a40.sbatch

# evaluation on all splits, greedy + 16 samples
python -m ilp_pfn.evaluate --ckpt runs/base/ckpt.pt --n 500 --k 16
```

Prior overrides are passed as JSON, e.g. `--prior '{"allow_negation": true}'`.

**Loss.** By default training is fully autoregressive: next-token loss on the rule plus, with
`--prompt_loss_weight` (default 1.0), next-token loss on the prompt. The two terms are averaged
separately, so the few rule tokens are not drowned out by the long prompt. Set
`--prompt_loss_weight 0` for rule-only training.

**Per-level evaluation.** At every evaluation, `--eval_levels default` (on by default) evaluates
200 fixed tasks from each curriculum level L0–L4 separately. It does this whatever distribution is
being trained on. Look for the `LEVELS step …` lines, or `eval.levels` in `metrics.jsonl`.

Minimal sanity tasks (`--curriculum configs/<name>.json`). Each has a single decision:

| Config | Task | Tokens (mean) |
|---|---|---|
| `tiny_unary.json` | 2 unary predicates, which one is `T`? (match one constant) | 33 |
| `tiny_binary.json` | 1 binary predicate, `p(X,Y)` or `p(Y,X)`? (match a pair) | 55 |
| `l0_only.json` | curriculum level 0 (both of the above, larger worlds) | 90 |

Role embeddings (`--role_emb`, on by default) add two learned vectors to every token: one for its
section (facts / bk / pos / neg / rule) and one for its position within an atom (predicate /
first argument / second argument). Both are computed from the token sequence itself, so they also
work during generation.

### Curriculum

```bash
python -m ilp_pfn.train --out_dir runs/cur --curriculum default --final_steps 50000 \
    --eval_every 1000 --n_val_decode 200 ...
```

Training goes through levels `L0_single_literal → L1_conjunctions → L2_bk_disjunction →
L3_near_full → L4_full` (defined in `ilp_pfn/curriculum.py`; pass a JSON file for your own).

- **Advancing.** At each evaluation, greedy consistency (`--cur_metric`) is measured on a fixed
  validation set of the *current* level. Training moves to the next level after `--cur_patience`
  consecutive evaluations at or above the level's threshold, with at least `--cur_min_steps` spent
  on the level. After `--cur_max_steps` it advances anyway, and the history marks this as `forced`.
- **Replay.** `--replay_frac` of every batch comes from earlier levels.
- **Learning rate.** Constant until the last level, then cosine-decayed over `--final_steps`.
- **Logs.** Advancements are printed as `ADVANCE {...}` and written to `metrics.jsonl`. During the
  curriculum, each evaluation also reports `final_level` metrics on the full prior.

## Code layout

| File | Content |
|---|---|
| `ilp_pfn/logic.py` | Datalog engine: dense-tensor clause evaluation, stratified negation, fixpoint |
| `ilp_pfn/prior.py` | `PriorConfig` and `sample_task`: world, BK rules, target rule, examples |
| `ilp_pfn/tokenizer.py` | serialisation, per-episode anonymous symbols, canonical rule form, parser |
| `ilp_pfn/verify.py` | checks a generated rule: valid / consistent / semantic / exact / held-out accuracy |
| `ilp_pfn/curriculum.py` | curriculum levels, replay mixture, performance-gated advancement |
| `ilp_pfn/splits.py` | evaluation splits (iid and distribution shifts), deterministic eval sets |
| `ilp_pfn/model.py` | decoder-only transformer with RoPE and a KV cache, sampling |
| `ilp_pfn/train.py`, `evaluate.py` | training loop (resumable) and evaluation CLI |

## Design (v1)

- **Logic**: function-free Datalog with predicates of arity at most 2. Negation is implemented but
  off by default. Recursion appears only in BK (transitive closure); target rules are not
  recursive.
- **Prior**:
  - 4–10 constants; 1–3 unary and 1–3 binary base predicates, each with fact density 0.1–0.4.
  - 0–3 derived BK predicates, plus a transitive closure with probability 0.3.
  - Target: 1–2 clauses, each with 1–3 literals and at most 1 existential variable.
  - Rejection conditions:
    - in BK and target clauses, the head variables are linked through the body (no
      `t(X,Y) :- p(X), q(Y)`);
    - unary targets only occur in worlds with at least 6 constants;
    - at least 2 positives and 2 negatives are shown;
    - every clause is minimal in its world;
    - every clause covers positives no other clause covers;
    - the positive rate is in [0.1, 0.9];
    - in 50% of tasks the target must use a derived predicate.
- **Input**: base facts, BK *rules* (not their consequences), and at most 10 positive and 10
  negative examples. The remaining tuples are held out as queries. Constants and predicates get
  random token ids in every episode, and facts, BK rules and examples are shuffled. Facts have no
  separators, because every atom starts with a predicate token. Each example carries its own
  label, and positives and negatives are interleaved:
  `<facts> p1 c14 p5 c6 <bk> <neg> T c6 <pos> T c5 <rule> T V0 :- p1 V0 . <eos>`
- **Target**: the canonical rule (variables numbered by first occurrence, literals and clauses
  sorted). The loss covers only the rule tokens.

## Initialisation

The model uses fan-in initialisation. Linear weights have std `1/sqrt(in_features)`, and residual
output projections are additionally divided by `sqrt(2 * n_layers)`. Token embeddings have std 1.
Role and section embeddings start small (std 0.02). The output head is untied.

The first version used GPT-2-style std 0.02 for all weights. With that, attention starts almost
exactly uniform, and on in-context lookup the model stayed at chance for thousands of steps. On a
2-layer "has this token appeared before?" probe after 600 steps:

| Initialisation | Accuracy |
|---|---|
| std 0.02 everywhere | 0.51 |
| fan-in | 0.98 |
| `nn.TransformerEncoder` reference | 0.999 |

Checkpoints trained with the old initialisation still load.

## Metrics

All metrics are reported for greedy decoding and for the best of k samples. The best sample is the
shortest one that is consistent with the shown examples.

- `valid`: the output parses and is safe and stratifiable.
- `consistent`: covers all shown positives and no shown negatives.
- `semantic`: has the same extension as the true rule over the whole world.
- `exact`: is the same canonical rule as the true one.
- `query_acc`: accuracy on held-out tuples (`majority_query_acc` is the trivial baseline).

Evaluation splits: `iid`, `more_constants` (12–16), `longer_bodies` (4 literals), `more_clauses`
(3), `deeper_bk` (4–5 derived predicates), `denser_facts` (0.4–0.6).

`semantic` and `exact` are upper-bounded below 1, because the shown examples often admit several
consistent rules. `consistent` is the primary measure of whether the model produces a correct
logic program.
