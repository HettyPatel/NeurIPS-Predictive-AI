# PAIEC entry: subject prior + LLM difficulty rating + three corrections

Predicts P(an AI system answers a benchmark item correctly) given the labels revealed so far.
NeurIPS 2026 Predictive AI Evaluation Competition (Stanford AIMS).
Team: Het Patel, Suki Wang, Dawon Ahn, Vagelis Papalexakis (UC Riverside / USC / Roblox).

Report: *Predicting LLM Benchmark Performance from Few Labels: LLM-Rated Item Difficulty and Hierarchical
Shrinkage for Cold-Start Evaluation* (TODO: add link). "The report" below means this paper.

The competition scored entries on its own server: it loaded `submission/model.py` and called
`predict(input, labeled)` on hidden benchmarks. That server may no longer be running. Nothing below needs it:
`evaluate.py` runs the same protocol locally on public data.

## Method

    P(correct) = sigmoid( 0.5 * prior(subject) + w_domain * rating(item) + d_bench + d_subject + d_item )

- **prior**: a 385 -> 256 -> 64 -> 1 network on the MiniLM embedding of the subject's *name* plus the item's log
  length, trained on the 160 public binary benchmarks of `measurement-db`. A per-benchmark bias is learned during
  training and discarded, so the network encodes relative subject strength only; the hidden benchmark's level comes
  from the labels. No item content: item embeddings ranked difficulty backwards on unseen benchmarks.
- **rating**: `claude-haiku-5-5` is asked what share of frontier models would get the item right. The logit of
  that share enters with a per-domain weight (knowledge 1.0, math 0.8, coding 0.6, legal 0.5, agentic and
  multimodal 0). This is the main part that moves budget 0 (0.245 -> 0.221).
- **d_bench, d_subject, d_item**: one-number logistic shifts fit by six Newton steps on the revealed labels, each
  shrunk toward 0 (strengths 2, 2, 0.5): all labels on the benchmark, this subject's own, other subjects' on this
  exact item. With many subjects per benchmark `d_item` is most of the gain over the hierarchical mean; with one or
  two it rarely activates.

### Scores (Brier ALC over label budgets 0/1/3/7/15/31; lower is better; 0.5 everywhere = 0.25)

| setting | ours | hierarchical mean |
|---|---|---|
| DEV, 13 held-out benchmarks, 819 pairs | **0.194** | 0.229 |

## Files

| file | purpose |
|---|---|
| `download.py` | downloads the dataset tables into `data/raw/` |
| `features.py` | builds `data/feat/<benchmark>.npz`: subject-name embeddings, item lengths, soft labels |
| `evaluate.py` | the competition protocol run locally, the DEV list, three baselines |
| `experiments.py` | reruns the report's DEV experiments: main comparison, shift ablation, cache-fix check, sparse coverage |
| `rate.py` | rates items with the LLM rater; `data/ratings.jsonl` is committed so evaluation needs no API key |
| `train_and_zip.py` | trains on all 160 benchmarks and builds `submission.zip` |
| `submission/model.py` | main method file; `predict(input, labeled)` is the entry point the competition called |
| `tests/test_shift_cache.py` | regression test: the cached shifts must depend on the whole reveal list |
| `notebooks/data.ipynb`, `notebooks/modelTest.ipynb`, `notebooks/testing.ipynb` | data exploration; component checks and the DEV evaluation; checks of the evaluation protocol, the baselines, the embeddings and one live rating call |

## Setup

```bash
conda create -n paiec python=3.11 -y && conda activate paiec
pip install torch
pip install sentence-transformers anthropic huggingface_hub pandas pyarrow numpy scipy jupyter
hf auth login            # measurement-db is gated: accept its terms on Hugging Face first
```

Tested with Python 3.11, torch 2.15.0, sentence-transformers 6.3.0, anthropic 2.12.0, huggingface_hub 2.4.0,
pandas 3.1.1, pyarrow 26.0.1, numpy 2.4.6 and scipy 1.17.1. If a later release breaks something, install these
versions.

**API key.** Needed only to rate new items, with `rate.py` or when `model.py` meets an item it has no rating for.
Put it in `submission/submission_config.json` as `{"anthropic_api_key": "<your key>"}` (gitignored), or set
`ANTHROPIC_API_KEY` in the environment. Without a key the model still runs and falls back to prior plus labels.

## Data and features

```bash
python download.py      # ~5 GB -> data/catalog.csv and data/raw/<benchmark>/formatted_tables/{subjects,items,responses}.parquet
                        # pinned to measurement-db commit 5407445 (2026-10-07); MEASUREMENT_DB_REVISION overrides
python features.py      # ~15 min on a GPU -> one .npz per benchmark; already-built ones are skipped
```

## How evaluation works (`evaluate.py`)

- **Training pool**: the 160 public, binary, item-level benchmarks (`usable_benchmarks()`).
- **DEV set**: 13 of them held out as pseudo-hidden benchmarks: hallusionbench, ikp, multi_swebench, mmlupro,
  editbench, swebench_live, researchcodebench, legalbench, tau2_bench, real_webagents, mtbbench,
  workbench_revisited, hle. For each DEV benchmark the prior is refit on the other 159, so the model never trains on
  a benchmark it is scored on.
- **Pairs**: a (subject, benchmark) pair is scored if the subject answered at least 80 distinct items. Its items are
  shuffled with a seed derived from the subject ID and split in half: labels are revealed from the first half, in
  order; the second half (up to 100 items) is scored. 819 pairs in total.
- **Budgets**: predictions with 0, 1, 3, 7, 15 and 31 revealed labels (nested). Brier score per budget;
  ALC = weighted mean with weights 0.1, 0.2, 0.2, 0.2, 0.2, 0.1; pairs weighted equally.
- `evaluate(predictor, DEV, train_pool=usable_benchmarks())` takes any object with `predict(input, labeled)` and
  optionally `fit(benchmark_names)` and `predict_many(inputs, labeled)`. Baselines `EmpiricalMean`, `ShrunkMean`
  and `HierarchicalMean` are in the same file.

## Run the DEV evaluation (no API calls)

```bash
python experiments.py main      # Table 1: baselines, shifts only, ours without the rating, ours
python experiments.py ablate    # Table 2: one shift removed at a time
python experiments.py cache     # checks that the cache fix leaves DEV scores unchanged
python experiments.py all       # the three above, training the 13 leave-one-out priors once
python experiments.py sparse    # Section 5: two randomly drawn subjects per benchmark, 20 draws
python experiments.py damping   # Section 3.1: the factor on the prior's logit, 0.3 to 1.0
python tests/test_shift_cache.py
```

Add `--max-eval-items 20` to score 20 items per pair, the setting the report's tables use; the default is up to
100. The trained leave-one-out priors are kept in `data/loo_nets/` and reused by later runs.

| DEV, 819 pairs, 20 items per pair | ALC | B0 | B31 |
|---|---|---|---|
| empirical mean | 0.2819 | 0.250 | 0.220 |
| shrunk mean | 0.2373 | 0.250 | 0.219 |
| hierarchical mean | 0.2285 | 0.250 | 0.218 |
| shifts only (no prior, no rating) | 0.2123 | 0.250 | 0.172 |
| ours without the rating | 0.2095 | 0.244 | 0.172 |
| ours | 0.1933 | 0.220 | 0.164 |

Example output from one run. Retraining the prior gives slightly different numbers each time (up to about
0.002), so your run and the report will not match to the last digit.

**Sparse coverage (`sparse`).** On every DEV benchmark two subjects are drawn at random and all others are
removed, so only those two subjects' labels are revealed or scored; 20 draws, 26 pairs each, every benchmark
weighted equally. Ours 0.214 against 0.227 for the hierarchical mean (difference 0.012 +- 0.001, ours ahead in
19 of 20 draws); without `d_item` 0.215. With all subjects and the same equal weighting of benchmarks: 0.204
against 0.222.

**Change since the submitted entry.** `model.py` refits the shifts whenever any revealed outcome changes; the
submitted version checked only part of the list. DEV scores are identical (`python experiments.py cache`).

To re-rate items instead of using the committed ratings: `python rate.py --dev-full` (about 22,000 Haiku calls,
roughly $6; cached by item text, so reruns rate only what is missing), then `python rate.py --analyze` for the
rating-vs-accuracy correlation table on items with at least three responses (average within-benchmark
Spearman +0.23).

## How the competition submission was built

This section records how the entry was packaged. It is not needed to reproduce the results above.

```bash
python train_and_zip.py      # trains the prior on all 160 benchmarks -> submission.zip
```

`train_and_zip.py` needs `submission/submission_config.json` to exist (see **API key** above), because the key
travelled inside the zip so that the model could rate unseen items on the competition server.

The zip holds six files: `model.py`, `weights.npz`, `meta.json`, `models.txt` (names the MiniLM encoder),
`requirements.txt`, `submission_config.json`. Items without a rating are rated when first seen and cached in
`_rating_cache.jsonl` next to the model, so each item costs one call (about $0.30 per 1,000 items).

The committed `submission/weights.npz` is the one in the submitted zip.

Settings shipped (`meta.json`): bench_strength 2.0, subject_strength 2.0, item_strength 0.5, prior_scale 0.5, the
domain weights above, live_rating true, max_live_ratings 60000.
