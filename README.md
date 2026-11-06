# PAIEC entry: subject prior + LLM difficulty rating + three corrections

Predicts P(an AI system answers a benchmark item correctly) given the labels revealed so far.
NeurIPS 2026 Predictive AI Evaluation Competition (Stanford AIMS).
Team: Het Patel, Suki Angeles, Dawon Ahn, Vagelis Papalexakis (UC Riverside / USC / Roblox).

## Method

    P(correct) = sigmoid( 0.5 * prior(subject) + w_domain * rating(item) + d_bench + d_subject + d_item )

- **prior**: a 385 -> 256 -> 64 -> 1 network on the MiniLM embedding of the subject's *name* plus the item's log
  length, trained on the 160 public binary benchmarks of `measurement-db`. A per-benchmark bias is learned during
  training and discarded, so the network encodes relative subject strength only; the hidden benchmark's level comes
  from the labels. No item content: item embeddings ranked difficulty backwards on unseen benchmarks.
- **rating**: `claude-haiku-5-5` is asked what share of frontier models would get the item right. The logit of
  that share enters with a per-domain weight (knowledge 1.0, math 0.8, coding 0.6, legal 0.5, agentic and
  multimodal 0). This is the only part that moves budget 0 (0.246 -> 0.221). Ratings are cached on disk and shared
  across the platform's worker processes; failures fall back to a neutral rating.
- **d_bench, d_subject, d_item**: one-number logistic shifts fit by six Newton steps on the revealed labels, each
  shrunk toward 0 (strengths 2, 2, 0.5): all labels on the benchmark, this subject's own, other subjects' on this
  exact item. With many subjects per benchmark `d_item` is most of the gain over the hierarchical mean; with one or
  two it rarely activates.

### Scores (Brier ALC over label budgets 0/1/3/7/15/31; lower is better; 0.5 everywhere = 0.25)

| setting | ours | hierarchical mean |
|---|---|---|
| DEV, 13 held-out benchmarks, 819 pairs | **0.194** | 0.229 |
| Codabench formative, seven runs of the same method (9 pairs each) | 0.169 - 0.209, mean 0.191 | 0.214 |

The formative board draws nine pairs per run and reshuffles the label order, which moves a score by about
+-0.015 on its own (one subject scored 0.216 and 0.242 in two runs of an identical zip). Method comparisons are
made on DEV, where every variant is scored on the same 819 pairs.

### Notes

- `d_item` pools other subjects' labels on the same item, so it contributes most when a benchmark has many
  scored subjects (as on DEV) and rarely activates when it has one or two (as on the hidden test); the rating
  and the other two shifts carry the method in that case.
- The rating is strong on knowledge and math items (ikp +0.91, mmlupro +0.58 within-benchmark Spearman with
  item accuracy), weak on coding and agentic tasks (+0.1 to +0.3), and uninformative on multimodal items because
  the image is not in the data; the domain weights reflect that. Opus 5.5 rates somewhat better (+0.35 vs +0.24
  average) for about 0.005 ALC at ten times the cost, so Haiku is shipped.
- Variants that were measured and not kept, all within 0.002 of this model on DEV or worse: item-text embeddings
  in the prior, hand-crafted subject features, matrix-factorisation priors, TabPFN, per-pair rating slopes,
  calibration layers, an uncertainty-sampling label-acquisition policy, prompt variants of the rater, open-weight
  and fine-tuned raters, and longer or excerpted item text for the rater.

## Files

| file | purpose |
|---|---|
| `download.py` | downloads the dataset tables into `data/raw/` |
| `features.py` | builds `data/feat/<benchmark>.npz`: subject-name embeddings, item lengths, soft labels |
| `evaluate.py` | the competition protocol run locally, the DEV list, three baselines |
| `rate.py` | rates items with the LLM rater; `data/ratings.jsonl` is committed so evaluation needs no API key |
| `train_and_zip.py` | trains on all 160 benchmarks and builds `submission.zip` |
| `submission/model.py` | the whole method; the platform imports it and calls `predict()` |
| `notebooks/data.ipynb`, `notebooks/modelTest.ipynb` | data exploration; component checks and the DEV evaluation |

## Setup

```bash
conda create -n paiec python=3.11 -y && conda activate paiec
pip install torch --index-url https://download.pytorch.org/whl/cu126      # or the CPU build
pip install sentence-transformers anthropic huggingface_hub pandas pyarrow numpy scipy jupyter
huggingface-cli login                                                    # measurement-db is gated: accept its terms first
```

**API key.** Needed only to rate items: the platform run, or `rate.py`. Put it in
`submission/submission_config.json` as `{"anthropic_api_key": "<your key>"}` (gitignored; the file travels inside
the zip so the platform can rate unseen items), or set `ANTHROPIC_API_KEY` in the environment. Without a key the
model still runs and falls back to prior plus labels.

## Data and features

```bash
python download.py      # ~5 GB -> data/catalog.csv and data/raw/<benchmark>/formatted_tables/{subjects,items,responses}.parquet
python features.py      # ~15 min on a GPU -> one .npz per benchmark; already-built ones are skipped
```

## How evaluation works (`evaluate.py`)

- **Training pool**: the 160 public, binary, item-level benchmarks (`usable_benchmarks()`).
- **DEV set**: 13 of them held out as pseudo-hidden benchmarks: hallusionbench, ikp, multi_swebench, mmlupro,
  editbench, swebench_live, researchcodebench, legalbench, tau2_bench, real_webagents, mtbbench,
  workbench_revisited, hle. For each DEV benchmark the prior is refit on the other 159, so the model never trains on
  a benchmark it is scored on.
- **Pairs**: a (subject, benchmark) pair is scored if the subject answered at least 80 distinct items. Its items are
  shuffled with a seed derived from the subject id and cut in half: labels are revealed from the first half, in
  order; the second half (up to 100 items) is scored. 819 pairs in total.
- **Budgets**: predictions with 0, 1, 3, 7, 15 and 31 revealed labels (nested). Brier score per budget;
  ALC = weighted mean with weights 0.1, 0.2, 0.2, 0.2, 0.2, 0.1; pairs weighted equally.
- `evaluate(predictor, DEV, train_pool=usable_benchmarks())` takes any object with `predict(input, labeled)` and
  optionally `fit(benchmark_names)` and `predict_many(inputs, labeled)`. Baselines `EmpiricalMean`, `ShrunkMean`
  and `HierarchicalMean` are in the same file.

## Run the DEV evaluation (~7 min on a GPU, no API calls)

```python
import sys; sys.path.insert(0, "submission")
import model as M
from evaluate import evaluate, DEV, usable_benchmarks, HierarchicalMean

m = M.Model(live_rating=False).load_ratings("data/ratings.jsonl")
r = evaluate(m, DEV, train_pool=usable_benchmarks(), verbose=True)   # Brier ALC = 0.1937 over 819 pairs
h = evaluate(HierarchicalMean(), DEV, verbose=False)                 # 0.2285
```

Expected per budget: B0 0.221, B1 0.207, B3 0.200, B7 0.190, B15 0.178, B31 0.165. Training is seeded; expect
agreement within about 0.001 across machines. The same run with component checks is in `notebooks/modelTest.ipynb`.

To re-rate items instead of using the committed ratings: `python rate.py --dev-full` (about 18,000 Haiku calls,
roughly $6; cached by item text, so reruns rate only what is missing), then `python rate.py --analyze` for the
rating-vs-accuracy correlation table (average within-benchmark Spearman +0.24).

## Train and build the submission (~3 min on a GPU)

```bash
python train_and_zip.py                                               # trains on all 160 benchmarks -> submission.zip
python <paiec_baseline>/check_submission_zip.py submission.zip        # organizers' checker -> "OK"
```

The zip holds six files: `model.py`, `weights.npz`, `meta.json`, `models.txt` (MiniLM, pre-downloaded by the
platform), `requirements.txt`, `submission_config.json`. On the platform, unseen items are rated live and cached in
`_rating_cache.jsonl` next to the model, so each item costs one call (about $0.30 per 1,000 items).

The committed `submission/weights.npz` is the one in the submitted zip. GPU training is not bit-deterministic, so
retraining gives different weights with the same DEV score to within 0.001; the prior is damped by 0.5 and the
result barely depends on it.

Settings shipped (`meta.json`): bench_strength 2.0, subject_strength 2.0, item_strength 0.5, prior_scale 0.5, the
domain weights above, live_rating true, max_live_ratings 60000.
