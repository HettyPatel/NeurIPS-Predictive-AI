"""Local evaluation script that mirrors competition protocol.

A *subject* is one AI system configuraiton. 
an *item* is one question 
a *reveal* (card) is one labeled example [[subject, item], 0 or 1] that the platform has handed over. 

predict(input, labeled) gets the target [subject, item] and the list of all the reveals so far. 

"""

import hashlib
from pathlib import Path
import copy
import random
from collections import defaultdict

import pandas as pd 

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CATALOG = ROOT / "data" / "catalog.csv"





#  the 8 subject fields the platform gives predict in its order
SUBJECT_FIELDS = ["normalized_name",
                  "provider",
                  "release_date",
                  "access_date",
                  "harness",
                  "harness_version",
                  "reasoning_effort",
                  "subject_features_extra"]

def clean(v):
    """platform rule: every field is a plain string missing becomes ''
    """
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def usable_benchmarks():
    """The 160 public, binary, item-level benchmarks: the training pool."""
    cat = pd.read_csv(CATALOG)
    mask = (cat.release == "public") & (cat.response_type == "binary") & (cat.granularity == "item")
    return sorted(cat.loc[mask, "dir"])



def load_benchmark(name):
    """ One benchmark -> list of records, one per response row:
    {subject_id, item_id, "subject": {8 fields}, "item" {4 fields}, "label": 0 or 1}
    The subject and item dicts are exactly what the platform passes to predict()."""

    t = RAW / name / "formatted_tables"
    subjects = pd.read_parquet(t / "subjects.parquet")
    items = pd.read_parquet(t / "items.parquet")
    responses = pd.read_parquet(t / "responses.parquet")

    responses = responses[responses.response.notna()] # only fields with recorded outcome. 

    # the platform anonymises the benchmark; a stable fake id is enough to group reveals by benchmark
    anon_id = "bench_" + hashlib.sha256(name.encode()).hexdigest()[:12]

    subject_dicts = {r.subject_id: {f: clean(getattr(r, f)) for f in SUBJECT_FIELDS} for r in subjects.itertuples()}

    item_dicts = {r.item_id: {"item_content": clean(r.content), "item_features": clean(r.item_features), "benchmark_id": anon_id} for r in items.itertuples()}

    records = []
    for r in responses.itertuples():

        item = dict(item_dicts[r.item_id], interactors=clean(r.interactors))
        records.append({
            "subject_id": r.subject_id,
            "item_id": r.item_id,
            "subject": subject_dicts[r.subject_id],
            "item": item,
            "label": int(r.response)
        })

    return records




BUDGETS = [0, 1, 3, 7, 15, 31]               # reveals per subject-benchmark pair
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]     # trapezoid rule over log2(budget + 1); sums to 1
MIN_ITEMS = 80                               # a pair needs this many distinct items to be scored (at least 80 distinct itemss on the benchmark must be answered.)


def _rng(*parts):
    """A random generator seeded from strings, so the split is identical on every run."""
    key = "|".join(str(p) for p in parts)
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))

def split_pairs(records, seed=0, min_items=MIN_ITEMS):
    """
    each subject's items are shuffled and cut in half: reveals come from the first half, in order;
    the second half is what gets scored. A Subject never sees a label for an item it is scored on. 
    """
    items_by_subject  = defaultdict(set)

    for r in records:
        items_by_subject[r["subject_id"]].add(r["item_id"])

    pairs = {}
    for subject_id, item_ids in items_by_subject.items():
        if len(item_ids) < min_items:
            continue

        ordered = sorted(item_ids)
        _rng(seed, subject_id).shuffle(ordered)

        half = len(ordered) // 2

        pairs[subject_id] = (ordered[:half], ordered[half:])

    return pairs



def index_responses(records):
    """
    {(subject_id, item_id): [records]} so repeated trials of a pair are found together.
    """
    idx = defaultdict(list)
    for r in records:
        idx[(r["subject_id"], r["item_id"])].append(r)
    return idx

def labeled_at(pairs, idx, budget):
    """
    The reveal list at a budget: the first budget acquisition items of every subject pooled into one list.
    """
    labeled = []
    for subject_id, (acq, _) in pairs.items():
        for item_id in acq[:budget]:
            r = idx[(subject_id, item_id)][0]
            labeled.append([[r["subject"], r["item"]], r["label"]])
    return labeled

def brier(preds, labels):
    return sum((p - y) ** 2 for p, y in zip(preds, labels)) / len(labels)


def alc(budget_scores):
    """Brier ALC for one pair from its {budget: brier} dict. Lower is better; 0.25 = always 0.5."""
    return sum(w * budget_scores[b] for w, b in zip(WEIGHTS, BUDGETS))


def evaluate_benchmark(predict, records, seed=0, max_eval_items=100, predict_many=None, min_items=MIN_ITEMS):
    """{subject_id: {budget: brier}} for one benchmark.
    For every budget: build the reveal list, then for every subject predict its evaluation items
    with that list. max_eval_items caps the scored items per pair for speed (100 vs all moved the
    DEV number by 0.0003). Every recorded trial of a pair is scored with the same prediction."""
    pairs = split_pairs(records, seed, min_items)
    idx = index_responses(records)
    scores = defaultdict(dict)
    for budget in BUDGETS:
        labeled = labeled_at(pairs, idx, budget)
        for subject_id, (_, eval_items) in pairs.items():
            if max_eval_items:
                eval_items = eval_items[:max_eval_items]
            labeled_copy = copy.deepcopy(labeled)                # a predictor must not mutate the reveals
            firsts = [idx[(subject_id, i)][0] for i in eval_items]
            inputs = [[dict(r["subject"]), dict(r["item"])] for r in firsts]
            if predict_many is not None:                         # batched path: same answers, one call
                ps = [float(p) for p in predict_many(inputs, labeled_copy)]
            else:
                ps = [float(predict(inp, labeled_copy)) for inp in inputs]
            preds, labels = [], []
            for item_id, p in zip(eval_items, ps):
                if not 0.0 <= p <= 1.0:
                    raise ValueError(f"prediction {p} outside [0, 1]")
                for t in idx[(subject_id, item_id)]:
                    preds.append(p); labels.append(t["label"])
            scores[subject_id][budget] = brier(preds, labels)
    return scores



DEV = ["hallusionbench", "ikp", "multi_swebench", "mmlupro", "editbench", "swebench_live",
       "researchcodebench", "legalbench", "tau2_bench", "real_webagents", "mtbbench",
       "workbench_revisited", "hle"]


def evaluate(predictor, benchmarks=DEV, train_pool=None, seed=0, max_eval_items=100, items_subset=None, min_items=MIN_ITEMS, verbose=True):
    """Leave one benchmark out evaluation every pair weighted equally as on the platform
    
    'predictor' is a predict(intput, labeled) function or an object with .predict and optionally .fit(benchmark_names) and .predict_many(input, labeled) if it has .fit it is refit on train_pool
    minus the held-out benchmark before that benchmark is scored so the model never trains on what it is scored on. exactly like the official evaluation protocol.
    """

    train_pool = train_pool or benchmarks # (benchmark, subject_id, {budget: brier})

    all_pairs = [] 

    for name in benchmarks:
        if hasattr(predictor, "fit"):
            predictor.fit([b for b in train_pool if b != name]) # fit on all benchmarks except the held-out one
            try:                                                 # release the GPU cache between fits (Windows counts it as commit)
                import gc, torch
                gc.collect(); torch.cuda.empty_cache()
            except ImportError:
                pass

        predict = predictor.predict if hasattr(predictor, "predict") else predictor
        predict_many = getattr(predictor, "predict_many", None)

        records = load_benchmark(name)

        if items_subset is not None: # only items that have an LLM rating
            keep = items_subset.get(name, set())
            records = [r for r in records if r["item_id"] in keep]

        scores = evaluate_benchmark(predict, records, seed, max_eval_items, predict_many, min_items)
        if not scores:
            print(f"  {name:28s} no subject has {min_items}+ items, skipped"); continue
        all_pairs += [(name, sid, s) for sid, s in scores.items()]

        if verbose:
            print(f"  {name:28s} pairs={len(scores):4d}  ALC={sum(alc(s) for s in scores.values()) / len(scores):.4f}")

    per_budget = {b: sum(s[b] for _, _, s in all_pairs) / len(all_pairs) for b in BUDGETS}
    overall = sum(alc(s) for _, _, s in all_pairs) / len(all_pairs)
    per_benchmark = {}
    for n in benchmarks:
        subset = [s for m, _, s in all_pairs if m == n]
        if subset:
            per_benchmark[n] = sum(alc(s) for s in subset) / len(subset)
    if verbose:
        print("  " + "  ".join(f"B{b}={per_budget[b]:.3f}" for b in BUDGETS))
        print(f"  Brier ALC = {overall:.4f} over {len(all_pairs)} pairs")
    return {"alc": overall, "per_budget": per_budget, "per_benchmark": per_benchmark, "n_pairs": len(all_pairs)}





#________________________________ BASELINES_________________________________#

class EmpiricalMean:
    """basic baseline: the subject's observed success rate on this benchmark, computed from its own reveals. 0.5 when it has none. No training, no shrinkag. """

    def predict(self, input, labeled=None):
        subject, item = input

        mine = [label for (s, it), label in (labeled or [])
                if s == subject and it["benchmark_id"] == item["benchmark_id"]]
        return sum(mine) / len(mine) if mine else 0.5


class ShrunkMean:
    """Empirical mean with a Beta prior: pretend to have already seen `alpha` successes and
    `beta` failures. One real reveal then moves the estimate from 0.5 only a little, instead
    of all the way to 0 or 1. With many reveals the prior washes out."""

    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha, self.beta = alpha, beta

    def predict(self, input, labeled=None):
        subject, item = input
        mine = [label for (s, it), label in (labeled or [])
                if s == subject and it["benchmark_id"] == item["benchmark_id"]]
        return (self.alpha + sum(mine)) / (self.alpha + self.beta + len(mine))



class HierarchicalMean:
    """Two levels of shrinkage. The benchmark's rate is estimated from ALL subjects' reveals on
    it, shrunk toward 0.5; the subject's rate is then shrunk toward that benchmark rate instead
    of toward 0.5. The first use of other subjects' reveals."""

    def __init__(self, subject_strength=8.0, benchmark_strength=4.0):
        self.subject_strength, self.benchmark_strength = subject_strength, benchmark_strength

    @staticmethod
    def shrink(labels, center, strength):
        """Mean of `labels`, pulled toward `center` as if `strength` extra observations sat there."""
        return (sum(labels) + center * strength) / (len(labels) + strength)

    def predict(self, input, labeled=None):
        subject, item = input
        bench_labels, my_labels = [], []
        for (s, it), label in labeled or []:
            if it["benchmark_id"] == item["benchmark_id"]:
                bench_labels.append(label)
                if s == subject:
                    my_labels.append(label)
        center = self.shrink(bench_labels, 0.5, self.benchmark_strength)
        return self.shrink(my_labels, center, self.subject_strength)

