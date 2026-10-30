"""Rate DEV items for difficulty with the LLM rater in submission/model.py, and check the ratings
against real item accuracy.  ->  data/ratings.jsonl

Run from the repo root:  python rate.py                   (80 items per DEV benchmark, a quick probe)
                         python rate.py --per-benchmark 500
                         python rate.py --dev-full        (every item evaluate.py's protocol touches, ~13k,
                                                           so the DEV score needs no API calls)
                         python rate.py --analyze         (report only, on what is already rated)
Rated items are cached by text: re-running only rates what is missing.
"""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "submission"))
from model import rater_client, rate_text, text_key, RATER_MODEL   # noqa: E402
from evaluate import RAW, DEV, load_benchmark, split_pairs           # noqa: E402

RATINGS = ROOT / "data" / "ratings.jsonl"


def load_ratings():
    """{text_key: record} of everything rated so far."""
    out = {}
    if RATINGS.exists():
        for line in RATINGS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); out[r["key"]] = r
    return out


def sample_items(benchmarks, per_benchmark):
    """Deterministic sample of items that were attempted at least 3 times, with their accuracy."""
    rows = []
    for b in benchmarks:
        t = RAW / b / "formatted_tables"
        items = pd.read_parquet(t / "items.parquet", columns=["item_id", "content"])
        resp = pd.read_parquet(t / "responses.parquet", columns=["item_id", "response"]).dropna()
        acc = resp.groupby("item_id").response.agg(["mean", "count"])
        items = items.merge(acc, left_on="item_id", right_index=True)
        items = items[items["count"] >= 3]
        rng = np.random.RandomState(int(hashlib.sha256(b.encode()).hexdigest()[:8], 16))
        pick = items.iloc[rng.choice(len(items), min(per_benchmark, len(items)), replace=False)]
        rows += [{"benchmark": b, "item_id": r.item_id, "text": str(r.content or ""),
                  "acc": float(r.mean), "n": int(r.count)} for r in pick.itertuples()]
    return rows


def protocol_items(benchmarks):
    """Every item evaluate.py can touch on these benchmarks: each pair's eval items plus its 31 label
    candidates, with the item's accuracy over all subjects. Rating these makes the DEV score
    reproducible with no API calls at evaluation time."""
    rows, seen = [], set()
    for b in benchmarks:
        recs = load_benchmark(b)
        text = {r["item_id"]: (r["item"].get("item_content") or "") for r in recs}
        acc = pd.DataFrame({"item_id": [r["item_id"] for r in recs], "y": [r["label"] for r in recs]}) \
                .groupby("item_id").y.agg(["mean", "count"])
        for _, (train_items, eval_items) in split_pairs(recs, 0).items():
            for i in list(eval_items[:100]) + list(train_items[:31]):
                if (b, i) not in seen:
                    seen.add((b, i))
                    rows.append({"benchmark": b, "item_id": i, "text": text[i],
                                 "acc": float(acc.loc[i, "mean"]), "n": int(acc.loc[i, "count"])})
    return rows


def rate_rows(rows, workers=8):
    client = rater_client()
    have = load_ratings()
    todo = [r for r in rows if text_key(r["text"]) not in have]
    print(f"{len(rows)} items, {len(todo)} to rate with {RATER_MODEL}", flush=True)
    t0 = time.time(); done = 0
    with RATINGS.open("a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(rate_text, client, r["text"]): r for r in todo}
        for fut in as_completed(futures):
            r = futures[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001  (refusals, unparseable replies: skip the item)
                print(f"  skipped {r['benchmark']}/{r['item_id']}: {e.__class__.__name__}", flush=True)
                continue
            f.write(json.dumps({"key": text_key(r["text"]), "benchmark": r["benchmark"], "item_id": r["item_id"],
                                "acc": r["acc"], "n": r["n"], "model": RATER_MODEL, **res}) + "\n")
            f.flush(); done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)}  {time.time() - t0:.0f}s", flush=True)
    print(f"rated {done} new items in {time.time() - t0:.0f}s")


def analyze():
    """Does the rating track difficulty? Spearman correlation with item accuracy, within each
    benchmark (item level) and across benchmarks on their means (benchmark level).
    Rows without an accuracy (ratings imported from elsewhere) are skipped."""
    from scipy.stats import spearmanr
    df = pd.DataFrame(load_ratings().values())
    print(f"\n{len(df)} rated items across {df.benchmark.nunique()} benchmarks\n")
    df = df.dropna(subset=["p_correct", "acc"])
    per = []
    for b, g in df.groupby("benchmark"):
        if len(g) >= 20:
            c = spearmanr(g.p_correct, g.acc).correlation; per.append(c)
            print(f"  {b:24s} n={len(g):4d}  item-level corr={c:+.3f}   mean rating {g.p_correct.mean():5.1f}  mean acc {100 * g.acc.mean():5.1f}")
    print(f"  average item-level corr: {np.mean(per):+.3f}")
    bm = df.groupby("benchmark").agg(rating=("p_correct", "mean"), acc=("acc", "mean"))
    print(f"\nbenchmark level ({len(bm)}): corr(mean rating, mean accuracy) = {spearmanr(bm.rating, bm.acc).correlation:+.3f}")
    print("domains:", df.domain.value_counts().to_dict())



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--per-benchmark", type=int, default=80)
    ap.add_argument("--dev-full", action="store_true", help="rate every item the DEV protocol uses")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    RATINGS.parent.mkdir(parents=True, exist_ok=True)
    if not args.analyze:
        rows = protocol_items(DEV) if args.dev_full else sample_items(DEV, args.per_benchmark)
        rate_rows(rows, args.workers)
    analyze()
