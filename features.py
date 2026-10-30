"""Turn each benchmark's tables into the arrays the model trains on -> data/feat/<benchmark>.npz

Per benchmark:
  - keep up to MAX_ITEMS items (deterministic sample) so embedding stays cheap
  - collapse repeated trials into one row per (subject, item) with the mean response as a soft label
  - embed every kept item's text and every subject's name with MiniLM (384 numbers each)

Run from the repo root (GPU recommended, ~15 min for all 160):  python features.py
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "submission"))
from model import embed, subject_name          # noqa: E402  (same functions the platform runs)
from evaluate import RAW, SUBJECT_FIELDS, clean, usable_benchmarks   # noqa: E402

FEAT = ROOT / "data" / "feat"
MAX_ITEMS = 2000


def build_one(name):
    t = RAW / name / "formatted_tables"
    subjects = pd.read_parquet(t / "subjects.parquet")
    items = pd.read_parquet(t / "items.parquet", columns=["item_id", "content"])
    resp = pd.read_parquet(t / "responses.parquet", columns=["subject_id", "item_id", "response"])
    resp = resp[resp.response.notna()]

    # deterministic item sample: same items every run, seeded from the benchmark name
    item_ids = sorted(items.item_id)
    rng = np.random.RandomState(int(hashlib.sha256(name.encode()).hexdigest()[:8], 16))
    if len(item_ids) > MAX_ITEMS:
        item_ids = sorted(rng.choice(item_ids, MAX_ITEMS, replace=False))
    items = items.set_index("item_id").loc[item_ids]

    # one row per (subject, item): mean over trials -> a soft label in [0, 1] # can add additional samples. 
    resp = resp[resp.item_id.isin(item_ids)]
    agg = resp.groupby(["subject_id", "item_id"]).response.mean().reset_index()
    subjects = subjects[subjects.subject_id.isin(agg.subject_id)].reset_index(drop=True)

    subj_dicts = [{f: clean(getattr(r, f)) for f in SUBJECT_FIELDS} for r in subjects.itertuples()]
    subj_pos = {sid: k for k, sid in enumerate(subjects.subject_id)}
    item_pos = {iid: k for k, iid in enumerate(item_ids)}
    texts = [clean(c) for c in items.content]

    np.savez_compressed(
        FEAT / f"{name}.npz",
        item_emb=embed(texts).astype(np.float16),                      # (n_items, 384)
        item_len=np.log1p([len(x) for x in texts]).astype(np.float32),  # (n_items,)
        subj_emb=embed([subject_name(s) for s in subj_dicts]).astype(np.float16),  # (n_subjects, 384)
        subj_json=np.array([json.dumps(s) for s in subj_dicts]),       # the dicts, for inspection
        subj_idx=agg.subject_id.map(subj_pos).to_numpy(np.int32),      # row -> subject position
        item_idx=agg.item_id.map(item_pos).to_numpy(np.int32),         # row -> item position
        y=agg.response.to_numpy(np.float32),                           # row -> soft label
    )
    return len(subj_dicts), len(item_ids), len(agg)


if __name__ == "__main__":
    FEAT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or usable_benchmarks()
    for k, name in enumerate(names, 1):
        if (FEAT / f"{name}.npz").exists():
            continue
        ns, ni, nr = build_one(name)
        print(f"[{k}/{len(names)}] {name}: {ns} subjects, {ni} items, {nr} rows", flush=True)
    print(f"done -> {FEAT}")


# ---------------------------------------------------------------- what ends up on disk
# One file per benchmark, data/feat/<benchmark>.npz, holding:
#   item_emb   (n_items, 384)     MiniLM embedding of each kept item's text, float16 to halve the size
#   item_len   (n_items,)         log(1 + characters) of the item text
#   subj_emb   (n_subjects, 384)  MiniLM embedding of each subject's name text
#   subj_json  (n_subjects,)      the 8-field subject dict as a JSON string, for inspection only
#   subj_idx   (n_rows,)          for each training row, which subject (position into subj_emb)
#   item_idx   (n_rows,)          for each training row, which item (position into item_emb)
#   y          (n_rows,)          for each training row, the soft label: fraction of trials correct
# A training row is one (subject, item) pair. The model gathers subj_emb[subj_idx] and
# item_len[item_idx] per row when it trains, so embeddings are stored once, not once per row.
# Already-built benchmarks are skipped, so the script can be rerun or interrupted safely.
