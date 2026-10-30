"""Download the training data from measurement-db on HuggingFace
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

REPO = "aims-foundations/measurement-db"
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"            # hf_hub_download mirrors the dataset's folder layout here
CATALOG = ROOT / "data" / "catalog.csv"



def fetch(path):
    """download one file into data/raw. None if the folder has no such file"""
    try:
        return hf_hub_download(REPO, path, repo_type="dataset", local_dir=RAW)
    except EntryNotFoundError:
        return None

def build_catalog():
    # one API call lists the top-level folders; listing every file would take minutes
    folders = sorted(e.path for e in HfApi().list_repo_tree(REPO, repo_type="dataset")
                     if e.path and not e.path.startswith(".") and "." not in e.path)
    print(f"{len(folders)} folders; fetching each benchmarks.parquet ...")
    with ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda f: fetch(f"{f}/formatted_tables/benchmarks.parquet"), folders))

    frames = []
    for path in sorted(RAW.glob("*/formatted_tables/benchmarks.parquet")):
        df = pd.read_parquet(path)
        df.insert(0, "dir", path.parts[-3])   # folder name = the benchmark name used everywhere
        frames.append(df)
    catalog = pd.concat(frames, ignore_index=True)
    catalog.to_csv(CATALOG, index=False)
    print(f"{len(catalog)} benchmarks -> {CATALOG}")


TABLES = ("subjects", "items", "responses")    # traces and assets are large and not needed


def usable_benchmarks():
    """The 160 public, binary, item-level benchmarks: what the hidden test set looks like."""
    cat = pd.read_csv(CATALOG)
    mask = (cat.release == "public") & (cat.response_type == "binary") & (cat.granularity == "item")
    return sorted(cat.loc[mask, "dir"])


def download_tables():
    names = usable_benchmarks()
    jobs = [f"{b}/formatted_tables/{t}.parquet" for b in names for t in TABLES]
    print(f"{len(names)} benchmarks x {len(TABLES)} tables = {len(jobs)} files (~5 GB)")
    with ThreadPoolExecutor(8) as pool:
        for k, _ in enumerate(pool.map(fetch, jobs), 1):
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}", flush=True)
    print(f"done -> {RAW}")


if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    if not CATALOG.exists():
        build_catalog()
    download_tables()