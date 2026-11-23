"""The DEV experiments behind the report's tables, one command each.

    python experiments.py main       # Table 1: ours, ours without the rating, shifts only, three baselines
    python experiments.py ablate     # Table 2: one shift removed at a time
    python experiments.py cache      # effect of the shift-cache fingerprint fix on the DEV score
    python experiments.py all        # the three above in one process, training the priors once
    python experiments.py sparse     # Section 5: DEV with two randomly drawn subjects per benchmark

Needs data/raw and data/feat (download.py, features.py). No API calls: ratings come from data/ratings.jsonl.
The leave-one-benchmark-out priors are trained once per held-out benchmark and shared by every variant in a
run, so variants differ only in what the experiment changes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "submission"))
import model as M                                                                    # noqa: E402
import evaluate as E                                                                 # noqa: E402
from evaluate import (BUDGETS, DEV, EmpiricalMean, HierarchicalMean, ShrunkMean,      # noqa: E402
                      evaluate, usable_benchmarks)

_load, _loaded = E.load_benchmark, {}
E.load_benchmark = lambda name: _loaded[name] if name in _loaded else _loaded.setdefault(name, _load(name))   # read each benchmark once
NETS = ROOT / "data" / "loo_nets"          # trained leave-one-out priors, kept between runs (gitignored)

RATINGS = ROOT / "data" / "ratings.jsonl"
NO_RATING = {k: 0.0 for k in ["knowledge", "reasoning_math", "coding", "legal_medical_finance",
                              "agentic_tool_use", "multimodal", "default"]}
OFF = 1e9          # a shift with this strength stays at 0: the way a shift is "removed"


class SharedPrior(M.Model):
    """The submitted model, with the trained prior network cached per training pool so that several
    variants evaluated in one run reuse the same leave-one-out networks."""
    _nets = {}

    def fit(self, sources, **kw):
        key = (tuple(sorted(sources)), self.hidden, self.epochs, self.lr, self.batch, self.seed)
        if key not in SharedPrior._nets:
            import hashlib
            f = NETS / (hashlib.sha256(repr(key).encode()).hexdigest()[:16] + ".npz")
            if f.exists():
                z = np.load(f)
                self.net = [(z[f"W{k}"], z[f"b{k}"]) for k in range(len(z.files) // 2)]
            else:
                super().fit(sources, **kw)
                NETS.mkdir(parents=True, exist_ok=True)
                np.savez(f, **{f"{n}{k}": a for k, (W, b) in enumerate(self.net) for n, a in (("W", W), ("b", b))})
            SharedPrior._nets[key] = self.net
        self.net = SharedPrior._nets[key]
        self.reset_run()
        return self


class ShiftsOnly(M.Model):
    """Item-level pooling without the prior or the rating: the three shifts on a flat logit of 0.
    Uses the revealed outcomes and nothing else, like the hierarchical mean, plus the item level."""
    def fit(self, sources, **kw):                # nothing to train
        return self

    def prior_logits(self, subjects, items):
        return np.zeros(len(items), dtype=np.float32)


class LegacyKey(SharedPrior):
    """The submitted fingerprint (entry 1031289): list length, first subject and label, and the first
    80 characters and the label of the last item. Kept here only to measure what the fix changes.
    Counts the calls where it reused shifts that the full fingerprint would have refit."""
    stale = calls = 0

    def reset_run(self):
        super().reset_run()
        self._full_key = None

    def _corrections(self, labeled):
        full = M.Model._reveal_key(labeled)
        LegacyKey.calls += 1
        if self._reveal_key(labeled) == self._labeled_key:      # the submitted code answers from the cache here
            LegacyKey.stale += full != self._full_key
            return self._corr
        self._full_key = full
        return super()._corrections(labeled)                     # refits, keyed by the submitted fingerprint

    @staticmethod
    def _reveal_key(labeled):
        return (len(labeled), M.subject_key(labeled[0][0][0]), labeled[0][1],
                (labeled[-1][0][1].get("item_content") or "")[:80], labeled[-1][1])


def ours(cls=SharedPrior, **kw):
    return cls(live_rating=False, **kw).load_ratings(RATINGS)


def run(name, predictor, args, quiet=False, **kw):
    r = evaluate(predictor, DEV, train_pool=usable_benchmarks(), seed=args.seed,
                 max_eval_items=args.max_eval_items, verbose=args.verbose and not quiet, **kw)
    if quiet:
        return r
    print(f"{name:34s} ALC {r['alc']:.4f}   " + "  ".join(f"B{b} {r['per_budget'][b]:.3f}" for b in BUDGETS)
          + f"   ({r['n_pairs']} pairs)", flush=True)
    return r


def main_table(args):
    run("empirical mean", EmpiricalMean(), args)
    run("shrunk mean", ShrunkMean(), args)
    run("hierarchical mean", HierarchicalMean(), args)
    run("shifts only (no prior, no rating)", ShiftsOnly(live_rating=False), args)
    run("ours without the rating", ours(rating_weight=dict(NO_RATING)), args)
    run("ours", ours(), args)


def ablate(args):
    run("ours", ours(), args)
    run("without d_bench", ours(bench_strength=OFF), args)
    run("without d_subject", ours(subject_strength=OFF), args)
    run("without d_item", ours(item_strength=OFF), args)
    run("without any shift", ours(bench_strength=OFF, subject_strength=OFF, item_strength=OFF), args)


def cache(args):
    new = run("full fingerprint (fixed)", ours(), args)
    old = run("submitted fingerprint", ours(LegacyKey), args)
    print(f"\nsubmitted fingerprint reused stale shifts on {LegacyKey.stale} of {LegacyKey.calls} calls")
    print(f"ALC difference (submitted - fixed): {old['alc'] - new['alc']:+.6f}")
    for b in BUDGETS:
        print(f"  B{b:<2d} {old['per_budget'][b] - new['per_budget'][b]:+.6f}")


def sparse(args):
    """DEV with only a few subjects per benchmark, as when a new benchmark has little coverage.
    For each draw, args.subjects subjects are chosen at random on every benchmark and all other
    subjects are removed, so their outcomes are neither revealed nor scored. Scores are averaged over
    the draws; every draw has the same number of pairs per benchmark."""
    models = {"hierarchical mean": HierarchicalMean(), "shifts only (no prior, no rating)": ShiftsOnly(live_rating=False),
              "ours without d_item": ours(item_strength=OFF), "ours": ours()}
    res = {n: [] for n in models}
    for draw in range(args.draws):
        for n, m in models.items():
            res[n].append(run(n, m, args, quiet=True, n_subjects=args.subjects, subject_seed=draw))
        print(f"draw {draw:2d}  " + "  ".join(f"{n} {r[-1]['alc']:.4f}" for n, r in res.items())
              + f"  ({res['ours'][-1]['n_pairs']} pairs)", flush=True)
    print(f"\n{args.subjects} subjects per benchmark, mean over {args.draws} draws (+- standard error)")
    for n, rs in res.items():
        a = np.array([r["alc"] for r in rs])
        print(f"{n:34s} ALC {a.mean():.4f} +- {a.std(ddof=1) / np.sqrt(len(a)):.4f}   "
              + "  ".join(f"B{b} {np.mean([r['per_budget'][b] for r in rs]):.3f}" for b in BUDGETS))
    d = np.array([h["alc"] - o["alc"] for h, o in zip(res["hierarchical mean"], res["ours"])])
    print(f"hierarchical mean - ours: {d.mean():+.4f} +- {d.std(ddof=1) / np.sqrt(len(d)):.4f}; ours better in {(d > 0).sum()} of {len(d)} draws")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiment", choices=["main", "ablate", "cache", "all", "sparse"])
    ap.add_argument("--max-eval-items", type=int, default=100, help="scored items per pair (the report uses 20)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true", help="print the per-benchmark scores")
    ap.add_argument("--subjects", type=int, default=2, help="sparse: subjects kept per benchmark")
    ap.add_argument("--draws", type=int, default=20, help="sparse: random draws of the subjects")
    args = ap.parse_args()
    if args.experiment == "sparse":
        sparse(args); sys.exit()
    todo = {"main": main_table, "ablate": ablate, "cache": cache}
    for name in (todo if args.experiment == "all" else [args.experiment]):   # "all" trains the priors once
        print(f"\n== {name} ==", flush=True)
        todo[name](args)
