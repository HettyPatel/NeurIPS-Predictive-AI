"""Regression test for the shift cache in submission/model.py.

The fitted shifts (d_bench, d_subject, d_item) are cached on a fingerprint of the reveal list.
The fingerprint must cover the whole list: two lists that differ in any card must not share shifts.

    python tests/test_shift_cache.py          (or: pytest tests/)

Needs numpy only. The MiniLM encoder is replaced by a deterministic stand-in, so nothing is downloaded.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "submission"))
import model as M  # noqa: E402


def _fake_embed(texts):
    """Deterministic unit vectors from a hash of the text, in place of the sentence encoder."""
    out = []
    for t in texts:
        seed = int(hashlib.sha256((t or " ").encode()).hexdigest()[:8], 16)
        v = np.random.default_rng(seed).standard_normal(384).astype(np.float32)
        out.append(v / np.linalg.norm(v))
    return np.stack(out)


M.embed = _fake_embed


def fresh():
    m = M.Model.load()
    m.live_rating = False          # no API calls: unrated items get a neutral rating
    return m


def subject(name):
    return {"normalized_name": name}


def item(text, bench="bench_a"):
    return {"benchmark_id": bench, "item_content": text}


S, OTHER = subject("model-x"), subject("model-y")
TARGET = [S, item("target item, never revealed")]
BASE = [([S, item("first item")], 1), ([S, item("middle item")], 1), ([S, item("last item")], 0)]


def variant(**change):
    """BASE with one card altered; the first and last cards and the length stay the same where possible."""
    cards = [([dict(s), dict(it)], y) for (s, it), y in BASE]
    kind, value = next(iter(change.items()))
    (s, it), y = cards[1]
    if kind == "label":
        cards[1] = ([s, it], value)
    elif kind == "text":
        cards[1] = ([s, item(value)], y)
    elif kind == "subject":
        cards[1] = ([value, it], y)
    elif kind == "bench":                       # every card moves to another benchmark
        cards = [([s, item(it["item_content"], value)], y) for (s, it), y in cards]
    elif kind == "last_tail":                   # last item differs only after its first 80 characters
        pad = "x" * 100
        cards[2] = ([s, item(pad + " one")], 0)
        return cards, [([dict(a), dict(b)], c) for (a, b), c in cards[:2]] + [([S, item(pad + " two")], 0)]
    return cards


def check(first, second, target=TARGET):
    """Predict with `first`, then with `second`, on one model; the second answer must equal a fresh fit."""
    cached = fresh()
    cached.predict(target, first)
    got = cached.predict(target, second)
    want = fresh().predict(target, second)
    assert abs(got - want) < 1e-9, f"stale shifts reused: cached {got:.4f}, fresh {want:.4f}"
    return got, want


def test_middle_label_changed():
    """The reviewers' case: three reveals, the middle label flipped."""
    got, _ = check(BASE, variant(label=0))
    assert abs(got - fresh().predict(TARGET, BASE)) > 1e-3, "the two lists should give different predictions"


def test_middle_item_changed():
    # the swapped item is the one being predicted for another subject, so d_item differs
    target = [OTHER, item("middle item")]
    check(BASE, variant(text="a different middle item"), target)


def test_middle_subject_changed():
    check(BASE, variant(subject=OTHER))


def test_benchmark_changed():
    check(BASE, variant(bench="bench_b"))


def test_last_item_differs_after_80_chars():
    a, b = variant(last_tail=None)
    pad = "x" * 100
    check(a, b, [OTHER, item(pad + " two")])


def test_same_list_is_reused():
    """An equal list (a deep copy, as the evaluator passes) must hit the cache, not refit."""
    m = fresh()
    m.predict(TARGET, BASE)
    corr = m._corr
    m.predict(TARGET, variant(label=1))         # equal content, new objects
    assert m._corr is corr


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for n, f in tests:
        f()
        print(f"ok  {n}")
    print(f"{len(tests)} passed")
