"""Item prior: a learned prior per subject, an LLM difficulty rating per item, and three
corrections estimated from the reveals. The platform imports this file and calls predict()."""

import json
import math
from pathlib import Path

import numpy as np

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"   # listed in models.txt: the platform pre-downloads it
HERE = Path(__file__).resolve().parent
MAX_CHARS = 2000                                      # text fed to the encoder; it truncates at 256 tokens anyway

#___________________ TEXT ENCODER ___________________#

_encoder = None

def get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import SentenceTransformer
        _encoder = SentenceTransformer(ENCODER)
    return _encoder

def embed(texts):
    """(n, 384) float32 unit vectors."""
    texts = [(t or " ")[:MAX_CHARS] for t in texts]
    return np.asarray(get_encoder().encode(texts, batch_size=256, normalize_embeddings=True,
                                           show_progress_bar=False), dtype=np.float32)


###__________________ SUBJECT NAME ___________________#
NAME_KEYS = ["normalized_name", "model_identifier", "source_model_label", "model_name",
             "source_model", "base_model", "display_name", "model"]

def subject_name(subject):
    """Best available name text. normalized_name is empty on most benchmarks; the name is then in
    subject_features_extra as key=value;key=value (or, rarely, JSON)."""
    if subject.get("normalized_name"):
        return subject["normalized_name"]
    extra = subject.get("subject_features_extra") or ""
    fields = {}
    if extra.startswith("{"):
        try:
            fields = json.loads(extra)
        except ValueError:
            fields = {}
    elif "=" in extra:
        fields = dict(kv.split("=", 1) for kv in extra.split(";") if "=" in kv)
    for k in NAME_KEYS:
        if isinstance(fields, dict) and fields.get(k):
            return str(fields[k]).strip()
    return subject.get("provider") or "unknown"


#__________ IDENTIFIERS (not features) ______#
# subject_key -> which subject is this?  (groups reveals by subject, caches per subject)
# text_key    -> which item is this?     (keys the embedding and rating caches)

def subject_key(subject):
    """Dict -> string, so a subject can be a dictionary key and compared to reveals."""
    return json.dumps(subject, sort_keys=True)

def text_key(text):
    """Stable short id for an item text; the rating caches are keyed by it."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


##_______________ LLM ITEM-DIFFICULTY RATER _________####
# Haiku is the cheapest model that rates well; Opus 5.5 rates slightly better (0.1803 vs 0.1849 on the
# DEV subset) at ~10x the cost. rate.py rates DEV items offline; on the platform unseen items are rated live.

RATER_MODEL = "claude-haiku-5-5"
RATER_MAX_CHARS = 3000
RATER_SYSTEM = """You estimate how hard a benchmark item is for current frontier AI language models
(GPT-4o / Claude 3.5 Sonnet / Gemini 1.5 Pro class and newer, as of 2025). The item may be a
question, a coding or agent task, a judgment prompt, or a multimodal task whose image you cannot see.

Reply with JSON only, no prose:
{"p_correct": <integer 0-100, your estimate of the percentage of such models that would answer
this item correctly or complete it successfully>,
 "difficulty": <integer 1-5; 1 = almost every model gets it right, 5 = almost none do>,
 "domain": <one of: "knowledge", "reasoning_math", "coding", "agentic_tool_use", "safety_alignment",
 "language_translation", "multimodal", "legal_medical_finance", "other">}"""


def rater_client():
    """Anthropic client. The key comes from the environment, or from submission_config.json next
    to this file (that is how the zip carries it onto the platform)."""
    import os
    import anthropic
    if not os.environ.get("ANTHROPIC_API_KEY"):
        cfg = HERE / "submission_config.json"
        if cfg.exists():
            os.environ["ANTHROPIC_API_KEY"] = json.loads(cfg.read_text()).get("anthropic_api_key", "")
    return anthropic.Anthropic(timeout=30.0, max_retries=1)


def rate_text(client, text):
    """One rater call -> {"p_correct": 0-100, "difficulty": 1-5, "domain": str}. Raises on failure."""
    import re
    msg = client.messages.create(
        model=RATER_MODEL, max_tokens=1500, output_config={"effort": "low"},
        system=RATER_SYSTEM, messages=[{"role": "user", "content": "Item:\n" + text[:RATER_MAX_CHARS]}])
    if msg.stop_reason == "refusal":
        raise RuntimeError("refusal")
    raw = next(b.text for b in msg.content if b.type == "text").strip()
    for blk in reversed(re.findall(r"\{[^{}]*\}", raw)):      # the answer is the last {...} block
        try:
            r = json.loads(blk.replace("'", '"'))
        except ValueError:
            continue
        if "p_correct" in r:
            return {"p_correct": float(r["p_correct"]), "difficulty": float(r.get("difficulty", 3)),
                    "domain": str(r.get("domain", "other"))}
    raise ValueError("no JSON rating in reply: " + raw[:80])


'''
rate.py --dev-full: 21,888 rated items; within-benchmark Spearman(rating, item accuracy) on the 9,300 with accuracy:

  editbench                n= 344  +0.280        multi_swebench           n= 741  +0.113
  hallusionbench           n= 791  +0.188        real_webagents           n= 210  +0.111
  hle                      n= 524  +0.129        researchcodebench        n= 207  +0.177
  ikp                      n= 688  +0.907        swebench_live            n= 374  +0.120
  legalbench               n=1664  +0.295        tau2_bench               n= 220  -0.068
  mmlupro                  n=3042  +0.576        workbench_revisited      n= 263  +0.336
  mtbbench                 n= 252  -0.088        average item-level corr: +0.237
benchmark level (13): corr(mean rating, mean accuracy) = +0.396

The rater is strong on knowledge/math items, weak on coding/agentic ones, and blind on multimodal ones
(the image is not in the data) and on items whose first 3000 characters are a shared preamble.
'''


#################### MODEL #############################
# P(subject answers item correctly) is built from three parts, all added in logit space:
#
#   1. prior      a small network, 385 -> 256 -> 64 -> 1, trained offline on the public benchmarks.
#                 Input: the subject's name embedding (384) + the item's log length (1).
#                 Output: how strong this subject is on a typical item. Nothing about item content:
#                 item embeddings in the prior ranked difficulty backwards on unseen benchmarks,
#                 and even a perfect subject prior is worth only ~0.016 ALC (oracle test on DEV).
#   2. rating     an LLM's estimate of how hard this item is (RATER_* above), as a logit weighted by
#                 the item's domain (trust knowledge/math ratings fully, coding/legal partly, agentic
#                 and multimodal not at all). The only signal that moves budget 0: 0.246 -> 0.221.
#   3. shifts     three corrections estimated from the revealed labels, each a one-number logistic
#                 fit shrunk toward 0 (Newton steps, a few lines of numpy):
#                   d_bench    from every reveal on the benchmark   (fixes the benchmark's level)
#                   d_subject  from this subject's own reveals      (fixes the subject's level)
#                   d_item     from other subjects' reveals on this exact item (needs >= 2 subjects
#                              on the benchmark; this is most of the gain when many subjects share it)
#
#   answer = sigmoid(prior_scale * prior + rating + d_bench + d_subject + d_item)
#
# Plumbing: fit() trains part 1 and runs only locally; save()/load() move the weights through
# weights.npz + meta.json. predict() runs on the platform with numpy plus one API call per unseen
# item for part 2. Ratings are cached in _rating_cache.jsonl next to this file so the platform's
# worker processes rate each item once; max_live_ratings caps spend. The shifts are refit only when
# the reveal list changes (fingerprint check), so predict() is cheap per call.
#
# Numbers (Brier ALC, lower is better; 0.5 everywhere = 0.25):
#   DEV, 13 held-out benchmarks, all subjects pooled:         0.194   (hierarchical mean 0.228)
#   DEV, 1-2 subjects per benchmark like the hidden test:     0.207   (hierarchical mean 0.229)
#   Codabench formative runs, 9 pairs each, +-0.015 noise:    0.187 - 0.209
# Everything else we tried (item embeddings, matrix factorisation, TabPFN, calibration, stronger or
# batched raters, per-pair slopes, an uncertainty-sampling acquisition policy: +0.001 on DEV but worse
# on the two paired platform comparisons, so not shipped) was within 0.002 or worse; see README.md.


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _fit_offset(z, y, precision, iters=6):
    """The d that maximises  sum log-lik(sigmoid(z + d), y) - precision/2 * d^2, by Newton steps.
    precision = strength of the pull toward 0; one label of evidence weighs about p(1-p) ~ 0.25."""
    if len(z) == 0:
        return 0.0
    d = 0.0
    for _ in range(iters):
        p = _sigmoid(z + d)
        g = precision * d - np.sum(y - p)          # gradient of the penalised negative log-lik
        h = precision + np.sum(p * (1.0 - p))      # its curvature
        d -= g / h
    return float(d)


class Model:
    def __init__(self, bench_strength=2.0, subject_strength=2.0, item_strength=0.5, prior_scale=0.5,
                 rating_weight=None, live_rating=True, max_live_ratings=60000,
                 hidden=(256, 64), epochs=4, lr=1e-3, batch=8192, seed=0):
        # shrinkage of the three shifts: larger = trust the reveals less (precision of a N(0, 1/s) prior)
        self.bench_strength = bench_strength
        self.subject_strength = subject_strength
        self.item_strength = item_strength
        self.prior_scale = prior_scale          # <1 damps the network's spread; 0.5 tuned on DEV
        # how much to trust the LLM rating per domain (logit of p_correct is multiplied by this)
        if rating_weight is None:
            rating_weight = {"knowledge": 1.0, "reasoning_math": 0.8, "coding": 0.6, "legal_medical_finance": 0.5,
                             "agentic_tool_use": 0.0, "multimodal": 0.0, "default": 0.3}
        self.rating_weight = rating_weight
        self.live_rating = live_rating          # rate unseen items through the API (platform); False locally
        self.max_live_ratings = max_live_ratings  # spend cap per process; beyond it unseen items get p = 0.5
        self.hidden, self.epochs, self.lr, self.batch, self.seed = hidden, epochs, lr, batch, seed
        # learned / cached state
        self.net = []                           # [(W, b), ...] numpy layers of the prior network
        self.ratings = {}                       # text_key -> {"p": p_correct in [0, 1], "domain": str}
        self._subj_cache = {}                   # subject_key -> name embedding (384,)
        self._rater = None                      # Anthropic client, created on first use
        self._rater_failures = 0
        self._live_calls = 0
        self._cache_pos = 0                     # bytes of the shared rating cache file already read
        self.reset_run()

    RATING_CACHE = HERE / "_rating_cache.jsonl"  # shared across the platform's worker processes

    def reset_run(self):
        """Per-run memory; the evaluator calls it once per benchmark."""
        self._labeled_key = None                # fingerprint of the reveal list the shifts were fit on
        self._corr = None                       # the fitted (d_bench, d_subject, d_item)

    # ------------------------------------------------------------ part 1: prior
    def _subject_emb(self, subjects):
        """(n, 384) name embeddings, computed once per distinct subject and cached."""
        keys = [subject_key(s) for s in subjects]
        todo = {k: s for k, s in zip(keys, subjects) if k not in self._subj_cache}
        if todo:
            for k, v in zip(todo, embed([subject_name(s) for s in todo.values()])):
                self._subj_cache[k] = v
        return np.stack([self._subj_cache[k] for k in keys])

    def _features(self, subjects, items):
        """(n, 385): name embedding + log item length. fit() builds the same columns from features.py's cache."""
        lens = np.log1p([len(it.get("item_content") or "") for it in items]).astype(np.float32)
        return np.column_stack([self._subject_emb(subjects), lens])

    @staticmethod
    def _mlp(layers, x):
        """Forward pass in numpy: ReLU hidden layers, linear output -> (n,) logits."""
        for W, b in layers[:-1]:
            x = np.maximum(x @ W + b, 0.0)
        W, b = layers[-1]
        return (x @ W + b)[:, 0]

    def prior_logits(self, subjects, items):
        """Damped network logit + the rating term (part 2), per (subject, item)."""
        z = self._mlp(self.net, self._features(subjects, items)) * self.prior_scale
        return z + np.asarray([self._rating_logit(it.get("item_content") or "") for it in items])

    # ------------------------------------------------------------ training (local only)
    def fit(self, sources, feat_dir=HERE.parent / "data" / "feat"):
        """Train the prior on the cached features of these benchmarks (features.py -> data/feat/<name>.npz)."""
        X, y, pair_id, bench_id = [], [], [], []
        for name in sources:
            z = np.load(feat_dir / f"{name}.npz")
            if len(z["y"]) == 0:
                continue                                    # a few benchmarks have no usable rows
            si, ii, b = z["subj_idx"], z["item_idx"], len(bench_id)
            # item_len is already log1p(characters) in the cache, matching _features()
            X.append(np.column_stack([z["subj_emb"][si].astype(np.float32), z["item_len"][ii].astype(np.float32)]))
            y.append(z["y"].astype(np.float32))
            pair_id.append(si.astype(np.int64) + 100_000 * b)  # unique (subject, benchmark) id
            bench_id.append(np.full(len(si), b, dtype=np.int64))
        X, y = np.concatenate(X), np.concatenate(y)
        pair_id, bench_id = np.concatenate(pair_id), np.concatenate(bench_id)
        _, inv, counts = np.unique(pair_id, return_inverse=True, return_counts=True)
        w = (1.0 / counts[inv]).astype(np.float32)
        w *= len(w) / w.sum()                               # mean weight 1, so the loss scale is unchanged
        self.net = self._train(X, y, w, bench_id, n_bench=len(sources))
        return self

    def _train(self, X, y, w, bench_id, n_bench):
        """Weighted logistic regression through a small MLP, with a throw-away per-benchmark bias."""
        import torch
        torch.manual_seed(self.seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        dims = [X.shape[1], *self.hidden, 1]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [torch.nn.Linear(a, b), torch.nn.ReLU()]
        net = torch.nn.Sequential(*layers[:-1]).to(dev)    # no ReLU after the output layer
        bias = torch.nn.Embedding(n_bench, 1).to(dev)      # benchmark level: used in training only
        torch.nn.init.zeros_(bias.weight)
        opt = torch.optim.Adam(list(net.parameters()) + list(bias.parameters()), lr=self.lr)
        Xt, yt, wt = (torch.as_tensor(a, device=dev) for a in (X, y, w))
        bt = torch.as_tensor(bench_id, device=dev)
        n = len(yt)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                logit = net(Xt[idx])[:, 0] + bias(bt[idx])[:, 0]
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[idx], reduction="none")
                        * wt[idx]).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        # hand the weights to numpy: torch stores Linear as (out, in), _mlp wants x @ W, so transpose
        return [(m.weight.detach().cpu().numpy().T.copy(), m.bias.detach().cpu().numpy().copy())
                for m in net if isinstance(m, torch.nn.Linear)]

    # ------------------------------------------------------------ persistence
    def save(self, folder=HERE):
        """weights.npz (the network) + meta.json (every setting the platform needs)."""
        arrays = {}
        for k, (W, b) in enumerate(self.net):
            arrays[f"W{k}"], arrays[f"b{k}"] = W, b
        np.savez(folder / "weights.npz", **arrays)
        meta = {"n_layers": len(self.net), "bench_strength": self.bench_strength,
                "subject_strength": self.subject_strength, "item_strength": self.item_strength,
                "prior_scale": self.prior_scale, "rating_weight": self.rating_weight,
                "live_rating": self.live_rating, "max_live_ratings": self.max_live_ratings}
        (folder / "meta.json").write_text(json.dumps(meta, indent=1))

    @classmethod
    def load(cls, folder=HERE):
        meta = json.loads((folder / "meta.json").read_text())
        n = meta.pop("n_layers")
        m = cls(**meta)
        z = np.load(folder / "weights.npz")
        m.net = [(z[f"W{k}"], z[f"b{k}"]) for k in range(n)]
        return m

    # ------------------------------------------------------------ part 2: LLM difficulty rating
    def load_ratings(self, path):
        """Offline ratings from rate.py (data/ratings.jsonl) or a shared cache file; p stored in [0, 1]."""
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                p = r["p"] if "p" in r else r["p_correct"] / 100.0
                self.ratings.setdefault(r["key"], {"p": float(p), "domain": r.get("domain", "other")})
        return self

    def _sync_rating_cache(self):
        """Pull in ratings other processes appended since we last looked (a partial last line is skipped)."""
        try:
            with self.RATING_CACHE.open("r", encoding="utf-8") as f:
                f.seek(self._cache_pos)
                for line in f:
                    if line.endswith("\n"):
                        r = json.loads(line)
                        self.ratings.setdefault(r["key"], {"p": r["p"], "domain": r["domain"]})
                        self._cache_pos += len(line.encode("utf-8"))
        except (OSError, ValueError):
            pass

    def _persist_rating(self, key, r):
        try:
            with self.RATING_CACHE.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "p": r["p"], "domain": r["domain"]}) + "\n")
        except OSError:
            pass                                            # read-only folder: memory cache only

    def _rate_live(self, text):
        """One API call -> (rating, ok). Any failure gives a neutral rating and ok = False."""
        try:
            if self._rater is None:
                self._rater = rater_client()
            res = rate_text(self._rater, text)
            return {"p": res["p_correct"] / 100.0, "domain": res["domain"]}, True
        except Exception as e:  # noqa: BLE001
            self._rater_failures += 1
            if self._rater_failures <= 3:
                print(f"rater failed ({e.__class__.__name__}): neutral rating used", flush=True)
            return {"p": 0.5, "domain": "other"}, False

    def _rating_record(self, text):
        """The rating for an item text: memory -> shared cache file -> live call (under the cap) -> None."""
        k = text_key(text)
        r = self.ratings.get(k)
        if r is None and self.live_rating:
            self._sync_rating_cache()
            r = self.ratings.get(k)
        if r is None and self.live_rating and self._live_calls < self.max_live_ratings:
            self._live_calls += 1
            r, ok = self._rate_live(text)
            self.ratings[k] = r                             # remember failures too, so we never retry an item
            if ok:
                self._persist_rating(k, r)                  # but only share successes with other workers
        return r

    def _rating_logit(self, text):
        """Domain-weighted logit of the rated p_correct; 0 when unrated or the domain has weight 0."""
        r = self._rating_record(text)
        if r is None:
            return 0.0
        w = self.rating_weight.get(r["domain"], self.rating_weight.get("default", 0.0))
        if not w:
            return 0.0
        p = min(max(r["p"], 0.02), 0.98)                    # clip so a confident 0 or 100 is not infinite
        return w * math.log(p / (1 - p))

    # ------------------------------------------------------------ part 3: shifts from the reveals
    def _corrections(self, labeled):
        """Fit d_bench, d_subject, d_item from the revealed (subject, item, label) cards.
        Cached on a fingerprint of the reveal list: predict() is called once per item with the same list."""
        key = (len(labeled), subject_key(labeled[0][0][0]), labeled[0][1],
               (labeled[-1][0][1].get("item_content") or "")[:80], labeled[-1][1])
        if key == self._labeled_key:
            return self._corr
        subjects = [c[0][0] for c in labeled]
        items = [c[0][1] for c in labeled]
        y = np.asarray([float(c[1]) for c in labeled], dtype=np.float32)
        z = self.prior_logits(subjects, items)               # prior + rating: the baseline to correct
        bench = np.asarray([it.get("benchmark_id", "") for it in items])
        skey = np.asarray([subject_key(s) for s in subjects])
        text = np.asarray([it.get("item_content") or "" for it in items])
        d_bench, d_subj, d_item = {}, {}, {}
        for b in set(bench):
            mb = bench == b
            d_bench[b] = _fit_offset(z[mb], y[mb], self.bench_strength)          # 1. benchmark level
            zb = z[mb] + d_bench[b]
            for s in set(skey[mb]):
                ms = skey[mb] == s
                d_subj[(b, s)] = _fit_offset(zb[ms], y[mb][ms], self.subject_strength)   # 2. subject level
            zs = zb + np.asarray([d_subj[(b, s)] for s in skey[mb]])
            for t in set(text[mb]):
                mt = text[mb] == t
                d_item[(b, t)] = _fit_offset(zs[mt], y[mb][mt], self.item_strength)      # 3. this exact item
        self._labeled_key, self._corr = key, (d_bench, d_subj, d_item)
        return self._corr

    # ------------------------------------------------------------ prediction
    def predict_many(self, inputs, labeled=None):
        """Probabilities for a batch of (subject, item) pairs given the revealed cards so far."""
        subjects = [s for s, _ in inputs]
        items = [it for _, it in inputs]
        z = self.prior_logits(subjects, items)
        if labeled:
            d_bench, d_subj, d_item = self._corrections(labeled)
            z = z + np.asarray([d_bench.get(it.get("benchmark_id", ""), 0.0)
                                + d_subj.get((it.get("benchmark_id", ""), subject_key(s)), 0.0)
                                + d_item.get((it.get("benchmark_id", ""), it.get("item_content") or ""), 0.0)
                                for s, it in inputs])
        return [float(min(max(p, 0.001), 0.999)) for p in _sigmoid(z)]   # never bet everything on one item

    def predict(self, input, labeled=None):
        return self.predict_many([input], labeled)[0]


# ------------------------------------------------------------------ platform entry point
_model = Model.load() if (HERE / "weights.npz").exists() and (HERE / "meta.json").exists() else Model()


def predict(input, labeled=None):
    """input = [subject dict, item dict]; labeled = [([subject, item], label), ...] revealed so far."""
    return _model.predict(input, labeled)
