"""Order-N byte context model with backoff.

Small, dependency-free, and trained in seconds from whatever corpus resembles
the data being carved.  It is the default answer to the two questions the rest
of the tool keeps asking:

    "is this decode plausible?"      (alignment ranking)
    "what byte belongs in this hole?" (constraint solving)

Both reduce to logprob(byte | preceding bytes), so one model serves both.  The
heavier predictors in this package implement the same interface.
"""
from __future__ import annotations

import math
import pickle
from collections import defaultdict
from pathlib import Path

UNIFORM = -8.0                      # log2(1/256)

# How often a pass over the corpus reports where it has got to. Training 17.9 MB
# of prose at order 6 takes over two minutes, and a caller that cannot say
# whether that is progress or a hang has to guess.
#
# A `progress` argument is always optional and is called as
# `progress(done, total, note)` in units of corpus bytes, monotonically, with
# `total` fixed for the whole run so a caller can render one bar across the
# several passes a training run makes. Raising from it aborts the training --
# which is how the interactive path cancels, and why nothing is assigned to a
# model until its pass has finished.
STEP = 1 << 16


def _trigram_counts(data: bytes, progress=None, base: int = 0,
                    total: int = 0) -> bytearray:
    """How often each of the 16.7 million trigrams occurs, saturating at 255.

    16 MB flat, and one pass. Saturation costs nothing: the only question ever
    asked of it is whether a count reached `min_obs`, which is 2.
    """
    counts = bytearray(1 << 24)
    for i in range(len(data) - 2):
        k = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2]
        if counts[k] < 255:
            counts[k] += 1
        if progress is not None and not i % STEP:
            progress(base + i, total, "counting trigrams")
    return counts


class ContextModel:
    """PPM-style backoff over byte contexts.

    Counts are kept for every context length 0..order.  Prediction walks from
    the longest context down until it finds one observed often enough to trust,
    which keeps rare long contexts from producing overconfident nonsense.
    """

    def __init__(self, order: int = 4, alpha: float = 0.12, min_obs: int = 2):
        self.order = order
        self.alpha = alpha
        self.min_obs = min_obs
        self._counts: list[dict[bytes, dict[int, int]]] = [
            defaultdict(dict) for _ in range(order + 1)]
        self._totals: list[dict[bytes, int]] = [defaultdict(int) for _ in range(order + 1)]
        self.trained_bytes = 0
        # What this was trained on. Empty means unknown, which is what a model
        # saved before provenance existed honestly is.
        self.provenance: dict = {}
        self.calibration = None       # predict/calibration.py, once measured

    def train(self, data: bytes, progress=None, base: int = 0,
              total: int = 0) -> "ContextModel":
        order = self.order
        counts, totals = self._counts, self._totals
        for i, b in enumerate(data):
            top = min(order, i)
            for o in range(top + 1):
                ctx = data[i - o:i]
                d = counts[o][ctx]
                d[b] = d.get(b, 0) + 1
                totals[o][ctx] += 1
            if progress is not None and not i % STEP:
                progress(base + i, total, "training")
        self.trained_bytes += len(data)
        return self

    @property
    def passes(self) -> int:
        """How many times `train_corpus` walks the corpus, so a caller can size
        a progress bar before the first byte is read: count the trigrams, train,
        then drop what cannot be read back."""
        return 1 if self.min_obs <= 1 else 3

    def train_corpus(self, data: bytes, progress=None, base: int = 0,
                     total: int = 0) -> "ContextModel":
        """Train on a corpus that is complete, storing only what will be read.

        Identical in every prediction to `train(data).prune()`, and it never
        allocates the difference. This is ziprec's trick from `mklang`, and it
        rests on a bound rather than a heuristic: a context of length n occurs
        no more often than any trigram inside it, so a context whose leading
        trigram was seen fewer than `min_obs` times cannot itself reach
        `min_obs`, and the lookup would back off past it anyway.

        The same bound lets the loop stop rather than skip. Every longer context
        at this position contains that same rare trigram, so once one is
        refused, all of them are.

        It matters where it costs most. 10.9 MB of ELF at order 6 needed 7.8 GB
        stored in full, because high-entropy data makes nearly every long
        context unique -- which is to say nearly every one of them was memory
        spent on something that could never be consulted.

        `data` must be the whole corpus: a trigram that is rare in one file and
        common across the rest is not rare, and filtering per file would drop it.
        """
        n = len(data)
        total = total or self.passes * n
        if self.min_obs <= 1:
            # Nothing would be filtered anyway. Two of the three passes do not
            # happen, so the bar is told the truth about what it is measuring.
            return self.train(data, progress, base, total)
        trigrams = _trigram_counts(data, progress, base, total)
        order, floor = self.order, self.min_obs
        counts, totals = self._counts, self._totals
        for i, b in enumerate(data):
            top = min(order, i)
            for o in range(top + 1):
                if o >= 3:
                    j = i - o
                    if trigrams[(data[j] << 16) | (data[j + 1] << 8)
                                | data[j + 2]] < floor:
                        break
                ctx = data[i - o:i]
                d = counts[o][ctx]
                d[b] = d.get(b, 0) + 1
                totals[o][ctx] += 1
            if progress is not None and not i % STEP:
                progress(base + n + i, total, "training")
        self.trained_bytes += n
        return self.prune(progress, base + 2 * n, total, n)

    def train_files(self, paths, max_bytes: int | None = None) -> "ContextModel":
        """Train on the whole of what you point at.

        `max_bytes` defaults to no limit. It used to default to 8 MB, which meant
        an operator who assembled a 30 MB corpus and pointed the tool at it got a
        model built from the first 8 MB in directory order and no word about the
        rest. Memory is a real constraint but this was the wrong instrument for
        it: 8 MB of prose and 8 MB of machine code differ by 2x in what they
        cost, so a byte count neither protects the expensive case nor lets the
        cheap one alone. `corpora.auto_order` handles it where the cost is
        actually known, by shortening the context and reporting that it did.
        """
        blob = b""
        budget = max_bytes if max_bytes is not None else None
        for p in paths:
            if budget is not None and budget <= 0:
                break
            chunk = Path(p).read_bytes()
            if budget is not None:
                chunk = chunk[:budget]
                budget -= len(chunk)
            blob += chunk
        return self.train_corpus(blob)

    def prune(self, progress=None, base: int = 0, total: int = 0,
              span: int = 0) -> "ContextModel":
        """Drop the contexts this model's own lookup can never consult.

        `logprob` and `distribution` back off past any context whose total is
        below `min_obs`, so every context stored below that threshold is memory
        spent on something that will never be read. Removing them is exactly
        behaviour-preserving: a missing context reads as total 0, which fails
        the same test and backs off to the same shorter context.

        It matters because that is most of the model. On English text the
        majority of 5- and 6-byte contexts occur exactly once, and this was
        keeping every one of them -- which is a large part of the difference
        between what ziprec can afford at order 6 and what this could. ziprec
        filters rare n-grams as it stores them (`mklang -m`, with a trigram
        pre-pass); this filters them once training is over.

        Call it only when training is finished: a context seen once in one file
        and once in the next is not rare, and pruning between files would throw
        it away.
        """
        for o in range(1, self.order + 1):
            totals, counts = self._totals[o], self._counts[o]
            dead = [ctx for ctx, t in totals.items() if t < self.min_obs]
            for ctx in dead:
                del totals[ctx]
                del counts[ctx]
            if progress is not None:
                # Charged against the corpus like the other passes, spread over
                # the orders, because what the caller is drawing is one bar.
                progress(base + span * o // max(1, self.order), total,
                         f"dropping contexts nothing can read (order {o})")
        return self

    def logprob(self, byte: int, left: bytes) -> float:
        if not self.trained_bytes:
            return UNIFORM
        counts, totals = self._counts, self._totals
        alpha = self.alpha
        for o in range(min(self.order, len(left)), -1, -1):
            ctx = bytes(left[len(left) - o:]) if o else b""
            total = totals[o].get(ctx, 0)
            if total >= self.min_obs:
                c = counts[o][ctx].get(byte, 0)
                return math.log2((c + alpha) / (total + alpha * 256))
        return UNIFORM

    def distribution(self, left: bytes) -> dict[int, float]:
        """Full next-byte distribution given the preceding bytes."""
        if not self.trained_bytes:
            return {}
        counts, totals = self._counts, self._totals
        alpha = self.alpha
        for o in range(min(self.order, len(left)), -1, -1):
            ctx = bytes(left[len(left) - o:]) if o else b""
            total = totals[o].get(ctx, 0)
            if total >= self.min_obs:
                d = counts[o][ctx]
                denom = total + alpha * 256
                return {b: (d.get(b, 0) + alpha) / denom for b in range(256)}
        return {}

    def score(self, data: bytes, limit: int | None = None) -> float:
        """Mean log2 probability per byte.  Higher (closer to 0) is more plausible."""
        n = len(data) if limit is None else min(len(data), limit)
        if not n:
            return UNIFORM
        total = 0.0
        for i in range(n):
            total += self.logprob(data[i], data[max(0, i - self.order):i])
        return total / n

    # ---- persistence --------------------------------------------------------

    def save(self, path: str | Path, provenance: dict | None = None) -> None:
        """Write the model, and what it was trained on.

        Pre-trained corpora here are operator-built, so a saved model IS the
        distribution format -- it gets copied between machines and reused months
        later. A model whose training data is unknown is a hypothesis generator
        with no parentage, and this tool's whole rule is that a hypothesis must
        never be indistinguishable from a fact. So the lineage travels with it.
        """
        Path(path).write_bytes(pickle.dumps({
            "order": self.order, "alpha": self.alpha, "min_obs": self.min_obs,
            "counts": [dict(c) for c in self._counts],
            "totals": [dict(t) for t in self._totals],
            "trained_bytes": self.trained_bytes,
            "provenance": provenance or {},
            "calibration": _cal_dict(self),
        }, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: str | Path) -> "ContextModel":
        blob = pickle.loads(Path(path).read_bytes())
        if blob.get("bidirectional"):
            return Bidirectional.load(path)
        m = cls(blob["order"], blob["alpha"], blob["min_obs"])
        m._counts = [defaultdict(dict, c) for c in blob["counts"]]
        m._totals = [defaultdict(int, t) for t in blob["totals"]]
        m.trained_bytes = blob["trained_bytes"]
        # Absent on models saved before provenance was recorded. An empty dict
        # is the honest answer there: unknown, rather than a claim of nothing.
        m.provenance = blob.get("provenance", {})
        m.calibration = _cal_load(blob)
        return m


def _cal_dict(model):
    """A model's calibration, as stored. None for a model never measured --
    which the floor then treats as raw confidence, and the screen says so."""
    cal = getattr(model, "calibration", None)
    return cal.to_dict() if cal is not None else None


def _cal_load(blob):
    from .calibration import Calibration
    return Calibration.from_dict(blob.get("calibration"))


class Bidirectional:
    """Two n-gram models: one reading forwards, one reading backwards.

    Taken from ziprec, whose `mklang` builds both by default and whose
    documentation says building only the forward half "cuts the file size in
    half but degrades reconstruction". The reason shows up sharply in a carved
    fragment: the unknown bytes that matter most sit at the edge of the
    pre-fragment window, where everything to the left is missing and cannot be
    recovered, while everything to the right is known. A forward model asked to
    predict there is conditioning on nothing.

    The reverse half is the same algorithm trained on the reversed corpus, so
    predicting byte p from the bytes that follow it is just the forward problem
    read the other way round.
    """

    def __init__(self, forward: ContextModel, reverse: ContextModel,
                 weight: float = 0.5):
        self.forward = forward
        self.reverse = reverse
        self.weight = weight          # 0 = forward only, 1 = reverse only
        self.provenance: dict = {}
        self.calibration = None       # predict/calibration.py, once measured

    @property
    def order(self) -> int:
        return self.forward.order

    @property
    def trained_bytes(self) -> int:
        return self.forward.trained_bytes

    @classmethod
    def train_on(cls, data: bytes, order: int = 4, progress=None,
                 **kw) -> "Bidirectional":
        # train_corpus rather than train(): the caller has handed over the whole
        # corpus, which is what lets the rare contexts be skipped instead of
        # stored and then dropped.
        fwd = ContextModel(order=order, **kw)
        rev = ContextModel(order=order, **kw)
        # One bar over both halves. Two models over the same bytes is twice the
        # work and the operator watching it has no reason to care which half is
        # running, only that it is still moving -- so the note says, and the
        # count does not restart.
        span = fwd.passes * len(data)
        total = span * 2

        def half(name):
            """`done` already carries the offset train_corpus was given, so
            this only names the half and holds `total` across both."""
            if progress is None:
                return None
            return lambda done, _tot, note: progress(done, total,
                                                     f"{name} · {note}")

        return cls(fwd.train_corpus(data, half("forwards"), 0, total),
                   rev.train_corpus(data[::-1], half("backwards"), span, total))

    @classmethod
    def train_files(cls, paths, order: int = 4, max_bytes: int | None = None,
                    progress=None, **kw) -> "Bidirectional":
        """No byte limit by default -- see `ContextModel.train_files`."""
        blob = b""
        budget = max_bytes
        paths = list(paths)             # counted in the note below
        for n, p in enumerate(paths, 1):
            if budget is not None and budget <= 0:
                break
            if progress is not None:
                # Reading is charged nothing: the bar it shares is in corpus
                # bytes and the corpus is not measured yet. The note is the
                # point -- on a directory sweep this is where the time goes
                # before any training starts.
                progress(0, 0, f"reading file {n} of {len(paths)}")
            chunk = Path(p).read_bytes()
            if budget is not None:
                chunk = chunk[:budget]
                budget -= len(chunk)
            blob += chunk
        return cls.train_on(blob, order, progress, **kw)

    def logprob(self, byte: int, left: bytes) -> float:
        """Forward only, for callers that hand over just a left context."""
        return self.forward.logprob(byte, left)

    def logprob_both(self, byte: int, left: bytes, right: bytes = b"") -> float:
        """Blend the two directions, using whichever has context to offer.

        With context on one side only the blend collapses to that side, which is
        the point: a byte at the edge of the window is judged by what follows it
        rather than written off.
        """
        if not right:
            return self.forward.logprob(byte, left)
        if not left:
            return self.reverse.logprob(byte, right[::-1])
        f = self.forward.logprob(byte, left)
        r = self.reverse.logprob(byte, right[::-1])
        return (1.0 - self.weight) * f + self.weight * r

    def score(self, data: bytes, limit: int | None = None) -> float:
        return self.forward.score(data, limit)

    def save(self, path: str | Path, provenance: dict | None = None) -> None:
        Path(path).write_bytes(pickle.dumps({
            "bidirectional": True, "weight": self.weight,
            "provenance": provenance or getattr(self, "provenance", {}) or {},
            "calibration": _cal_dict(self),
            "forward": pickle.dumps({
                "order": self.forward.order, "alpha": self.forward.alpha,
                "min_obs": self.forward.min_obs,
                "counts": [dict(c) for c in self.forward._counts],
                "totals": [dict(t) for t in self.forward._totals],
                "trained_bytes": self.forward.trained_bytes}),
            "reverse": pickle.dumps({
                "order": self.reverse.order, "alpha": self.reverse.alpha,
                "min_obs": self.reverse.min_obs,
                "counts": [dict(c) for c in self.reverse._counts],
                "totals": [dict(t) for t in self.reverse._totals],
                "trained_bytes": self.reverse.trained_bytes}),
        }, protocol=pickle.HIGHEST_PROTOCOL))

    @classmethod
    def load(cls, path: str | Path) -> "Bidirectional":
        blob = pickle.loads(Path(path).read_bytes())

        def rebuild(raw):
            b = pickle.loads(raw)
            m = ContextModel(b["order"], b["alpha"], b["min_obs"])
            m._counts = [defaultdict(dict, c) for c in b["counts"]]
            m._totals = [defaultdict(int, t) for t in b["totals"]]
            m.trained_bytes = b["trained_bytes"]
            return m

        m = cls(rebuild(blob["forward"]), rebuild(blob["reverse"]),
                blob.get("weight", 0.5))
        m.provenance = blob.get("provenance", {})
        m.calibration = _cal_load(blob)
        return m
