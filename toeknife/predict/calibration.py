"""Confidence, measured: what a model's confidence delivers on text it never saw.

The solver's confidence is a softmax over per-occurrence scores. It ranks
candidates, but it is not a probability, and the same 0.995 floor delivered 99%
precision on a licence-text carve, 65% on a phrack one, and about 19% inside a
lost packet's gap. A floor the operator sets is a
promise about how many fills will be right, and only a measurement can keep it.

So a model is calibrated on text it was not trained on. Held-out documents are
damaged two ways -- the tool's standard carve (sync-flushed, first table
destroyed, resynced as f3 would: predict/carves.py) and a fixed-Huffman stream
with its front cut off, which is how the phrack demo specimen was made -- and
every open class is scored once against the known answer. A monotone fit maps confidence to the precision actually seen there,
separately for short holes and for long unread runs, since the two regimes do
not share a curve at all. The fit travels inside the model file, so the floor
means the same thing on every machine the model is copied to.

Training holds the documents back itself (`holdout`), so every model trained
here arrives calibrated without the operator running anything.
"""
from __future__ import annotations

import time
import zlib
from collections import Counter
from dataclasses import dataclass, field

from .solver import LONG_RUN, measured_text

# Compressed bytes cut from the front of each held-out document. 64 to 256 is
# the range of a carve that lost its start: enough to strand the window, not so
# much that nothing of the document is left to score.
CUTS = (64, 128, 256)
# The lost-packet case: one segment of the IPv4 default MSS.
PACKET = 536
# Where the packet goes, as fractions of the stream. The n-gram is rarely sure
# next to a gap: across 48 held-out phrack documents and these five positions,
# 171 near-gap classes of 106,586 reached confidence 0.999. Five positions
# gave the near curve's top steps 2-5x the classes the old two (1/3, 1/2) did,
# out-of-fold error 0.0244 -> 0.0225, for 18% more calibration time.
PACKET_AT = ((1, 4), (1, 3), (1, 2), (2, 3), (3, 4))
# A calibration this thin says more about its documents than about the model.
MIN_CLASSES = 60
# Held-out documents: every Nth file, at most this many, each at most this long.
# 24 spread across the corpus, not the first 12 in sorted order: on phrack
# those were p10-3 to p29-9, 120,797 bytes of early issues, and on 18 documents
# none of the fits saw, their curve's mean error was 0.129 against 0.027 for 24
# spread out (a 50% floor near a gap delivered 29%). More cuts per document
# measured no better.
EVERY, CAP, DOC_BYTES = 20, 24, 32_768


@dataclass
class Calibration:
    """Confidence -> measured precision, per regime, with its own evidence."""
    short: list = field(default_factory=list)  # [(lowest confidence, precision, classes)]
    long: tuple | None = None                  # (precision, classes) in runs >= LONG_RUN
    # Classes with an occurrence inside an interior gap -- a lost packet, not a
    # carve's window -- on their own curve. Measured from
    # held-out documents with a packet-sized chunk cut out of the middle.
    near: list = field(default_factory=list)
    classes: int = 0
    cases: int = 0
    held_out: list = field(default_factory=list)
    date: str = ""
    # "slm" for a byte model's curve, keyed on its confidence; "" for an
    # n-gram's. Each loader takes only its own: read as the other's, a curve
    # promises a precision nobody measured for that number.
    kind: str = ""

    def precision(self, confidence: float, run: int = 0,
                  near: bool = False) -> float | None:
        """What a class at this confidence, in this regime, measured. None
        where nothing was measured, which is not the same as zero."""
        if run >= LONG_RUN:
            return self.long[0] if self.long else None
        # Step edges are stored to 6 places (_curve); a lookup is compared the
        # same way, or 0.9999996 falls below a step stored as 1.0.
        confidence = round(confidence, 6)
        steps = self.short
        if near:
            if not self.near:
                # Calibrated before the gap regime existed: the long-run
                # figure is the honest stand-in, not the carve curve.
                return self.long[0] if self.long else None
            steps = self.near
        got = None
        for lo, prec, _n in steps:
            if confidence >= lo:
                got = prec
        return got

    def best(self) -> float:
        return max((p for _lo, p, _n in self.short), default=0.0)

    def resting(self, floor: float, near: bool = False) -> int:
        """Held-out classes under the steps a floor reaches, in one regime.

        What a floor's promise was measured on. Near a gap it is small at any
        high floor: the n-gram is rarely sure there, and phrackarchive-o6-cal
        put its 90% near-gap step on 9 classes.
        """
        steps = self.near if near else self.short
        return sum(n for _lo, p, n in steps if p >= floor)

    def summary(self) -> str:
        return (f"calibrated on {self.classes:,} held-out classes from "
                f"{len(self.held_out)} document(s); best measured precision "
                f"{measured_text(self.best())}")

    def to_dict(self) -> dict:
        return {"version": 1, "short": [list(b) for b in self.short],
                "long": list(self.long) if self.long else None,
                "near": [list(b) for b in self.near],
                "classes": self.classes, "cases": self.cases,
                "held_out": list(self.held_out), "date": self.date,
                **({"kind": self.kind} if self.kind else {})}

    @classmethod
    def from_dict(cls, d: dict | None,
                  kind: str = "") -> "Calibration | None":
        if not d or d.get("version") != 1 or d.get("kind", "") != kind:
            return None
        return cls([tuple(b) for b in d.get("short") or []],
                   tuple(d["long"]) if d.get("long") else None,
                   [tuple(b) for b in d.get("near") or []],
                   d.get("classes", 0), d.get("cases", 0),
                   list(d.get("held_out") or []), d.get("date", ""),
                   kind)


def fit_filler(rows) -> Calibration | None:
    """Rows of (confidence, run, near, correct) for a byte model -> its curve.

    The byte model's own confidence, the solver's per-term softmax over its
    scores: on short holes in p38-5, p35-12 and p49-14, which it never saw, a
    curve fit this way passed 1,030 classes at an 80% floor and 99.4% were
    right, where the n-gram's gate in "both" mode passed 72 at 90%. Splitting
    the curve by whether the n-gram agreed bought nothing (1,079 at 98.5%), so
    there is one.
    """
    cal = fit(rows)
    if cal is not None:
        cal.kind = "slm"
        cal.date = time.strftime("%Y-%m-%d")
    return cal


def _laplace(right: int, n: int) -> float:
    """Precision pulled toward a half by one right and one wrong: five classes
    all right is not 100%, and must not read as a floor of 99.5% cleared."""
    return (right + 1) / (n + 2)


def _curve(pairs) -> list:
    """(confidence, correct) pairs -> monotone smoothed steps.

    Pool-adjacent-violators over the pairs sorted by confidence, so higher
    confidence never maps to lower precision, then each pooled block smoothed.
    """
    # One confidence is one block, counted whole before any pooling. Sorted, a
    # tie puts its wrong answers first; merged on the fly, PAV pooled those
    # down into the block below before the right ones arrived, and the right
    # ones then stood alone at ~100%. On the byte model's fit set, 16,725
    # classes at confidence 1.0 were 75.5% right and the curve said 99.99%.
    # Rounded to the 6 places a step edge is stored and looked up at, so a
    # tie is a tie at that resolution too.
    groups: list[list] = []
    for conf, ok in sorted((round(c, 6), bool(o)) for c, o in pairs):
        if groups and groups[-1][0] == conf:
            groups[-1][1] += int(ok)
            groups[-1][2] += 1
        else:
            groups.append([conf, int(ok), 1])
    blocks: list[list] = []                 # [lowest conf, right, n]
    for g in groups:
        blocks.append(g)
        # >=, not >: equal neighbours pool too, or a run of single right
        # answers stays sixty one-class blocks that each smooth to 67%.
        while len(blocks) > 1 and (blocks[-2][1] / blocks[-2][2]
                                   >= blocks[-1][1] / blocks[-1][2]):
            lo, r, n = blocks.pop()
            blocks[-1][1] += r
            blocks[-1][2] += n
    steps = [(round(lo, 6), round(_laplace(r, n), 4), n) for lo, r, n in blocks]
    # Smoothing can undo monotonicity between a large block and a small one
    # above it; the floor must never get easier to clear by being surer.
    fixed, best = [], 0.0
    for lo, p, n in steps:
        best = max(best, p)
        if fixed and fixed[-1][1] == best:
            # Same precision as the step below: one step, counting both.
            fixed[-1] = (fixed[-1][0], best, fixed[-1][2] + n)
        else:
            fixed.append((lo, best, n))
    return fixed


def fit(rows) -> Calibration | None:
    """Rows of (confidence, run, correct) or (confidence, run, near, correct)
    -> a calibration with one curve per regime.

    Short holes in decoded text get a curve, and so do classes next to an
    interior gap: the two do not share one, and pooling them would let a
    carve's good figures carry fills near a lost packet. Long-run rows get one
    figure, because inside a long gap the confidence does not rank at all
    and a curve would be fitting noise.
    """
    norm = [(r[0], r[1], False, r[2]) if len(r) == 3 else tuple(r)
            for r in rows]
    short = [(c, bool(ok)) for c, run, near, ok in norm
             if run < LONG_RUN and not near]
    near = [(c, bool(ok)) for c, run, near, ok in norm
            if run < LONG_RUN and near]
    long = [bool(ok) for c, run, near, ok in norm if run >= LONG_RUN]
    if len(short) + len(near) + len(long) < MIN_CLASSES:
        return None
    return Calibration(
        short=_curve(short),
        long=(round(_laplace(sum(long), len(long)), 4), len(long)) if long else None,
        near=_curve(near) if len(near) >= MIN_CLASSES // 2 else [],
        classes=len(short) + len(near) + len(long))


def holdout(paths, every: int = EVERY, cap: int = CAP):
    """Split a file list into training files and held-out documents.

    Every `every`th file, deterministically, so the same corpus always holds
    out the same files and a model can be rebuilt to the same calibration. A
    corpus of fewer than `every` files gives nothing: it is split by bytes
    instead, through `holdout_blob`, by the caller.
    """
    paths = sorted(str(p) for p in paths)
    if len(paths) < every:
        return paths, []
    picks = paths[every // 2::every]
    # Evenly across the candidates when there are more than `cap`, so a big
    # corpus is tested on all of itself rather than on its first files.
    held = (picks if len(picks) <= cap else
            [picks[i * len(picks) // cap] for i in range(cap)])
    out = set(held)
    train = [p for p in paths if p not in out]
    docs = []
    for p in held:
        try:
            with open(p, "rb") as fh:
                docs.append((p, fh.read(DOC_BYTES)))
        except OSError:
            continue
    return train, docs


def holdout_blob(blob: bytes, share: float = 0.05, cap: int = CAP):
    """The tail of one training blob, cut into documents, held out."""
    n = int(len(blob) * share)
    if n < 4096:
        return blob, []
    tail = blob[-n:]
    size = max(4096, min(DOC_BYTES, n // cap))
    docs = [(f"tail[{i}]", tail[i:i + size]) for i in range(0, n, size)
            if len(tail[i:i + size]) >= 2048][:cap]
    return blob[:-n], docs


def _sessions(doc: bytes, model, cuts=CUTS, packets=PACKET_AT,
              carved: bool = True):
    """The damaged versions of one held-out document, as sessions.

    `cuts`, `packets` and `carved` choose the damage; the defaults are what
    every calibration uses. Research scores one kind alone through them.
    """
    from ..core.session import Session
    from .carves import carve, resynced
    if carved:
        try:
            s = resynced(carve(doc))
            if s is not None:
                s.model = model
                yield s
        except Exception:
            pass
    co = zlib.compressobj(9, zlib.DEFLATED, -15, 9, zlib.Z_FIXED)
    raw = co.compress(doc) + co.flush()
    for k in cuts:
        if len(raw) > k + 512:
            try:
                yield Session(raw[k:], mode="auto", profile="auto",
                              model=model)
            except Exception:
                continue
    # A lost packet: gzip'd, with a 536-byte chunk cut out at each of
    # PACKET_AT, the way a capture loses a segment. This is the case where a
    # carve's calibration does not hold.
    from ..core.gap import from_holes
    co = zlib.compressobj(9, zlib.DEFLATED, 16 + 15)
    gz = co.compress(doc) + co.flush()
    for num, den in packets:
        at = 10 + (len(gz) - 18) * num // den
        if len(gz) < at + PACKET + 1024:
            continue
        try:
            asm = from_holes(gz[:at] + gz[at + PACKET:], [(at, PACKET)],
                             10, 8, len(doc))
            s = Session.from_trace(asm.trace)
            s.model = model
            yield s
        except Exception:
            continue


def _offset(session, doc: bytes) -> int | None:
    """Where tape position 0 sits in the document, by voting on known runs."""
    buf, mask = session.tape.recovered()
    votes = Counter()
    i = 0
    while i < len(buf) - 16:
        if all(mask[i:i + 16]):
            j = doc.find(bytes(buf[i:i + 16]))
            if j >= 0:
                votes[j - i] += 1
            i += 16
        else:
            i += 1
    if not votes:
        return None
    off, n = votes.most_common(1)[0]
    return off if n >= 3 else None


def measure(model, docs, progress=None, cuts=CUTS):
    """Score every open class of every damaged held-out document once.

    Through the same assembly the solver uses everywhere, so the confidence
    measured is the confidence ^f will read. Returns (rows, cases).
    """
    rows, cases = [], 0
    for n, (name, doc) in enumerate(docs):
        if progress is not None:
            progress(n, len(docs), f"calibrating: document {n + 1} of "
                                   f"{len(docs)}")
        for s in _sessions(doc, model, cuts):
            cases += _score(s, doc, model, rows)
    return rows, cases


def _score(s, doc, model, rows) -> int:
    """One damaged session's open classes, against the document. 1 if used."""
    from . import corpora
    from .profiles import build
    from .solver import HoleSolver
    off = _offset(s, doc)
    if off is None or not s.tape.holes():
        return 0
    mixture, _ = build(s, corpus_model=model, self_train=False,
                       order=corpora.resolve_order(
                           "auto", s.tape.stats().get("known", 0)))
    # propose() annotates run and near exactly as ^f will see them.
    for p in HoleSolver(s, mixture).propose():
        t = p.anchor + off
        if 0 <= t < len(doc):
            rows.append((p.confidence, p.run, p.near, doc[t] == p.value))
    return 1


def measure_filler(filler, docs, ngram=None, progress=None, **damage):
    """`measure` for a byte model. Rows are (confidence, run, near, correct).

    `ngram` only builds the solver's mixture, which the pool and the evidence
    positions come from; the byte model's scores do not read it. Without one,
    each damaged document trains its own on what it recovered, as a fill with
    no corpus loaded does.
    """
    from . import corpora
    from .profiles import build
    from .solver import HoleSolver
    rows, cases = [], 0
    for n, (name, doc) in enumerate(docs):
        if progress is not None:
            progress(n, len(docs), f"calibrating: document {n + 1} of "
                                   f"{len(docs)}")
        for s in _sessions(doc, ngram, **damage):
            off = _offset(s, doc)
            if off is None or not s.tape.holes():
                continue
            mixture, _ = build(s, corpus_model=ngram, self_train=ngram is None,
                               order=corpora.resolve_order(
                                   "auto", s.tape.stats().get("known", 0)))
            for p in HoleSolver(s, mixture, filler=filler,
                                mode="slm").propose():
                t = p.anchor + off
                if 0 <= t < len(doc):
                    rows.append((p.confidence, p.run, p.near,
                                 doc[t] == p.value))
            cases += 1
    return rows, cases


def calibrate_filler(filler, docs, ngram=None,
                     progress=None) -> Calibration | None:
    """Measure a byte model on held-out documents and fit its curve."""
    if not docs:
        return None
    rows, cases = measure_filler(filler, docs, ngram, progress)
    cal = fit_filler(rows)
    if cal is not None:
        cal.cases = cases
        cal.held_out = [str(name) for name, _ in docs]
    return cal


def save_filler_calibration(src, dst, cal: Calibration) -> None:
    """Write a byte-model checkpoint with its calibration inside.

    `dst` may be `src`: a training run calibrates its own checkpoint in place.
    Written to a temporary name and moved, so a crash mid-write leaves the
    checkpoint that cost days as it was.
    """
    import os
    import torch
    blob = torch.load(src, weights_only=False, map_location="cpu")
    if not isinstance(blob, dict) or "cfg" not in blob:
        raise ValueError(f"{src} is not a ToeKnife byte model")
    blob["calibration"] = cal.to_dict()
    tmp = f"{dst}.partial"
    torch.save(blob, tmp)
    os.replace(tmp, dst)


def filler_holdout(paths, limit: int):
    """Split a byte model's corpus into files to train on and held-out text.

    Whole files first, by the n-gram's rule (`holdout`): article text is what
    a model is used on, and the byte model's curve fit on article tails tops
    out at 88.5% where whole articles came out 98-100%. A corpus too small to
    give 4 files
    falls back to text past `limit`, which the trainer never reads.
    Returns (training paths, [(name, bytes)]).
    """
    train, docs = holdout(paths)
    if len(docs) >= 4:
        return train, docs
    train = sorted(str(p) for p in paths)
    return train, tail_windows(train, limit)


def tail_windows(paths, start: int, cap: int = CAP):
    """`DOC_BYTES` from `start` in each file long enough, spread over the list."""
    long = [p for p in paths if _size(p) >= start + 8192]
    picks = (long if len(long) <= cap else
             [long[i * len(long) // cap] for i in range(cap)])
    docs = []
    for p in picks:
        try:
            docs.append(read_held_out(f"{p}@{start}"))
        except OSError:
            continue
    return docs


def read_held_out(spec: str):
    """`FILE` or `FILE@OFFSET` -> (spec, up to DOC_BYTES from there).

    The offset form is for text past what a trainer read: slmtrain reads the
    first 48,000 bytes of each file, so `p10-3.txt@48000` is text the model
    trained beside and never on.
    """
    path, at, off = str(spec).rpartition("@")
    if not at or not off.isdigit():
        path, off = str(spec), "0"
    with open(path, "rb") as fh:
        fh.seek(int(off))
        return str(spec), fh.read(DOC_BYTES)


def calibrate(model, docs, progress=None) -> Calibration | None:
    """Measure and fit. None when there was too little to say anything."""
    if not docs:
        return None
    rows, cases = measure(model, docs, progress)
    cal = fit(rows)
    if cal is not None:
        cal.cases = cases
        cal.held_out = [str(name) for name, _ in docs]
        cal.date = time.strftime("%Y-%m-%d")
    return cal


def train_files(paths, order: int, forward_only: bool = False,
                progress=None):
    """Train on all but the held-out files, then calibrate on those.

    Returns (model, calibration-or-None, training files). The model is the one
    trained without the held-out files, so its calibration describes it
    exactly; the price is ~5% less training data.
    """
    from .context import Bidirectional, ContextModel
    from .corpora import record_training_rate
    started = time.monotonic()
    train, docs = holdout(paths)
    if len(docs) < 4:
        # Too few files to hold whole ones back and still measure anything;
        # hold back the tail of the corpus instead.
        train = sorted(str(p) for p in paths)
        docs = []
    if not docs:
        blob = b"".join(open(p, "rb").read() for p in train)
        body, docs = holdout_blob(blob)
        model = (ContextModel(order=order).train(body) if forward_only
                 else Bidirectional.train_on(body, order=order,
                                             progress=progress))
    else:
        model = (ContextModel(order=order).train_files(train) if forward_only
                 else Bidirectional.train_files(train, order=order,
                                                progress=progress))
    model.calibration = calibrate(model, docs, progress)
    record_training_rate(order, sum(_size(p) for p in paths),
                         time.monotonic() - started)
    return model, model.calibration, train


def _size(path) -> int:
    import os
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def train_blob(blob: bytes, order: int, progress=None):
    """`train_files` for a corpus that arrives as one blob (a fetch, a type)."""
    from .context import Bidirectional
    from .corpora import record_training_rate
    started = time.monotonic()
    body, docs = holdout_blob(blob)
    model = Bidirectional.train_on(body, order=order, progress=progress)
    model.calibration = calibrate(model, docs, progress)
    record_training_rate(order, len(blob), time.monotonic() - started)
    return model, model.calibration


def find_held_out(names, roots=()):
    """A run's recorded held-out names, found again on this machine.

    Each is a path as recorded, `FILE@OFFSET` included, and is looked for
    there first, then by file name under `roots`. Returns (specs, missing).
    """
    from pathlib import Path
    by_name = None
    found, missing = [], []
    for n in names:
        path, at, off = str(n).rpartition("@")
        if not at or not off.isdigit():
            path, at, off = str(n), "", ""
        if not Path(path).is_file():
            if by_name is None:
                by_name = {}
                for root in roots or ():
                    root = Path(root).expanduser()
                    for f in ([root] if root.is_file() else
                              sorted(root.rglob("*"))):
                        if f.is_file():
                            by_name.setdefault(f.name, str(f))
            path = by_name.get(Path(path).name)
        if path is None:
            missing.append(str(n))
        else:
            found.append(path + (f"@{off}" if at else ""))
    return found, missing


def recorded_held_out(path) -> list:
    """The held-out names a checkpoint carries; [] when it carries none."""
    import torch
    blob = torch.load(path, weights_only=False, map_location="cpu")
    return list(blob.get("held_out") or []) if isinstance(blob, dict) else []


def calibrated_path(src):
    """`M.tks` -> `M-cal.tks` beside it."""
    from pathlib import Path
    src = Path(src)
    return src.with_name(src.stem + "-cal" + src.suffix)


def calibrate_checkpoint(src, specs, ngram=None, progress=None):
    """Measure the byte model at `src` on `specs` and write `M-cal.tks`.

    A copy beside the original: the model that was measured stays on disk.
    Returns (destination, calibration); the calibration is None, and nothing
    is written, when the text gave too few classes to fit.
    """
    from .bytemodel import ClassFiller
    dst = calibrated_path(src)
    if dst.exists():
        raise FileExistsError(f"{dst} already exists")
    docs = [d for d in (read_held_out(s) for s in specs) if len(d[1]) >= 2048]
    if not docs:
        raise ValueError("no held-out document has 2 KB of text to damage")
    cal = calibrate_filler(ClassFiller.load(src), docs, ngram, progress)
    if cal is not None:
        save_filler_calibration(src, dst, cal)
    return dst, cal
