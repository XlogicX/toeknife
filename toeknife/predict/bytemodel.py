"""The byte filler: a predictor that reads the tape instead of a byte string.

It is a different KIND of predictor from
everything else in this package and the difference is not cosmetic.

`base.Predictor` is `logprob(byte, left)` -- left context, as plain bytes.  A
byte model cannot be expressed that way for two reasons.  It needs both sides,
and more importantly it needs to say **"an unknown byte sits here"**, which no
`bytes` object can carry.  UNKNOWN is a token this model was trained on; that
convention is what every off-the-shelf checkpoint measured was lacking, and
it is what the 42-to-72 gap was made of.

Faking it by splicing the known text around a hole is the failure that result
measured: "for" + hole + "forensic" reads as "forforensic", and the model then
punishes the correct answer for a bigram the data never contained.  Every
language-model harness in that document broke this rule and lost roughly half its
accuracy each time.

So it plugs in at the CLASS level instead, where the solver already aggregates
evidence over a hole's occurrences.  `ClassFiller.class_logprobs` answers a whole
equality class at once: k occurrences are k independent readings of one byte, and
they vote.
"""
from __future__ import annotations

from pathlib import Path

VOCAB, UNKNOWN, PAD = 258, 256, 257


def _torch():
    """Imported on use, never on import.

    Most sessions never touch a byte model and torch costs seconds to import.
    Nothing in this module runs at collection time for that reason.
    """
    import torch
    return torch


class ByteFiller:
    """The architecture, as trained.  Built lazily so importing costs nothing."""

    def __new__(cls, *a, **k):
        return _build(*a, **k)


def _build(d_model=256, layers=6, heads=4, window=129, dropout=0.1):
    torch = _torch()
    import torch.nn as nn

    class _ByteFiller(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(VOCAB, d_model)
            self.pos = nn.Embedding(window, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=4 * d_model,
                batch_first=True, norm_first=True, dropout=dropout)
            self.body = nn.TransformerEncoder(layer, layers)
            self.head = nn.Linear(d_model, 256)
            self.window = window
            self.cfg = dict(d_model=d_model, layers=layers, heads=heads,
                            window=window)

        def forward(self, ids):
            h = self.body(self.embed(ids)
                          + self.pos(torch.arange(ids.shape[1])))
            return self.head(h)

    return _ByteFiller()


def window_ids(tape, centre: int, window: int):
    """One window of tape, as the model sees it.

    Read from the TAPE, so an unknown reads as UNKNOWN and no answer can leak in
    through the input.  PAD is a separate token from UNKNOWN: off the end of the
    output and "unknown byte here" are different claims, and collapsing them
    would teach the model that the edges of the tape are holes.
    """
    torch = _torch()
    half = window // 2
    ids = torch.full((window,), PAD, dtype=torch.long)
    for i in range(window):
        q = centre - half + i
        if q < tape.min_pos or q >= tape.length:
            continue
        v = tape.value_at(q)
        ids[i] = UNKNOWN if v is None else v
    return ids


class ClassFiller:
    """A loaded byte model, answering one equality class at a time.

    Not a `Predictor` and deliberately not shaped like one -- see the module
    docstring. The solver holds this separately from its ranking model.
    """

    def __init__(self, model, window: int, provenance=None, batch: int = 64):
        self.model = model
        self.window = window
        self.provenance = provenance or {}
        self.batch = batch
        # How fast this model runs on THIS machine, accumulated from the work it
        # actually does rather than from a figure measured somewhere else. The
        # first thing asked of a loaded model is sanity_check, which pushes a
        # couple of hundred windows through it, so a rate exists before the
        # operator can start anything.
        self.windows_done = 0
        self.seconds_spent = 0.0
        # What this model's confidence measured on held-out text, when the
        # checkpoint carries it (predict/calibration.py fit_filler).
        self.calibration = None
        self.path = None                # the checkpoint it was loaded from

    def rate(self) -> float | None:
        """Windows per second, or None before it has done any work."""
        if self.windows_done < 32 or self.seconds_spent <= 0:
            return None
        return self.windows_done / self.seconds_spent

    def note_work(self, windows: int, seconds: float) -> None:
        self.windows_done += windows
        self.seconds_spent += seconds

    @classmethod
    def load(cls, path) -> "ClassFiller":
        torch = _torch()
        blob = torch.load(Path(path), weights_only=False, map_location="cpu")
        if not isinstance(blob, dict) or "cfg" not in blob:
            raise ValueError(f"{path} is not a ToeKnife byte model")
        cfg = dict(blob["cfg"])
        model = _build(**cfg)
        model.load_state_dict(blob["model"])
        model.eval()
        out = cls(model, cfg["window"], blob.get("provenance"))
        out.path = str(path)
        from .calibration import Calibration
        out.calibration = Calibration.from_dict(blob.get("calibration"),
                                                kind="slm")
        return out

    def class_logprobs(self, tape, positions):
        """Summed log-probabilities over a class's occurrences, as 256 floats.

        The same aggregation the solver applies to the n-gram: k occurrences are
        k independent readings of one byte, and they vote. Summed rather than
        averaged here; the solver divides by the count it used, the way it does
        for every other predictor, so the confidence threshold sees one scale.
        """
        torch = _torch()
        positions = list(positions)
        if not positions:
            return None
        import time
        total = torch.zeros(256)
        half = self.window // 2
        t0 = time.monotonic()
        with torch.no_grad():
            for i in range(0, len(positions), self.batch):
                chunk = positions[i:i + self.batch]
                ids = torch.stack([window_ids(tape, p, self.window)
                                   for p in chunk])
                total += self.model(ids)[:, half].log_softmax(-1).sum(0)
        self.note_work(len(positions), time.monotonic() - t0)
        return total

    def describe(self) -> str:
        p = self.provenance or {}
        bits = [p.get("params", "?"), f"window {self.window}"]
        if p.get("corpus"):
            bits.append(f"trained on {p['corpus']}")
        return "byte model, " + ", ".join(bits)

IGNORE = -100


class Windows:
    """Samples (ids, targets) windows from a pool of carves.

    Targets sit only at positions the tape does not know: training the model to
    reproduce bytes it can already read would teach it to copy rather than infer.
    """

    def __init__(self, carves, window=129, rng=None):
        self.window = window
        self.rng = rng or random.Random(0)
        self.pool = []
        for c in carves:
            positions = [p for h in c.session.tape.holes() for p in h.positions
                         if 0 <= c.at + p < len(c.truth)]
            if positions:
                self.pool.append((c, positions))
        self.total = sum(len(p) for _c, p in self.pool)

    def one(self):
        torch = _torch()
        c, positions = self.rng.choice(self.pool)
        centre = self.rng.choice(positions)
        tape, truth, at = c.session.tape, c.truth, c.at
        ids = window_ids(tape, centre, self.window)
        tgt = torch.full((self.window,), IGNORE, dtype=torch.long)
        half = self.window // 2
        for i in range(self.window):
            if ids[i] != UNKNOWN:
                continue
            j = at + (centre - half + i)
            if 0 <= j < len(truth):
                tgt[i] = truth[j]
        return ids, tgt

    def batch(self, size):
        torch = _torch()
        pairs = [self.one() for _ in range(size)]
        return (torch.stack([p[0] for p in pairs]),
                torch.stack([p[1] for p in pairs]))


def sample_windows(carves, window, rng=None):
    """The training sampler, kept beside the model it feeds.

    Targets sit only at positions the tape does not know: training the model to
    reproduce bytes it can already read would teach it to copy rather than
    infer.
    """
    return Windows(carves, window, rng)


def sanity_check(filler, tape, sample: int = 192, floor: int = 32):
    """Is this byte model any use on THIS capture, or is it actively harmful?

    The n-gram's rule, measured the byte model's way. `profiles.sanity_check`
    scores a corpus model on the bytes already recovered and warns below -8.0,
    because a model trained on the wrong kind of data does not degrade to
    ignorance -- it degrades past it, to -17.2 bits per byte against the -8.0 of
    knowing nothing, and confidence is what the solver's threshold reads.

    A byte model cannot be scored that way: it does not answer
    `logprob(byte, left)` and there is nothing to hand it but the tape. So the
    tape is used directly. A position whose value is already KNOWN is hidden --
    the window is built and its centre overwritten with UNKNOWN -- and the model
    is asked to fill it. That is exactly the question it will be asked about the
    real holes, with the difference that the answer is already on the tape.

    Held-out by construction, and it needs no corpus, no ground truth and no
    second file. What it cannot see is the part of the capture that is missing,
    so it measures fitness for this data rather than accuracy on the holes.

    Returns (mean log2 probability per byte, top-1 accuracy, a verdict).
    """
    torch = _torch()
    half = filler.window // 2
    known = [p for p in range(tape.min_pos, tape.length)
             if tape.value_at(p) is not None]
    if len(known) < floor:
        return 0.0, 0.0, "too few recovered bytes to judge the model against"
    step = max(1, len(known) // sample)
    picked = known[::step][:sample]

    import time
    total, hits = 0.0, 0
    t0 = time.monotonic()
    with torch.no_grad():
        for i in range(0, len(picked), filler.batch):
            chunk = picked[i:i + filler.batch]
            ids = torch.stack([window_ids(tape, p, filler.window)
                               for p in chunk])
            # Hide the answer that is sitting in the middle of each window.
            ids[:, half] = UNKNOWN
            out = filler.model(ids)[:, half].log_softmax(-1)
            for row, pos in zip(out, chunk):
                truth = tape.value_at(pos)
                total += float(row[truth])
                hits += int(int(row.argmax()) == truth)

    filler.note_work(len(picked), time.monotonic() - t0)

    import math
    bits = total / len(picked) / math.log(2)
    top1 = hits / len(picked)
    if bits < -8.0:
        return bits, top1, (
            f"this byte model scores {bits:.1f} bits per byte on the recovered "
            f"data, worse than the -8.0 of knowing nothing, and gets "
            f"{top1:.0%} of them right. It was trained on the wrong kind of "
            f"data and will be confidently wrong. Use a model trained on "
            f"something that resembles this stream, or the n-gram alone")
    if bits < -6.5:
        return bits, top1, (
            f"this byte model scores {bits:.1f} bits per byte on the recovered "
            f"data, barely better than the -8.0 of knowing nothing; a model "
            f"trained on a closer corpus would help a great deal")
    return bits, top1, ""


def fill_cost(filler, tape, max_occurrences: int = 24) -> tuple[int, float]:
    """How much work filling every open class is, and how long that will take.

    The byte model is roughly 40x slower per hole than the n-gram, which is
    settled as acceptable for an opt-in thorough pass -- but only if the operator
    sees the number before starting a fill rather than while waiting on one.

    Work is counted the way the solver spends it: one forward window per
    occurrence, capped at the solver's own `max_occurrences`, which is why the
    honest count is not simply the number of unknown positions. The cap costs
    nothing: 24 well-chosen readings measured 6.5 points of bytes better than
    all of them.

    Returns (windows, seconds), and seconds is None-ish -- 0.0 -- when the model
    has not done enough work yet to have a measured rate.
    """
    windows = sum(min(len(h.positions), max_occurrences) for h in tape.holes())
    rate = filler.rate()
    return windows, (windows / rate if rate else 0.0)
