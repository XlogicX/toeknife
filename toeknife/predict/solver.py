"""Filling unknown byte classes with a model.

The unknowns are equality classes, not positions, and that is the whole
advantage. A class appearing at k places in the output gets k independent
readings of what belongs there, from k different surrounding contexts, all
voting on one byte. Ordinary infilling gets one context per unknown.

Scoring runs in both directions. A causal model asks how likely a byte is given
what precedes it, which on its own ignores everything after the hole, and the
bytes after a hole are usually known and usually decisive. So a candidate is
also charged for how well it predicts the bytes that follow it.

Context stops at a hole rather than skipping over it. Splicing across an unknown
invents adjacencies that are not in the data: "for" followed by a hole followed
by "forensic" reads as "forforensic", and the model then punishes the correct
answer for a bigram the data never contained.

Nothing here commits anything. The solver returns proposals with confidence and
the caller decides, because a byte that silently became known because a model
liked it is worse than a hole.

Confidence ranks well and is miscalibrated in absolute terms. Measured on a
carved licence-text stream with a held-out order-4 model, against ground truth:

    threshold   classes filled   precision of new fills
      0.90            163                  60.7%
      0.99            163                  90.8%
      0.995           100                  99.0%
      0.999            44                  97.7%
      0.9995           25                 100.0%

So the number is a dial, not a probability, and the curve is not even monotonic:
0.995 beats 0.999 on this data because the tail is small enough to be noisy. The
default sits at 0.995, which was chosen by measuring rather than guessing -- an
earlier default of 0.99 turned out to give 90.8% where 0.995 gives 99.0% for
only a third fewer fills. How much it buys depends mostly on how much
of the output was already known: a hole surrounded by other holes has no context
to judge it by, and no threshold rescues that.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from ..core.tape import SOLVED, SOLVED_PROP, UNKNOWN, Hole


def pct(value: float) -> str:
    """Format a threshold without rounding it into a lie.

    0.995 through "%.0f%%" is "100%", which reads as "the model has to be
    certain" when the real bar is 99.5% and the difference between those two is
    the entire dial.  Keep enough digits to tell the rungs apart.
    """
    return f"{value * 100:.4g}%"


def confidence_text(value: float) -> str:
    """A confidence, never rounded up into a claim of certainty.

    A proposal at 0.999991 printed to one decimal is "100.0%", which reads as
    the model being sure.  It is not sure, the figure is not a probability in
    the first place, and certainty is the one thing this number must not imply.
    Only a confidence that really is 1.0 gets to print 100%.
    """
    text = f"{value * 100:.1f}%"
    return ">99.9%" if text == "100.0%" and value < 1.0 else text


def measured_text(value: float) -> str:
    """A measured precision, cut to a tenth of a percent, never rounded up.

    It sits beside a floor, and "80%" for 79.79% read as clearing an 80% floor
    that it does not.
    """
    return f"{math.floor(value * 1000 + 1e-9) / 10:.1f}%"


def readable_neighbours(tape, pos: int, window: int = 16) -> int:
    """How many of this position's neighbours can actually be read.

    The measure that decides which occurrence of an unknown class is worth
    looking at, whether by a model or by a person. Shared rather than defined
    twice: the solver was scoring the first occurrences and the panel was
    previewing the first occurrence, and both were wrong for the same reason.
    """
    return sum(1 for q in range(pos - window, pos + window + 1)
               if q != pos and tape.value_at(q) is not None)


def grounding(tape, positions, window: int) -> float:
    """How much of a class's context is evidence rather than the model's own
    earlier fills: the share of known neighbours that were read, copied or
    typed, over all known neighbours of the positions it was scored at.

    0.0 when nothing around it is known at all. A fill whose context is mostly
    model fills is the model agreeing with itself, which is how a 0.995 floor
    filled a lost packet's gap at 28.8% precision.
    """
    real = model = 0
    for pos in positions:
        for q in range(pos - window, pos + window + 1):
            if q == pos:
                continue
            prov = tape.prov_at(q)
            if prov == UNKNOWN:
                continue
            if prov in (SOLVED, SOLVED_PROP):
                model += 1
            else:
                real += 1
    return real / (real + model) if real + model else 0.0


# A run of output this long with nothing read from the capture is outside the
# regime the confidence floor was measured in. Measured on the held-out phrack
# article: classes whose every occurrence sits in such a
# run were right 18.7% of the time (489 of 2,618, carve and lost packet
# together); below it a carve's classes were right about 67%, and its 16-256
# byte runs 92-100%, which is why the line is not drawn lower.
LONG_RUN = 256
LONG_RUN_PRECISION = "19%"

# A zlib preset dictionary is not running text: it is strings common in the
# data, typically a word list in frequency order, and the floor was measured
# on running text. On the manual's workflow H specimen (the phrack article
# against its 200 commonest words) the calibrated phrack model at a 0.8 floor
# filled 220 of 1,110 dictionary classes and 106 were right; at 0.9 it filled
# 38, 10 right (re-measured). One specimen.
DICTIONARY_PRECISION = "48%"
DICTIONARY_FLOOR = "80%"

# What each confidence band delivered, measured: the held-out phrack article
# p49-14 front-truncated (stack.bin), phrackarchive-o6, every open class scored
# once against the real text. Re-measured 2026-10-28 after the fragment start
# fix: the earlier figures (57% at 0.995) were dragged
# down by misread start bytes, not by the model. One carve, one model -- shown
# beside a fill so the number reads as evidence, not as a probability.
MEASURED_BANDS = (
    (0.995, 1.01, "100%", 15), (0.99, 0.995, "86%", 7), (0.90, 0.99, "95%", 40),
    (0.70, 0.90, "77%", 82), (0.50, 0.70, "45%", 22), (0.0, 0.50, "19%", 21),
)


def measured_band(confidence: float):
    """(precision, classes) measured for this confidence band, or None."""
    for lo, hi, prec, n in MEASURED_BANDS:
        if lo <= confidence < hi:
            return prec, n
    return None


def _not_evidence(prov) -> bool:
    return prov in (UNKNOWN, SOLVED, SOLVED_PROP)


def evidence_runs(tape) -> dict[int, int]:
    """For every position not read from the capture, how long the unbroken run
    of such positions around it is.

    Model fills count as not read. Otherwise each fill would split a long gap
    into short runs, and the classes inside it would look like short holes one
    round later -- the snowball this exists to stop.
    """
    runs: dict[int, int] = {}
    start = None
    for pos in range(tape.min_pos, tape.length + 1):
        inside = pos < tape.length and _not_evidence(tape.prov_at(pos))
        if inside and start is None:
            start = pos
        elif not inside and start is not None:
            n = pos - start
            for q in range(start, pos):
                runs[q] = n
            start = None
    return runs


def interior_gaps(tape) -> set[int]:
    """Positions inside a long run with decoded output on both sides of it.

    A lost packet or a bad sector, as opposed to the unread window before a
    carve's first byte. The two need telling apart: on the phrack carve every
    class touches the window and the 90% band delivered 87.8%, while next to a
    lost packet's gap the same band delivered 52.5% scored once and 35% after
    ^f filled them in turn.
    """
    out: set[int] = set()
    start = None
    for pos in range(tape.min_pos, tape.length + 1):
        inside = pos < tape.length and _not_evidence(tape.prov_at(pos))
        if inside and start is None:
            start = pos
        elif not inside and start is not None:
            if (pos - start >= LONG_RUN and start > tape.min_pos
                    and pos < tape.length):
                out.update(range(start, pos))
            start = None
    return out


def class_run(runs: dict[int, int], positions) -> int:
    """The shortest evidence-free run any occurrence of a class sits in: one
    occurrence in decoded text is enough to judge it by."""
    return min((runs.get(p, 0) for p in positions), default=0)


def most_readable(tape, positions, window: int = 16) -> int:
    """The occurrence of a class with the most context around it.

    A class's early occurrences are usually the ones inside the lost region,
    where every neighbour is also unknown. Its later occurrences are the
    back-reference copies, which sit in text that decoded fine. Ties break on
    position so the choice is stable.
    """
    return min(positions,
               key=lambda p: (-readable_neighbours(tape, p, window), p))


def floor_calibration(mode: str, ngram_calibration, filler=None):
    """The calibration a floor in this fill mode is measured against.

    The n-gram's in "ngram" mode; the byte model's own once it has one.
    Without one, "both" gates on the n-gram's confidence and so on its curve,
    and "slm" has nothing: a curve fit on the n-gram's confidence says nothing
    about the byte model's.
    """
    if mode == "ngram" or filler is None:
        return ngram_calibration
    fcal = getattr(filler, "calibration", None)
    if fcal is not None:
        return fcal
    return ngram_calibration if mode == "both" else None


@dataclass
class Proposal:
    """One candidate fill for one hole class."""
    anchor: int
    positions: tuple[int, ...]
    value: int
    score: float                       # total log2 probability, summed over evidence
    confidence: float                  # softmax over candidate values, 0..1
    runners: tuple[tuple[int, float], ...] = ()   # (value, score) alternatives
    source: str = "ngram"              # which predictor this answer came from
    agreement: bool | None = None      # both consulted: did they name the same byte?
    grounded: float = 1.0              # share of its context that is evidence, not fills
    run: int = 0                       # unread run its best occurrence sits in
    near: bool = False                 # some occurrence lies in an interior gap
    # What this confidence measured on held-out text for this model, in this
    # regime (predict/calibration.py). None for an uncalibrated model, which
    # leaves the floor reading raw confidence.
    precision: float | None = None

    @property
    def gate(self) -> float:
        """The number the floor is compared with."""
        return self.confidence if self.precision is None else self.precision

    @property
    def size(self) -> int:
        return len(self.positions)

    def label(self) -> str:
        ch = repr(chr(self.value)) if 0x20 <= self.value < 0x7F else f"0x{self.value:02x}"
        return f"{ch} at {self.anchor}"

    def summary(self) -> str:
        alts = " ".join(
            (repr(chr(v)) if 0x20 <= v < 0x7F else f"0x{v:02x}")
            for v, _ in self.runners[:3])
        return (f"{self.anchor:>8} x{self.size:<3} {self.label():<16} "
                f"conf {confidence_text(self.confidence):>6}  next: {alts}")


@dataclass
class SolveResult:
    filled: list[Proposal] = field(default_factory=list)
    remaining: int = 0
    rounds: int = 0
    stopped: str = ""
    # Classes that cleared the floor but were held back because they sit only
    # inside long runs of unread output. Nonzero means the run stopped on the
    # regime, not on the floor, and "continue anyway" has something to do.
    held: int = 0
    # Classes that cleared the floor but hold a zlib preset dictionary byte.
    held_dictionary: int = 0

    @property
    def positions_filled(self) -> int:
        return sum(p.size for p in self.filled)


class HoleSolver:
    def __init__(self, session, model, *, order: int | None = None,
                 lookahead: int = 3, max_occurrences: int = 24,
                 candidates: int = 40, min_confidence: float = 0.995,
                 max_rounds: int = 60, temperature: float = 0.35,
                 batch: int = 32, filler=None, mode: str = "ngram",
                 calibration=None):
        self.session = session
        self.model = model
        # A byte model does not go in `model`. It is not a Predictor: it reads
        # the tape, where UNKNOWN is a token it was trained on, and answers a
        # whole class at once. See predict/bytemodel.py for why that cannot be
        # squeezed into logprob(byte, left).
        self.filler = filler
        # The corpus model's own calibration, when it has one. Passed rather
        # than read off `model`, which is the mixture: the calibration was fit
        # on the mixture's confidence but belongs to the corpus model it wraps.
        self.calibration = calibration
        if mode not in ("ngram", "slm", "both"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode != "ngram" and filler is None:
            raise ValueError(f"mode {mode!r} needs a byte model")
        self.mode = mode
        self.order = order if order is not None else getattr(model, "order", 4)
        self.lookahead = lookahead
        self.max_occurrences = max_occurrences
        self.n_candidates = candidates
        self.min_confidence = min_confidence
        self.max_rounds = max_rounds
        self.temperature = temperature
        self.batch = max(1, batch)
        self._candidates = self._candidate_bytes()

    @property
    def tape(self):
        return self.session.tape

    # ---- candidate byte values ---------------------------------------------

    def _candidate_bytes(self) -> list[int]:
        """Which byte values are worth scoring at all.

        A dynamic block's Huffman table names every byte that can appear as a
        literal anywhere in the block, and roughly how often, which is a prior
        derived from the artefact rather than from a corpus. Where that is
        available it bounds the search; otherwise fall back to what the stream
        already shows plus the model's own frequent bytes.
        """
        prior: dict[int, float] = {}
        for block in getattr(self.session.trace, "blocks", ()):
            table = getattr(block, "litlen", None)
            if table is not None and getattr(table, "codes", None):
                for sym, weight in table.byte_prior().items():
                    prior[sym] = max(prior.get(sym, 0.0), weight)
        if prior:
            return sorted(prior, key=lambda b: -prior[b])

        seen = Counter()
        tape = self.tape
        for p in range(min(tape.length, 200000)):
            v = tape.value_at(p)
            if v is not None:
                seen[v] += 1
        if getattr(self.model, "trained_bytes", 0):
            for b, p in self.model.distribution(b"").items():
                seen[b] += 0  # ensure present
            ranked = sorted(range(256),
                            key=lambda b: -(seen.get(b, 0) + self.model.logprob(b, b"")))
        else:
            ranked = [b for b, _ in seen.most_common()] or list(range(256))
        top = [b for b, _ in seen.most_common(self.n_candidates)]
        for b in ranked:
            if len(top) >= self.n_candidates:
                break
            if b not in top:
                top.append(b)
        return top

    # ---- scoring -------------------------------------------------------------

    def _left_context(self, pos: int) -> bytes:
        """Up to `order` known bytes before pos.  A hole resets the run."""
        out = bytearray()
        tape = self.tape
        for q in range(pos - self.order, pos):
            v = tape.value_at(q)
            if v is None:
                out.clear()
            else:
                out.append(v)
        return bytes(out)

    def _score_value(self, positions, value: int) -> tuple[float, int]:
        """(total log2 probability, number of evidence terms) for this byte.

        Evidence is one term for the byte itself at each occurrence, plus one
        for each known byte it then has to predict.
        """
        model, tape = self.model, self.tape
        both = getattr(model, "logprob_both", None)
        total = 0.0
        terms = 0
        for pos in positions:
            left = self._left_context(pos)
            if both is not None:
                # A model that reads backwards as well can judge a byte with
                # nothing at all on its left, which is exactly where the holes
                # that matter sit.
                total += both(value, left, self._right_context(pos))
            else:
                total += model.logprob(value, left)
            terms += 1
            # Charge the candidate for the bytes that follow it too: a causal
            # model would otherwise ignore the right-hand context entirely.
            ctx = bytearray(left)
            ctx.append(value)
            for j in range(1, self.lookahead + 1):
                nxt = tape.value_at(pos + j)
                if nxt is None:
                    break
                total += model.logprob(nxt, bytes(ctx[-self.order:]))
                ctx.append(nxt)
                terms += 1
        return total, terms

    def _evidence_positions(self, hole: Hole) -> list[int]:
        """The occurrences worth scoring, richest context first.

        It used to be `hole.positions[:max_occurrences]` -- the first N, in output
        order -- and on a real recovery that is close to the worst possible
        choice. A class's early occurrences are usually the ones inside the lost
        region, where every neighbour is also unknown and the model is being asked
        to judge a byte against nothing. Its later occurrences are the
        back-reference copies, which sit in text that decoded fine.

        Measured on a carve with 513 unknown classes: scoring the first 24
        occurrences answered "space" at a flat 0.860 for two thirds of them,
        which is the mixture falling back on its prior. Ranking by how much
        context each occurrence actually has took the same model up from 42% on
        the same 40 classes.
        """
        if len(hole.positions) <= self.max_occurrences:
            return list(hole.positions)
        window = max(4, self.order)
        ranked = sorted(hole.positions,
                        key=lambda p: (-readable_neighbours(self.tape, p, window),
                                       p))
        return ranked[:self.max_occurrences]

    def rank_values(self, hole: Hole, top: int | None = None) -> list[int]:
        """Candidate bytes for this class, best first."""
        positions = self._evidence_positions(hole)
        scored = sorted(((v, self._score_value(positions, v)[0])
                         for v in self._pool(hole)), key=lambda t: -t[1])
        values = [v for v, _ in scored]
        return values if top is None else values[:top]

    def _pool(self, hole: Hole) -> list[int]:
        """Which byte values are worth scoring for this class.

        A predictor that carries constraints gets to eliminate first.  Ranking a
        byte that cannot occur wastes the ranking and, worse, lets a
        confident-looking impossibility through.  Constraints are evaluated at
        every occurrence and intersected, since a class has to satisfy all of
        the places it appears at once.
        """
        ask = getattr(self.model, "candidates", None)
        if ask is None:
            return self._candidates
        allowed: set[int] | None = None
        tape = self.tape
        for pos in self._evidence_positions(hole):
            right = bytearray()
            for j in range(1, self.lookahead + 1):
                v = tape.value_at(pos + j)
                if v is None:
                    break
                right.append(v)
            got = ask(self._left_context(pos), bytes(right))
            if got is None:
                continue
            allowed = set(got) if allowed is None else (allowed & got)
        if not allowed:
            return self._candidates
        pool = [v for v in self._candidates if v in allowed]
        return pool or self._candidates

    def _right_context(self, pos: int) -> bytes:
        out = bytearray()
        for j in range(1, self.lookahead + 1):
            v = self.tape.value_at(pos + j)
            if v is None:
                break
            out.append(v)
        return bytes(out)

    def _score_batched(self, hole: Hole, pool):
        """Ask a predictor that can score a whole candidate set at once.

        A model that runs a forward pass per candidate is unusable here: the
        solver asks for tens of thousands of scores on a modest fragment, which
        is minutes rather than seconds.  Predictors exposing rank() get the
        whole set in one call per occurrence instead.

        Two things have to be got right, and neither was at first.

        A candidate the predictor declined to score is not a candidate that
        scored zero.  Defaulting it to zero makes it beat every byte that was
        actually scored, since real log-probabilities are negative, and a pile
        of tied unscorable bytes then splits the confidence between them so
        nothing clears the threshold at all.  Declined candidates are dropped.

        And the scale has to match.  A byte n-gram returns a log-probability for
        one byte; a sequence model returns one for a whole context, hundreds of
        bits in magnitude, of which all but a little is the shared prefix.  Only
        differences between candidates carry information, so each position's
        scores are taken relative to that position's best and divided by the
        bytes they actually cover.  That puts both kinds of predictor on the
        scale the confidence threshold assumes.
        """
        rank = getattr(self.model, "rank", None)
        if rank is None or not getattr(self.model, "prefers_batched", True):
            return None          # cheap enough to score properly, one at a time
        totals: dict[int, float] = {}
        usable: set[int] | None = None
        scored = 0
        for pos in self._evidence_positions(hole):
            right = self._right_context(pos)
            got = rank(self._left_context(pos), pool, right)
            if not got:
                continue
            scored += 1
            best = max(score for _, score in got)
            width = 1 + len(right)
            here = {value: (score - best) / width for value, score in got}
            usable = set(here) if usable is None else (usable & set(here))
            for value, relative in here.items():
                totals[value] = totals.get(value, 0.0) + relative
        if not scored or not usable:
            return []          # the predictor declined; do not grind the slow path
        return [(v, totals[v], totals[v] / scored) for v in usable]

    def _ngram_scored(self, hole: Hole, pool):
        """What the ranking model says, batched where it can be."""
        scored = self._score_batched(hole, pool)
        if scored is None:              # no batched path: score one at a time
            positions = self._evidence_positions(hole)
            scored = []
            for v in pool:
                total, terms = self._score_value(positions, v)
                scored.append((v, total, total / max(terms, 1)))
        return scored

    def _filler_scored(self, hole: Hole, pool):
        """The byte model's answer for one class, on the solver's own scale.

        Restricted to `pool`, because the pool is where the constraints live: a
        dynamic block's Huffman table names the bytes that can appear as
        literals at all, and a fill outside that set is not unlikely, it is
        impossible. The model is free to prefer such a byte and must not be
        allowed to spend the class on it.

        Converted from nats to log2 so the confidence threshold, which was
        calibrated on the n-gram, sees one scale from both predictors.
        """
        import math

        positions = self._evidence_positions(hole)
        totals = self.filler.class_logprobs(self.tape, positions)
        if totals is None:
            return []
        n = max(len(positions), 1)
        out = []
        for v in pool:
            total = float(totals[v]) / math.log(2)
            out.append((v, total, total / n))
        return out

    def propose_one(self, hole: Hole) -> Proposal | None:
        positions = self._evidence_positions(hole)
        pool = self._pool(hole)

        if self.mode == "ngram":
            scored, source, agrees = self._ngram_scored(hole, pool), "ngram", None
        elif self.mode == "slm":
            scored, source, agrees = self._filler_scored(hole, pool), "slm", None
        else:
            # Both consulted. The byte model's answer is the one taken, because
            # where they disagree it is right far more often -- 67 against 22 on
            # p38-5 and 84 against 5 on p35-12. What the n-gram adds is a
            # confidence signal: agreement covers about three quarters of the
            # bytes and is right on 72.8% to 96.9% of them. Mixing the two
            # scores is NOT done: the weight that measured best
            # on one carve landed 6.5 points below the byte model alone on a
            # held-out one.
            scored = self._filler_scored(hole, pool)
            other = self._ngram_scored(hole, pool)
            source = "slm"
            if scored and other:
                agrees = (max(scored, key=lambda x: x[1])[0]
                          == max(other, key=lambda x: x[1])[0])
            else:
                agrees = None

        if not scored:
            return None
        scored.sort(key=lambda t: -t[1])
        best_v, best_total, best_mean = scored[0]
        if self.mode == "both" and self._filler_calibrated:
            # The byte model carries its own curve, so its confidence is the
            # number the floor reads, as in "slm" mode: at an 80% floor it
            # filled 1,030 classes of three articles it never saw at 99.4%,
            # where the gate below passed 72 at 90%. Agreement is still
            # recorded; a curve split on it measured no better
            # (predict/calibration.py fit_filler).
            conf = self._confidence(scored)
        elif self.mode == "both":
            # The threshold is a precision promise, and only the n-gram's
            # confidence was ever calibrated to keep it. The byte model's
            # scores are far sharper, so the same softmax saturates on them:
            # on a phrack capture with a 536 B packet lost, ^f at 0.995 filled
            # 87% of classes at 24.1% precision, and no reading of the byte
            # model's own confidence did better than 42% at any threshold.
            # So a class is
            # as sure as the n-gram is, and only where the two name the same
            # byte; where they disagree it is 0, and f4 still offers it.
            conf = self._confidence(other) if agrees else 0.0
        else:
            conf = self._confidence(scored)
        ground = grounding(self.tape, positions, max(4, self.order))
        return Proposal(hole.anchor, hole.positions, best_v, best_total, conf,
                        tuple((v, t) for v, t, _ in scored[1:5]),
                        source=source, agreement=agrees, grounded=ground)

    @property
    def _filler_calibrated(self) -> bool:
        return getattr(self.filler, "calibration", None) is not None

    @property
    def floor_calibration(self):
        """What the floor is measured against in this mode, or None."""
        return floor_calibration(self.mode, self.calibration, self.filler)

    def annotate(self, p: Proposal, runs, gaps) -> Proposal:
        """Regime and measured precision, exactly as ^f will read them."""
        p.run = class_run(runs, p.positions)
        p.near = any(q in gaps for q in p.positions)
        curve = self.floor_calibration
        if curve is None:
            return p
        p.precision = curve.precision(p.confidence, p.run, p.near)
        if p.near and p.grounded < 1.0 and curve.long:
            # The gap curve was measured with real bytes around each class.
            # Once model fills are part of the context it no longer applies:
            # near a lost packet, fills scored on each other went from 75%
            # predicted to 31% right. On a carve the same staging helps,
            # so this is for the gap regime only.
            p.precision = min(p.precision or 0.0, curve.long[0])
        return p

    def _confidence(self, scored) -> float:
        """Softmax over the best candidates' per-evidence-term scores.

        Per term, not total: totals run to tens of bits once a class appears in
        a dozen places, and a softmax over those saturates at 1.0 for whatever
        came first, which makes the threshold meaningless and fills everything
        in one round.
        """
        if not scored:
            return 0.0
        top = sorted(scored, key=lambda t: -t[1])[:12]
        m = top[0][2]
        weights = [2.0 ** ((mean - m) / self.temperature) for _, _, mean in top]
        return weights[0] / sum(weights)

    def propose(self, holes=None, limit: int | None = None,
                progress=None) -> list[Proposal]:
        """Rank every unfilled class by how confident its best value is.

        `progress(done, total, note)` is optional and counts classes scored --
        1,742 of them takes 2.25 seconds against an order-5 model, which is long
        enough that a caller may want to say so. It may raise to abort.
        """
        holes = list(holes if holes is not None else self.tape.holes())
        # Most evidence first: a class seen in many places is cheaper to be sure
        # about, and filling it unlocks context for its neighbours.
        holes.sort(key=lambda h: -h.size)
        if limit:
            holes = holes[:limit]
        out = []
        # Runs count model fills as unread, so they only move when the operator
        # types; once per call is enough.
        runs = evidence_runs(self.tape) if holes else {}
        gaps = interior_gaps(self.tape) if holes else set()
        for n, hole in enumerate(holes):
            if progress is not None:
                progress(n, len(holes), f"scoring class {n + 1} of {len(holes)}")
            p = self.propose_one(hole)
            if p is not None:
                out.append(self.annotate(p, runs, gaps))
        out.sort(key=lambda p: (-p.gate, -p.confidence, -p.size))
        return out

    # ---- the search -----------------------------------------------------------

    def solve(self, min_confidence: float | None = None,
              max_rounds: int | None = None, on_round=None,
              progress=None, batch: int | None = None,
              long_runs: bool = False,
              dictionary: bool = False) -> SolveResult:
        """Fill the surest classes first, propagate, re-score what moved, repeat.

        `min_confidence` is a floor, not a schedule.  Each round takes the
        `batch` most confident classes in the whole tape, so the descent runs
        from the surest byte down to the threshold and stops there -- one call
        does what walking a confidence ladder by hand used to.

        Ordering is most of the accuracy.  Filling every class above the
        threshold at once scores them all against the tape as it was *before*
        any of them landed, so at a low threshold the doubtful fills go in
        beside the sure ones and then poison the context for later rounds.
        Measured on a phrack carve at a 0.70 floor, against ground truth:

            batch size    classes filled   precision of new fills
            all at once        414              72.3%
            32                 392              80.7%
            1                  386              80.4%

        The gain is in staging at all, not in staging finely: one-at-a-time is
        no better than 32 and takes 4.7x as long, because a fill only reaches
        its neighbours through their context and `_neighbours_of` already
        re-scores exactly those.  32 is where accuracy and wall-clock meet.
        """
        threshold = self.min_confidence if min_confidence is None else min_confidence
        rounds = self.max_rounds if max_rounds is None else max_rounds
        size = self.batch if batch is None else max(1, batch)
        # A round either fills at least one class or breaks out, so holes fall
        # monotonically and the cap is a backstop against oscillation, not the
        # thing that ends the search.  It still has to scale with the batch: a
        # low floor on a large tape can leave thousands of classes fillable at
        # `size` a round, and a fixed 60 would quietly stop with some of them
        # still standing above the floor -- a silent short recovery, which is
        # the one failure this tool must not have.
        rounds = max(rounds, len(self.tape.holes()) // size + 2)
        result = SolveResult()
        # Scores survive between rounds: a fill changes the context of its
        # neighbours and nothing else, so re-scoring the whole tape every round
        # would be paying for the same numbers repeatedly.
        scores: dict[int, Proposal | None] = {}
        queued = [0]            # classes above the floor at the last count

        for rnd in range(rounds):
            holes = {h.anchor: h for h in self.tape.holes()}
            if not holes:
                result.stopped = "no holes left"
                break
            for anchor in [a for a in scores if a not in holes]:
                del scores[anchor]
            stale = [h for a, h in holes.items() if a not in scores]

            def overall(note):
                """The whole run's bar: filled against filled plus what is
                queued above the floor. Unknown until round 1 has counted."""
                if rnd == 0 and not result.filled and not queued[0]:
                    return 0, 0, note
                return (len(result.filled), len(result.filled) + queued[0],
                        note)

            def round_progress(done, total, note, _r=rnd, _n=len(stale)):
                # The number of rounds is not knowable in advance, so there are
                # two bars: the run as a whole on top, and each round's own
                # scoring below it, kept full once done (progress.py). Round 1
                # scores every open class; later rounds only the neighbours of
                # the last fills.
                what = (f"scoring all {_n:,} open classes" if _r == 0 else
                        f"re-scoring {_n:,} neighbours of the last fills")
                progress(*overall(f"round {_r + 1} · {len(result.filled):,} "
                                  f"filled · {what}"),
                         stage=(_r + 1, done, total))

            for p in self.propose(
                    stale, progress=None if progress is None else round_progress):
                scores[p.anchor] = p
            for h in stale:                     # propose() drops the unscorable
                scores.setdefault(h.anchor, None)
            if progress is not None:
                progress(*overall(f"round {rnd + 1} · scored"),
                         stage=(rnd + 1, len(stale), len(stale)))

            ready = sorted((p for p in scores.values()
                            if p is not None and p.gate >= threshold),
                           key=lambda p: (-p.gate, -p.confidence, -p.size))
            if not long_runs and ready:
                # Held back, not dropped: they cleared the floor, and the
                # operator may still want them -- `long_runs=True` is the
                # "continue anyway". See LONG_RUN for what they measured.
                short = [p for p in ready if p.run < LONG_RUN]
                result.held = len(ready) - len(short)
                ready = short
            if (not dictionary and ready
                    and getattr(self.session, "has_dictionary", False)):
                # A class holding a dictionary byte (a negative position on a
                # stream with a DICTID) is outside the floor's regime. Held
                # back like the long runs; `dictionary=True` overrides.
                text = [p for p in ready
                        if min(holes[p.anchor].positions) >= 0]
                result.held_dictionary = len(ready) - len(text)
                ready = text
            queued[0] = len(ready)
            if progress is not None and ready:
                # The fill queue as the bar: filled against filled plus what
                # clears the floor right now. It can grow when fills lift new
                # classes over the floor, so the round count is a floor too.
                left = -(-len(ready) // size)
                progress(len(result.filled), len(result.filled) + len(ready),
                         f"round {rnd + 1} · {len(result.filled):,} filled · "
                         f"{len(ready):,} above the floor now · at least "
                         f"{left} round{'s' if left != 1 else ''} to go",
                         stage=(rnd + 1, len(stale), len(stale)))
            if not ready:
                if result.held_dictionary:
                    result.stopped = (
                        f"the {result.held_dictionary} classes left above "
                        f"{pct(threshold)} hold bytes of the preset "
                        f"dictionary, a word list rather than text, where "
                        f"fills measured {DICTIONARY_PRECISION} right")
                elif result.held:
                    result.stopped = (
                        f"the {result.held} classes left above "
                        f"{pct(threshold)} sit only inside runs of "
                        f"{LONG_RUN}+ bytes nothing was read from, where "
                        f"fills measured {LONG_RUN_PRECISION} right")
                elif self.floor_calibration is not None:
                    # The floor is measured precision now, so say what this
                    # model can reach here rather than only that it did not.
                    # pct, the floor's own format: 0.8957 as "90%" told the
                    # operator a 90% floor would fill it, and it did not.
                    reach = max((p.gate for p in scores.values()
                                 if p is not None), default=0.0)
                    result.stopped = (
                        f"nothing here measured {pct(threshold)} right for "
                        f"this model; the best open class measured "
                        f"{pct(reach)}, so a floor at or under that fills it")
                else:
                    result.stopped = (f"nothing above {pct(threshold)} "
                                      f"confidence")
                break

            touched_positions = []
            before = len(result.filled)
            for p in ready[:size]:
                if self.tape.value_at(p.anchor) is not None:
                    continue            # an earlier fill in this round resolved it
                try:
                    moved = self.session.set_guess(p.anchor, p.value, SOLVED)
                except (ValueError, KeyError):
                    scores.pop(p.anchor, None)
                    continue
                result.filled.append(p)
                touched_positions.extend(moved)
                note = getattr(self.session, "note_fill", None)
                if note is not None:
                    from ..core.session import FillNote
                    note(p.anchor, FillNote(
                        how={"ngram": "n-gram", "slm": "byte model",
                             "both": "both"}[self.mode],
                        confidence=p.confidence, agreement=p.agreement,
                        runners=tuple(v for v, _ in p.runners),
                        grounded=p.grounded, run=p.run, floor=threshold,
                        precision=p.precision, near=p.near))
            result.rounds = rnd + 1
            # What is still queued from this round's count, so the bar does
            # not jump back while the next round re-scores.
            queued[0] = max(0, len(ready) - (len(result.filled) - before))
            if on_round:
                on_round(result, len(ready))
            if not touched_positions:
                result.stopped = "no class could be applied"
                break
            for anchor in self._neighbours_of(touched_positions):
                scores.pop(anchor, None)
        else:
            result.stopped = f"hit the {rounds}-round limit"

        result.remaining = len(self.tape.holes())
        return result

    def _neighbours_of(self, positions) -> set[int]:
        """Anchors of classes whose context a fill at these positions changed."""
        reach = self.order + self.lookahead
        window = set()
        for p in positions:
            window.update(range(p - reach, p + reach + 1))
        out = set()
        for hole in self.tape.holes():
            if any(q in window for q in hole.positions):
                out.add(hole.anchor)
        return out


def train_from_session(session, extra_paths=(), order: int = 4):
    """A model trained on the stream's own *decoded* bytes, plus any corpus.

    Self-adaptation is worth something on its own: the readable part of a stream
    is usually the best available sample of the unreadable part.

    The model's own fills are excluded.  Training on those teaches it its own
    guesses as ground truth: a wrong fill is learned as fact and then votes for
    itself in every other context it fits.  Measured on a phrack carve,
    self-training on everything readable cost 18 points of precision against not
    self-training at all, and 5 of those 18 were this -- the fills contaminating
    their own evidence.

    An operator's typed byte is not that loop.  It is evidence from outside the
    model, which is the whole reason it is worth typing, and it stays: feeding
    corrections back took one comparison from 5 classes at 33% to 21 at 53%.
    """
    from .context import ContextModel
    from ..core.tape import SOLVED, SOLVED_PROP
    model = ContextModel(order=order)
    if extra_paths:
        model.train_files(extra_paths)
    hypotheses = (SOLVED, SOLVED_PROP)
    tape = session.tape
    run = bytearray()
    # From min_pos, not 0: a carve resynced mid-stream has back-references
    # reaching before its own start, and those bytes decoded like any other.
    for pos in range(tape.min_pos, tape.length):
        value = tape.value_at(pos)
        if value is not None and tape.prov_at(pos) not in hypotheses:
            run.append(value)
        elif run:
            model.train(bytes(run))
            run.clear()
    if run:
        model.train(bytes(run))
    return model
