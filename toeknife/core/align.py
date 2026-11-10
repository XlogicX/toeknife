"""Finding where a fragment's bitstream actually starts.

DEFLATE is not byte aligned, so a carved fragment begins at an unknown bit
offset, probably part way through a Huffman symbol.  The original tool walked
offsets one at a time and asked the operator "does this look right?" up to 26
times.  This ranks every offset instead and shows its reasoning, so the operator
confirms a shortlist rather than auditioning the whole space.

Nothing here decides for you.  Every candidate keeps its component scores so the
UI can say *why* it ranked something first, and a wrong-looking winner is one
keypress away from being overruled.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from .events import HARD_FAILURES, Trace
from .huffman import HuffmanTable
from .inflate import Decoder
from .tables import MAX_SYMBOL_BITS

TEXTISH = frozenset(list(range(0x20, 0x7F)) + [0x09, 0x0A, 0x0D])


@dataclass
class Candidate:
    bit_offset: int
    trace: Trace
    score: float
    parts: dict[str, float] = field(default_factory=dict)
    subset_of: int | None = None      # this decode is a tail of that offset's

    @property
    def status(self) -> str:
        return self.trace.status

    def summary(self) -> str:
        t = self.trace
        st = t.tape.stats()
        bits = f"{t.bits_left} bit{'s' if t.bits_left != 1 else ''} left"
        tail = f"  tail of bit {self.subset_of}" if self.subset_of is not None else ""
        return (f"bit {self.bit_offset:<3} {self.score:6.2f}  "
                f"{t.n_literals:>4} lit  {t.n_matches:>3} match  "
                f"{st['out_len']:>6} out  {st['holes']:>3} holes  "
                f"{bits:>14}  {t.status}{tail}")


def _printable_ratio(literals: list[int]) -> float:
    if not literals:
        return 0.0
    return sum(1 for v in literals if v in TEXTISH) / len(literals)


def _entropy_score(literals: list[int]) -> float:
    """1.0 for a maximally skewed literal histogram, 0.0 for uniform.

    Normalised against log2(n) rather than 8 bits, because a 40-symbol fragment
    cannot exhibit 8 bits of entropy no matter how random it is.
    """
    n = len(literals)
    if n < 8:
        return 0.0
    counts = Counter(literals)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values())
    hmax = math.log2(min(256, n))
    return max(0.0, 1.0 - h / hmax) if hmax else 0.0


def _prior_score(literals: list[int], prior: dict[int, float]) -> float:
    """Mean log-likelihood of the literals under a byte prior, vs uniform.

    The prior normally comes from a dynamic block's own Huffman table, which
    encodes the byte frequencies of the data that produced it.  That makes this
    the strongest available signal whenever a header fragment is in hand.
    """
    if not literals or not prior:
        return 0.0
    floor = 1e-6
    ll = sum(math.log2(prior.get(v, floor)) for v in literals) / len(literals)
    return max(0.0, min(1.0, (ll + 8.0) / 8.0))


def _model_score(trace: Trace, model) -> float:
    """How many bits the model saves over uniform, across the whole fragment.

    Deliberately a total and not a mean.  A mean rewards decoding *less*: an
    alignment that recovers forty bytes which happen to be slightly more
    predictable beats one that recovers forty-seven, and the shorter decode is
    the worse answer.  Measuring bits saved against a uniform baseline, spread
    over the fragment's length, rewards explaining more and explaining it well
    at the same time.

    A hole truncates the context rather than being spliced out.  Splicing
    invents bigrams that are not in the data ("for" + "forensic" reads as
    "forforensic") and the model then punishes the correct decode for them.
    """
    if model is None:
        return 0.0
    tape = trace.tape
    order = getattr(model, "order", 4)
    total, n = 0.0, 0
    ctx = bytearray()
    for p in range(tape.length):
        v = tape.value_at(p)
        if v is None:
            ctx.clear()
            continue
        total += model.logprob(v, bytes(ctx[-order:]))
        n += 1
        ctx.append(v)
    if not n:
        return 0.0
    bits_saved = total - (-8.0 * n)
    return max(0.0, min(1.0, bits_saved / max(len(trace.data) * 8, 1)))


def _structure_score(trace: Trace) -> float:
    """Hard evidence from the format itself, independent of what the data means."""
    s = 0.0
    if trace.status in ("eob", "end"):
        s += 1.0
        # A fragment that really ends at end-of-block should leave only the
        # final byte's padding behind.
        s += 1.0 if trace.bits_left < 8 else max(0.0, 0.6 - trace.bits_left / 512)
    elif trace.status == "truncated":
        s += 0.35                       # plausible for a fragment cut mid-stream
    elif trace.status == "limit":
        s += 0.2
    if trace.status in HARD_FAILURES:
        s -= 1.5
    return s


def score_trace(trace: Trace, prior: dict[int, float] | None = None,
                profile: str = "auto", model=None) -> tuple[float, dict[str, float]]:
    literals = [s.literal for s in trace.symbols if s.kind == "literal"]
    total_bits = max(1, len(trace.data) * 8)
    parts = {
        "structure": _structure_score(trace),
        "printable": _printable_ratio(literals),
        "entropy": _entropy_score(literals),
        "prior": _prior_score(literals, prior or {}),
        "model": _model_score(trace, model),
        # How much of the fragment this alignment actually accounts for.  Among
        # decodes that are all individually valid -- and on a self-synchronising
        # code there are always several -- the one that explains more of the
        # bits is the better explanation, and recovers strictly more data.
        "coverage": trace.bits_consumed / total_bits,
    }
    if profile == "text":
        w = {"structure": 3.0, "printable": 2.5, "entropy": 0.3, "prior": 1.0,
             "coverage": 1.2, "model": 4.0}
    elif profile == "binary":
        w = {"structure": 3.0, "printable": 0.2, "entropy": 1.2, "prior": 2.0,
             "coverage": 1.2, "model": 4.0}
    else:                                # auto: let both speak, structure leads
        w = {"structure": 3.0, "printable": 1.4, "entropy": 0.8, "prior": 1.6,
             "coverage": 1.2, "model": 4.0}
    return sum(w[k] * v for k, v in parts.items()), parts


def rank(data: bytes, litlen: HuffmanTable | None = None,
         dist: HuffmanTable | None = None, max_offset: int = MAX_SYMBOL_BITS,
         profile: str = "auto", max_symbols: int | None = None,
         prior: dict[int, float] | None = None, model=None) -> list[Candidate]:
    """Score every plausible starting bit offset, best first.

    Scans up to MAX_SYMBOL_BITS offsets rather than the 8 you might expect.
    Offsets 0..7 cover every bit phase, but if the fragment opens part way
    through a symbol the correct phase can still fail on that first partial
    code, so the scan has to be able to step over a whole leading symbol.
    """
    if prior is None and litlen is not None:
        prior = litlen.byte_prior()
    dec = Decoder(data, strict_window=False)
    limit = min(max_offset, max(1, len(data) * 8))
    cands: list[Candidate] = []
    for off in range(limit):
        trace = dec.fragment(off, litlen, dist, max_symbols=max_symbols)
        score, parts = score_trace(trace, prior, profile, model)
        cands.append(Candidate(off, trace, score, parts))

    _annotate_subsets(cands)
    cands.sort(key=lambda c: (-c.score, c.bit_offset))
    return cands


def _annotate_subsets(cands: list[Candidate]) -> None:
    """Mark candidates whose decode is merely the tail of an earlier one.

    Huffman codes are self-synchronising, so bit 3, bit 11 and bit 19 of a
    fragment routinely land on the same symbol stream with a symbol or two
    shaved off the front.  Decoding is deterministic from a bit position, so
    offset B is a strict tail of offset A exactly when B is one of the bit
    positions at which A started a symbol.  That is an exact test, not a
    heuristic, and it annotates rather than hides: a tail alignment is still a
    real alignment, it just recovers less.
    """
    by_offset = sorted(cands, key=lambda c: c.bit_offset)
    starts = {c.bit_offset: {s.bit_start for s in c.trace.symbols} for c in by_offset}
    for c in by_offset:
        for other in by_offset:
            if other.bit_offset >= c.bit_offset:
                break
            if c.bit_offset in starts[other.bit_offset]:
                c.subset_of = other.bit_offset
                break


def best(data: bytes, **kw) -> Candidate | None:
    r = rank(data, **kw)
    return r[0] if r else None
