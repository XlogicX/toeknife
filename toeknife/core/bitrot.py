"""Repairing a header whose bits are present but wrong.

The other loss cases are completion problems: bits are missing and something has
to supply them. Bit rot is a correction problem. Every bit is there and some
small number of them are inverted, so the search space is the flips over the
header rather than the tables over the alphabet, a difference measured at a
hundred orders of magnitude.

What was also measured is the part an operator has to be told: **small does
not mean unambiguous.** Undoing the damage is guaranteed to be among the one-flip
candidates, because inverting the flipped bit is one of the flips enumerated. It
is usually not the only candidate that parses cleanly, and nothing available here
separates the true repair from a coincidental one. So this returns every survivor
with what it produces, and never picks.

What separates them is the container, when there is one. A gzip member or zip
entry records the output's size and checksum, and every survivor's output can
be held against both. On the Phrack article, 82 one-bit flips across a dynamic
header broke the decode; 63 left more than one survivor (up to 54), the true
repair was not first in 26, and the CRC passed exactly one survivor every
time: the true one. So survivors are checked and the
ones that match go first. Without a recorded checksum the list stays unranked.

The rates were measured by classifying each candidate against the true output,
which is exactly what an operator does not have. This is the same enumeration with the answer key
removed: a candidate survives if it decodes cleanly to the end, which is a
property of the bitstream and not of anybody's expectations.
"""
from __future__ import annotations

import itertools
import time
import zlib
from dataclasses import dataclass, field

from .inflate import Decoder

# A clean landing. Anything else means the flip did not produce a stream.
LANDED = ("end", "eob")


@dataclass(frozen=True)
class Candidate:
    """One set of flipped bits that leaves the block parsing cleanly."""
    bits: tuple[int, ...]
    status: str
    out_len: int
    holes: int
    known: int
    # The container's size and checksum held against this output: True when
    # both match, False when either does not, None with nothing recorded.
    matches: bool | None = None

    @property
    def label(self) -> str:
        where = ", ".join(f"bit {b:,}" for b in self.bits)
        mark = {True: "  checksum ok", False: "  checksum fails",
                None: ""}[self.matches]
        return f"{where} → {self.out_len:,} B, {self.known:,} known{mark}"


@dataclass
class Repair:
    """What the enumeration found, and what it cost to find it."""
    span: tuple[int, int]                    # the bit range searched
    k: int
    tried: int
    total: int
    candidates: list[Candidate] = field(default_factory=list)
    capped: bool = False
    seconds: float = 0.0                     # wall time of the search

    def next_cost(self) -> tuple[int, float]:
        """Flips, and seconds at this search's own rate, for one more bit
        over the same span: every pair after a single, and so on."""
        lo, hi = self.span
        n, k = hi - lo, self.k + 1
        flips = 1
        for i in range(k):
            flips = flips * (n - i) // (i + 1)
        per = self.seconds / self.tried if self.tried else 0.0
        return flips, flips * per

    @property
    def unique(self) -> bool:
        return len(self.candidates) == 1

    @property
    def matching(self) -> list[Candidate]:
        return [c for c in self.candidates if c.matches]

    @property
    def checked(self) -> bool:
        return any(c.matches is not None for c in self.candidates)

    def summary(self) -> str:
        lo, hi = self.span
        head = (f"{self.tried:,} of {self.total:,} {self.k}-bit flips over "
                f"{hi - lo:,} header bits")
        if self.checked:
            n, m = len(self.candidates), len(self.matching)
            if m == 1:
                return (f"{head}: {n} parse, and 1 matches the recorded size "
                        f"and checksum. That one is the original.")
            if m == 0:
                return (f"{head}: {n} parse, and none matches the recorded "
                        f"size and checksum. The damage is elsewhere or more "
                        f"than {self.k} bit(s).")
            return (f"{head}: {n} parse, and {m} match the recorded size and "
                    f"checksum; they produce different output, so the "
                    f"checksum can't choose between them.")
        if not self.candidates:
            return (f"{head}: nothing parsed cleanly. Either the damage is not "
                    f"in this span or it is more than {self.k} bit(s).")
        if self.unique:
            return (f"{head}: exactly one survivor. That is the strongest this "
                    f"route gets; f8 is what proves it.")
        return (f"{head}: {len(self.candidates)} survivors. The true repair is "
                f"among them when the damage is {self.k} bit(s), and nothing "
                f"here says which one it is.")


def header_span(session, block=None) -> tuple[int, int] | None:
    """The bits that carry the table, which is where rot has to be undone.

    From the block's first bit to the first symbol it emitted: the header is
    exactly what sits between them. Deliberately not one bit further. A flip in
    the BODY usually still parses -- it corrupts one symbol rather than the code
    that reads every symbol -- so extending the span into the body would bury the
    real repair under hundreds of survivors that merely decode.

    The default block is the LAST one in the trace, which on a damaged capture is
    the block the decode stopped in. When the header failed to parse there are no
    symbols to bound it, and the bound is where the decoder gave up: a bit can
    only affect reads at or after itself, so whatever choked it lies at or before
    that point.

    Not restricted to dynamic blocks, because two of the bits in question are
    BTYPE itself: rot there turns a dynamic block into a fixed or stored one, and
    a repair route that looked only for dynamic blocks could not see the damage
    it was there to undo.
    """
    blocks = list(session.trace.blocks)
    if block is None:
        block = blocks[-1] if blocks else None
    if block is None:
        return None
    body = [s for s in session.trace.symbols if s.block == block.index]
    end = body[0].bit_start if body else getattr(block, "bit_end", 0)
    if end <= block.bit_start:
        # Nothing parsed at all, so there is no measured end. Three bits is the
        # least a block header can be, and BTYPE is inside them.
        end = min(block.bit_start + 3, len(session.data) * 8)
    if end <= block.bit_start:
        return None
    return block.bit_start, end


def flip(data: bytes, bits) -> bytes:
    out = bytearray(data)
    for b in bits:
        out[b >> 3] ^= 1 << (b & 7)
    return bytes(out)


def _score(data: bytes, start_bit: int, stream=None):
    """Decode from `start_bit` and say whether it landed, and whether what it
    produced matches the container's record."""
    try:
        trace = Decoder(data, strict_window=False).stream(start_bit)
    except Exception:
        return None
    if trace.status not in LANDED:
        return None
    st = trace.tape.stats()
    return (trace.status, st["out_len"], st["holes"], st["known"],
            _matches(trace.tape, stream))


def _matches(tape, stream) -> bool | None:
    if stream is None:
        return None
    size = getattr(stream, "uncompressed_size", None)
    crc = getattr(stream, "crc32", None)
    adler = getattr(stream, "adler32", None)
    if size is None and crc is None and adler is None:
        return None
    if size is not None and len(tape) != size:
        return False
    if crc is None and adler is None:
        return None                  # the size alone proves nothing
    buf, mask = tape.recovered()
    if not all(mask):
        return False
    if crc is not None:
        return zlib.crc32(buf) & 0xFFFFFFFF == crc
    return zlib.adler32(buf) & 0xFFFFFFFF == adler


def spans(session) -> list[tuple[int, int]]:
    """Header spans to try, the block the decode stopped in first.

    More than one, because the block that looks like the problem is not always
    the one carrying the damage. Flipping BFINAL on a final block makes the
    decoder read on past the end, so the trace's last block is a phantom and the
    real header is behind it. Trying each block's header in turn costs nothing --
    a header is a few hundred bits -- and the alternative is widening the span
    into the body, which buries the answer.
    """
    out, seen = [], set()
    for block in reversed(list(session.trace.blocks)):
        got = header_span(session, block)
        if got and got not in seen:
            seen.add(got)
            out.append(got)
    return out


def repair(session, k: int = 1, span=None, cap: int = 500_000,
           progress=None) -> Repair:
    """Every k-bit flip over the header that leaves the block parsing cleanly.

    `progress(done, total, note)` is optional and may raise to abort, which is
    how the interactive path cancels a k=2 sweep that is taking too long.
    """
    if span is None:
        tried_any = None
        for candidate_span in spans(session) or [None]:
            if candidate_span is None:
                return Repair((0, 0), k, 0, 0)
            got = repair(session, k, candidate_span, cap, progress)
            if got.candidates:
                return got
            tried_any = tried_any or got
        return tried_any or Repair((0, 0), k, 0, 0)
    lo, hi = span
    positions = range(lo, hi)
    n = len(positions)
    total = 1
    for i in range(k):
        total = total * (n - i) // (i + 1)
    start = session.bit_offset

    out = Repair(span, k, 0, total)
    started = time.monotonic()
    for bits in itertools.combinations(positions, k):
        if out.tried >= cap:
            out.capped = True
            break
        out.tried += 1
        if progress is not None and not out.tried % 256:
            progress(out.tried, min(total, cap),
                     f"{out.tried:,} of {min(total, cap):,} flips, "
                     f"{len(out.candidates)} surviving")
        got = _score(flip(session.data, bits), start,
                     getattr(session, "stream", None))
        if got is None:
            continue
        status, out_len, holes, known, matches = got
        out.candidates.append(Candidate(bits, status, out_len, holes, known,
                                        matches))
    # A checksum match first; then most output: a flip that reaches the end
    # of the block and produces the most bytes is the one worth looking at
    # first. Without a checksum that is an ordering, not a ranking -- see the
    # module docstring.
    out.candidates.sort(key=lambda c: (c.matches is not True, -c.known,
                                       -c.out_len, c.bits))
    out.seconds = time.monotonic() - started
    return out


def apply(session, candidate: Candidate) -> bytes:
    """The capture with this candidate's flips undone."""
    return flip(session.data, candidate.bits)
