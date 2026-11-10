"""Filling the holes from a copy the operator suspects is the original.

A lost packet is not recoverable from a model: one 536 B segment hides
kilobytes of output as a solid run, and no corpus tried filled more than a few
percent of it right. What a real investigation often
has instead is a candidate -- the same page fetched later, a cached copy, the
public document -- and a different question: did this capture carry that text?

The capture can answer it, and better than a model ever could. Every byte that
decoded is a test the candidate must pass, and the container's checksum is a
test of everything at once. So the candidate is aligned to what decoded,
checked against it, used to fill every open class, and the CRC rules. A pass
is proof the lost bytes were exactly these. A fail says where the candidate
contradicts what the capture actually holds, which is what a different version
or an edited copy looks like.

Fills go in as the operator's own guesses (yellow), not as model fills: the
candidate is something the operator asserted, and the colour has to say so.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .tape import GUESS

K = 16                  # bytes per alignment probe
MIN_VOTES = 3           # probes that must agree before an offset is believed


@dataclass
class CopyFill:
    """What happened when a candidate copy was laid over the capture."""
    offset: int | None = None           # candidate index = output position + offset
    votes: int = 0
    checked: int = 0                    # decoded bytes the candidate covered
    contradictions: list = field(default_factory=list)   # output positions
    filled: int = 0                     # classes filled from the candidate
    positions: int = 0                  # output positions those classes cover
    uncovered: int = 0                  # classes the candidate does not reach
    split: int = 0                      # classes the candidate disagrees with itself on
    verdict: object = None              # Session.verify() afterwards
    # The capture's own decode already failed its checksum before the copy
    # was laid over it: its bytes, not only the copy's, may be wrong.
    decode_failed: bool = False
    # The copy on its own, cut at the alignment to the recorded size, checked
    # against the recorded checksum: None when there is nothing to check it
    # against. A True here proves the copy is the original whatever the
    # capture's own bytes say, which is the one proof left when damage put
    # wrong bytes on the tape.
    copy_matches: bool | None = None
    reason: str = ""
    # The copy was placed by sliding a window of the recorded size along it
    # until the CRC matched, because nothing decoded to align it on.
    window: bool = False

    @property
    def proven(self) -> bool:
        v = self.verdict
        return v is not None and v.ok is True

    @property
    def refuted(self) -> bool:
        """Ruled out, by the checksum or before it: a contradiction with what
        decoded, or the copy disagreeing with itself where the capture proves
        two bytes equal, is as final as a CRC mismatch."""
        v = self.verdict
        return bool(self.contradictions or self.split
                    or (v is not None and v.ok is False))

    def figures(self) -> str:
        """The counts alone, for a screen that states the verdict itself."""
        if self.offset is None:
            return self.reason
        head = (f"aligned at offset {self.offset:+,} ({self.votes} matching "
                f"runs); checked {self.checked:,} decoded bytes, "
                f"{len(self.contradictions):,} contradicted; filled "
                f"{self.filled:,} classes ({self.positions:,} positions)")
        if self.uncovered:
            head += f"; left {self.uncovered:,} the copy does not reach"
        if self.split and self.copy_matches:
            # The copy is the original, so the ties it breaks were made by
            # garbage decoded from the damage, not by the text.
            head += (f". {self.split:,} class(es) tie bytes together that "
                     f"the copy holds as different values: the damage made "
                     f"those ties")
        elif self.split:
            head += (f". It disagrees with itself on {self.split:,} "
                     f"class(es) the capture proves are one byte, so it "
                     f"differs from this text there")
        return head

    def summary(self) -> str:
        if self.offset is None:
            return self.reason
        if self.window:
            v = self.verdict
            size = v.size_expected if v is not None else None
            span = (f"bytes {self.offset:,}-{self.offset + size - 1:,}"
                    if size else f"from byte {self.offset:,}")
            head = (f"nothing decoded to line the copy up against; {span} "
                    f"of it match the recorded size and CRC32, so they are "
                    f"this entry's original content")
            if self.filled:
                head += (f"; the {self.positions:,} bytes are on the tape as "
                         f"yours")
            return head
        head = self.figures()
        v = self.verdict
        if v is None:
            return head + ". Nothing recorded a checksum, so the fit cannot " \
                "be proven"
        if self.copy_matches and self.contradictions:
            return head + (". The copy on its own matches the recorded size "
                           "and checksum, so it is the original; the "
                           f"{len(self.contradictions):,} byte(s) where it "
                           "differs are where the capture is damaged")
        if v.ok is False and self.decode_failed and self.contradictions:
            return head + (". The capture's own decode already failed its "
                           "checksum; if this copy is the original, the "
                           f"{len(self.contradictions):,} differing byte(s) "
                           "are where the capture is damaged")
        if v.ok is True and self.contradictions:
            return head + (". The checksum agrees, so the missing bytes were "
                           "exactly this copy's; but the copy contradicts the "
                           f"capture in {len(self.contradictions):,} decoded "
                           "byte(s): a different "
                           "version or an edited copy")
        if v.ok is True:
            return head + ". The checksum agrees: the missing bytes were " \
                "exactly this copy's"
        if v.ok is False:
            where = (" -- and since nothing that decoded contradicts it, the "
                     "difference is inside the lost bytes themselves"
                     if not self.contradictions else "")
            return head + ". The checksum disagrees: this copy is not what " \
                "was lost, at least not all of it" + where
        return head + ". Holes remain, so the checksum cannot rule yet"


def align(session, candidate: bytes):
    """Where the candidate sits against the output: (offset, votes) or None.

    Every K-byte run of decoded output that occurs exactly once in the
    candidate votes for one offset. A copy with a header, a trailer or a
    shifted start still lines up; one that shares nothing with what decoded
    does not, and saying so beats filling from it.
    """
    index: dict[bytes, int] = {}
    dup = set()
    for i in range(len(candidate) - K + 1):
        k = candidate[i:i + K]
        if k in index:
            dup.add(k)
        else:
            index[k] = i
    tape = session.tape
    votes = Counter()
    pos = 0
    while pos <= tape.length - K:
        run = [tape.value_at(p) for p in range(pos, pos + K)]
        if None in run:
            pos += run.index(None) + 1
            continue
        k = bytes(run)
        if k in index and k not in dup:
            votes[index[k] - pos] += 1
        pos += K
    if not votes:
        return None
    off, n = votes.most_common(1)[0]
    return (off, n) if n >= MIN_VOTES else None


def fill(session, candidate: bytes, name: str = "") -> CopyFill:
    """Align, check, fill every open class it reaches, and let the CRC rule.

    One undo group: a candidate that turns out wrong comes back out with a
    single ctrl+z.
    """
    out = CopyFill()
    got = align(session, candidate)
    if got is None:
        # Nothing decoded to line it up against, but the container may still
        # record the size and CRC: then a slice of the copy can match them on
        # its own, which proves it whatever the capture held (a gzip member
        # whose table is gone decodes to nothing and still has its trailer).
        stream = getattr(session, "stream", None)
        size = getattr(stream, "uncompressed_size", None)
        crc = getattr(stream, "crc32", None)
        at = (find_window(candidate, size, crc)
              if size and crc is not None else None)
        if at is None:
            out.reason = ("the copy does not line up with anything that "
                          f"decoded: no {K}-byte run of the output occurs in "
                          f"it")
            if size and crc is not None:
                out.reason += (f", and no {size:,}-byte slice of it has the "
                               f"recorded CRC32")
            return out
        out.offset, out.copy_matches, out.window = at, True, True
        tape = session.tape
        if not len(tape):
            # The recorded size says how long the output is; the matching
            # slice says what it holds. On the tape as the operator's bytes,
            # one ctrl+z from gone, so ^w writes the entry like any other.
            tape.emit_unknown(size)
            session.begin_group()
            try:
                for hole in tape.holes():
                    session.set_guess(hole.anchor, candidate[at + hole.anchor],
                                      GUESS, source=f"copy {name}".strip())
                    out.filled += 1
                    out.positions += hole.size
            finally:
                session.end_group()
        out.verdict = session.verify()
        return out
    out.offset, out.votes = got
    before = session.verify()
    out.decode_failed = (before is not None and before.ok is False)
    out.copy_matches = copy_matches(session, candidate, out.offset)
    tape = session.tape
    for pos in range(tape.min_pos, tape.length):
        v = tape.value_at(pos)
        c = pos + out.offset
        if v is None or not 0 <= c < len(candidate):
            continue
        out.checked += 1
        if candidate[c] != v:
            out.contradictions.append(pos)

    session.begin_group()
    try:
        for hole in tape.holes():
            vals = {candidate[p + out.offset] for p in hole.positions
                    if 0 <= p + out.offset < len(candidate)}
            if not vals:
                out.uncovered += 1
                continue
            if len(vals) > 1:
                out.split += 1
                continue
            try:
                session.set_guess(hole.anchor, vals.pop(), GUESS,
                                  source=f"copy {name}".strip())
            except (ValueError, KeyError):
                out.split += 1
                continue
            out.filled += 1
            out.positions += hole.size
    finally:
        session.end_group()
    out.verdict = session.verify()
    return out


def copy_matches(session, candidate: bytes, offset: int) -> bool | None:
    """Whether the copy, read from where output 0 aligns, is exactly what the
    container recorded: its size, then its CRC-32 or Adler-32."""
    import zlib
    stream = getattr(session, "stream", None)
    size = getattr(stream, "uncompressed_size", None)
    crc = getattr(stream, "crc32", None)
    adler = getattr(stream, "adler32", None)
    if size is None or (crc is None and adler is None):
        return None
    if offset < 0:
        return False
    span = candidate[offset:offset + size]
    if len(span) != size:
        return False
    if crc is not None:
        return zlib.crc32(span) & 0xFFFFFFFF == crc
    return zlib.adler32(span) & 0xFFFFFFFF == adler


def find_window(candidate: bytes, size: int, crc: int) -> int | None:
    """The first offset where `size` bytes of the candidate have this CRC32.

    A rolling CRC: CRC32 is affine over GF(2), so dropping the window's first
    byte is one XOR with a table of that byte followed by `size` zeros.
    """
    import zlib
    if size <= 0 or size > len(candidate):
        return None
    zeros = zlib.crc32(bytes(size))
    out = [zlib.crc32(bytes([b]) + bytes(size)) ^ zeros for b in range(256)]
    c = zlib.crc32(candidate[:size])
    if c == crc:
        return 0
    crc32 = zlib.crc32
    for i in range(len(candidate) - size):
        c = crc32(candidate[i + size:i + size + 1], c) ^ out[candidate[i]]
        if c == crc:
            return i + 1
    return None


def regions(positions, gap: int = 8) -> list[tuple[int, int]]:
    """Contradicted positions as (first, last) runs, for showing where."""
    runs: list[list[int]] = []
    for p in sorted(positions):
        if runs and p - runs[-1][1] <= gap:
            runs[-1][1] = p
        else:
            runs.append([p, p])
    return [tuple(r) for r in runs]
