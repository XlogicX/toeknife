"""Repairing flipped bits with a chunk's own CRC.

A PNG keeps a CRC32 over every chunk's type and data: the compressed bytes,
not the pixels. When a bit flips inside an IDAT chunk the stream may still
decode to the end, quietly wrong, and only the Adler-32 at the end says so.
The chunk CRC says which chunk, and since CRC32 is linear over GF(2) it also
says which bit.

Flipping bit i of a message changes its CRC by a fixed amount D_i that
depends only on i and the message length. The damage changed the CRC by
S = crc(chunk as found) ^ stored CRC. One flipped bit is an i with D_i == S;
two are a pair with D_i ^ D_j == S, found in one pass with a table of the
D_i. CRC32 tells every one- and two-bit error from none at these lengths, so
a chunk that fails its CRC with one or two flips has candidates; a candidate
counts only when the stream it makes also matches its Adler-32, which settles
the rare pair of two-bit patterns sharing a syndrome.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field


@dataclass
class Fix:
    segment: int                # 0-based index into the stream's segments
    bits: tuple[int, ...]       # bit offsets inside the segment's data
    where: str = "data"         # data, or the stored CRC itself
    proven: bool | None = None  # the stream's Adler-32 agreed

    def label(self, total: int) -> str:
        if self.where == "crc":
            return (f"chunk {self.segment + 1} of {total}: the stored CRC has "
                    f"one bit wrong; the data is intact")
        bits = ", ".join(f"byte {b >> 3:,} bit {b & 7}" for b in self.bits)
        return f"chunk {self.segment + 1} of {total}: {bits}"


@dataclass
class ChunkRepair:
    bad: list[int] = field(default_factory=list)      # failing segments
    fixes: list[Fix] = field(default_factory=list)    # proven ones first
    candidates: int = 0
    reason: str = ""

    def summary(self, total: int) -> str:
        if not self.bad:
            return self.reason or "every chunk matches its CRC"
        proven = [f for f in self.fixes if f.proven]
        names = ", ".join(str(k + 1) for k in self.bad)
        if not self.fixes:
            return (f"chunk(s) {names} of {total} fail their CRC and no one- "
                    f"or two-bit flip restores it: more than two bits are "
                    f"wrong, or the chunk's length is")
        if proven:
            return (f"{len(proven)} repair(s) restore the chunk CRC and the "
                    f"stream's Adler-32: "
                    + "; ".join(f.label(total) for f in proven))
        return (f"{self.candidates} flip(s) restore the chunk CRC, and none "
                f"makes the stream match its Adler-32")


def _deltas(prefix: bytes, n: int) -> list[int]:
    """D_i for every bit of an n-byte data field behind `prefix` (the chunk
    type, which the CRC also covers)."""
    buf = bytearray(len(prefix) + n)
    base = zlib.crc32(bytes(buf))
    out = []
    for i in range(n * 8):
        at = len(prefix) + (i >> 3)
        buf[at] ^= 1 << (i & 7)
        out.append(zlib.crc32(buf) ^ base)
        buf[at] ^= 1 << (i & 7)
    return out


def flips(prefix: bytes, chunk: bytes, stored: int) -> list[Fix]:
    """Every one- or two-bit change to `chunk` that gives it CRC `stored`.
    Segment index is filled in by the caller."""
    syndrome = (zlib.crc32(prefix + chunk) & 0xFFFFFFFF) ^ stored
    if not syndrome:
        return []
    if bin(syndrome).count("1") == 1:
        # The stored CRC took the hit; nothing in the data needs to change.
        return [Fix(-1, (), "crc")]
    d = _deltas(prefix, len(chunk))
    out = [Fix(-1, (i,)) for i, v in enumerate(d) if v == syndrome]
    first = {}
    for i, v in enumerate(d):
        first.setdefault(v, i)
    for i, v in enumerate(d):
        j = first.get(v ^ syndrome)
        if j is not None and j > i:
            out.append(Fix(-1, (i, j)))
    return out


def bad_segments(session) -> list[int]:
    """Segments whose bytes in the file, with any applied repair, fail their
    stored CRC."""
    stream = getattr(session, "stream", None)
    data = repaired_file(session) or getattr(session, "file_data", None)
    crcs = getattr(stream, "segment_crcs", None)
    if not crcs or data is None or not stream.segments:
        return []
    return [k for k, ((off, ln), (ctype, stored))
            in enumerate(zip(stream.segments, crcs))
            if zlib.crc32(ctype + data[off:off + ln]) & 0xFFFFFFFF != stored]


def _segment_starts(stream) -> list[int]:
    at, out = 0, []
    for _off, ln in stream.segments:
        out.append(at)
        at += ln
    return out


def repair(session, file_data: bytes) -> ChunkRepair:
    """Find the flips that restore each failing chunk's CRC, and keep those
    that also make the stream match its Adler-32. Touches nothing."""
    stream = session.stream
    out = ChunkRepair()
    crcs = getattr(stream, "segment_crcs", None)
    if not crcs or not stream.segments:
        out.reason = "this stream has no per-chunk CRC"
        return out
    starts = _segment_starts(stream)
    for k, ((off, ln), (ctype, stored)) in enumerate(zip(stream.segments, crcs)):
        chunk = file_data[off:off + ln]
        if zlib.crc32(ctype + chunk) & 0xFFFFFFFF == stored:
            continue
        out.bad.append(k)
        for fix in flips(ctype, chunk, stored):
            fix.segment = k
            out.candidates += 1
            if fix.where == "crc":
                fix.proven = None
            else:
                fix.proven = _adler_ok(session, stream, starts[k], fix.bits)
            out.fixes.append(fix)
    out.fixes.sort(key=lambda f: (f.proven is not True, len(f.bits)))
    return out


def _flipped(session, stream, start: int, bits) -> bytes | None:
    """The session's DEFLATE bytes with these bits of a segment flipped, or
    None when a bit falls in the zlib header or Adler-32 instead."""
    data = bytearray(session.data)
    for b in bits:
        at = start + (b >> 3) - stream.header_skip
        if not 0 <= at < len(data):
            return None
        data[at] ^= 1 << (b & 7)
    return bytes(data)


def _adler_ok(session, stream, start: int, bits) -> bool | None:
    data = _flipped(session, stream, start, bits)
    if data is None or stream.adler32 is None:
        return None
    try:
        out = zlib.decompressobj(-15).decompress(data)
    except zlib.error:
        return False
    return zlib.adler32(out) & 0xFFFFFFFF == stream.adler32


def apply(session, fix: Fix) -> None:
    """Put the repaired bytes in the session and decode again, and remember
    where they are in the file so the file itself can be written repaired."""
    if fix.where == "crc":
        return
    stream = session.stream
    start = _segment_starts(stream)[fix.segment] - stream.header_skip
    bits = [start * 8 + b for b in fix.bits]
    if any(not 0 <= b < len(session.data) * 8 for b in bits):
        return                          # in the zlib header or Adler-32
    off = stream.segments[fix.segment][0]
    session.flip_bits(bits, [(off + (b >> 3), b & 7) for b in fix.bits])


def repaired_file(session) -> bytes | None:
    """The input file with every applied repair flipped back, or None when
    nothing was repaired."""
    fixes = getattr(session, "file_fixes", None)
    data = getattr(session, "file_data", None)
    if not fixes or data is None:
        return None
    out = bytearray(data)
    for at, bit in fixes:
        out[at] ^= 1 << bit
    return bytes(out)
