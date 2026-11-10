"""What a container hands back: located streams, and the constraints on them.

Finding the compressed bytes is the boring half.  The valuable half is that
wrappers record what the data was *before* compression -- its length and its
checksum -- which turns "is this decode right?" from a judgement call into
arithmetic, and bounds how much output a damaged stream should have produced.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

# Encodings a located stream can be in.
RAW = "raw"          # bare DEFLATE, no wrapper
ZLIB = "zlib"        # RFC 1950 header + Adler-32 trailer
GZIP = "gzip"        # RFC 1952 member header + CRC32/ISIZE trailer


@dataclass
class Stream:
    """One compressed stream located inside a container."""
    name: str
    container: str
    offset: int                          # byte offset of the compressed data
    length: int | None = None            # compressed length, None if not recorded
    encoding: str = RAW
    method: int = 8
    method_name: str = "deflate"
    # Constraints recovered from the wrapper.
    crc32: int | None = None
    adler32: int | None = None
    uncompressed_size: int | None = None
    encrypted: bool = False
    source: str = "central"              # where the metadata came from
    notes: list[str] = field(default_factory=list)
    # True when back-references legitimately reach outside this stream's own
    # output, as with a zlib preset dictionary.  Without this the decoder treats
    # the first such reference as corruption and stops.
    references_outside: bool = False
    # zlib's DICTID: the Adler-32 of the preset dictionary, which proves a
    # candidate dictionary on its own.
    dictid: int | None = None
    # A stream is not always one contiguous run of bytes.  A PNG's pixel data is
    # a single zlib stream split across however many IDAT chunks the encoder
    # felt like emitting, with a length, a type and a CRC wedged between each
    # pair.  Segments are (offset, length) in the file, concatenated in order.
    segments: list[tuple[int, int]] | None = None
    # Bytes of wrapper to drop once the segments are joined: a zlib header at
    # the front, its Adler-32 at the back.
    header_skip: int = 0
    trailer_skip: int = 0
    # A CRC32 the wrapper keeps over each segment's compressed bytes, as
    # (chunk type, stored CRC): a PNG chunk's CRC covers its type and data.
    # It says which segment holds damage, and with one or two flipped bits,
    # which bits (core/chunkcrc.py).
    segment_crcs: list[tuple[bytes, int]] | None = None
    # Runs of the stream that are KNOWN to be absent, as (position within the
    # joined stream, byte count).  Only a capture can populate this: a file is
    # either all there or truncated, while TCP numbers every byte and so says
    # exactly how wide a hole is.  `slice` refuses to join across one unless
    # asked, because splicing the two sides together would produce a stream that
    # looks continuous and is not.
    missing: list[tuple[int, int]] = field(default_factory=list)

    @property
    def bit_offset(self) -> int:
        return self.offset * 8

    @property
    def decodable(self) -> bool:
        """Whether ToeKnife can even try.  Says nothing about whether it works."""
        return self.method in (8, 9) and not self.encrypted

    @property
    def why_not(self) -> str:
        if self.encrypted:
            return "entry is encrypted"
        if self.method == 0:
            return "stored, not compressed"
        if self.method not in (8, 9):
            return f"method {self.method} ({self.method_name}) is not DEFLATE"
        return ""

    @property
    def spans(self) -> list[tuple[int, int]]:
        if self.segments is not None:
            return self.segments
        return [(self.offset, self.length if self.length is not None else -1)]

    def slice(self, data: bytes, allow_gaps: bool = False) -> bytes:
        """The DEFLATE bytes, with segments joined and any wrapper removed.

        A length of None runs to the end of the file, which is what gzip and
        zlib need since neither records how long its compressed data is.
        """
        if self.missing and not allow_gaps:
            lost = sum(n for _p, n in self.missing)
            raise ValueError(
                f"{lost} byte(s) of this stream are missing in "
                f"{len(self.missing)} place(s); joining the surviving runs "
                f"would splice them into a stream that looks continuous and is "
                f"not. Pass allow_gaps=True if that is what you want.")
        if self.segments is None:
            end = (len(data) if self.length is None
                   else min(len(data), self.offset + self.length))
            body = data[self.offset:end]
        else:
            body = b"".join(data[off:off + ln] for off, ln in self.segments)
        stop = len(body) - self.trailer_skip if self.trailer_skip else len(body)
        return body[self.header_skip:max(self.header_skip, stop)]

    def summary(self) -> str:
        size = "?" if self.length is None else str(self.length)
        out = "?" if self.uncompressed_size is None else str(self.uncompressed_size)
        crc = "-" if self.crc32 is None else f"{self.crc32:08x}"
        flags = "".join(("E" if self.encrypted else "-",
                         "D" if self.decodable else "-"))
        return (f"{self.offset:>10} {size:>10} {out:>10} {crc:>9} "
                f"{self.method_name:<9} {flags} {self.name}")


@dataclass
class Verdict:
    """The result of checking a decode against what the wrapper promised."""
    size_expected: int | None = None
    size_got: int | None = None
    crc_expected: int | None = None
    crc_got: int | None = None
    adler_expected: int | None = None
    adler_got: int | None = None
    complete: bool = False               # every output byte known
    ok: bool | None = None               # None when nothing could be checked

    def summary(self) -> str:
        bits = []
        if self.size_expected is not None:
            match = "ok" if self.size_got == self.size_expected else "MISMATCH"
            bits.append(f"size {self.size_got}/{self.size_expected} {match}")
        for name, want, got in (("crc", self.crc_expected, self.crc_got),
                                ("adler", self.adler_expected, self.adler_got)):
            if want is None:
                continue
            if not self.complete:
                bits.append(f"{name} unchecked (output still has unknown bytes)")
            else:
                match = "ok" if got == want else "MISMATCH"
                bits.append(f"{name} {got:08x}/{want:08x} {match}")
        return "  ".join(bits) or "nothing to check against"


def verify(stream: Stream, tape) -> Verdict:
    """Check a decoded tape against the container's recorded size and checksum.

    A checksum is only meaningful once every output byte is known, so a partial
    recovery reports the size comparison and leaves the checksum alone rather
    than reporting a failure that means nothing.
    """
    v = Verdict(size_expected=stream.uncompressed_size, size_got=len(tape),
                crc_expected=stream.crc32, adler_expected=stream.adler32)
    buf, mask = tape.recovered()
    v.complete = all(mask)
    if v.complete:
        if v.crc_expected is not None:
            v.crc_got = zlib.crc32(buf) & 0xFFFFFFFF
        if v.adler_expected is not None:
            v.adler_got = zlib.adler32(buf) & 0xFFFFFFFF
    checks = []
    if v.size_expected is not None:
        checks.append(v.size_got == v.size_expected)
    if v.complete and v.crc_expected is not None:
        checks.append(v.crc_got == v.crc_expected)
    if v.complete and v.adler_expected is not None:
        checks.append(v.adler_got == v.adler_expected)
    v.ok = all(checks) if checks else None
    # A size match over output that still has unknown bytes is not a pass. The
    # only check that ran was "the right number of bytes came out", and the
    # bytes themselves are partly unknown -- reporting that as verified would
    # be the tool making exactly the claim it exists to distrust.
    if v.ok and not v.complete:
        v.ok = None
    return v


@dataclass
class ContainerReport:
    kind: str
    streams: list[Stream] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    structure: list[str] = field(default_factory=list)

    @property
    def decodable(self) -> list[Stream]:
        return [s for s in self.streams if s.decodable]

    def header_line(self) -> str:
        return (f"{'offset':>10} {'comp':>10} {'uncomp':>10} {'crc32':>9} "
                f"{'method':<9} fl name")
