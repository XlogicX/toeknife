"""Finding block boundaries inside a fragment whose own block header is gone.

If a fragment starts part way through a dynamic block, the Huffman table that
defines its symbols is behind you and the bits are unreadable.  Reconstructing
that table from the body is not practical: the search space is code lengths 0-15
over up to 286 symbols, and Ralf Brown, who wrote two DFRWS papers on recovering
corrupt DEFLATE, concluded it was "intractible in the general case, but usable
when the Huffman trees are known".

The way out is not to rebuild the table but to skip to the next one.  Streams
carry block headers periodically, and each dynamic header brings a fresh table,
so everything from the next boundary onward is readable without any outside
help.  This module finds those boundaries.

Four signatures, strongest first:

  sync-flush  a `00 00 FF FF` marker.  Z_SYNC_FLUSH and Z_FULL_FLUSH emit an
              empty stored block, so a new header starts byte-aligned right
              after it.  Cheap to find and close to certain.
  stored      a stored block header, where LEN and NLEN must be complements.
              Sixteen bits of redundancy makes this very hard to hit by chance.
  dynamic     a dynamic header whose code-length code and both Huffman tables
              build cleanly.  Structured enough to be worth scanning for, but
              it does fire on noise, so candidates are decode-validated.
  fixed       three bits.  Meaningless on its own, off by default, and only
              ever reported after validation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bitreader import BitEOF, BitReader
from .events import Trace
from .huffman import HuffmanError
from .inflate import Decoder

SYNC_MARKER = b"\x00\x00\xff\xff"

KIND_SYNC = "sync-flush"
KIND_STORED = "stored"
KIND_DYNAMIC = "dynamic"
KIND_FIXED = "fixed"

# How much a signature is worth before any decoding is attempted.
CONFIDENCE = {KIND_SYNC: 0.95, KIND_STORED: 0.9, KIND_DYNAMIC: 0.5, KIND_FIXED: 0.05}

# How often a bit-by-bit sweep reports where it is, when anyone is asking. A
# quarter of a 95 KB capture is 190,000 bit positions and takes seconds, so a
# caller that wants to draw a bar needs to hear from it more than once.
PROGRESS_STEP = 4096


@dataclass
class SyncPoint:
    bit_offset: int
    kind: str
    note: str = ""
    status: str = ""
    symbols: int = 0
    out_bytes: int = 0
    holes: int = 0
    blocks: int = 0
    trace: Trace | None = field(default=None, repr=False)

    @property
    def byte_offset(self) -> int:
        return self.bit_offset // 8

    @property
    def confidence(self) -> float:
        return CONFIDENCE.get(self.kind, 0.0)

    @property
    def score(self) -> float:
        """Signature strength plus how much decoding from here actually yields."""
        s = self.confidence * 100
        if self.status in ("end", "eob"):
            s += 40
        elif self.status == "truncated":
            s += 10
        s += min(self.out_bytes, 100000) / 1000.0
        return s

    def summary(self) -> str:
        at = f"bit {self.bit_offset} (byte {self.byte_offset}"
        at += f".{self.bit_offset % 8})" if self.bit_offset % 8 else ")"
        return (f"{at:<28} {self.kind:<11} {self.out_bytes:>8} B out  "
                f"{self.blocks:>3} blk  {self.holes:>5} holes  {self.status:<10} {self.note}")


def _sync_points(data: bytes, start_bit: int, end_bit: int):
    pos = data.find(SYNC_MARKER, start_bit // 8)
    while pos >= 0:
        bit = (pos + len(SYNC_MARKER)) * 8
        if bit >= end_bit:
            return
        if bit >= start_bit:
            yield SyncPoint(bit, KIND_SYNC, f"after flush marker at byte {pos}")
        pos = data.find(SYNC_MARKER, pos + 1)


def _stored_points(data: bytes, start_bit: int, end_bit: int, progress=None,
                   base: int = 0, total: int = 0, note: str = ""):
    for bit in range(start_bit, end_bit):
        if progress is not None and not bit % PROGRESS_STEP:
            progress(bit - base, total, note)
        br = BitReader(data, bit)
        try:
            br.read(1)
            if br.read(2) != 0:
                continue
            br.align_to_byte()
            ln = br.read(16)
            nln = br.read(16)
        except BitEOF:
            return
        if ln != (~nln) & 0xFFFF or ln == 0:
            continue                      # LEN==0 is the flush marker, already found
        yield SyncPoint(bit, KIND_STORED, f"LEN={ln} NLEN ok")


def _table_points(data: bytes, start_bit: int, end_bit: int, want_fixed: bool,
                  progress=None, base: int = 0, total: int = 0,
                  note: str = ""):
    dec = Decoder(data, strict_window=False)
    for bit in range(start_bit, end_bit):
        if progress is not None and not bit % PROGRESS_STEP:
            progress(bit - base, total, note)
        br = BitReader(data, bit)
        try:
            br.read(1)
            btype = br.read(2)
        except BitEOF:
            return
        if btype == 2:
            try:
                trace = dec.stream(bit, max_symbols=0)
            except (HuffmanError, ValueError, BitEOF):
                continue
            block = trace.blocks[0] if trace.blocks else None
            if block is None or block.litlen is None or trace.status == "bad-table":
                continue
            yield SyncPoint(bit, KIND_DYNAMIC,
                            f"HLIT={block.hlit} HDIST={block.hdist} HCLEN={block.hclen}")
        elif btype == 1 and want_fixed:
            yield SyncPoint(bit, KIND_FIXED, "fixed-Huffman header (weak signature)")


def scan(data: bytes, kinds=(KIND_SYNC, KIND_STORED, KIND_DYNAMIC),
         start_bit: int = 0, max_bit: int | None = None,
         validate: bool = True, min_out: int = 1,
         max_results: int = 200, progress=None) -> list[SyncPoint]:
    """Locate candidate block boundaries, best first.

    Candidates are decode-validated by default: a signature that decodes to
    nothing is a false positive no matter how it looked.  That second stage is
    what makes the dynamic-header scan usable, since the header alone fires on
    random data roughly once every four thousand bits.

    `progress(done, total, note)` is optional and is called in bit offsets
    through the two stages this walks: trying every bit position, then decoding
    what survived. It may raise to abort the scan -- nothing is committed until
    the list is returned, so unwinding costs only the work done.
    """
    end_bit = len(data) * 8 if max_bit is None else min(max_bit, len(data) * 8)
    found: dict[int, SyncPoint] = {}
    # Two stages over the same axis: find, then validate.
    total = 2 * max(1, end_bit - start_bit)

    def offer(sp: SyncPoint) -> None:
        prev = found.get(sp.bit_offset)
        if prev is None or sp.confidence > prev.confidence:
            found[sp.bit_offset] = sp

    if KIND_SYNC in kinds:
        for sp in _sync_points(data, start_bit, end_bit):
            offer(sp)
    if KIND_STORED in kinds:
        for sp in _stored_points(data, start_bit, end_bit, progress,
                                 start_bit, total, "looking for stored blocks"):
            offer(sp)
    if KIND_DYNAMIC in kinds or KIND_FIXED in kinds:
        for sp in _table_points(data, start_bit, end_bit, KIND_FIXED in kinds,
                                progress, start_bit, total,
                                "looking for block headers"):
            offer(sp)

    points = sorted(found.values(), key=lambda p: p.bit_offset)
    if not validate:
        return points

    dec = Decoder(data, strict_window=False)
    kept = []
    for n, sp in enumerate(points, 1):
        if progress is not None:
            # Candidate n of N, charged against the second half of the axis:
            # this stage decodes, so it is the slow one per item.
            progress(end_bit - start_bit
                     + (end_bit - start_bit) * n // max(1, len(points)),
                     total, f"decoding candidate {n} of {len(points)}")
        try:
            trace = dec.stream(sp.bit_offset)
        except Exception:
            continue
        st = trace.tape.stats()
        sp.trace = trace
        sp.status = trace.status
        sp.symbols = len(trace.symbols)
        sp.out_bytes = st["out_len"]
        sp.holes = st["holes"]
        sp.blocks = len(trace.blocks)
        if sp.kind == KIND_STORED and not _stored_continues(trace, data):
            continue
        if sp.out_bytes >= min_out:
            kept.append(sp)
    kept.sort(key=lambda p: (-p.score, p.bit_offset))
    return kept[:max_results]


def _stored_continues(trace, data: bytes) -> bool:
    """Is a stored-header candidate a real block, judged by what follows it?

    A stored block "decodes" whatever bytes come after its header, so the
    test the other kinds pass -- it produced output -- says nothing here, and
    LEN/NLEN is only 16 bits. In carved-midblock.bin one fired inside a
    dynamic block's body and copied 30,513 bytes of compressed data onto the
    tape as known, ranked first because it produced the most (wrap-up #5).
    So the stream has to carry on correctly after it: a final block leaves
    at most a trailer behind it, and any other is followed by a block that
    decodes. One that runs off the end of the data cannot be checked, and is
    not kept.
    """
    blocks = trace.blocks
    if not blocks or blocks[0].status != "eob":
        return False
    first = blocks[0]
    if first.bfinal:
        return len(data) - (first.bit_end + 7) // 8 <= 8
    return len(blocks) > 1 and blocks[1].status in ("eob", "truncated") \
        and blocks[1].out_end > blocks[1].out_start


def best_resync(data: bytes, **kw) -> SyncPoint | None:
    r = scan(data, **kw)
    return r[0] if r else None
