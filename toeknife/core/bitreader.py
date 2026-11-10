"""LSB-first bit reader over a byte buffer.

DEFLATE packs bits starting at the least significant bit of each byte, but
Huffman codes are written most-significant-bit first within that stream.  So
integers (extra bits, HLIT, HDIST, LEN) are read LSB-first, while Huffman codes
are accumulated MSB-first one bit at a time.  Both live here.

Positions are absolute bit offsets into the buffer, which is what makes an
arbitrary-alignment fragment decode expressible at all.
"""
from __future__ import annotations


class BitEOF(Exception):
    """Ran off the end of the buffer."""


class BitReader:
    __slots__ = ("data", "pos", "nbits")

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.nbits = len(data) * 8
        self.pos = pos

    @property
    def remaining(self) -> int:
        return self.nbits - self.pos

    def bit(self) -> int:
        p = self.pos
        if p >= self.nbits:
            raise BitEOF
        self.pos = p + 1
        return (self.data[p >> 3] >> (p & 7)) & 1

    def read(self, n: int) -> int:
        """Read n bits as an LSB-first integer."""
        if n == 0:
            return 0
        if self.pos + n > self.nbits:
            raise BitEOF
        v = 0
        p = self.pos
        data = self.data
        for i in range(n):
            v |= ((data[p >> 3] >> (p & 7)) & 1) << i
            p += 1
        self.pos = p
        return v

    def peek(self, n: int) -> int:
        """Peek n bits as an LSB-first integer, zero-padded past end of data."""
        v = 0
        p = self.pos
        data = self.data
        limit = self.nbits
        for i in range(n):
            if p >= limit:
                break
            v |= ((data[p >> 3] >> (p & 7)) & 1) << i
            p += 1
        return v

    def skip(self, n: int) -> None:
        if self.pos + n > self.nbits:
            raise BitEOF
        self.pos += n

    def align_to_byte(self) -> int:
        """Advance to the next byte boundary.  Returns bits discarded."""
        pad = (-self.pos) % 8
        self.skip(pad)
        return pad

    def read_bytes(self, n: int) -> bytes:
        """Read n whole bytes.  Only valid when byte-aligned (stored blocks)."""
        if self.pos & 7:
            raise ValueError("read_bytes requires byte alignment")
        start = self.pos >> 3
        if start + n > len(self.data):
            raise BitEOF
        self.pos += n * 8
        return self.data[start:start + n]


def bit_string(data: bytes, start: int, end: int) -> str:
    """Render bits [start, end) in stream order as '0'/'1' characters.

    This is the order the decoder consumed them, which is what the inspector
    pane should show.  For a Huffman code that reads left-to-right as the
    canonical MSB-first code; for an extra-bits field it reads LSB-first, so it
    looks reversed relative to its numeric value.  That is not a bug, it is what
    is physically in the file.
    """
    out = []
    nbits = len(data) * 8
    for p in range(start, min(end, nbits)):
        out.append("1" if (data[p >> 3] >> (p & 7)) & 1 else "0")
    return "".join(out)
