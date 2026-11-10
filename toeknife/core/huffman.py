"""Canonical Huffman code construction and decoding (RFC 1951 section 3.2.2).

Decoding uses a flat lookup table indexed by the next MAX bits peeked from the
stream, so each symbol costs one array index instead of the original tool's
linear scan over every code in the table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .bitreader import BitEOF, BitReader
from .tables import MAX_CODE_BITS


class HuffmanError(Exception):
    pass


def _reverse(value: int, nbits: int) -> int:
    r = 0
    for _ in range(nbits):
        r = (r << 1) | (value & 1)
        value >>= 1
    return r


@dataclass
class HuffmanTable:
    """A canonical Huffman code built from per-symbol code lengths.

    lengths[sym] == 0 means the symbol is absent from the alphabet, which is
    itself information: in a dynamic block it says that byte value never appears
    as a literal anywhere in the block.
    """

    lengths: tuple[int, ...]
    codes: dict[int, tuple[int, int]] = field(default_factory=dict)  # sym -> (code, nbits)
    max_bits: int = 0
    complete: bool = True
    oversubscribed: bool = False
    _lut: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self):
        used = [l for l in self.lengths if l]
        if not used:
            # Legal: an empty distance tree means the block contains no matches.
            self.max_bits = 0
            self.complete = True
            return
        self.max_bits = max(used)
        if self.max_bits > MAX_CODE_BITS:
            raise HuffmanError(f"code length {self.max_bits} exceeds {MAX_CODE_BITS}")

        bl_count = [0] * (self.max_bits + 1)
        for l in self.lengths:
            if l:
                bl_count[l] += 1

        # Kraft inequality: left == 0 means exactly complete, > 0 incomplete,
        # < 0 oversubscribed (impossible to build).
        left = 1
        for b in range(1, self.max_bits + 1):
            left <<= 1
            left -= bl_count[b]
            if left < 0:
                self.oversubscribed = True
                raise HuffmanError("oversubscribed Huffman table")
        self.complete = left == 0

        next_code = [0] * (self.max_bits + 2)
        code = 0
        for b in range(1, self.max_bits + 1):
            code = (code + bl_count[b - 1]) << 1
            next_code[b] = code

        for sym, length in enumerate(self.lengths):
            if not length:
                continue
            c = next_code[length]
            next_code[length] += 1
            self.codes[sym] = (c, length)

    def _build_lut(self) -> None:
        """Flat lookup table indexed by the next max_bits peeked from the stream.

        Built on first use rather than at construction.  A 15-bit table is 32768
        entries, and scanning damaged data for block headers builds thousands of
        tables purely to ask whether they are well formed, without ever decoding
        a symbol through them.
        """
        lut = [0] * (1 << self.max_bits)
        for sym, (code, length) in self.codes.items():
            # The stream delivers a code MSB-first, so as an LSB-first peek it
            # appears bit-reversed.  Fill every slot agreeing in the low
            # `length` bits.
            rev = _reverse(code, length)
            packed = (sym << 5) | length
            for i in range(rev, 1 << self.max_bits, 1 << length):
                lut[i] = packed
        self._lut = lut

    def decode(self, br: BitReader) -> tuple[int, int, int]:
        """Decode one symbol.  Returns (symbol, nbits, canonical_code)."""
        if not self.max_bits:
            raise HuffmanError("decode from empty Huffman table")
        if not self._lut:
            self._build_lut()
        packed = self._lut[br.peek(self.max_bits)]
        nbits = packed & 31
        if nbits == 0:
            raise HuffmanError("no valid code at this position")
        if br.remaining < nbits:
            raise BitEOF
        br.skip(nbits)
        sym = packed >> 5
        return sym, nbits, self.codes[sym][0]

    def symbols(self) -> list[int]:
        return sorted(self.codes)

    def code_str(self, sym: int) -> str:
        code, nbits = self.codes[sym]
        return format(code, f"0{nbits}b")

    def byte_prior(self) -> dict[int, float]:
        """Approximate byte frequencies implied by the code lengths.

        A canonical Huffman code assigns length L to a symbol of probability
        roughly 2**-L, so a dynamic block's literal table is a free frequency
        model of the data that produced it.  Useful both for ranking alignments
        and as a prior when guessing unknown bytes.
        """
        raw = {sym: 2.0 ** -length
               for sym, length in enumerate(self.lengths)
               if length and sym < 256}
        total = sum(raw.values())
        if not total:
            return {}
        return {sym: p / total for sym, p in raw.items()}


def fixed_litlen() -> HuffmanTable:
    from .tables import FIXED_LITLEN_LENGTHS
    return HuffmanTable(FIXED_LITLEN_LENGTHS)


def fixed_dist() -> HuffmanTable:
    from .tables import FIXED_DIST_LENGTHS
    return HuffmanTable(FIXED_DIST_LENGTHS)


_FIXED_CACHE: dict[str, HuffmanTable] = {}


def structural_tables(a, eob_len: int, length_lens, dist_lens):
    """Tables for a code set whose literal MEANINGS are not known.

    A recovered code set says how many literals sit at each code length, what
    length the end-of-block code has, and the lengths of the 29 length codes and
    30 distance codes.  That is strictly less than a table -- it never says WHICH
    byte any literal stands for -- but it is exactly enough to parse, and the
    reason is worth stating because it is not obvious:

    Canonical Huffman hands out code words in (length, symbol) order, and every
    literal symbol is below 256, which is below end-of-block, which is below
    every length code.  So which code words fall to literals, which to
    end-of-block and which to the length codes is fixed by the COUNTS alone.
    Any assignment with the same counts therefore yields the same partition and
    the same code words for everything that is not a literal, and decoding the
    real bitstream with it recovers the real parse.

    What it cannot recover is which literal is which, so the literals are given
    placeholder identities: `(code length, index among literals of that length)`,
    the only name a literal has before its meaning is known.  Returns
    `(litlen, dist, identity)` where identity maps the placeholder symbol id used
    in the returned table to that name.
    """
    lengths = [0] * 286
    identity: dict[int, tuple[int, int]] = {}
    s = 0
    for L in range(1, MAX_CODE_BITS + 1):
        for i in range(a[L]):
            if s >= 256:
                raise HuffmanError(
                    f"code set claims more than 256 literals (at length {L})")
            lengths[s] = L
            identity[s] = (L, i)
            s += 1
    lengths[256] = eob_len
    for j, ln in enumerate(length_lens):
        lengths[257 + j] = ln
    return (HuffmanTable(tuple(lengths)), HuffmanTable(tuple(dist_lens)),
            identity)


def fixed_tables() -> tuple[HuffmanTable, HuffmanTable]:
    """Fixed-mode tables, built once and shared (alignment scanning rebuilds a
    lot otherwise)."""
    if "litlen" not in _FIXED_CACHE:
        _FIXED_CACHE["litlen"] = fixed_litlen()
        _FIXED_CACHE["dist"] = fixed_dist()
    return _FIXED_CACHE["litlen"], _FIXED_CACHE["dist"]
