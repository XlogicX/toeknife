"""Decoded structure, annotated with the exact bits it came from.

Every field records its bit span so the inspector pane can show, for any symbol
or header field the cursor lands on, precisely which bits in the file encode it.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class BitField:
    """One named run of bits and what it decoded to."""
    label: str
    start: int
    end: int
    detail: str = ""

    @property
    def nbits(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class Symbol:
    """A literal, a match, or an end-of-block marker."""
    kind: str                       # 'literal' | 'match' | 'eob'
    out_pos: int
    bit_start: int
    bit_end: int
    block: int = 0
    literal: int | None = None
    length: int | None = None
    distance: int | None = None
    fields: tuple[BitField, ...] = ()
    # Which entry of each alphabet was actually read. `literal` and `length` are
    # what the symbol MEANS; these are which code carried it, and the two do not
    # always determine each other -- length 258 is legal as symbol 285 or as 284
    # with its extra bits maxed, and a literal decoded from a recovered code set
    # has a code with no meaning attached at all. Anything counting per-symbol
    # frequency has to read this rather than infer it backwards from the value.
    code_sym: int | None = None     # litlen alphabet: 0..285
    dist_sym: int | None = None     # distance alphabet: 0..29, matches only

    @property
    def out_len(self) -> int:
        if self.kind == "literal":
            return 1
        if self.kind == "match":
            return self.length or 0
        return 0

    @property
    def nbits(self) -> int:
        return self.bit_end - self.bit_start

    def label(self) -> str:
        if self.kind == "literal":
            v = self.literal
            # A literal decoded with a recovered code set has a code but no
            # meaning yet, and saying "0x00" for that would be inventing one.
            if v is None:
                return "literal (unassigned)"
            if 0x20 <= v < 0x7F:
                return f"literal {chr(v)!r}"
            return f"literal 0x{v:02x}"
        if self.kind == "match":
            return f"match {self.length},{self.distance}"
        return "end of block"


@dataclass(slots=True)
class CodeLengthOp:
    """One instruction in a dynamic block's code-length stream."""
    op: str                         # 'lens' | 'repeat' | 'zeros'
    value: int                      # the code length written
    count: int                      # how many alphabet entries it filled
    index: int                      # first alphabet index filled
    bit_start: int
    bit_end: int


@dataclass(slots=True)
class BlockInfo:
    index: int
    bit_start: int
    bit_end: int = 0
    bfinal: int = 0
    btype: int = -1
    btype_name: str = "?"
    out_start: int = 0
    out_end: int = 0
    fields: list[BitField] = field(default_factory=list)
    clen_ops: list[CodeLengthOp] = field(default_factory=list)
    litlen: object | None = None            # HuffmanTable
    dist: object | None = None              # HuffmanTable
    codelen: object | None = None           # HuffmanTable for the length code
    hlit: int = 0
    hdist: int = 0
    hclen: int = 0
    stored_len: int = 0
    n_literals: int = 0
    n_matches: int = 0
    status: str = "ok"

    @property
    def nbits(self) -> int:
        return self.bit_end - self.bit_start


@dataclass(slots=True)
class Trace:
    """The result of decoding from one starting bit offset."""
    data: bytes
    bit_start: int
    bit_end: int
    blocks: list[BlockInfo]
    symbols: list[Symbol]
    tape: object                    # Tape
    status: str
    error: str | None = None
    headerless: bool = False
    # Set when this trace was decoded from a recovered code set, where a literal
    # symbol id is a placeholder -- (code length, index among literals of that
    # length) -- and NOT a byte value. Anything that puts a name to a literal has
    # to check: reading symbol 0x41 as 'A' here would invent a meaning the
    # bitstream never carried, which is the one thing this tool must not do.
    identity: dict[int, tuple[int, int]] | None = None

    @property
    def bits_consumed(self) -> int:
        return self.bit_end - self.bit_start

    @property
    def bits_left(self) -> int:
        return len(self.data) * 8 - self.bit_end

    @property
    def n_literals(self) -> int:
        return sum(1 for s in self.symbols if s.kind == "literal")

    @property
    def n_matches(self) -> int:
        return sum(1 for s in self.symbols if s.kind == "match")

    def block_at_out(self, pos: int) -> BlockInfo | None:
        """The block that produced output position pos (binary search).

        Blocks tile the output in order and never overlap, so this is the same
        search as symbol_at_out one level up.  Negative positions belong to the
        pre-fragment window, which predates every block we can see.
        """
        if pos < 0:
            return None
        lo, hi = 0, len(self.blocks) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            b = self.blocks[mid]
            if pos < b.out_start:
                hi = mid - 1
            elif pos >= b.out_end:
                lo = mid + 1
            else:
                return b
        return None

    def symbol_at_out(self, pos: int) -> Symbol | None:
        """The symbol responsible for output position pos (binary search)."""
        lo, hi = 0, len(self.symbols) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s = self.symbols[mid]
            if pos < s.out_pos:
                hi = mid - 1
            elif pos >= s.out_pos + max(s.out_len, 1):
                lo = mid + 1
            else:
                return s
        return None


# Statuses that mean "this alignment cannot be right"
HARD_FAILURES = frozenset({
    "bad-code", "bad-litlen", "bad-dist", "bad-distance",
    "bad-table", "reserved-btype", "bad-stored-len",
})
