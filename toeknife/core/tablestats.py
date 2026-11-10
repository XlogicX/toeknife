"""What a block's Huffman tables say, and what the block then did with them.

A dynamic block carries its own code table, and a canonical Huffman code assigns
a short code to what the encoder expected often. So the table is a frequency
statement about the data that produced it, readable without decoding a single
byte of the body -- which is why `HuffmanTable.byte_prior` already exists and is
used to rank alignments.

This module puts numbers on that statement and, separately, counts what the block
actually emitted. Both are facts, and they are different facts. The declared
share is a property of the header; the measured count is a property of the body.
Keeping them in two columns is the point: a table that disagrees with its own
block is how a foreign or wrongly-recovered table shows itself, and a table that
agrees says nothing more than "this header belongs to this body".

Nothing here decodes anything or reaches for a model. It reads a trace that has
already been decoded, so it cannot fail on data the decoder survived.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .tables import DISTANCE_BASE, DISTANCE_EXTRA, LENGTH_BASE, LENGTH_EXTRA

# Bands for the heat display, in bits, absolute rather than relative to the
# table in front of you. A 6-bit code is the same claim -- one symbol in 64 --
# in every block ever encoded, so two blocks can be compared. The cost is that a
# table using only long codes reads as uniformly cold, which is true.
HEAT_EDGES = (4, 6, 8, 10)


def heat_band(bits: int) -> int:
    """0 for the hottest band, 4 for the coldest, -1 for a symbol not in the
    alphabet at all."""
    if bits <= 0:
        return -1
    for band, edge in enumerate(HEAT_EDGES):
        if bits <= edge:
            return band
    return len(HEAT_EDGES)


def band_label(band: int) -> str:
    lo = 1 if band == 0 else HEAT_EDGES[band - 1] + 1
    hi = HEAT_EDGES[band] if band < len(HEAT_EDGES) else None
    span = f"{lo}-{hi}b" if hi else f"{lo}b+"
    if hi:
        return f"{span}  one in {2 ** lo}-{2 ** hi}"
    return f"{span}  one in {2 ** lo} or rarer"


ESCAPES = {0x09: "\\t", 0x0A: "\\n", 0x0D: "\\r", 0x00: "nul"}


def _ch(v: int) -> str:
    """A byte value named so that a space cannot be mistaken for nothing."""
    if 0x20 <= v < 0x7F:
        return f"'{chr(v)}'"
    if v in ESCAPES:
        return f"'{ESCAPES[v]}'"
    return ""


def litlen_name(sym: int) -> str:
    if sym < 256:
        return f"0x{sym:02x} {_ch(sym)}".rstrip()
    if sym == 256:
        return "end of block"
    i = sym - 257
    if i >= len(LENGTH_BASE):
        return f"sym {sym} invalid"
    base, extra = LENGTH_BASE[i], LENGTH_EXTRA[i]
    if not extra:
        return f"length {base}"
    return f"length {base}-{base + (1 << extra) - 1}"


def dist_name(sym: int) -> str:
    if sym >= len(DISTANCE_BASE):
        # 30 and 31 never occur in valid data but are representable in fixed
        # mode, which makes them a misalignment signal rather than a bug.
        return f"sym {sym} invalid"
    base, extra = DISTANCE_BASE[sym], DISTANCE_EXTRA[sym]
    if not extra:
        return f"distance {base}"
    return f"distance {base}-{base + (1 << extra) - 1}"


def codelen_name(sym: int) -> str:
    if sym < 16:
        return f"code length {sym}"
    return {16: "repeat last, 3-6×", 17: "zeros, 3-10",
            18: "zeros, 11-138"}.get(sym, f"sym {sym} invalid")


@dataclass(frozen=True)
class SymbolStat:
    sym: int
    name: str
    bits: int
    declared: float                 # 2**-bits: this symbol's share of the code
    count: int                      # times the block actually used it
    measured: float                 # count / everything emitted from this table

    @property
    def band(self) -> int:
        return heat_band(self.bits)


@dataclass(frozen=True)
class Alphabet:
    """One table, its declared shape, and what the block did with it."""
    key: str                        # 'litlen' | 'dist' | 'codelen'
    name: str
    lengths: tuple[int, ...]        # per-symbol code length, 0 = absent
    stats: list[SymbolStat]         # ranked: shortest code first
    kraft: float                    # sum of 2**-bits over the alphabet
    complete: bool
    emitted: int                    # symbols of this alphabet in the block
    partial: bool                   # the block never reached its end-of-block
    per_length: list[tuple[int, int, float]]   # (bits, symbols, code share)
    byte_space: bool                # worth drawing as a 16x16 map of 0..255
    counted: bool                   # whether measuring this was possible at all
    note: str = ""

    @property
    def used(self) -> int:
        return sum(1 for L in self.lengths if L)

    @property
    def size(self) -> int:
        return len(self.lengths)

    @property
    def min_bits(self) -> int:
        return min((L for L in self.lengths if L), default=0)

    @property
    def max_bits(self) -> int:
        return max(self.lengths, default=0)

    @property
    def unused(self) -> int:
        return self.size - self.used

    @property
    def absent_bytes(self) -> int:
        """Byte values with no code at all, which is the number worth saying:
        it counts literals only, where "no code" means "never appears in this
        block". Above 255 the alphabet is length codes, and a length code with no
        code just means the encoder found no match of that size."""
        return sum(1 for v in range(min(256, self.size))
                   if not self.lengths[v])


def _rank(lengths, counts: Counter, emitted: int, namer) -> list[SymbolStat]:
    out = []
    for sym, bits in enumerate(lengths):
        if not bits:
            continue
        n = counts.get(sym, 0)
        out.append(SymbolStat(sym, namer(sym), bits, 2.0 ** -bits, n,
                              n / emitted if emitted else 0.0))
    # Shortest code first, and within one length the one the block leaned on
    # most. Same-length symbols have identical declared shares, so measured use
    # is the only thing that separates them -- and with nothing measured, symbol
    # order keeps it deterministic.
    out.sort(key=lambda s: (s.bits, -s.count, s.sym))
    return out


def _per_length(lengths) -> list[tuple[int, int, float]]:
    per = Counter(L for L in lengths if L)
    return [(L, per[L], per[L] * 2.0 ** -L) for L in sorted(per)]


def _body_counts(symbols) -> tuple[Counter, Counter, bool, int, int]:
    """Litlen and distance symbol counts for one block's symbols."""
    lit, dist = Counter(), Counter()
    saw_eob = False
    for s in symbols:
        if s.kind == "eob":
            saw_eob = True
        # code_sym is what the decoder read out of the alphabet. Deriving it
        # back from the literal value or the match length would be a guess:
        # length 258 is legal two ways, and a literal from a recovered code set
        # has no value at all.
        if s.code_sym is not None:
            lit[s.code_sym] += 1
        if s.dist_sym is not None:
            dist[s.dist_sym] += 1
    return lit, dist, saw_eob, sum(lit.values()), sum(dist.values())


def _clen_counts(ops) -> Counter:
    """Which code-length codes the header used to write the other two tables.

    These are emitted in the header rather than the body, so they are counted
    from the ops the decoder recorded. The op does not keep which symbol it came
    from, but the run lengths settle it: 17 writes 3 to 10 zeros and 18 writes
    11 to 138, and the ranges do not overlap.
    """
    out = Counter()
    for op in ops:
        if op.op == "lens":
            out[op.value] += 1
        elif op.op == "repeat":
            out[16] += 1
        else:
            out[17 if op.count <= 10 else 18] += 1
    return out


def unassigned_namer(identity):
    """Names for a literal alphabet whose meanings were never recovered.

    A code set fixes which code words fall to literals without saying which byte
    each one stands for, so the only name a literal has is the one the recovery
    gave it: its code length and its index among literals of that length.
    """
    def name(sym: int) -> str:
        if sym in identity:
            L, i = identity[sym]
            return f"literal {L}/{i} unassigned"
        if sym == 256:
            return "end of block"
        return litlen_name(sym)
    return name


def alphabets(trace, block) -> list[Alphabet]:
    """Every Huffman table this block carries, with its statistics.

    Empty for a stored block, which has no tables -- and for a fixed block this
    returns the fixed tables, which are worth looking at exactly once: they are
    the same in every fixed block ever written, and their flatness is why a
    fixed block's implied frequencies say nothing about its contents.
    """
    mine = [s for s in trace.symbols if s.block == block.index]
    lit_counts, dist_counts, saw_eob, lit_emitted, dist_emitted = \
        _body_counts(mine)
    partial = not saw_eob
    out = []

    if block.litlen is not None:
        # A recovered code set has literal codes but no literal meanings, so
        # there is no byte space to draw and no byte to name. The codes and their
        # lengths are still entirely real, which is why the rest of this survives
        # the distinction.
        ident = getattr(trace, "identity", None)
        namer = unassigned_namer(ident) if ident else litlen_name
        out.append(Alphabet(
            "litlen",
            "literal/length code" + (" (meanings unrecovered)" if ident else ""),
            tuple(block.litlen.lengths),
            _rank(block.litlen.lengths, lit_counts, lit_emitted, namer),
            _kraft(block.litlen.lengths), block.litlen.complete,
            lit_emitted, partial, _per_length(block.litlen.lengths),
            byte_space=not ident, counted=True,
            note="" if not ident else
                 "this code set was recovered without its assignment: the "
                 "codes and their lengths are real, which byte each one means "
                 "is not known, and nothing here will guess one"))

    if block.dist is not None:
        note = ("" if block.dist.codes else
                "empty, which is legal and says the block has no matches")
        out.append(Alphabet(
            "dist", "distance code", tuple(block.dist.lengths),
            _rank(block.dist.lengths, dist_counts, dist_emitted, dist_name),
            _kraft(block.dist.lengths), block.dist.complete,
            dist_emitted, partial, _per_length(block.dist.lengths),
            byte_space=False, counted=True, note=note))

    if block.codelen is not None:
        counts = _clen_counts(block.clen_ops)
        out.append(Alphabet(
            "codelen", "code-length code", tuple(block.codelen.lengths),
            _rank(block.codelen.lengths, counts, sum(counts.values()),
                  codelen_name),
            _kraft(block.codelen.lengths), block.codelen.complete,
            sum(counts.values()), False, _per_length(block.codelen.lengths),
            byte_space=False, counted=True,
            note="used in the header, not the body: this is the code that "
                 "wrote the other two tables"))
    return out


def _kraft(lengths) -> float:
    return sum(2.0 ** -L for L in lengths if L)
