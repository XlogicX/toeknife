"""The output tape: byte positions plus the equality constraints between them.

The central idea of the rewrite.  An LZ77 back-reference is not "copy some
bytes we might not have", it is an assertion that two output positions hold the
same byte.  Modelling it that way turns fragment recovery into a constraint
problem instead of a string-splicing problem.

Positions are integers.  The fragment's first decoded byte is position 0, so
back-references that reach behind the start of the fragment land on negative
positions: that is the unknown pre-fragment sliding window, addressed directly
rather than approximated with '?' padding.

Bytes whose value follows from a literal are resolved immediately during decode
and stored in a flat bytearray.  Only positions that transitively depend on the
unknown window enter the union-find, so a complete stream (no holes) costs
nothing beyond a normal inflate.
"""
from __future__ import annotations

from dataclasses import dataclass

from .tables import MAX_WINDOW

UNKNOWN = 0
LITERAL = 1          # emitted verbatim by a literal symbol
COPY = 2             # resolved through a back-reference to a known byte
GUESS = 3            # the position a guess was entered at
GUESS_PROP = 4       # resolved by a guess entered elsewhere in the same class
SOLVED = 5           # the position a model fill was applied at
SOLVED_PROP = 6      # resolved by a model fill elsewhere in the same class
RESUMED = 7          # decoded through damage: may be garbage, unproven
BORROWED = 8         # decoded with a table from outside this stream, unproven
GAPPED = 9           # copied across a gap whose width was searched for, unproven
# Read from the bitstream, but resting on something the bitstream does not
# prove. One colour on screen; the inspector names which.
UNPROVEN = (RESUMED, BORROWED, GAPPED)

PROV_NAMES = {UNKNOWN: "unknown", LITERAL: "literal", COPY: "copy",
              GUESS: "guess", GUESS_PROP: "guess-propagated",
              SOLVED: "model fill", SOLVED_PROP: "model fill (propagated)",
              RESUMED: "decoded near damage (unproven)",
              BORROWED: "decoded with a table from elsewhere (unproven)",
              GAPPED: "copied across a searched gap (unproven)"}

# What a model suggested is not what the operator asserted, and neither is what
# the bitstream said.  Keeping them apart is the point.
PROPAGATED = {GUESS: GUESS_PROP, SOLVED: SOLVED_PROP,
              GUESS_PROP: GUESS_PROP, SOLVED_PROP: SOLVED_PROP}


@dataclass(slots=True)
class Hole:
    """One unresolved equality class: a single unknown byte, seen at N places.

    `code` names the literal Huffman code every position in the class was
    emitted by, when that is why they are equal.  It is None for a class held
    together only by back-references.  The distinction is not about how much the
    equality can be trusted -- both are facts read off the bitstream -- but about
    what resolving it will do: a copy chain is usually a handful of positions,
    while a common literal code can be thousands, so one entered byte can light
    up a large fraction of the output at once.
    """
    root: int
    positions: tuple[int, ...]
    code: tuple[int, int] | None = None

    @property
    def size(self) -> int:
        return len(self.positions)

    @property
    def anchor(self) -> int:
        """Lowest position in the class, used as its stable identity."""
        return self.positions[0]


class WindowError(Exception):
    """A back-reference reached further back than the 32K window allows."""


class Tape:
    def __init__(self, window: int = MAX_WINDOW):
        self.window = window
        self.out = bytearray()          # positions 0..length-1
        self.prov = bytearray()         # parallel provenance codes
        self.length = 0
        self.pre_val: dict[int, int] = {}    # resolved negative positions
        self.pre_prov: dict[int, int] = {}
        self.min_pos = 0
        # Union-find over unresolved positions only.
        self._parent: dict[int, int] = {}
        self._members: dict[int, list[int]] = {}
        # Literal codes whose meaning is not known: code -> first position it
        # emitted, and how many times it fired.  The count is kept separately
        # because resolving a class discards its member list.
        self._code_first: dict[tuple[int, int], int] = {}
        self._code_count: dict[tuple[int, int], int] = {}
        # Set while replaying symbols decoded near damage. Where the damage ends
        # cannot be read off the data, so the first symbols after it may be
        # decoded from garbage, and back-references carry that garbage
        # anywhere later (332 wrong bytes, the last 36 KB on, on a 200-byte
        # burst). Every byte written under it is stored as
        # this provenance rather than LITERAL or COPY.
        self.taint: int | None = None

    # ---- construction, called by the decoder -------------------------------

    def emit_literal(self, value: int) -> int:
        pos = self.length
        self.out.append(value)
        self.prov.append(self.taint or LITERAL)
        self.length += 1
        return pos

    def emit_literal_unknown(self, code: tuple[int, int]) -> int:
        """A literal whose CODE is known but whose VALUE is not.

        This is the state a recovered code set leaves behind.  Recovering the
        table's shape tells you how many literals sit at each code length, and
        that is enough to parse -- to know where every symbol starts and ends and
        which ones are matches -- but not which byte any literal stands for.  The
        meanings are a separate unknown, and they are recovered separately.

        Two positions emitted by the SAME literal code hold the same byte.  That
        is an equality constraint of exactly the kind this class already models,
        so it goes into the same union-find as a back-reference and needs no new
        vocabulary: identify one occurrence, by guess or by a copy from known
        text, and every other occurrence of that code follows, propagating with
        the provenance the identification came with.
        """
        p = self.length
        self.out.append(0)
        self.prov.append(UNKNOWN)
        self.length += 1
        self._code_count[code] = self._code_count.get(code, 0) + 1
        first = self._code_first.get(code)
        if first is None:
            self._code_first[code] = p
            self._parent[p] = p
            self._members[p] = [p]
            return p
        known = self.value_at(first)
        if known is not None:
            # The code's meaning is already settled, so this occurrence is not a
            # hole at all.  It inherits how that meaning was arrived at, demoted
            # one step: a guess stays a guess, a bitstream fact stays a fact.
            src = self.prov_at(first)
            self.out[p] = known
            self.prov[p] = PROPAGATED.get(src, src)
        else:
            self._union(p, first)
        return p

    def emit_unknown(self, count: int) -> int:
        """Append output that was produced by data we do not have.

        This is the gap between two fragments of one stream: we know bytes came
        out there, and how many, but not what they were.  Each is its own
        equality class from the start, so a later back-reference reaching into
        the gap joins that class rather than creating a second unknown for the
        same byte.
        """
        start = self.length
        for _ in range(count):
            p = self.length
            self.out.append(0)
            self.prov.append(UNKNOWN)
            self.length += 1
            self._parent[p] = p
            self._members[p] = [p]
        return start

    def emit_copy(self, distance: int, length: int) -> int:
        """Apply a back-reference.  Returns the starting output position.

        Overlapping copies (distance < length) need no special case: position
        p is unioned with p-distance, which was itself just written, so the
        repeat falls out of the constraint chain.
        """
        start = self.length
        if distance > self.window:
            raise WindowError(f"distance {distance} exceeds {self.window}-byte window")
        for _ in range(length):
            p = self.length
            s = p - distance
            if s < self.min_pos:
                self.min_pos = s
            self.out.append(0)
            self.prov.append(UNKNOWN)
            self.length += 1
            sv = self.value_at(s)
            if sv is not None:
                self.out[p] = sv
                # A copy of an unproven byte is no better than its source.
                src = self.prov[s] if s >= 0 else self.pre_prov.get(s)
                self.prov[p] = self.taint or (src if src in UNPROVEN
                                              else COPY)
            else:
                self._union(p, s)
        return start

    # ---- reading ----------------------------------------------------------

    def value_at(self, pos: int) -> int | None:
        if pos < 0:
            return self.pre_val.get(pos)
        if pos >= self.length:
            return None
        return self.out[pos] if self.prov[pos] != UNKNOWN else None

    def prov_at(self, pos: int) -> int:
        if pos < 0:
            return self.pre_prov.get(pos, UNKNOWN)
        if pos >= self.length:
            return UNKNOWN
        return self.prov[pos]

    def __len__(self) -> int:
        return self.length

    # ---- union-find over unresolved positions ------------------------------

    def _find(self, p: int) -> int:
        parent = self._parent
        r = p
        while parent.get(r, r) != r:
            r = parent[r]
        while parent.get(p, p) != p:      # path compression
            parent[p], p = r, parent[p]
        return r

    def _union(self, a: int, b: int) -> None:
        ra, rb = self._find(a), self._find(b)
        members = self._members
        ma = members.setdefault(ra, [ra])
        if ra == rb:
            if a not in ma:
                ma.append(a)
            return
        mb = members.setdefault(rb, [rb])
        if len(ma) < len(mb):                  # union by size
            ra, rb, ma, mb = rb, ra, mb, ma
        self._parent[rb] = ra
        ma.extend(mb)
        del members[rb]

    # ---- holes and resolution ----------------------------------------------

    def _code_by_root(self) -> tuple[dict[int, tuple[int, int]], list[tuple]]:
        """Which unresolved class belongs to which literal code.

        Also returns the classes that two DIFFERENT literal codes both landed
        in.  In one dynamic table two literal codes are two different byte
        values, so a back-reference tying them together is a contradiction, and
        the parse that produced it cannot be right.
        """
        roots: dict[int, tuple[int, int]] = {}
        clashes: list[tuple] = []
        for code, first in sorted(self._code_first.items()):
            if self.value_at(first) is not None:
                continue
            r = self._find(first)
            if r in roots:
                clashes.append((roots[r], code))
            else:
                roots[r] = code
        return roots, clashes

    def holes(self) -> list[Hole]:
        """Every unresolved equality class, ordered by first appearance."""
        by_root, _clashes = self._code_by_root()
        out = []
        for root, members in self._members.items():
            positions = tuple(sorted(set(members)))
            out.append(Hole(root, positions, by_root.get(root)))
        out.sort(key=lambda h: h.anchor)
        return out

    def assignment(self) -> dict[tuple[int, int], dict]:
        """Read off what each literal code has turned out to mean.

        The point of the exercise.  A recovered code set gives the parse and
        withholds the meanings; every byte that becomes known -- copied in from
        text the operator already had, entered by hand, or filled by a model --
        settles one code for the whole block, and this is the ledger of which
        ones have been settled and on what basis.

        Keyed by (code length, index among literals of that length), which is
        the only name a literal has when its value is unknown.  `provenance` is
        carried through so a code pinned by an operator's guess never reads the
        same as one pinned by the bitstream.
        """
        out = {}
        for code, first in sorted(self._code_first.items()):
            v = self.value_at(first)
            out[code] = dict(value=v,
                             provenance=self.prov_at(first) if v is not None
                             else UNKNOWN,
                             occurrences=self._code_count[code],
                             first=first)
        return out

    def assignment_conflicts(self) -> list[tuple]:
        """Pairs of literal codes the constraints force to the same byte.

        Two codes in one table are two byte values, so this is never legal.  It
        is the only self-check this representation gets for free, and a
        non-empty result means the code set being decoded with is wrong.
        """
        _roots, clashes = self._code_by_root()
        seen: dict[int, tuple[int, int]] = {}
        for code, first in sorted(self._code_first.items()):
            v = self.value_at(first)
            if v is None:
                continue
            if v in seen:
                clashes.append((seen[v], code))
            else:
                seen[v] = code
        return clashes

    def hole_at(self, pos: int) -> Hole | None:
        if self.value_at(pos) is not None:
            return None
        if pos not in self._parent and pos not in self._members:
            return None
        root = self._find(pos)
        members = self._members.get(root)
        if members is None:
            return None
        by_root, _clashes = self._code_by_root()
        return Hole(root, tuple(sorted(set(members))), by_root.get(root))

    def resolve(self, pos: int, value: int, provenance: int = GUESS) -> list[int]:
        """Assign a byte to pos and every position equal to it.

        Returns the positions that changed.  Refuses to overwrite a byte that
        is already known: those are facts from the bitstream, not guesses.
        """
        if pos >= self.length or (pos < 0 and pos < self.min_pos):
            raise ValueError(f"position {pos} is outside the output "
                             f"({self.min_pos}..{self.length - 1})")
        if self.value_at(pos) is not None:
            raise ValueError(f"position {pos} is already known, not a hole")
        root = self._find(pos)
        members = self._members.pop(root, [root])
        touched = sorted(set(members))
        prop = PROPAGATED.get(provenance, GUESS_PROP)
        for m in touched:
            self._parent.pop(m, None)
            self._store(m, value, provenance if m == pos else prop)
        return touched

    def unresolve(self, positions: list[int]) -> None:
        """Undo a resolve(), putting the class back into the unknown pool."""
        if not positions:
            return
        root = positions[0]
        for p in positions:
            self._store_unknown(p)
            self._parent[p] = root
        self._parent[root] = root
        self._members[root] = list(positions)

    def _store(self, pos: int, value: int, provenance: int) -> None:
        if pos < 0:
            self.pre_val[pos] = value
            self.pre_prov[pos] = provenance
        else:
            self.out[pos] = value
            self.prov[pos] = provenance

    def _store_unknown(self, pos: int) -> None:
        if pos < 0:
            self.pre_val.pop(pos, None)
            self.pre_prov.pop(pos, None)
        else:
            self.out[pos] = 0
            self.prov[pos] = UNKNOWN

    # ---- summary -----------------------------------------------------------

    def stats(self) -> dict:
        holes = self.holes()
        codes = self.assignment()
        return {
            "out_len": self.length,
            "min_pos": self.min_pos,
            "holes": len(holes),
            "hole_positions": sum(h.size for h in holes),
            "known": sum(1 for i in range(self.length) if self.prov[i] != UNKNOWN),
            "codes": len(codes),
            "codes_known": sum(1 for c in codes.values() if c["value"] is not None),
        }

    def recovered(self, fill: int = 0x3F) -> tuple[bytes, bytes]:
        """(bytes, mask) for positions 0..length-1.  mask[i] is 1 if known."""
        buf = bytearray(self.out)
        mask = bytearray(self.length)
        for i in range(self.length):
            if self.prov[i] == UNKNOWN:
                buf[i] = fill
            else:
                mask[i] = 1
        return bytes(buf), bytes(mask)

    def window_bytes(self, fill: int = 0x3F) -> tuple[bytes, bytes, int]:
        """The reconstructed pre-fragment window, as (bytes, mask, start_pos)."""
        if self.min_pos >= 0:
            return b"", b"", 0
        n = -self.min_pos
        buf = bytearray(n)
        mask = bytearray(n)
        for i in range(n):
            pos = self.min_pos + i
            v = self.pre_val.get(pos)
            if v is None:
                buf[i] = fill
            else:
                buf[i] = v
                mask[i] = 1
        return bytes(buf), bytes(mask), self.min_pos
