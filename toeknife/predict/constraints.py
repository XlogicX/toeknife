"""Ruling bytes out, as opposed to ranking them.

A constraint is only entitled to say no when the answer is genuinely impossible.
Two are, and both are free:

UTF-8 is a prefix code with a rigid shape. A byte in the middle of a multi-byte
sequence must be a continuation byte, and a byte following a complete character
must not be. That eliminates most of the alphabet at most positions in text, and
it does so without any model or corpus at all. ziprec used the detected encoding
the same way, noting that UTF-16 text has a zero in every other byte.

A dynamic block's Huffman table names every byte that can appear as a literal
anywhere in that block. A byte with no code in the table was never emitted as a
literal there. This is derived from the artefact rather than assumed about it.

Both refuse to answer rather than guess when they cannot tell, because an empty
candidate set stops the solver dead and a wrong one sends it somewhere false.
"""
from __future__ import annotations

ALL = frozenset(range(256))
ASCII_TEXT = frozenset(list(range(0x20, 0x7F)) + [0x09, 0x0A, 0x0D])
CONTINUATION = frozenset(range(0x80, 0xC0))
# Bytes that can never appear anywhere in valid UTF-8.
UTF8_NEVER = frozenset({0xC0, 0xC1}) | frozenset(range(0xF5, 0x100))
UTF8_LEAD = {b: (2 if 0xC2 <= b <= 0xDF else 3 if 0xE0 <= b <= 0xEF else
                 4 if 0xF0 <= b <= 0xF4 else 0) for b in range(256)}


class Utf8:
    """Bytes that keep the surrounding text valid UTF-8.

    Works backwards from the preceding bytes: if they end part way through a
    multi-byte character, only a continuation byte can follow, and if they end
    on a boundary a continuation byte is impossible.
    """

    name = "utf-8"

    def __init__(self, strict: bool = True):
        self.strict = strict

    def _pending(self, left: bytes) -> int | None:
        """How many continuation bytes the tail of `left` is still waiting for.

        None when the preceding bytes are not valid UTF-8 at all, in which case
        this constraint has no business ruling anything out.
        """
        for back in range(1, min(len(left), 4) + 1):
            b = left[-back]
            if b < 0x80:
                return 0 if back == 1 else None
            if b in CONTINUATION:
                continue
            need = UTF8_LEAD.get(b, 0)
            if not need:
                return None                  # 0xC0, 0xC1, 0xF5.. : not UTF-8
            return need - back if need >= back else None
        return None if len(left) >= 4 else 0

    def allowed(self, left: bytes, right: bytes = b"") -> set[int] | None:
        pending = self._pending(left)
        if pending is None:
            return None                      # cannot tell, so do not pretend to
        if pending > 0:
            return set(CONTINUATION)
        ok = set(ALL - CONTINUATION - UTF8_NEVER)
        if not self.strict:
            return ok
        # A lead byte has to be followed by the continuations it promises, and
        # `right` may already say what follows.
        if right:
            nxt = right[0]
            if nxt in CONTINUATION:
                ok = {b for b in ok if UTF8_LEAD.get(b, 0) >= 2}
            else:
                ok = {b for b in ok if b < 0x80}
        return ok


class Alphabet:
    """Bytes a block's Huffman table actually has a code for.

    In a dynamic block, a literal with no code was never emitted. Fixed blocks
    give every byte a code and so say nothing, and this reports no opinion
    rather than a set of all 256.
    """

    name = "table alphabet"

    def __init__(self, allowed_bytes):
        self._set = set(allowed_bytes)

    @classmethod
    def from_trace(cls, trace):
        found: set[int] = set()
        informative = False
        for block in getattr(trace, "blocks", ()):
            table = getattr(block, "litlen", None)
            if table is None or not getattr(table, "codes", None):
                continue
            here = {s for s in table.codes if s < 256}
            if len(here) < 256:
                informative = True
            found |= here
        return cls(found) if informative and found else None

    def allowed(self, left: bytes, right: bytes = b"") -> set[int] | None:
        return set(self._set) if self._set else None


class Fixed:
    """A flat set, for when the operator knows something the tool does not."""

    def __init__(self, values, name: str = "fixed"):
        self._set = set(values)
        self.name = name

    def allowed(self, left: bytes, right: bytes = b"") -> set[int] | None:
        return set(self._set)


def combine(constraints, left: bytes, right: bytes = b"") -> set[int] | None:
    """Intersect whatever has an opinion.

    An intersection that comes out empty means the constraints contradict each
    other, which is a statement about them and not about the data, so it is
    discarded rather than acted on.
    """
    out: set[int] | None = None
    for c in constraints:
        got = c.allowed(left, right)
        if got is None:
            continue
        out = set(got) if out is None else (out & got)
    if out is not None and not out:
        return None
    return out
