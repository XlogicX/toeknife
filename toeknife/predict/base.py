"""What a predictor is, and what a constraint is.

Two different things get consulted when deciding an unknown byte, and keeping
them apart matters.

A *predictor* ranks. It answers logprob(byte, left_context) and is always
willing to answer, however badly. Everything soft lives here: byte n-grams, a
Huffman table's implied frequencies, a language model.

A *constraint* eliminates. It answers whether a byte is possible at all, and it
is only allowed to say no when the answer is genuinely impossible rather than
unlikely. A byte that would make the surrounding text invalid UTF-8 is
impossible. A byte the block's Huffman table has no code for cannot have been
emitted as a literal in that block. Neither is a matter of degree.

Ranking a candidate that cannot occur wastes the ranking, and worse, lets a
confident-looking wrong answer through. So constraints run first and predictors
choose among what survives.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

UNIFORM = -8.0                      # log2(1/256)


@runtime_checkable
class Predictor(Protocol):
    """Anything that can rank bytes in context."""

    order: int

    def logprob(self, byte: int, left: bytes) -> float:
        """log2 P(byte | left).  Never raises; returns UNIFORM when it has
        nothing to say."""
        ...


@runtime_checkable
class Constraint(Protocol):
    """Anything that can rule bytes out."""

    def allowed(self, left: bytes, right: bytes = b"") -> set[int] | None:
        """Which byte values are possible here.

        None means "no opinion", which is different from the empty set and much
        more common.  A constraint that cannot tell must say so rather than
        guess, because an empty set stops the solver dead.
        """
        ...

    @property
    def name(self) -> str:
        ...


class Uniform:
    """A predictor that knows nothing.  Useful as a floor and in tests."""

    order = 0

    def logprob(self, byte: int, left: bytes) -> float:
        return UNIFORM

    def distribution(self, left: bytes) -> dict[int, float]:
        return {b: 1 / 256 for b in range(256)}
