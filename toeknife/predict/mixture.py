"""Combining predictors, and combining them with constraints.

No single predictor is right for a carved stream. A corpus model knows the
language but not this file. A model trained on the stream's own recovered bytes
knows this file but has seen very little of it. A dynamic block's Huffman table
knows which bytes occur and roughly how often but nothing about order. They fail
in different places, which is the case for mixing rather than choosing.

Mixed in the log domain with weights, which is a product of experts rather than
a vote: a predictor that is confident and wrong drags the result, so weights
matter and the defaults are deliberately timid about anything trained on very
little data.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .base import UNIFORM


@dataclass
class Weighted:
    predictor: object
    weight: float = 1.0
    label: str = ""


class Mixture:
    """A weighted product of predictors, with optional hard constraints."""

    def __init__(self, parts, constraints=(), floor: float = -11.0):
        self.parts = [p if isinstance(p, Weighted) else Weighted(p) for p in parts]
        self.constraints = list(constraints)
        self.floor = floor
        self.order = max((getattr(p.predictor, "order", 0) for p in self.parts),
                         default=4)
        # The solver's batched path exists for predictors too slow to call once
        # per candidate, and it pays for that with weaker evidence: it cannot
        # charge a candidate for the known bytes that follow it. A mixture of
        # cheap n-grams should not opt into that merely because it happens to
        # expose rank(), so say whether any part actually needs it.
        self.prefers_batched = any(hasattr(p.predictor, "rank")
                                   for p in self.parts)

    def logprob(self, byte: int, left: bytes) -> float:
        total = weight = 0.0
        for part in self.parts:
            if not part.weight:
                continue
            total += part.weight * part.predictor.logprob(byte, left)
            weight += part.weight
        if not weight:
            return UNIFORM
        return total / weight

    def logprob_both(self, byte: int, left: bytes, right: bytes = b"") -> float:
        """The same blend, letting any part that reads backwards use `right`.

        The solver prefers this whenever a model offers it, because the holes
        that matter sit at the edge of the pre-fragment window where there is no
        left context at all.  Mixing must not be what throws that away: a part
        with a reverse half is asked for both directions, and one without falls
        back to its forward answer.
        """
        total = weight = 0.0
        for part in self.parts:
            if not part.weight:
                continue
            both = getattr(part.predictor, "logprob_both", None)
            lp = (both(byte, left, right) if both is not None
                  else part.predictor.logprob(byte, left))
            total += part.weight * lp
            weight += part.weight
        if not weight:
            return UNIFORM
        return total / weight

    def candidates(self, left: bytes, right: bytes = b"") -> set[int] | None:
        from .constraints import combine
        return combine(self.constraints, left, right)

    def rank(self, left: bytes, candidates=None,
             right: bytes = b"") -> list[tuple[int, float]]:
        """Score candidate bytes in place, best first.

        The parameter order is the batched protocol the solver calls and that
        HFPredictor.rank and ByT5Predictor.rank implement: (left, candidates,
        right).  A mixture that disagreed with them was handed the candidate
        pool as its right-hand context and the right-hand context as its
        candidate set, so it ranked the bytes that follow the hole as if they
        were the answers -- and did it at confidence 1.0.

        A part with its own rank() is asked once for the whole set, which is the
        point of the protocol for a model too slow to call per candidate.  A
        part without one is scored per candidate through logprob_both, so the
        reverse direction survives being mixed.
        """
        allowed = self.candidates(left, right)
        pool = list(range(256) if candidates is None else candidates)
        if allowed is not None:
            pool = [v for v in pool if v in allowed]
        if not pool:
            return []
        totals: dict[int, float] = {}
        weights: dict[int, float] = {}
        for part in self.parts:
            if not part.weight:
                continue
            own = getattr(part.predictor, "rank", None)
            if own is not None:
                got = dict(own(left, pool, right))
            else:
                both = getattr(part.predictor, "logprob_both", None)
                got = {v: (both(v, left, right) if both is not None
                           else part.predictor.logprob(v, left))
                       for v in pool}
            for v, score in got.items():
                totals[v] = totals.get(v, 0.0) + part.weight * score
                weights[v] = weights.get(v, 0.0) + part.weight
        scored = [(v, totals[v] / weights[v]) for v in totals if weights[v]]
        scored.sort(key=lambda t: -t[1])
        return scored

    def describe(self) -> str:
        bits = [f"{p.label or type(p.predictor).__name__}x{p.weight:g}"
                for p in self.parts if p.weight]
        if self.constraints:
            bits.append("constrained by " +
                        ", ".join(c.name for c in self.constraints))
        return " + ".join(bits) or "nothing"


class TablePrior:
    """The byte frequencies implied by a dynamic block's Huffman code lengths.

    A canonical code gives length L to a symbol of probability roughly 2**-L, so
    a block's own table is a frequency model of the data that produced it. It is
    derived from the artefact rather than assumed about it, which makes it worth
    having even though it knows nothing about order.
    """

    order = 0

    def __init__(self, prior: dict[int, float]):
        self._log = {b: math.log2(p) for b, p in prior.items() if p > 0}

    @classmethod
    def from_trace(cls, trace):
        """None when no block carries a table that says anything.

        A fixed-Huffman block gives every byte a code of almost the same length,
        so its implied frequencies are near-flat and carry no information.
        Mixing that in only dilutes whatever else is there.
        """
        merged: dict[int, float] = {}
        informative = False
        for block in getattr(trace, "blocks", ()):
            table = getattr(block, "litlen", None)
            if table is None or not getattr(table, "codes", None):
                continue
            if len({s for s in table.codes if s < 256}) < 256:
                informative = True
            for sym, weight in table.byte_prior().items():
                merged[sym] = max(merged.get(sym, 0.0), weight)
        if not merged or not informative:
            return None
        total = sum(merged.values())
        return cls({b: p / total for b, p in merged.items()})

    def logprob(self, byte: int, left: bytes) -> float:
        return self._log.get(byte, -14.0)
