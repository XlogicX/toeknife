"""Alternating a model with a checksum until neither can make progress.

Layering two *models* gains nothing: the solver already loops internally, and a
second opinion re-reads the same evidence. Measured, a descending threshold
ladder is exactly equal to a single pass at its lowest rung.

Layering a model with a *checksum* is different, because the checksum is not an
opinion. It is 32 bits of arithmetic that were not in the loop before. And the
two feed each other: every class the model fills is one fewer unknown in the
linear system, and four unknowns or fewer are solvable outright; every class the
checksum pins is new context for the model to read.

So the loop runs until a full round changes nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Round:
    index: int
    by_model: int = 0
    by_checksum: int = 0
    remaining: int = 0
    note: str = ""

    @property
    def progress(self) -> int:
        return self.by_model + self.by_checksum

    def summary(self) -> str:
        return (f"round {self.index}: model filled {self.by_model}, checksum "
                f"pinned {self.by_checksum}, {self.remaining} classes left"
                + (f" ({self.note})" if self.note else ""))


@dataclass
class LoopResult:
    rounds: list = field(default_factory=list)
    started_with: int = 0
    verified: bool | None = None       # None means it was never actually checked
    contradicted: bool = False

    @property
    def filled(self) -> int:
        return sum(r.progress for r in self.rounds)

    @property
    def remaining(self) -> int:
        return self.rounds[-1].remaining if self.rounds else self.started_with

    def summary(self) -> str:
        out = (f"{self.filled} of {self.started_with} classes filled over "
               f"{len(self.rounds)} round(s), {self.remaining} left")
        if self.contradicted:
            # The most useful thing the checksum does when it cannot solve: it
            # can still prove that something already filled in is wrong.
            return out + ("; THE CHECKSUM CONTRADICTS WHAT IS ALREADY FILLED "
                          "IN, so at least one of these bytes is wrong")
        if self.verified is True:
            return out + "; the checksum confirms every recovered byte"
        if self.verified is False:
            return out + "; the checksum DISAGREES with the result"
        return out + ("; not checked against a checksum (holes remain, or the "
                      "container recorded none)")


def solve_loop(session, model, min_confidence: float = 0.995,
               max_rounds: int = 12, shortlist: int = 16,
               max_combinations: int = 1 << 21, on_round=None,
               long_runs: bool = False, calibration=None,
               dictionary: bool = False, filler=None,
               mode: str = "ngram") -> LoopResult:
    """Alternate the model and the checksum until a round changes nothing.

    `filler` and `mode` are the byte model and fill mode ^f uses. Without
    them the loop filled with the n-gram alone while ^f, on the same screen,
    filled with the byte model.
    """
    from .crcsolve import solve_by_shortlist, solve_from_crc
    from .solver import HoleSolver
    from ..core.tape import SOLVED

    result = LoopResult(started_with=len(session.tape.holes()))
    for index in range(1, max_rounds + 1):
        rnd = Round(index)
        before = len(session.tape.holes())
        if not before:
            break

        session.begin_group()
        model_result = HoleSolver(session, model,
                                  min_confidence=min_confidence,
                                  calibration=calibration, filler=filler,
                                  mode=mode if filler is not None
                                  else "ngram").solve(
                                      long_runs=long_runs,
                                      dictionary=dictionary)
        session.end_group()
        rnd.by_model = len(model_result.filled)

        # The checksum comes second on purpose: the model has just removed
        # unknowns from the linear system, and below five it is exact.
        holes = len(session.tape.holes())
        if holes:
            exact = solve_from_crc(session)
            assignment = None
            if exact.ok and not exact.free_bits:
                assignment = exact.assignment
                rnd.note = "checksum solved it outright"
            elif exact.ok or "no CRC32" in exact.reason:
                got = solve_by_shortlist(session, model, per_class=shortlist,
                                         max_combinations=max_combinations)
                if got.ok:
                    assignment = got.assignment
                    rnd.note = got.summary()
                else:
                    rnd.note = got.reason
            else:
                rnd.note = exact.reason
                if "contradicts" in exact.reason:
                    result.contradicted = True
            if assignment:
                session.begin_group()
                for anchor, value in sorted(assignment.items()):
                    if session.tape.value_at(anchor) is None:
                        session.set_guess(anchor, value, SOLVED)
                        rnd.by_checksum += 1
                session.end_group()

        rnd.remaining = len(session.tape.holes())
        result.rounds.append(rnd)
        if on_round:
            on_round(rnd)
        if rnd.remaining == before:
            break                       # a whole round changed nothing

    verdict = session.verify()
    if verdict is not None and verdict.complete and (
            verdict.crc_got is not None or verdict.adler_got is not None):
        # Only a checksum that was actually computed and compared counts as
        # confirmation.  A size match on output that still contains unknown
        # bytes confirms nothing, and reporting it as though it did is worse
        # than reporting nothing.
        result.verified = verdict.ok
    return result
