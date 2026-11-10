"""The hole solver: multi-occurrence voting, propagation, and not lying."""
import binascii

import pytest

from toeknife.core.session import Session
from toeknife.core.tape import GUESS, SOLVED, SOLVED_PROP, Tape
from toeknife.predict.context import ContextModel
from toeknife.predict.solver import HoleSolver, train_from_session

# The README fragment: the tail of a fixed-Huffman compression of FULL, carved
# mid-symbol.  Real data, known ground truth, and the shape the solver is for:
# most bytes known, a few classes open.
FULL = (b"Some numbers are one, two, three, and four. four is the only of "
        b"those that has four letters in it")
FRAGMENT = binascii.unhexlify(
    "2f2dd203930a99c50a2519a90af97939950af9690a2519f9c5a90a251989250a"
    "1989c5102539a9252520e332f314324b00")

# English using the same vocabulary, but never the target sentence.
CORPUS = (b"one two three four five six seven eight nine ten. "
          b"the word four has four letters in it, which is unusual. "
          b"some numbers are one, two, three. others are larger. "
          b"four letters make up many short words. letters are symbols. "
          b"of those that remain, only one is the answer we want. "
          b"it is the only one of those that has that property. ") * 60


def model(order=4, corpus=CORPUS):
    return ContextModel(order=order).train(corpus)


def session():
    return Session(FRAGMENT, mode="fragment", profile="text")


# ---- scoring ---------------------------------------------------------------

def test_a_class_is_scored_across_all_its_occurrences():
    """The point of the union-find: k occurrences are k independent votes."""
    s = session()
    solver = HoleSolver(s, model())
    hole = next(h for h in s.tape.holes() if h.size >= 2)
    _, terms_one = solver._score_value(hole.positions[:1], ord("e"))
    _, terms_many = solver._score_value(hole.positions, ord("e"))
    assert terms_many > terms_one, "more occurrences must contribute more evidence"


def test_context_stops_at_a_hole_rather_than_splicing_over_it():
    """Splicing invents adjacencies the data never contained: 'for' before a
    hole before 'forensic' would read as 'forforensic'."""
    t = Tape()
    for b in b"abc":
        t.emit_literal(b)
    t.emit_copy(distance=6, length=3)      # positions 3,4,5 unknown
    for b in b"xy":
        t.emit_literal(b)
    fake = Session.__new__(Session)
    fake.trace = type("T", (), {"tape": t, "blocks": ()})()
    solver = HoleSolver.__new__(HoleSolver)
    solver.session = fake
    solver.order = 4
    assert solver._left_context(6) == b"", "context must not reach across the gap"
    assert solver._left_context(3) == b"abc"


def test_scoring_uses_the_bytes_after_the_hole_too():
    """A causal model reading only leftward ignores the right-hand context, and
    after a hole the right-hand context is usually known and usually decisive."""
    s = session()
    with_look = HoleSolver(s, model(), lookahead=3)
    without = HoleSolver(s, model(), lookahead=0)
    # A hole whose next byte is known.  Holes in the middle of a run of holes
    # have no right-hand context to read, which is exactly why the ones at the
    # edges are the ones that get solved first.
    hole = next(h for h in s.tape.holes()
                if any(s.tape.value_at(p + 1) is not None for p in h.positions))
    _, terms_a = with_look._score_value(hole.positions, ord("f"))
    _, terms_b = without._score_value(hole.positions, ord("f"))
    assert terms_a > terms_b


# ---- the headline result ---------------------------------------------------

def test_solver_recovers_the_readme_fragment_exactly():
    """The workflow the original tool needed a hand-written guesses.txt for.

    Seven unknown byte classes covering seventeen output positions; the answers
    are 'ers ' and ' fo'.  Nothing about the target sentence is in the corpus.
    """
    s = session()
    assert s.tape.stats()["holes"] == 7
    solver = HoleSolver(s, model())
    s.begin_group()
    result = solver.solve(min_confidence=0.9)
    s.end_group()

    assert not s.tape.holes(), "every class should be filled"
    buf, mask = s.recovered()
    assert all(mask)
    out = bytes(buf)
    start = FULL.index(b"ur. four is")
    assert out == FULL[start:start + len(out)], out

    wbuf, _, wstart = s.window()
    win = bytes(wbuf).decode("latin1")
    assert wstart == -31
    assert win.startswith("ers "), win
    assert win.endswith(" fo"), win
    assert result.positions_filled == 17


# ---- behaviour -------------------------------------------------------------

def test_filling_one_class_fills_every_position_in_it():
    s = session()
    solver = HoleSolver(s, model())
    hole = max(s.tape.holes(), key=lambda h: h.size)
    assert hole.size > 1
    p = solver.propose_one(hole)
    s.set_guess(p.anchor, p.value, SOLVED)
    for pos in hole.positions:
        assert s.tape.value_at(pos) == p.value


def test_model_fills_are_not_disguised_as_operator_guesses():
    s = session()
    hole = max(s.tape.holes(), key=lambda h: h.size)
    s.set_guess(hole.anchor, ord("z"), SOLVED)
    assert s.tape.prov_at(hole.anchor) == SOLVED
    assert all(s.tape.prov_at(p) == SOLVED_PROP
               for p in hole.positions if p != hole.anchor)

    s2 = session()
    h2 = max(s2.tape.holes(), key=lambda h: h.size)
    s2.set_guess(h2.anchor, ord("z"))
    assert s2.tape.prov_at(h2.anchor) == GUESS


def test_a_solver_run_undoes_as_one_action():
    s = session()
    before = s.tape.stats()["holes"]
    s.begin_group()
    result = HoleSolver(s, model()).solve(min_confidence=0.9)
    s.end_group()
    assert len(result.filled) > 1
    assert s.tape.stats()["holes"] < before
    assert s.undo()
    assert s.tape.stats()["holes"] == before, "one undo must revert the whole run"
    assert s.redo()
    assert s.tape.stats()["holes"] < before


def test_bytes_already_known_are_never_touched():
    """Everything the decoder established stays established, and stays correct."""
    base = session()
    b0, m0 = base.recovered()
    s = session()
    HoleSolver(s, model()).solve(min_confidence=0.9)
    buf, mask = s.recovered()
    for i, known in enumerate(m0):
        if known:
            assert mask[i]
            assert buf[i] == b0[i], "the solver must not overwrite decoded bytes"


def test_solver_declines_when_the_model_does_not_know_the_data():
    """Declining is the correct answer, not a failure.  A byte that silently
    became known because a model liked it is worse than a hole."""
    s = session()
    before = s.tape.stats()["holes"]
    ignorant = ContextModel(order=4).train(bytes(range(256)) * 40)
    result = HoleSolver(s, ignorant).solve(min_confidence=0.999)
    assert len(result.filled) < before
    assert "confidence" in result.stopped


def test_raising_the_threshold_fills_less():
    loose, tight = session(), session()
    a = HoleSolver(loose, model()).solve(min_confidence=0.5)
    b = HoleSolver(tight, model()).solve(min_confidence=0.99999)
    assert a.positions_filled >= b.positions_filled


def test_proposals_are_ordered_by_confidence():
    s = session()
    proposals = HoleSolver(s, model()).propose()
    assert proposals
    confs = [p.confidence for p in proposals]
    assert confs == sorted(confs, reverse=True)
    assert all(0.0 <= c <= 1.0 for c in confs)


def test_self_training_learns_from_the_streams_own_bytes():
    """The readable part of a stream is often the best sample of the part that
    is missing, and needs no corpus."""
    s = session()
    m = train_from_session(s, order=3)
    assert m.trained_bytes > 0
    real = m.score(b"is the only of those that has")
    noise = m.score(bytes(range(256)))
    assert real > noise + 1.0, f"real={real} noise={noise}"


def test_self_training_refuses_to_learn_from_its_own_guesses():
    """A fill is a hypothesis.  Training on it teaches the model its own guess as
    ground truth, and a wrong one then votes for itself everywhere it fits --
    measured at 5 points of precision on a phrack carve."""
    s = session()
    before = train_from_session(s, order=3).trained_bytes

    hole = next(h for h in s.tape.holes() if h.size >= 2)
    s.set_guess(hole.anchor, ord("Q"), SOLVED)      # a byte the stream never had

    after = train_from_session(s, order=3)
    assert after.trained_bytes == before, "a SOLVED fill became training data"
    # And the filter is not simply off: the decoded bytes still got through.
    assert before > 0


def test_self_training_does_learn_from_what_the_operator_typed():
    """A typed byte is evidence from outside the model, not the model's own
    output looping back, and it is the reason corrections are worth typing."""
    s = session()
    before = train_from_session(s, order=3).trained_bytes
    hole = next(h for h in s.tape.holes() if h.size >= 2)
    s.set_guess(hole.anchor, ord("e"), GUESS)
    assert train_from_session(s, order=3).trained_bytes > before


def test_a_solve_fills_the_surest_first_and_stops_at_the_floor():
    """The threshold is a floor, not a cutoff.  Filling everything above it at
    once scores every class against the tape as it was before any of them
    landed; descending means each fill is context for the next."""
    s = session()
    result = HoleSolver(s, model(), batch=1).solve(min_confidence=0.5)
    assert result.filled
    # batch=1 means one class per round, so rounds track fills.
    assert result.rounds >= len(result.filled)
    # Nothing below the floor was ever applied, whether it ran out of classes
    # above the floor or ran out of holes altogether.
    assert all(p.confidence >= 0.5 for p in result.filled)
    assert result.stopped in ("nothing above 50% confidence", "no holes left")


def test_the_round_cap_scales_with_the_work_it_has_to_do():
    """A fixed 60-round cap plus a batch size stops with classes still standing
    above the floor, which reads as a finished recovery and is not one."""
    s = session()
    result = HoleSolver(s, model(), batch=1).solve(min_confidence=0.5)
    assert "limit" not in result.stopped, result.stopped


def test_solver_is_a_no_op_on_a_complete_stream():
    full = binascii.unhexlify(
        "8dcbd10980400c04d156b600b10b0b0961258133caada0767fda811fef676096"
        "dbb6a311629d564eac7bffb0940e2b6b8f5213ae480fa4de04fe7ae601")
    s = Session(full)
    assert not s.tape.holes()
    result = HoleSolver(s, model()).solve()
    assert result.filled == []
    assert result.stopped == "no holes left"


def test_the_threshold_is_reported_at_the_value_it_actually_is():
    """0.995 through .0% is "100%", which told the operator the model had to be
    certain when the bar was 99.5%.  The rungs either side have to stay apart."""
    from toeknife.predict.solver import pct
    assert pct(0.995) == "99.5%"
    assert pct(0.999) == "99.9%"
    assert pct(0.99) == "99%"
    assert pct(0.9) == "90%"
    assert pct(0.0) == "0%"
    assert len({pct(t) for t in (0.9, 0.95, 0.99, 0.995, 0.999)}) == 5

    # and the refusal itself, which is the message that reached the operator
    stopped = HoleSolver(session(), model(), min_confidence=0.9999).solve().stopped
    assert stopped == "nothing above 99.99% confidence"


def test_a_confidence_is_never_rounded_up_into_certainty():
    """The panel showed a 99.9991% proposal as "100.0%", which is the same claim
    of certainty the threshold message was making, one column over."""
    from toeknife.predict.solver import confidence_text
    assert confidence_text(0.999991) == ">99.9%"
    assert confidence_text(0.99995) == ">99.9%"
    assert confidence_text(0.999) == "99.9%"
    assert confidence_text(0.5) == "50.0%"
    assert confidence_text(1.0) == "100.0%"      # genuinely the only candidate


def test_most_readable_picks_the_occurrence_with_context():
    """The measure that decides which occurrence of a class is worth looking at,
    for a model and for a person. Shared by the solver and the panel because both
    had the same first-occurrence bug."""
    from toeknife.core.tape import Tape
    from toeknife.predict.solver import most_readable, readable_neighbours
    tape = Tape()
    # 40 unknown bytes, then readable text, then one more unknown inside it.
    for _ in range(40):
        tape.emit_unknown(1)
    for b in b"the quick brown fox jumps over":
        tape.emit_literal(b)
    early, late = 5, 55
    assert readable_neighbours(tape, early) == 0, "buried in the void"
    assert readable_neighbours(tape, late) > 20
    assert most_readable(tape, [early, late]) == late
    assert most_readable(tape, [late, early]) == late, "order must not matter"


def test_the_solver_scores_the_readable_occurrences_not_the_first_ones():
    """Measured consequence: on a carve whose classes begin inside the lost
    region, scoring `positions[:24]` judged every byte against nothing and the
    mixture fell back on its prior."""
    from toeknife.core.tape import Tape
    from toeknife.predict.solver import HoleSolver
    from toeknife.predict.context import ContextModel

    session = type("S", (), {})()
    tape = Tape()
    for _ in range(40):
        tape.emit_unknown(1)
    for b in b"the quick brown fox jumps over the lazy dog":
        tape.emit_literal(b)
    session.tape = tape
    session.trace = type("T", (), {"blocks": ()})()
    solver = HoleSolver(session, ContextModel(order=4).train(b"the quick brown "),
                        max_occurrences=2)
    hole = type("H", (), {"positions": tuple(range(0, 40)) + (50, 60),
                          "anchor": 0, "size": 42})()
    got = solver._evidence_positions(hole)
    assert set(got) == {50, 60}, f"scored {got} instead of the readable pair"


class TestTheByteModelSeam:
    """A byte model answers a whole class off the tape, so it is held apart from
    the ranking model rather than mixed into it."""

    class _Filler:
        """A stand-in that names one byte, so the seam can be tested without
        torch or a 77 MB checkpoint."""

        def __init__(self, favourite: int, window: int = 129):
            self.favourite = favourite
            self.window = window
            self.asked = []

        def class_logprobs(self, tape, positions):
            self.asked.append(tuple(positions))
            out = [-20.0] * 256
            out[self.favourite] = -0.01
            return out

    def _solver(self, **kw):
        return HoleSolver(session(), model(), **kw)

    def test_a_mode_that_needs_a_byte_model_refuses_without_one(self):
        with pytest.raises(ValueError, match="needs a byte model"):
            HoleSolver(session(), model(), mode="slm")
        with pytest.raises(ValueError, match="unknown mode"):
            HoleSolver(session(), model(), mode="sideways")

    def test_slm_mode_answers_from_the_byte_model(self):
        s = self._solver(filler=self._Filler(ord("e")), mode="slm")
        hole = next(iter(s.tape.holes()))
        filler = s.filler
        p = s.propose_one(hole)
        assert p is not None and p.source == "slm"
        assert p.agreement is None, "nothing to agree with in slm mode"
        assert filler.asked, "the byte model was never consulted"

    def test_both_mode_takes_the_byte_model_and_records_agreement(self):
        """Where they disagree the byte model is right far more often -- 67 to
        22 on p38-5 -- so its answer is taken and the n-gram becomes a
        confidence signal rather than a vote."""
        base = self._solver()
        hole = next(iter(base.tape.holes()))
        ngram_pick = base.propose_one(hole)

        same = self._solver(filler=self._Filler(ngram_pick.value),
                            mode="both").propose_one(hole)
        assert same.value == ngram_pick.value
        assert same.agreement is True
        assert same.source == "slm"

        other = next(v for v in base._pool(hole) if v != ngram_pick.value)
        differ = self._solver(filler=self._Filler(other),
                              mode="both").propose_one(hole)
        assert differ.agreement is False
        assert differ.value != ngram_pick.value, "the byte model's answer wins"

    def test_both_mode_is_only_as_sure_as_the_ngram_and_only_when_they_agree(self):
        """The byte model's scores saturate the softmax the threshold was
        calibrated on: ^f at 0.995 filled 87% of a phrack gap at 24.1%
        precision. A disagreement is never confident, and
        an agreement is exactly as confident as the n-gram alone."""
        base = self._solver()
        hole = next(iter(base.tape.holes()))
        ngram_pick = base.propose_one(hole)

        same = self._solver(filler=self._Filler(ngram_pick.value),
                            mode="both").propose_one(hole)
        assert same.confidence == pytest.approx(ngram_pick.confidence)

        other = next(v for v in base._pool(hole) if v != ngram_pick.value)
        differ = self._solver(filler=self._Filler(other),
                              mode="both").propose_one(hole)
        assert differ.confidence == 0.0, "a disagreement must never clear ^f"

    def test_the_byte_model_cannot_fill_a_byte_the_block_forbids(self):
        """The pool carries the constraints. A byte with no code in the block's
        table was not emitted as a literal, however much the model likes it."""
        solver = self._solver(filler=self._Filler(0), mode="slm")
        hole = next(iter(solver.tape.holes()))
        pool = set(solver._pool(hole))
        p = solver.propose_one(hole)
        if 0 not in pool:
            assert p is None or p.value != 0
        assert p is None or p.value in pool

    @staticmethod
    def _cal(right: bool):
        from toeknife.predict import calibration as C
        return C.fit_filler([(0.9, 0, False, right)] * 80
                            + [(0.1, 0, False, not right)] * 80)

    def test_slm_mode_never_reads_the_ngram_curve(self):
        """The n-gram's curve is fit on the n-gram's confidence. Read off the
        byte model's it promises a precision nobody measured."""
        from toeknife.predict import calibration as C
        ngram_cal = C.fit([(c / 100, 0, True) for c in range(100)])
        s = self._solver(filler=self._Filler(ord("e")), mode="slm",
                         calibration=ngram_cal)
        assert s.floor_calibration is None
        p = s.propose()[0]
        assert p.precision is None and p.gate == p.confidence

    def test_both_mode_without_a_byte_curve_keeps_the_ngram_gate(self):
        from toeknife.predict import calibration as C
        ngram_cal = C.fit([(c / 100, 0, True) for c in range(100)])
        s = self._solver(filler=self._Filler(ord("e")), mode="both",
                         calibration=ngram_cal)
        assert s.floor_calibration is ngram_cal

    def test_a_calibrated_byte_model_brings_its_own_curve(self):
        """In both modes the floor reads the byte model's confidence on the
        byte model's curve, whether or not the n-gram agrees."""
        base = self._solver()
        hole = next(iter(base.tape.holes()))
        pick = base.propose_one(hole).value
        other = next(v for v in base._pool(hole) if v != pick)
        for mode in ("slm", "both"):
            for value in (pick, other):
                f = self._Filler(value)
                f.calibration = self._cal(right=True)
                s = self._solver(filler=f, mode=mode)
                assert s.floor_calibration is f.calibration
                p = next(q for q in s.propose() if q.anchor == hole.anchor)
                assert p.confidence > 0.5, "its own confidence, not the gate"
                assert p.precision == f.calibration.precision(p.confidence)


def test_the_stop_names_a_reach_the_floor_can_actually_meet():
    """A best class at 89.57% printed as "90%" promised that a 90% floor
    would fill it, and a 90% floor fills nothing."""
    from toeknife.predict.calibration import Calibration
    cal = Calibration(short=[(0.0, 0.8957, 100)], near=[(0.0, 0.8957, 100)],
                      long=(0.8957, 100), classes=300)
    s = HoleSolver(session(), model(), calibration=cal)
    r = s.solve(min_confidence=0.9)
    assert not r.filled
    assert "89.57%" in r.stopped, r.stopped


def test_a_measured_precision_never_rounds_up_past_a_floor():
    """f4 showed 79.79% as "80%" beside an 80% floor it does not clear."""
    from toeknife.predict.solver import measured_text
    assert measured_text(0.7979) == "79.7%"
    assert measured_text(0.8) == "80.0%"
    assert measured_text(0.9955) == "99.5%"
