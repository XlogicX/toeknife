"""Layering: the model with the checksum, and both directions of the n-gram."""
import zlib

import pytest

from toeknife.containers import parse
from toeknife.core.session import Session
from toeknife.predict.context import Bidirectional, ContextModel
from toeknife.predict.pipeline import solve_loop
from toeknife.predict.solver import HoleSolver

BODY = b"forensic analysis of compressed data, repeatedly and at length. " * 300
CORPUS = (b"forensic work on compressed streams. analysis of data at length. "
          b"the forensic analyst reads compressed data repeatedly. ") * 200


def dict_session(tail: bytes, with_crc: bool = True):
    """Holes made from a preset dictionary, the easiest way to get a few. The
    loop is told to fill them (dictionary=True): what these tests check is
    the loop, and ^f holding dictionary bytes back is tested elsewhere."""
    zdict = b"\xa5" * 80 + tail
    co = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY,
                          zdict=zdict)
    blob = co.compress(BODY) + co.flush()
    sm = parse(blob).streams[0]
    if with_crc:
        sm.crc32 = zlib.crc32(BODY) & 0xFFFFFFFF
    sm.uncompressed_size = len(BODY)
    return Session(sm.slice(blob), mode="stream", stream=sm), zdict


def model():
    return Bidirectional.train_on(CORPUS, order=4)


# ---- both directions --------------------------------------------------------

def test_a_reverse_model_judges_a_byte_with_nothing_on_its_left():
    """ziprec builds both directions and says forward-only "degrades
    reconstruction".  The reason: the holes that matter sit at the edge of the
    pre-fragment window, where the left side is permanently missing."""
    bi = Bidirectional.train_on(CORPUS, order=4)
    # 'a' should be likely before "nalysis", and the forward model cannot say so
    with_right = bi.logprob_both(ord("a"), b"", b"nalysis")
    forward_only = bi.forward.logprob(ord("a"), b"")
    assert with_right > forward_only
    assert with_right > bi.logprob_both(0xFE, b"", b"nalysis")


def test_with_context_on_both_sides_it_uses_both():
    bi = Bidirectional.train_on(CORPUS, order=4)
    a = bi.logprob_both(ord("l"), b"ana", b"ysis")
    b = bi.logprob_both(0xFE, b"ana", b"ysis")
    assert a > b


def test_a_bidirectional_model_survives_a_round_trip(tmp_path):
    bi = Bidirectional.train_on(CORPUS, order=3)
    path = tmp_path / "bi.tkm"
    bi.save(path)
    back = ContextModel.load(path)          # dispatches on the saved flag
    assert isinstance(back, Bidirectional)
    assert back.order == 3
    for ctx in (b"ana", b"", b"for"):
        assert back.logprob(ord("l"), ctx) == pytest.approx(bi.logprob(ord("l"), ctx))
    assert back.logprob_both(ord("a"), b"", b"nal") == pytest.approx(
        bi.logprob_both(ord("a"), b"", b"nal"))


def test_the_solver_uses_both_directions_when_offered():
    session, _ = dict_session(b"ensic", with_crc=False)
    hole = session.tape.holes()[0]
    solver = HoleSolver(session, model())
    assert solver._score_value(hole.positions[:3], ord("e"))[1] > 0


# ---- the loop ---------------------------------------------------------------

def test_the_loop_solves_and_says_the_checksum_confirmed_it():
    session, zdict = dict_session(b"ensic")
    result = solve_loop(session, model(), dictionary=True)
    assert result.remaining == 0
    assert result.verified is True
    assert "confirms every recovered byte" in result.summary()
    window, _, _ = session.window()
    assert bytes(window) == zdict[-len(window):]


def test_the_loop_reports_when_the_checksum_contradicts_the_fills():
    """The most useful thing a checksum does when it cannot solve: prove that
    something already filled in is wrong."""
    session, zdict = dict_session(b"c analysis")
    result = solve_loop(session, model(), dictionary=True)
    assert result.filled > 0
    assert result.contradicted
    assert "CONTRADICTS" in result.summary()
    window, _, _ = session.window()
    assert bytes(window) != zdict[-len(window):], (
        "the fixture is meant to be one the model gets wrong")


def test_the_loop_never_claims_confirmation_it_did_not_get():
    """Regression.  verify() returns ok on a size match while both checksums are
    unchecked, and reporting that as confirmation is a false assurance."""
    session, _ = dict_session(b"c analysis")
    result = solve_loop(session, model(), dictionary=True)
    assert session.tape.holes(), "this fixture should not finish"
    assert result.verified is None, "unchecked is not confirmed"
    assert "confirms" not in result.summary()


def test_the_loop_stops_when_a_round_changes_nothing():
    session, _ = dict_session(b"c analysis")
    result = solve_loop(session, model(), max_rounds=12, dictionary=True)
    assert len(result.rounds) < 12
    assert result.rounds[-1].progress == 0


def test_the_loop_is_a_no_op_with_nothing_unknown():
    import gzip as gzip_mod
    blob = gzip_mod.compress(BODY)
    sm = parse(blob).streams[0]
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    assert not session.tape.holes()
    result = solve_loop(session, model(), dictionary=True)
    assert result.filled == 0


class _Asked:
    """A byte model that records being asked, and prefers no byte."""

    calibration = None

    def __init__(self):
        self.asked = []

    def class_logprobs(self, tape, positions):
        self.asked.append(tuple(positions))
        return [-5.0] * 256


def test_the_loop_fills_with_the_byte_model_when_given_one():
    """^f used the byte model and the model+checksum loop on the same screen
    silently used the n-gram."""
    session, _ = dict_session(b"ensic")
    filler = _Asked()
    solve_loop(session, model(), dictionary=True, filler=filler, mode="slm",
               max_rounds=1)
    assert filler.asked, "the loop's fill round must consult the byte model"


def test_the_loop_without_a_byte_model_is_the_ngram_as_before():
    session, zdict = dict_session(b"ensic")
    result = solve_loop(session, model(), dictionary=True, mode="slm")
    assert result.remaining == 0, "mode without a filler falls back to n-gram"
