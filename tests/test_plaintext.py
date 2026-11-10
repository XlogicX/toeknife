"""Judging a decode against known plaintext: the assertions, and the refusal."""
import random
import zlib

import pytest

from toeknife.core import plaintext as pt
from toeknife.core.session import Session

TEXT = b"the quick brown fox jumps over the lazy dog. " * 400


def raw_deflate(data):
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9)
    return co.compress(data) + co.flush()


def scattered(data, rho, seed=7):
    """Plaintext the way an operator actually has it: scattered, not a slab."""
    rng = random.Random(seed)
    return {i: b for i, b in enumerate(data) if rng.random() < rho}


@pytest.fixture
def session():
    return Session(raw_deflate(TEXT), mode="stream")


class TestAssertions:
    def test_a_correct_decode_contradicts_nothing(self, session):
        v = pt.judge(session, scattered(TEXT, 0.4))
        assert v.testable > pt.FLOOR
        assert v.bad == 0 and v.consistent

    def test_both_ends_must_be_known_for_an_assertion_to_count(self, session):
        """With only the near end there is nothing to compare against but the
        candidate's own decode, which is the candidate vouching for itself."""
        sparse = pt.judge(session, scattered(TEXT, 0.02))
        dense = pt.judge(session, scattered(TEXT, 0.6))
        assert dense.testable > sparse.testable

    def test_plaintext_that_breaks_the_copy_relations_is_contradicted(self):
        """Values chosen independently destroy out[a] == out[a-d], and that is
        what this half of the check exists to notice."""
        s = Session(raw_deflate(TEXT), mode="stream")
        rng = random.Random(3)
        wrong = {i: rng.randrange(256) for i in range(0, 6000)}
        v = pt.judge(s, wrong)
        assert v.bad > 0 and not v.consistent
        assert "contradicted" in v.summary()

    def test_a_uniform_relabelling_is_invisible_here(self):
        """The match half is assignment-free by design: it tests the copy
        STRUCTURE, not the byte values. So XORing every supplied byte with the
        same mask preserves every assertion and passes cleanly.

        That is a real limit and not a bug. It is also why the literal side of
        the research verifier exists, and why a table that satisfies this half
        can still be wrong about what any given code MEANS.
        """
        s = Session(raw_deflate(TEXT), mode="stream")
        masked = {i: b ^ 0xFF for i, b in scattered(TEXT, 0.4).items()}
        v = pt.judge(s, masked)
        assert v.testable > pt.FLOOR
        assert v.bad == 0


class TestAbstention:
    def test_too_little_plaintext_refuses_rather_than_passing(self, session):
        """Every dictionary false accept was stopped by this refusal, so it
        must never be reported as a pass."""
        v = pt.judge(session, {0: TEXT[0], 1: TEXT[1]})
        assert not v.ruled
        assert not v.consistent
        assert "cannot rule" in v.summary()
        assert "cannot rule" in v.summary()

    def test_the_floor_is_where_the_verdict_changes(self, session):
        v = pt.judge(session, scattered(TEXT, 0.4))
        assert v.ruled and v.testable >= pt.FLOOR

    def test_an_abstention_reports_no_byte_classes(self, session):
        """Nothing has been established, so nothing may be vouched for."""
        v = pt.judge(session, {0: TEXT[0]})
        assert v.counts == {}


class TestClasses:
    def test_supplied_bytes_are_given_not_grounded(self, session):
        known = scattered(TEXT, 0.4)
        classes = pt.classify(session, known)
        for pos in list(known)[:50]:
            if pos < len(classes):
                assert classes[pos] == pt.GIVEN

    def test_every_byte_gets_exactly_one_class(self, session):
        classes = pt.classify(session, scattered(TEXT, 0.4))
        assert len(classes) == session.tape.length
        assert set(classes) <= {pt.GIVEN, pt.GROUNDED, pt.FLOATING,
                                pt.PRE, pt.UNFILLED}

    def test_no_evidence_means_nothing_is_grounded(self, session):
        """A caller who brought no plaintext gets the honest answer."""
        empty = bytearray(session.tape.length)
        classes = pt.classify(session, {}, reach=empty)
        assert pt.GROUNDED not in classes

    def test_a_copy_is_only_as_good_as_its_source(self, session):
        """18's false accept was a corroborated position whose SOURCE was not,
        so the chain is what gets checked, not the destination."""
        known = scattered(TEXT, 0.4)
        reach = pt.combined_reach(session, known)
        classes = pt.classify(session, known, reach)
        for sym in session.trace.symbols:
            if sym.kind != "match" or not sym.length:
                continue
            d = sym.distance or 0
            for i in range(min(sym.length, 8)):
                p, src = sym.out_pos + i, sym.out_pos + i - d
                if 0 <= src < len(classes) and p < len(classes):
                    if classes[p] == pt.GROUNDED:
                        assert classes[src] in (pt.GIVEN, pt.GROUNDED)

    def test_backward_reach_covers_what_forward_reach_cannot(self, session):
        """source_reach is the only thing that reaches output the plaintext
        never touches; without it the grounded class barely exists."""
        known = scattered(TEXT, 0.4)
        fwd = pt.reach_map(session, known)
        both = pt.combined_reach(session, known)
        assert sum(both) >= sum(fwd)


class TestReporting:
    def test_no_single_rate_is_ever_printed(self, session):
        """The error rate is a property of the block, so a percentage here
        would be a lie on one page and pessimism on another."""
        text = pt.render(session, scattered(TEXT, 0.4))
        assert "%" not in text
        assert "property of the block" in text

    def test_consistent_is_not_reported_as_correct(self, session):
        text = pt.render(session, scattered(TEXT, 0.4))
        assert "Consistent is not correct" in text

    def test_counts_are_reported_for_each_class(self, session):
        text = pt.render(session, scattered(TEXT, 0.4))
        assert "given" in text and "grounded" in text and "floating" in text
