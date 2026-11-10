"""The chain behind a byte, and the odds attached to a related capture."""
import zlib

import pytest

from toeknife.core import provenance, related
from toeknife.core.session import Session

TEXT = b"the quick brown fox jumps over the lazy dog. " * 40
# zlib emits a FIXED block below a few KB, and a fixed block carries no
# table to lift, so the related-capture tests need a bigger sample.
BIG = b"the quick brown fox jumps over the lazy dog. " * 400
# Jaccard counts DISTINCT tokens, so repeating nine words a thousand times still
# has nine of them and the overlap guard rightly abstains. Content that can be
# compared has to have vocabulary, not just length.
VARIED = b" ".join(b"word%04d item%04d field%04d" % (i, i * 7, i * 13)
                   for i in range(400))


def raw_deflate(data):
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9)
    return co.compress(data) + co.flush()


@pytest.fixture
def session():
    return Session(raw_deflate(TEXT), mode="stream")


class TestChain:
    def test_a_literal_rests_on_itself(self, session):
        # Position 0 cannot be a copy: there is nothing behind it.
        c = provenance.chain(session, 0)
        assert len(c.links) == 1
        assert c.links[0].kind == "literal"

    def test_a_copy_walks_back_to_a_literal(self, session):
        c = provenance.chain(session, 500)
        assert len(c.links) > 1
        assert c.links[0].kind == "copy"
        assert c.root.kind == "literal"
        assert c.root.pos < 500

    def test_the_chain_reports_the_hop_count(self, session):
        c = provenance.chain(session, 500)
        assert "hop" in c.rests_on()

    def test_a_long_chain_is_truncated_but_still_names_its_root(self, session):
        short = provenance.chain(session, 500, limit=3)
        assert len(short.links) <= 4          # limit links, plus the root
        assert short.root is not None
        assert short.truncated >= 0

    def test_values_agree_along_the_chain(self, session):
        """A copy is the byte it copied; if that ever stops holding, the tape
        and the trace disagree and every vouch built on them is void."""
        c = provenance.chain(session, 500)
        values = {ln.value for ln in c.links}
        assert len(values) == 1

    def test_render_mentions_every_link(self, session):
        text = provenance.render(session, 500)
        c = provenance.chain(session, 500)
        for ln in c.links:
            assert f"out {ln.pos}" in text

    def test_an_unknown_byte_says_so_rather_than_claiming_the_stream(self):
        """It is not 'read from the stream' -- nothing determined it."""
        s = Session(raw_deflate(TEXT)[30:])
        pos = next((i for i in range(s.tape.length)
                    if s.tape.value_at(i) is None), None)
        if pos is None:
            pytest.skip("this fragment left no unknown positions")
        text = provenance.render(s, pos)
        assert "not recovered" in text


class TestRelated:
    def test_overlap_is_one_for_identical_content(self):
        assert related.overlap(TEXT, TEXT) == 1.0

    def test_overlap_is_zero_for_disjoint_content(self):
        assert related.overlap(b"alpha beta", b"gamma delta") == 0.0

    def test_expectation_covers_the_measured_bands(self):
        assert "almost never" in related.expectation(0.5)
        assert "usually" in related.expectation(0.9995)

    def test_asset_content_carries_its_own_caveat(self):
        """The curve is calibrated on HTML and assets sit lower."""
        text = related.expectation(0.995, kind="css")
        assert "css" in text and "lower than HTML" in text

    def test_it_refuses_to_predict_from_too_little_recovered_content(self):
        """Quoting a calibrated rate off three tokens would dress noise up as
        a prediction, which is the failure the whole design is against."""
        rep = related.report(b"one two", raw_deflate(TEXT))
        assert rep["overlap"] is None
        assert "not predictable" in rep["expectation"]

    def test_it_predicts_once_there_is_enough_to_compare(self):
        rep = related.report(VARIED, raw_deflate(VARIED))
        assert rep["overlap"] == 1.0
        assert rep["usable"]

    def test_repetition_is_not_vocabulary(self):
        """Length does not make content comparable: nine words repeated a
        thousand times is still nine words, and the curve needs more."""
        rep = related.report(BIG, raw_deflate(VARIED))
        assert rep["overlap"] is None

    def test_a_table_is_lifted_from_the_related_capture(self):
        litlen, dist = related.tables_from(VARIED)
        assert litlen is not None and dist is not None

    def test_a_capture_too_small_for_a_dynamic_block_says_why(self):
        """zlib uses a fixed table below a few KB, and a fixed block carries
        none -- the commonest reason this route cannot run, so it is named."""
        rep = related.report(VARIED, raw_deflate(TEXT))
        assert not rep["usable"]
        assert "fixed table" in rep["why_not"]

    def test_content_arrives_through_any_wrapper(self):
        import gzip
        assert related.plain_bytes(gzip.compress(TEXT)) == TEXT
        assert related.plain_bytes(zlib.compress(TEXT)) == TEXT
        assert related.plain_bytes(raw_deflate(TEXT)) == TEXT
        assert related.plain_bytes(TEXT) == TEXT


class TestWhoAssertedIt:
    """Which of the operator, the model and the bitstream put a byte there.

    The tool's rule is that these never look alike, and two paths were losing
    the distinction after the fact rather than at the point of assertion: undo
    read the provenance out of a record it had just deleted, and a saved session
    wrote the values without them.
    """

    FRAGMENT = bytes.fromhex(
        "2f2dd203930a99c50a2519a90af97939950af9690a2519f9c5a90a251989250a"
        "1989c5102539a9252520e332f314324b00")

    def fragment_session(self):
        return Session(self.FRAGMENT, mode="fragment", profile="text")

    def test_undo_gives_a_model_fill_back_as_the_models(self):
        from toeknife.core.tape import GUESS, SOLVED
        s = self.fragment_session()
        pos = s.tape.holes()[0].anchor
        s.set_guess(pos, ord("x"), SOLVED)
        assert s.tape.prov_at(pos) == SOLVED

        # The operator overrules it, then changes their mind.
        s.begin_group()
        s.clear_guess(pos)
        s.set_guess(pos, ord("y"), GUESS)
        s.end_group()
        assert s.tape.prov_at(pos) == GUESS

        assert s.undo()
        assert s.tape.value_at(pos) == ord("x")
        assert s.tape.prov_at(pos) == SOLVED, \
            "undo credited the operator with a byte the model proposed"

    def test_retracting_takes_back_the_models_bytes_and_only_those(self):
        """The whole point of ctrl+e: a fill closes its class, so a byte the model
        got wrong early cannot be dislodged by context that arrives afterwards.
        Reopening has to spare what the operator typed, or it is just undo."""
        from toeknife.core.tape import GUESS, SOLVED
        s = self.fragment_session()
        holes = s.tape.holes()
        assert len(holes) >= 3, "need a few classes to tell them apart"
        mine, models, other = (h.anchor for h in holes[:3])
        s.set_guess(mine, ord("e"), GUESS)
        s.set_guess(models, ord("x"), SOLVED)
        s.set_guess(other, ord("z"), SOLVED)

        took = s.retract_model_fills()
        assert sorted(took) == sorted([models, other])
        assert s.tape.value_at(mine) == ord("e"), "the operator's byte went too"
        assert s.tape.prov_at(mine) == GUESS
        assert s.tape.value_at(models) is None
        assert s.tape.value_at(other) is None
        # And the classes are open again, which is what makes a refill possible.
        assert {h.anchor for h in s.tape.holes()} >= {models, other}

        # One ctrl+z puts the model's bytes back, still as the model's.
        assert s.undo()
        assert s.tape.value_at(models) == ord("x")
        assert s.tape.prov_at(models) == SOLVED
        assert s.tape.value_at(other) == ord("z")

    def test_retracting_with_nothing_of_the_models_changes_nothing(self):
        from toeknife.core.tape import GUESS
        s = self.fragment_session()
        pos = s.tape.holes()[0].anchor
        s.set_guess(pos, ord("e"), GUESS)
        before = len(s.tape.holes())
        assert s.retract_model_fills() == []
        assert len(s.tape.holes()) == before
        assert s.tape.value_at(pos) == ord("e")

    def test_redo_keeps_the_provenance_too(self):
        from toeknife.core.tape import SOLVED
        s = self.fragment_session()
        pos = s.tape.holes()[0].anchor
        s.set_guess(pos, ord("x"), SOLVED)
        s.clear_guess(pos)
        assert s.undo() and s.tape.prov_at(pos) == SOLVED
        assert s.redo(), "redo should re-clear it"
        assert s.tape.value_at(pos) is None
        assert s.undo() and s.tape.prov_at(pos) == SOLVED

    def test_a_saved_session_remembers_who_asserted_each_byte(self, tmp_path):
        from toeknife.core.tape import GUESS, SOLVED
        s = self.fragment_session()
        holes = s.tape.holes()
        mine, models = holes[0].anchor, holes[1].anchor
        s.set_guess(mine, ord("m"), GUESS)
        s.set_guess(models, ord("s"), SOLVED)
        path = tmp_path / "session.json"
        s.save(path)

        back = self.fragment_session()
        back.load(path)
        assert back.tape.value_at(mine) == ord("m")
        assert back.tape.value_at(models) == ord("s")
        assert back.tape.prov_at(mine) == GUESS
        assert back.tape.prov_at(models) == SOLVED, \
            "reloading promoted a model fill to the operator's own byte"

    def test_a_file_from_before_this_reads_its_bytes_as_guesses(self, tmp_path):
        """Version 2 wrote no provenance. Guess is the honest reading: the file
        cannot say who asserted the byte, so it must not claim the model did."""
        import json
        from toeknife.core.tape import GUESS
        s = self.fragment_session()
        pos = s.tape.holes()[0].anchor
        s.set_guess(pos, ord("z"))
        path = tmp_path / "old.json"
        s.save(path)
        blob = json.loads(path.read_text())
        blob["version"] = 2
        del blob["provenance"]
        path.write_text(json.dumps(blob))

        back = self.fragment_session()
        back.load(path)
        assert back.tape.value_at(pos) == ord("z")
        assert back.tape.prov_at(pos) == GUESS
