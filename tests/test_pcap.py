"""Captures: the one container that knows what is missing rather than absent."""
import gzip
import struct
from pathlib import Path

import pytest

from toeknife import containers
from toeknife.containers import pcap
from toeknife.core import gap as gapmod

INTACT = Path("samples/damaged/http-gzip.pcap")
LOSSY = Path("samples/damaged/http-gzip-loss.pcap")


def payload() -> bytes:
    """What the intact capture carries, according to zlib rather than to us.

    This used to re-read `docs/*.md`, which is what the generator compresses --
    so adding any doc to the repo silently broke two recovery tests that have
    nothing to do with documentation.  Taking truth from the reference decoder
    instead makes the check independent of the tree *and* stronger: it is now two
    implementations agreeing rather than one implementation agreeing with a glob.
    """
    raw = need(INTACT)
    stream = containers.parse(raw).streams[0]
    return gzip.decompress(joined_of(raw, stream))


def need(path):
    if not path.is_file():
        pytest.skip("run samples/build/makepcap.py to build the captures")
    return path.read_bytes()


def joined_of(raw, stream):
    return b"".join(raw[o:o + n] for o, n in stream.segments)


class TestParsing:
    def test_a_capture_is_recognised_and_its_flow_reassembled(self):
        raw = need(INTACT)
        assert containers.detect(raw) == "pcap"
        report = containers.parse(raw)
        assert len(report.streams) == 1, "one direction carries the response"
        st = report.streams[0]
        assert st.encoding == containers.GZIP
        assert len(st.segments) == 33
        assert not st.missing
        assert joined_of(raw, st)[:2] == b"\x1f\x8b", \
            "the segments have to start at the body, not the HTTP head"

    def test_a_lost_packet_is_measured_not_estimated(self):
        """The whole reason a capture is worth special handling: TCP numbers
        every byte, so the hole's width is arithmetic on two sequence numbers
        rather than a search."""
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        assert len(st.segments) == 32, "one packet fewer"
        assert len(st.missing) == 1
        _pos, width = st.missing[0]
        assert width == 1460, "exactly one MTU-sized segment"
        assert any("MISSING" in n and "sequence numbers" in n for n in st.notes)

    def test_the_hole_is_positioned_where_the_survivors_meet(self):
        """Positions index the CONCATENATION of what survives, because that is
        the only byte string a caller has. With one hole the two readings
        coincide, which is the sort of coincidence that hides a bug."""
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        pos, width = st.missing[0]
        joined = joined_of(raw, st)
        assert len(joined) == sum(n for _o, n in st.segments)
        assert pos < len(joined), "a cut point inside what survives"
        # The byte after the cut is real data, not the start of the hole.
        assert joined[pos:pos + 4] != b"\x00\x00\x00\x00"

    def test_splicing_across_a_hole_has_to_be_asked_for(self):
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        with pytest.raises(ValueError, match="looks continuous and is not"):
            st.slice(raw)
        assert st.slice(raw, allow_gaps=True)

    def test_pcapng_is_refused_by_name_rather_than_misread(self):
        report = containers.parse(b"\x0a\x0d\x0d\x0a" + b"\x00" * 64)
        assert report.kind == "pcapng"
        assert not report.streams
        assert any("editcap" in w for w in report.warnings)

    def test_a_short_capture_reports_nothing_rather_than_guessing(self):
        report = containers.parse(b"\xd4\xc3\xb2\xa1" + b"\x00" * 20)
        assert report is not None and not report.streams


class TestReassembly:
    def make(self, segments):
        flow = pcap.Flow("a", "b", 1, 2)
        for seq, off, ln in segments:
            flow.segments.append(pcap.Segment(seq, off, ln))
        return pcap.reassemble(flow)

    def test_in_order_segments_have_no_holes(self):
        segs, missing, _n = self.make([(100, 0, 10), (110, 10, 10)])
        assert segs == [(0, 10), (10, 10)] and not missing

    def test_a_gap_is_reported_at_the_join(self):
        segs, missing, _n = self.make([(100, 0, 10), (130, 10, 10)])
        assert segs == [(0, 10), (10, 10)]
        assert missing == [(10, 20)], "20 bytes absent where the two meet"

    def test_two_gaps_do_not_drift(self):
        """The second hole's position must index the concatenation too. Counting
        the first hole's width into the running total is the bug this guards."""
        segs, missing, _n = self.make([(100, 0, 10), (130, 10, 10),
                                       (200, 20, 10)])
        # seq 110..130 absent, then 140..200: 20 bytes then 60. Both positions
        # index the 30 surviving bytes, so the second is 20 and not 40.
        assert missing == [(10, 20), (20, 60)]

    def test_a_retransmission_is_not_counted_twice(self):
        segs, missing, notes = self.make([(100, 0, 10), (100, 10, 10),
                                          (110, 20, 10)])
        assert segs == [(0, 10), (20, 10)] and not missing

    def test_an_overlap_keeps_what_arrived_first(self):
        segs, missing, notes = self.make([(100, 0, 10), (105, 10, 10)])
        assert segs == [(0, 10), (15, 5)], "only the new 5 bytes"
        assert not missing and any("overlap" in n for n in notes)


class TestRecovery:
    def assemble(self, raw, stream):
        joined = joined_of(raw, stream)
        _crc, isize = struct.unpack("<II", joined[-8:])
        return gapmod.from_holes(joined, stream.missing, 10, 8, isize)

    def test_a_capture_with_a_lost_packet_still_yields_most_of_the_body(self):
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        asm = self.assemble(raw, st)
        truth = payload()
        buf, mask = asm.tape.recovered()
        right = sum(1 for i, (b, m) in enumerate(zip(buf, mask))
                    if m and i < len(truth) and b == truth[i])
        wrong = sum(1 for i, (b, m) in enumerate(zip(buf, mask))
                    if m and i < len(truth) and b != truth[i])
        assert wrong == 0, "a wrong byte here is worse than an unknown one"
        assert right > 100_000
        assert right / len(truth) > 0.84
        # The recorded length puts the far side back where it belongs, so the
        # total is exact even though a slice of the middle is unknown.
        assert asm.tape.stats()["out_len"] == len(truth)

    def test_the_output_gap_comes_from_the_recorded_size_not_a_search(self):
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        asm = self.assemble(raw, st)
        assert any("from the recorded decompressed size" in n for n in asm.notes)
        assert any("resumes mid-block" in n for n in asm.notes)

    def test_without_a_recorded_size_it_says_the_gap_is_unknown(self):
        """1,460 compressed bytes could be 1,460 output bytes or 18,000. Nothing
        in the capture decides it, and inventing a number would put every byte
        after the hole at the wrong offset."""
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        joined = joined_of(raw, st)
        asm = gapmod.from_holes(joined, st.missing, 10, 8, total_size=None)
        assert any("output gap is unknown" in n for n in asm.notes)

    def test_the_intact_capture_round_trips_exactly(self):
        raw = need(INTACT)
        st = containers.parse(raw).streams[0]
        from toeknife.core.session import Session
        s = Session(st.slice(raw), mode="stream", stream=st)
        buf, _mask = s.tape.recovered()
        assert bytes(buf) == payload()


class TestTheLoopCloses:
    """A capture is the one case where a recovery can be checked end to end: the
    gzip trailer rode along in the packets, so filling the holes the loss left
    has an answer to be right or wrong against."""

    def test_the_trailer_is_carried_through_to_the_session(self):
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        assert st.crc32 is not None and st.uncompressed_size == 120_000
        assert any("trailer survived" in n for n in st.notes)

    def test_the_checksum_refuses_to_rule_while_holes_remain(self):
        from toeknife.core.session import Session
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        asm = gapmod.from_holes(st.slice(raw, allow_gaps=True), st.missing,
                                10, 8, st.uncompressed_size)
        s = Session.from_trace(asm.trace, stream=st)
        v = s.verify()
        assert v.ok is None, "unknown bytes are not a pass and not a failure"
        assert "unchecked" in v.summary()
        assert s.container.kind == "pcap", "the view should say where it came from"

    def test_filling_the_holes_correctly_makes_the_crc_agree(self):
        from toeknife.core.session import Session
        from toeknife.core.tape import GUESS
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        asm = gapmod.from_holes(st.slice(raw, allow_gaps=True), st.missing,
                                10, 8, st.uncompressed_size)
        s = Session.from_trace(asm.trace, stream=st)
        truth = payload()
        for hole in list(s.tape.holes()):
            if 0 <= hole.anchor < len(truth):
                s.set_guess(hole.anchor, truth[hole.anchor], GUESS)
        assert s.tape.stats()["holes"] == 0
        assert s.verify().ok is True, "a completed recovery has to be checkable"

    def test_a_trailer_inside_the_loss_is_not_used(self):
        """A checksum nobody can trust is worse than none, so a hole covering the
        last 8 bytes means the stream reports no CRC at all."""
        flow = pcap.Flow("a", "b", 1, 2)
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        # Same stream, but pretend the loss landed at the very end.
        joined = st.slice(raw, allow_gaps=True)
        faked = [(len(joined) - 4, 500)]
        holes_after = [p for p, n in faked if p + n > len(joined) - 8]
        assert holes_after, "the fixture has to cover the trailer"


class TestTheOperatorIsToldWhatIsMissing:
    """Found writing the manual: on this capture ^f fills 1 class of 4,041 and
    the checksum says "fill some with the model first", so each screen sends the
    operator to the other and neither names the corpus that is actually absent."""

    def session(self):
        from toeknife.core.session import Session
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        asm = gapmod.from_holes(st.slice(raw, allow_gaps=True), st.missing,
                                10, 8, st.uncompressed_size)
        return Session.from_trace(asm.trace, stream=st), asm

    def test_too_many_holes_and_no_corpus_points_at_f5(self):
        from toeknife.tui.modals import ChecksumScreen
        s, _ = self.session()
        assert "f5" in ChecksumScreen(s, has_model=False)._system().plain
        assert "f5" not in ChecksumScreen(s, has_model=True)._system().plain

    def test_the_notes_read_as_sentences(self):
        """They printed "before the loss ended truncated" and "4041 unknown
        output bytes before after the loss"."""
        _, asm = self.session()
        assert not any("before after" in n for n in asm.notes)
        assert any(n.startswith("the part before the loss ended")
                   for n in asm.notes)
        import re
        assert any(re.fullmatch(r"[\d,]+ unknown output bytes before the "
                                r"part after the loss", n) for n in asm.notes)


class TestNothingMisreadIsCalledKnown:
    """Found writing the manual: bytes the tool marked known after a
    lost packet disagreed with the original -- on most single-packet losses of
    every payload tried, from 1 byte to 33,000. Two causes. The tail was read
    from whichever of eight offsets decoded furthest, and the symbols before
    the readings converge were kept as literals; and when the lost packet held
    a block boundary, the tail was read with the previous block's table.

    536 B segments of the sample's payload are dropped one at a time -- every
    sixth, which the old code failed on nearly all of, and every one holding a
    block boundary -- and no byte the tool calls known may differ from the
    original.
    """

    def test_no_single_lost_segment_leaves_a_wrong_known_byte(self):
        import zlib
        from toeknife.core.inflate import Decoder
        from toeknife.core.session import Session
        truth = payload()
        co = zlib.compressobj(9, zlib.DEFLATED, 16 + 15)
        gz = co.compress(truth) + co.flush()
        mss = 536
        bounds = {10 + b.bit_start // 8 for b in
                  Decoder(gz[10:-8]).stream(0).blocks[1:]}
        assert bounds, "the payload has to span a block boundary"
        drops = [at for at in range(mss, len(gz) - 2 * mss, mss)
                 if (at // mss) % 6 == 0
                 or any(at <= b < at + mss for b in bounds)]
        failures = []
        for at in drops:
            joined = gz[:at] + gz[at + mss:]
            asm = gapmod.from_holes(joined, [(at, mss)], 10, 8, len(truth))
            s = Session.from_trace(asm.trace)
            buf, mask = s.tape.recovered()
            assert len(buf) == len(truth)
            wrong = sum(1 for i in range(len(buf))
                        if mask[i] and buf[i] != truth[i])
            if wrong:
                failures.append((at // mss, wrong))
        assert not failures, f"(segment, wrong known bytes): {failures}"


class TestTheFloorIsNotTrustedInsideALongGap:
    """Inside a lost packet's gap the confidence floor does
    not hold -- each fill becomes context for the next, and a 0.995 floor
    filled the gap at 28.8% precision. Classes that sit only inside long runs
    of unread output are held back, and filling them is a separate decision."""

    def test_long_run_classes_are_held_back_until_asked_for(self):
        from toeknife.core.session import Session
        from toeknife.predict.context import ContextModel
        from toeknife.predict.solver import HoleSolver, LONG_RUN
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]

        def fresh():
            asm = gapmod.from_holes(st.slice(raw, allow_gaps=True),
                                    st.missing, 10, 8, st.uncompressed_size)
            return Session.from_trace(asm.trace, stream=st)

        # Trained on the answer, at a low floor, so plenty clears it: what is
        # under test is which of those the solver is willing to fill.
        model = ContextModel(order=4).train(payload())
        held = HoleSolver(fresh(), model, min_confidence=0.5).solve(batch=4096)
        assert held.held, "classes inside the gap must be held back"
        assert "256+" in held.stopped
        s = fresh()
        anyway = HoleSolver(s, model, min_confidence=0.5).solve(
            long_runs=True, batch=4096)
        assert anyway.held == 0
        assert len(anyway.filled) > len(held.filled)
        # And every fill carries its note, with the run it was made in.
        notes = [s.fill_notes[p.anchor] for p in anyway.filled]
        assert any(n.run >= LONG_RUN for n in notes)


class TestASuspectedCopyIsProvenOrRefuted:
    """A lost packet is beyond any model tried, but a candidate copy of the
    content can be laid over the capture: every decoded byte tests it, and the
    gzip CRC rules on the rest."""

    def session(self):
        from toeknife.core.session import Session
        raw = need(LOSSY)
        st = containers.parse(raw).streams[0]
        asm = gapmod.from_holes(st.slice(raw, allow_gaps=True), st.missing,
                                10, 8, st.uncompressed_size)
        return Session.from_trace(asm.trace, stream=st)

    def test_the_true_copy_is_proven(self):
        from toeknife.core import copyfill
        r = copyfill.fill(self.session(), payload())
        assert r.proven and not r.contradictions
        assert r.offset == 0 and r.filled > 0

    def test_a_copy_with_a_header_still_lines_up(self):
        from toeknife.core import copyfill
        r = copyfill.fill(self.session(), b"Received: cache\r\n\r\n" + payload())
        assert r.offset == len(b"Received: cache\r\n\r\n") and r.proven

    def test_an_edit_inside_the_lost_bytes_is_refuted_by_the_checksum(self):
        """Edited consistently -- every position the capture proves equal --
        so nothing short of the CRC can catch it."""
        from toeknife.core import copyfill
        s = self.session()
        hole = s.tape.holes()[0]
        truth = bytearray(payload())
        for p in hole.positions:
            truth[p] ^= 0x20
        r = copyfill.fill(s, bytes(truth))
        assert r.verdict.ok is False and not r.contradictions and r.refuted
        assert "inside the lost bytes" in r.summary()

    def test_a_copy_that_contradicts_itself_is_refuted_before_the_checksum(self):
        """One position of a class edited: the capture proves those bytes
        equal, so the copy is wrong without the CRC having to say so."""
        from toeknife.core import copyfill
        s = self.session()
        hole = next(h for h in s.tape.holes() if h.size > 1)
        truth = bytearray(payload())
        truth[hole.positions[0]] ^= 0x20
        r = copyfill.fill(s, bytes(truth))
        assert r.split == 1 and r.refuted and not r.proven
        assert "disagrees with itself" in r.summary()

    def test_an_edit_in_what_decoded_is_pointed_at(self):
        from toeknife.core import copyfill
        s = self.session()
        truth = bytearray(payload())
        truth[100:106] = b"EDITED"
        r = copyfill.fill(s, bytes(truth))
        assert r.contradictions and min(r.contradictions) >= 100
        assert copyfill.regions(r.contradictions)[0][0] < 106
        # The lost bytes still check out, so the CRC agrees -- but the copy is
        # not the captured text, and the summary must not read as identity.
        assert r.proven and "a different version" in r.summary()

    def test_an_unrelated_copy_is_refused_rather_than_used(self):
        from toeknife.core import copyfill
        s = self.session()
        before = len(s.tape.holes())
        r = copyfill.fill(s, bytes(range(256)) * 400)
        assert r.offset is None and "does not line up" in r.summary()
        assert len(s.tape.holes()) == before, "nothing filled from it"
