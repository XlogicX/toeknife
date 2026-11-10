"""Undoing bit rot in a header: every survivor, and none of them chosen."""
import zlib

import pytest

from toeknife.core import bitrot
from toeknife.core.session import Session

BODY = b"forensic analysis of compressed evidence, at some length. " * 90


def clean() -> bytes:
    co = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    return co.compress(BODY) + co.flush()


def rotted(offset: int):
    """(damaged bytes, the bit that was flipped) for a bit inside the header."""
    comp = clean()
    lo, hi = bitrot.header_span(Session(comp, mode="stream"))
    bit = lo + offset
    assert bit < hi, "the fixture has to damage a header bit"
    return bitrot.flip(comp, [bit]), bit


def test_the_span_is_the_header_and_stops_there():
    """Not one bit into the body. A body flip corrupts one symbol rather than the
    code that reads every symbol, so it usually still parses -- widening the span
    would bury the real repair under survivors that merely decode."""
    comp = clean()
    s = Session(comp, mode="stream")
    lo, hi = bitrot.header_span(s)
    first = min(sym.bit_start for sym in s.trace.symbols)
    assert lo == s.trace.blocks[0].bit_start
    assert hi == first, "the span has to end where the body begins"


def test_the_true_repair_is_always_among_the_survivors():
    """Guaranteed by construction: inverting the flipped bit is one of the flips
    enumerated. The measurement is how many OTHERS survive with it."""
    for offset in (0, 3, 30, 61, 100):
        damaged, bit = rotted(offset)
        s = Session(damaged, mode="stream")
        r = bitrot.repair(s, k=1)
        assert r.candidates, f"nothing survived for bit {bit}"
        assert any(bitrot.apply(s, c) == clean() for c in r.candidates), \
            f"the true undo of bit {bit} is not in the candidate set"


def test_a_flipped_bfinal_is_found_behind_the_phantom_block():
    """BFINAL rot makes the decoder read on past the end, so the trace's last
    block is a phantom and the real header is behind it. Trying each block's
    header in turn is why this is found at all."""
    comp = clean()
    damaged = bitrot.flip(comp, [0])
    s = Session(damaged, mode="stream")
    assert len(bitrot.spans(s)) > 1, "expected a phantom block to search past"
    r = bitrot.repair(s, k=1)
    assert any(bitrot.apply(s, c) == comp for c in r.candidates)


def test_survivors_are_ordered_by_what_they_recover_and_nothing_more():
    damaged, _bit = rotted(61)
    s = Session(damaged, mode="stream")
    r = bitrot.repair(s, k=1)
    knowns = [c.known for c in r.candidates]
    assert knowns == sorted(knowns, reverse=True)


def test_the_summary_never_claims_a_survivor_is_the_answer():
    """The measured finding is that small does not mean unambiguous, and the
    wording has to carry that or the route overstates itself."""
    damaged, _bit = rotted(61)
    s = Session(damaged, mode="stream")
    r = bitrot.repair(s, k=1)
    text = r.summary()
    if r.unique:
        assert "candidate rather than a certificate" in text
    else:
        assert "nothing here says which one it is" in text


def test_nothing_surviving_says_so_rather_than_guessing():
    """Two bits of damage against a one-bit search: the honest answer is that
    the span or the budget was wrong, not a candidate."""
    comp = clean()
    lo, _hi = bitrot.header_span(Session(comp, mode="stream"))
    s = Session(bitrot.flip(comp, [lo + 30, lo + 44]), mode="stream")
    r = bitrot.repair(s, k=1)
    assert not r.candidates
    assert "nothing parsed cleanly" in r.summary()
    assert "more than 1 bit" in r.summary()


def test_apply_undoes_exactly_the_flips_and_touches_nothing_else():
    damaged, bit = rotted(61)
    s = Session(damaged, mode="stream")
    r = bitrot.repair(s, k=1)
    true = next(c for c in r.candidates if bitrot.apply(s, c) == clean())
    assert true.bits == (bit,)
    assert bitrot.apply(s, true) == clean()
    assert s.data == damaged, "apply must not mutate the session"


def test_a_capture_with_nothing_parsed_returns_an_empty_result():
    s = Session(b"\xff" * 64, mode="fragment")
    r = bitrot.repair(s, k=1)
    assert isinstance(r, bitrot.Repair)
    assert r.tried >= 0 and not r.candidates


def test_progress_is_reported_and_can_stop_the_search():
    damaged, _bit = rotted(61)
    s = Session(damaged, mode="stream")
    seen = []
    bitrot.repair(s, k=1, progress=lambda d, t, n: seen.append((d, t, n)))

    class Stop(Exception):
        pass

    def cb(done, total, note):
        raise Stop(note)

    with pytest.raises(Stop):
        # A span long enough to reach the first report at 256 flips.
        bitrot.repair(s, k=1, span=(0, 4000), progress=cb)


def test_the_cap_is_reported_rather_than_silently_applied():
    damaged, _bit = rotted(61)
    s = Session(damaged, mode="stream")
    r = bitrot.repair(s, k=1, span=(0, 500), cap=10)
    assert r.capped and r.tried == 10
    assert "10 of" in r.summary()


def test_the_shipped_specimen_recovers_completely_and_verifies():
    """samples/damaged/bitrot-header.gz is the demonstration case: one flipped
    bit, nothing decoded, and a gzip CRC32 to settle whether the repair was the
    right one rather than merely a parsing one."""
    from pathlib import Path
    from toeknife import containers
    spec = Path("samples/damaged/bitrot-header.gz")
    if not spec.is_file():
        pytest.skip("run samples/build/makedamaged.py to build the specimens")
    raw = spec.read_bytes()
    st = containers.parse(raw).decodable[0]
    s = Session(st.slice(raw), mode="stream", stream=st)

    assert s.trace.status == "bad-table"
    assert s.tape.stats()["out_len"] == 0, "nothing decodes before the repair"
    assert s.verify().ok is False

    r = bitrot.repair(s, k=1)
    assert r.unique, f"expected one survivor, got {len(r.candidates)}"
    s.data = bitrot.apply(s, r.candidates[0])
    s.trace = None
    s.select_alignment(s.bit_offset)
    assert s.trace.status == "end"
    assert s.tape.stats()["out_len"] == 48_000
    assert s.verify().ok is True, "the CRC is what makes this a recovery"


# ---- the container's checksum picks among the survivors ---------------

def checked(offset: int):
    """A rotted stream inside a container that recorded its size and CRC."""
    from toeknife.containers.base import Stream
    damaged, bit = rotted(offset)
    st = Stream(name="t", container="gzip", offset=0, length=len(damaged),
                crc32=zlib.crc32(BODY) & 0xFFFFFFFF,
                uncompressed_size=len(BODY))
    return Session(damaged, mode="stream", stream=st), bit


def test_with_a_checksum_the_true_repair_is_the_one_that_matches():
    for offset in range(3, 60):
        s, bit = checked(offset)
        if s.trace.status in bitrot.LANDED:
            continue
        r = bitrot.repair(s, k=1)
        if len(r.candidates) < 2:
            continue
        assert [c.bits for c in r.matching] == [(bit,)]
        assert r.candidates[0].bits == (bit,), "a match goes first"
        assert all(c.matches is False for c in r.candidates[1:])
        assert "1 matches the recorded size and checksum" in r.summary()
        return
    pytest.fail("no header flip in range left more than one survivor")


def test_without_a_checksum_nothing_is_marked():
    damaged, _bit = rotted(61)
    r = bitrot.repair(Session(damaged, mode="stream"), k=1)
    assert all(c.matches is None for c in r.candidates)
    assert not r.checked


def test_the_command_line_applies_the_one_that_matches(tmp_path, capsys):
    """--bitrot lists the survivors and applies one only when the checksum
    picks exactly one; --verify then says whether that made the original."""
    import struct
    from toeknife.cli import main
    for offset in range(3, 60):
        damaged, bit = rotted(offset)
        if Session(damaged, mode="stream").trace.status in bitrot.LANDED:
            continue
        gz = tmp_path / "rot.gz"
        gz.write_bytes(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + damaged
                       + struct.pack("<II", zlib.crc32(BODY) & 0xFFFFFFFF,
                                     len(BODY)))
        assert main([str(gz), "--bitrot", "--verify"]) == 0
        err = capsys.readouterr().err
        assert f"applied bit {bit:,}" in err
        return
    pytest.fail("no header flip in range broke the decode")


def test_two_flipped_bits_are_found_by_the_checksum():
    """One flip finds nothing, every pair finds the two, and the CRC
    picks them out of the pairs that merely parse."""
    from toeknife.containers.base import Stream
    comp = clean()
    lo, hi = bitrot.header_span(Session(comp, mode="stream"))
    st = Stream(name="t", container="gzip", offset=0, length=len(comp),
                crc32=zlib.crc32(BODY) & 0xFFFFFFFF,
                uncompressed_size=len(BODY))
    for a, b in ((lo + 20, lo + 70), (lo + 33, hi - 9), (lo + 5, lo + 61)):
        s = Session(bitrot.flip(comp, [a, b]), stream=st)
        one = bitrot.repair(s, k=1)
        if one.matching:
            continue                    # a single flip happened to repair it
        # The span ends where the damaged parse gave up, which can be short
        # of the header's end; a second bit past that is out of reach.
        slo, shi = one.span
        if not (slo <= a < shi and slo <= b < shi):
            continue
        flips, secs = one.next_cost()
        n = shi - slo
        assert flips == n * (n - 1) // 2 and secs >= 0
        two = bitrot.repair(s, k=2)
        assert [c.bits for c in two.matching] == [(a, b)]
        return
    pytest.skip("every pair tried was repairable with one flip")
