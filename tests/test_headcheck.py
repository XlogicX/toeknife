"""Where the damage starts in a decode that stopped on it."""
import zlib

from toeknife.containers.base import Stream
from toeknife.core import headcheck
from toeknife.core.session import Session
from toeknife.core.inflate import Decoder
from toeknife.core.tape import RESUMED

from tests.test_resume import burst, text


def damaged(seed, strategy=zlib.Z_DEFAULT_STRATEGY):
    raw = text()
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 8, strategy)
    clean = co.compress(raw) + co.flush()
    blob = burst(clean, seed=seed)
    stream = Stream(name="t", container="zip", offset=0, length=len(blob),
                    crc32=zlib.crc32(raw) & 0xFFFFFFFF,
                    uncompressed_size=len(raw))
    return raw, clean, Session(blob, stream=stream)


def first_touched(session, clean):
    """Output position of the first symbol whose bits reach the burst."""
    at = next(i for i, (a, b) in enumerate(zip(clean, session.data)) if a != b)
    return next(s.out_pos for s in session.trace.symbols if s.bit_end > at * 8)


def wrong_facts(session, raw):
    tape = session.tape
    return [p for p in range(min(len(tape), len(raw)))
            if tape.prov_at(p) != RESUMED and tape.value_at(p) is not None
            and tape.value_at(p) != raw[p]]


def test_the_replay_finds_the_level_and_cuts_before_the_damage():
    raw, clean, s = damaged(1231)
    hc = s.headcheck
    # Levels 6-9 make the same parse of this text; any of them identifies it.
    assert hc is not None and hc.how == "replay" and 6 <= hc.level <= 9
    assert hc.cut <= first_touched(s, clean) <= hc.stop
    assert not wrong_facts(s, raw)
    assert all(s.tape.prov_at(p) == RESUMED for p in range(hc.cut, hc.stop))


def test_a_long_run_of_garbage_before_the_stop_is_caught():
    # Seed 1243: the garbage decoded as valid symbols for about 5,000 bytes
    # before one was invalid, and 4,780 wrong bytes went on the tape as facts.
    raw, clean, s = damaged(1243)
    assert s.headcheck.how == "replay"
    assert s.headcheck.stop - first_touched(s, clean) > 4000
    assert not wrong_facts(s, raw)


def test_an_encoder_the_replay_cannot_reproduce_falls_back_to_the_block():
    raw, clean, s = damaged(1231, zlib.Z_HUFFMAN_ONLY)
    assert s.trace.status not in headcheck.CLEAN
    hc = s.headcheck
    assert hc.how == "block"
    assert hc.cut <= first_touched(s, clean)
    assert not wrong_facts(s, raw)


def test_a_clean_decode_has_nothing_unproven():
    raw = text(5000)
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    blob = co.compress(raw) + co.flush()
    s = Session(blob, stream=Stream(name="t", container="zip", offset=0,
                                    length=len(blob),
                                    uncompressed_size=len(raw)))
    assert s.headcheck is None
    assert RESUMED not in s.tape.prov


def test_header_damage_leaves_no_wrong_byte_as_fact():
    # A wrong table garbles the block from its first symbol, and the replay
    # reproduces the garbled parse for a while; stack-rot.gz showed `^olume`
    # and `BugTraz` in green before the cut until the block-start rule.
    raw = text()
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    clean = co.compress(raw) + co.flush()
    first = Decoder(clean).stream(0).symbols[0].bit_start
    stopped = 0
    for bit in range(3, first):
        bad = bytearray(clean)
        bad[bit >> 3] ^= 1 << (bit & 7)
        tr = Decoder(bytes(bad), strict_window=True).stream(0)
        hc = headcheck.check(tr)
        if hc is None:
            continue
        stopped += 1
        tape = headcheck.tape_with_cut(tr.symbols, hc.cut)
        assert not [p for p in range(min(len(tape), len(raw)))
                    if tape.prov_at(p) != RESUMED
                    and tape.value_at(p) != raw[p]], f"bit {bit}"
    assert stopped > 10
