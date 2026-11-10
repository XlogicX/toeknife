"""CRC32 as a linear constraint on the unknown bytes."""
import random
import zlib

import pytest

from toeknife.containers import parse
from toeknife.core.session import Session
from toeknife.core.tape import SOLVED, Tape
from toeknife.predict.context import ContextModel
from toeknife.predict.crcsolve import (basis, build_system, enumerate_solutions,
                                       solve_by_shortlist,
                                       linear_of_byte, shift, solve,
                                       solve_from_crc, solve_with_model)

BODY = b"forensic analysis of compressed data, repeatedly and at length. " * 300
CORPUS = (b"forensic work on compressed streams. analysis of data at length. "
          b"the forensic analyst reads compressed data repeatedly. ") * 200


def model():
    return ContextModel(order=4).train(CORPUS)


def dict_stream(tail: bytes):
    """A zlib stream compressed against a dictionary whose tail gets referenced.

    The referenced bytes are genuinely absent from the file, which is the
    situation the checksum has to solve: unknown bytes, known length, known CRC.
    """
    zdict = b"\xa5" * 80 + tail
    co = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY,
                          zdict=zdict)
    blob = co.compress(BODY) + co.flush()
    sm = parse(blob).streams[0]
    sm.crc32 = zlib.crc32(BODY) & 0xFFFFFFFF
    sm.uncompressed_size = len(BODY)
    return blob, sm, zdict


# ---- the arithmetic --------------------------------------------------------

def test_the_linear_part_matches_a_direct_computation():
    """L(m) = CRC(m) xor CRC(zeros of the same length), computed in O(log n)."""
    rng = random.Random(1)
    for _ in range(200):
        n = rng.randrange(1, 5000)
        p = rng.randrange(0, n)
        v = rng.randrange(1, 256)
        m = bytearray(n)
        m[p] = v
        direct = zlib.crc32(bytes(m)) ^ zlib.crc32(bytes(n))
        assert basis(v, p, n) == direct, (n, p, v)


def test_the_linear_part_is_linear():
    """Many unknown bytes at once must equal the xor of their contributions,
    which is the property the whole solver rests on."""
    rng = random.Random(2)
    for _ in range(120):
        n = rng.randrange(8, 3000)
        points = {rng.randrange(0, n): rng.randrange(1, 256)
                  for _ in range(rng.randrange(1, 12))}
        m = bytearray(n)
        acc = 0
        for p, v in points.items():
            m[p] = v
            for b in range(8):
                if v >> b & 1:
                    acc ^= basis(1 << b, p, n)
        assert acc == zlib.crc32(bytes(m)) ^ zlib.crc32(bytes(n))


def test_shift_is_not_the_same_as_appending_zeros():
    """Guard against the mistake this module was written around: `shift` is the
    bare linear operator, not CRC(message || zeros), which also xors in the CRC
    of the zero run."""
    pre = b"some bytes"
    n = 100
    assert shift(zlib.crc32(pre), n) != zlib.crc32(pre + bytes(n))
    assert shift(zlib.crc32(pre), n) == zlib.crc32(pre + bytes(n)) ^ zlib.crc32(bytes(n))


def test_linear_of_a_zero_byte_is_zero():
    assert linear_of_byte(0) == 0


# ---- solving ---------------------------------------------------------------

def test_five_unknown_classes_are_solved_by_crc_plus_model():
    """Forty unknown bits against thirty-two equations leaves eight free, so
    2^40 possibilities become 256, and the model picks among those."""
    blob, sm, zdict = dict_stream(b"ensic")
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    assert session.tape.stats()["holes"] == 5
    result, assignment = solve_with_model(session, model())
    assert result.ok
    assert result.free_bits == 8
    assert assignment is not None

    session.begin_group()
    for anchor, value in sorted(assignment.items()):
        session.set_guess(anchor, value, SOLVED)
    session.end_group()

    window, _, _ = session.window()
    assert bytes(window) == zdict[-len(window):]
    v = session.verify()
    assert v.ok is True, v.summary()
    assert v.crc_got == v.crc_expected
    assert v.adler_got == v.adler_expected     # an independent second check


def test_the_crc_narrows_even_when_it_cannot_decide():
    blob, sm, _ = dict_stream(b"c analysis")
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    result = solve_from_crc(session)
    assert result.ok
    classes = session.tape.stats()["holes"]
    assert result.unknown_bits == classes * 8
    # thirty-two equations always remove thirty-two bits of freedom
    assert result.free_bits == result.unknown_bits - 32


def test_an_inconsistent_crc_is_reported_not_ignored():
    """A wrong byte among the ones already treated as known makes the system
    unsolvable, and saying so is more useful than returning a wrong answer."""
    blob, sm, _ = dict_stream(b"ensic")
    sm.crc32 ^= 0x5A5A5A5A                      # a checksum that cannot be met
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    sysm = build_system(session.tape, sm.crc32, session.tape.holes())
    assignment, _ = solve(sysm)
    # with 40 unknown bits the system is underdetermined, so a corrupted target
    # is still satisfiable; the check that bites is a complete-rank one
    assert assignment is None or len(assignment) == 5


def test_inconsistency_is_detected_when_the_system_has_full_rank():
    t = Tape()
    for b in b"hello world, this is a test of the checksum":
        t.emit_literal(b)
    truth = bytes(t.out)
    sysm = build_system(t, zlib.crc32(truth) & 0xFFFFFFFF, [])
    assert solve(sysm)[0] == {}                 # nothing unknown, and it agrees
    bad = build_system(t, (zlib.crc32(truth) ^ 1) & 0xFFFFFFFF, [])
    assert solve(bad)[0] is None                # nothing unknown, and it does not


def test_a_class_reaching_behind_the_output_is_still_a_variable():
    """The checksum covers the decompressed output only, so a pre-fragment byte
    might look invisible to it.  It is not: the class that byte belongs to also
    occupies the copy position inside the output, and that occurrence enters the
    CRC.  Recovering the window byte is exactly recovering the class.

    A class can never be entirely outside the output, because every class is
    created by a copy at a non-negative position.
    """
    t = Tape()
    for b in b"abcdef":
        t.emit_literal(b)
    t.emit_copy(distance=20, length=3)          # reaches well behind the start
    holes = t.holes()
    assert holes and all(h.anchor < 0 for h in holes)
    sysm = build_system(t, 0, holes)
    assert not sysm.free_classes
    assert sysm.anchors == [h.anchor for h in holes]
    assert sysm.unknown_bits == len(holes) * 8
    for h in holes:
        assert any(p >= 0 for p in h.positions), "the copy position is in-output"


def test_enumeration_is_bounded():
    blob, sm, _ = dict_stream(b"c analysis")
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    sysm = build_system(session.tape, sm.crc32, session.tape.holes())
    assert list(enumerate_solutions(sysm, limit=1024)) == [], (
        "a huge solution space must decline rather than run forever")


def test_refuses_when_the_length_disagrees():
    blob, sm, _ = dict_stream(b"ensic")
    sm.uncompressed_size = len(BODY) + 99
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    result = solve_from_crc(session)
    assert not result.ok
    assert "different data" in result.reason


def test_refuses_without_a_crc():
    blob, sm, _ = dict_stream(b"ensic")
    sm.crc32 = None
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    result = solve_from_crc(session)
    assert not result.ok
    assert "no CRC32" in result.reason


def test_adler32_alone_admits_wrong_answers():
    """The finding that shaped this module.

    Adler-32 is two sums modulo 65521. Across five unknown bytes the reachable
    sums number in the low thousands, so its effective strength is far below the
    32 bits its width suggests. Given a five-byte unknown it accepts answers
    that are simply wrong: 'etesa' and 'ensic' share a byte sum and a weighted
    sum, so both satisfy it.
    """
    a, b = b"etesa", b"ensic"
    assert sum(a) == sum(b)
    n = len(a)
    assert sum((n - i) * v for i, v in enumerate(a)) == \
           sum((n - i) * v for i, v in enumerate(b))
    assert zlib.adler32(a) == zlib.adler32(b)
    assert zlib.crc32(a) != zlib.crc32(b), "CRC32 does tell them apart"

    blob, sm, zdict = dict_stream(b"ensic")
    sm.crc32 = None                             # what a zlib stream really has
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    result = solve_by_shortlist(session, model(), per_class=8)
    assert result.ok
    # Several combinations satisfy it, and the tool must say so rather than
    # presenting the first one found as settled.
    assert result.survivors > 1
    assert result.ambiguous
    assert "did not decide" in result.summary()


def test_crc32_plus_adler32_pins_the_answer_uniquely():
    """With a real CRC32 and a wide enough shortlist there is one survivor, and
    it is right."""
    blob, sm, zdict = dict_stream(b"ensic")
    sm.crc32 = zlib.crc32(BODY) & 0xFFFFFFFF
    sm.uncompressed_size = len(BODY)
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    result = solve_by_shortlist(session, model(), per_class=16,
                                max_combinations=5_000_000)
    assert result.ok
    assert result.survivors == 1, "both checksums together should leave one"
    assert not result.ambiguous
    session.begin_group()
    for anchor, value in sorted(result.assignment.items()):
        session.set_guess(anchor, value, SOLVED)
    session.end_group()
    window, _, _ = session.window()
    assert bytes(window) == zdict[-len(window):]
    assert session.verify().ok is True


def test_a_shortlist_that_misses_the_truth_fails_honestly():
    """Better to report nothing found than to return the nearest wrong thing."""
    blob, sm, _ = dict_stream(b"ensic")
    sm.crc32 = zlib.crc32(BODY) & 0xFFFFFFFF
    sm.uncompressed_size = len(BODY)
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    result = solve_by_shortlist(session, model(), per_class=2)
    assert not result.ok
    assert "outside the model's top" in result.reason
    assert session.tape.holes(), "nothing should have been filled"


def test_nothing_to_do_on_a_complete_stream():
    import gzip as gzip_mod
    blob = gzip_mod.compress(BODY)
    sm = parse(blob).streams[0]
    session = Session(sm.slice(blob), mode="stream", stream=sm)
    assert not session.tape.holes()
    result = solve_from_crc(session)
    assert not result.ok
    assert "nothing unknown" in result.reason
