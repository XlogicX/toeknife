"""Reading on past mid-block damage."""
import random
import zlib

import pytest

from toeknife.containers.base import Stream
from toeknife.core import copyfill, resume
from toeknife.core.session import Session
from toeknife.core.tape import COPY, LITERAL, RESUMED, UNKNOWN
from toeknife.predict.crcsolve import solve_by_shortlist, solve_from_crc

def text(n=60000, seed=7):
    """Words drawn from a 3,000-word made-up vocabulary.

    A small vocabulary is no good here: its codes are so short and its
    distances so near that a 200-byte burst decodes as valid symbols to the
    end, and the decode never stops at all.
    """
    rng = random.Random(seed)
    letters = "etaoinshrdlucmfwypvbgkjqxz"
    vocab = ["".join(rng.choice(letters) for _ in range(rng.randint(2, 9)))
             for _ in range(3000)]
    out, size = [], 0
    while size < n:
        w = rng.choice(vocab)
        out.append(w)
        size += len(w) + 1
    return " ".join(out).encode()[:n]


def burst(blob, count=200, seed=1234):
    rng = random.Random(seed)
    blob = bytearray(blob)
    start = len(blob) // 2
    for i in range(count):
        blob[start + i] = rng.randrange(256)
    return bytes(blob)


@pytest.fixture(scope="module")
def specimen():
    raw = text()
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    blob = burst(co.compress(raw) + co.flush())
    stream = Stream(name="t", container="zip", offset=0, length=len(blob),
                    crc32=zlib.crc32(raw) & 0xFFFFFFFF,
                    uncompressed_size=len(raw))
    return raw, blob, stream


def fresh(specimen):
    raw, blob, stream = specimen
    return Session(blob, stream=stream)


def test_the_damaged_decode_stops_and_the_route_applies(specimen):
    s = fresh(specimen)
    assert s.trace.status not in ("end", "eob", "truncated")
    ok, why = resume.applies(s)
    assert ok, why


def test_the_tail_is_placed_by_the_recorded_size(specimen):
    raw = specimen[0]
    s = fresh(specimen)
    head = len(s.tape)
    got = resume.resume(s)
    assert got.trace is not None, got.reason
    resume.apply(s, got)
    assert len(s.tape) == len(raw)
    # Every literal read after the true resynchronisation lands on its true
    # position. Garbage decoded from the burst sits near the stop, so the
    # last half of the tail is far past it.
    late = [sym for sym in s.trace.symbols
            if sym.kind == "literal" and sym.out_pos > head + len(raw) // 4]
    assert late
    assert all(raw[sym.out_pos] == sym.literal for sym in late)


def test_everything_read_past_the_stop_is_unproven(specimen):
    s = fresh(specimen)
    head = len(s.tape)
    cut = s.headcheck.cut                   # the run before the stop
    resume.apply(s, resume.resume(s))
    tape = s.tape
    assert all(tape.prov_at(p) in (LITERAL, COPY) for p in range(cut))
    assert all(tape.prov_at(p) == RESUMED for p in range(cut, head))
    assert all(tape.prov_at(p) in (RESUMED, UNKNOWN)
               for p in range(head, len(tape)))
    assert any(tape.prov_at(p) == RESUMED for p in range(head, len(tape)))


def test_the_checksum_is_not_solved_around_unproven_bytes(specimen):
    s = fresh(specimen)
    resume.apply(s, resume.resume(s))
    assert not solve_from_crc(s).ok
    assert "unproven" in solve_from_crc(s).reason
    assert not solve_by_shortlist(s, model=None).ok


def test_the_original_copy_is_proven_on_its_own(specimen):
    raw = specimen[0]
    s = fresh(specimen)
    resume.apply(s, resume.resume(s))
    r = copyfill.fill(s, raw)
    assert r.copy_matches is True
    assert r.contradictions                 # the damage, shown where it is
    assert "it is the original" in r.summary()


def test_an_edited_copy_is_not_proven(specimen):
    raw = bytearray(specimen[0])
    raw[100] ^= 1
    s = fresh(specimen)
    resume.apply(s, resume.resume(s))
    assert copyfill.fill(s, bytes(raw)).copy_matches is False


def test_a_clean_decode_does_not_offer_it():
    raw = text(5000)
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    blob = co.compress(raw) + co.flush()
    stream = Stream(name="t", container="zip", offset=0, length=len(blob),
                    crc32=zlib.crc32(raw) & 0xFFFFFFFF,
                    uncompressed_size=len(raw))
    ok, why = resume.applies(Session(blob, stream=stream))
    assert not ok and "cleanly" in why


def test_past_the_damaged_block_a_later_header_takes_over():
    # Seed 1255: no reading with the stopped block's table reaches the end
    # (one used to stop 30 KB early on a final block decoded from garbage).
    raw = text()
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    blob = burst(co.compress(raw) + co.flush(), seed=1255)
    stream = Stream(name="t", container="zip", offset=0, length=len(blob),
                    crc32=zlib.crc32(raw) & 0xFFFFFFFF,
                    uncompressed_size=len(raw))
    s = Session(blob, stream=stream)
    got = resume.resume(s)
    assert got.trace is not None and got.header
    resume.apply(s, got)
    assert len(s.tape) == len(raw)
    late = [sym for sym in s.trace.symbols
            if sym.kind == "literal" and sym.out_pos >= len(raw) - 10000]
    assert late and all(raw[sym.out_pos] == sym.literal for sym in late)


def test_without_a_recorded_size_it_says_so(specimen):
    raw, blob, stream = specimen
    bare = Stream(name="t", container="zip", offset=0, length=len(blob),
                  crc32=stream.crc32)
    got = resume.resume(Session(blob, stream=bare))
    assert got.trace is None and "size" in got.reason


def test_a_copy_is_found_by_its_checksum_when_nothing_decoded():
    """A destroyed table decodes to nothing, so nothing on the tape can line a
    copy up; the recorded size and CRC32 find the slice of it that matches."""
    raw = text(20000, seed=3)
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    blob = bytearray(co.compress(raw) + co.flush())
    for i in range(10, 70):
        blob[i] ^= 0xFF
    stream = Stream(name="t", container="gzip", offset=0, length=len(blob),
                    crc32=zlib.crc32(raw) & 0xFFFFFFFF,
                    uncompressed_size=len(raw))
    s = Session(bytes(blob), stream=stream)
    assert len(s.tape) == 0
    copy = b"x" * 1234 + raw + b"y" * 99
    r = copyfill.fill(s, copy)
    assert (r.window, r.offset, r.copy_matches) == (True, 1234, True)
    assert "bytes 1,234-21,233" in r.summary()
    edited = bytearray(copy)
    edited[5000] ^= 1
    assert copyfill.fill(Session(bytes(blob), stream=stream),
                         bytes(edited)).offset is None


def test_the_rolling_crc_finds_what_brute_force_finds():
    rng = random.Random(9)
    for _ in range(200):
        c = bytes(rng.randrange(256) for _ in range(rng.randrange(1, 200)))
        n = rng.randrange(1, len(c) + 1)
        i = rng.randrange(0, len(c) - n + 1)
        crc = zlib.crc32(c[i:i + n])
        first = next(j for j in range(len(c) - n + 1)
                     if zlib.crc32(c[j:j + n]) == crc)
        assert copyfill.find_window(c, n, crc) == first
