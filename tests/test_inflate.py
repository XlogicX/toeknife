"""Differential tests: our decoder must agree with zlib, byte for byte."""
import random
import zlib

import pytest

from toeknife.core.inflate import Decoder
from toeknife.core.tables import (BTYPE_DYNAMIC, BTYPE_FIXED, BTYPE_STORED,
                                  DISTANCE_BASE, DISTANCE_EXTRA, LENGTH_BASE,
                                  LENGTH_EXTRA)


def raw_deflate(data: bytes, level=6, strategy=zlib.Z_DEFAULT_STRATEGY,
                flushes: int = 0) -> bytes:
    co = zlib.compressobj(level, zlib.DEFLATED, -15, 9, strategy)
    if not flushes:
        return co.compress(data) + co.flush()
    out = b""
    step = max(1, len(data) // (flushes + 1))
    for i in range(0, len(data), step):
        out += co.compress(data[i:i + step])
        out += co.flush(zlib.Z_FULL_FLUSH)
    return out + co.flush()


def corpus(rng: random.Random) -> list[bytes]:
    words = [b"forensic", b"deflate", b"huffman", b"analysis", b"toeknife",
             b"the", b"a", b" ", b".", b"\n", b"0123", b"\x00\x01\x02"]
    samples = [
        b"",
        b"a",
        b"a" * 300,                                     # long overlapping match
        b"ab" * 500,                                    # distance 2 overlap
        bytes(range(256)) * 4,
        b"Example sentance for forensic analysis, " * 20,
        bytes(rng.randrange(256) for _ in range(5000)),  # incompressible
        b"".join(rng.choice(words) for _ in range(2000)),
        bytes(rng.choice((0, 0, 0, 0xFF, 0x90)) for _ in range(3000)),
    ]
    # something that forces a large distance, exercising distance code 29
    big = bytes(rng.randrange(256) for _ in range(30000))
    samples.append(big + b"MARKERMARKERMARKER" + big[:200] + b"MARKERMARKERMARKER")
    return samples


@pytest.mark.parametrize("level,strategy", [
    (0, zlib.Z_DEFAULT_STRATEGY),      # stored blocks
    (6, zlib.Z_FIXED),                 # fixed Huffman
    (9, zlib.Z_DEFAULT_STRATEGY),      # dynamic Huffman
    (1, zlib.Z_HUFFMAN_ONLY),          # literals only, empty distance tree
    (6, zlib.Z_RLE),                   # distance 1 only
])
def test_matches_zlib(level, strategy):
    rng = random.Random(1234)
    for sample in corpus(rng):
        comp = raw_deflate(sample, level, strategy)
        trace = Decoder(comp).stream()
        assert trace.status == "end", (
            f"level={level} strategy={strategy} len={len(sample)} "
            f"status={trace.status} error={trace.error}")
        assert bytes(trace.tape.out) == sample, (
            f"output mismatch, level={level} strategy={strategy} len={len(sample)}")
        assert not trace.tape.holes(), "a complete stream must have no holes"


def test_multiple_blocks():
    rng = random.Random(7)
    data = b"".join(rng.choice([b"alpha ", b"beta ", b"gamma "]) for _ in range(4000))
    comp = raw_deflate(data, 6, zlib.Z_DEFAULT_STRATEGY, flushes=4)
    trace = Decoder(comp).stream()
    assert trace.status == "end"
    assert bytes(trace.tape.out) == data
    assert len(trace.blocks) > 1


def test_block_types_all_reachable():
    seen = set()
    rng = random.Random(1)
    for level, strategy, data in [
        (0, zlib.Z_DEFAULT_STRATEGY, bytes(rng.randrange(256) for _ in range(500))),
        (6, zlib.Z_FIXED, b"hello hello hello world"),
        # zlib only bothers with a dynamic table once there is enough varied
        # material to make one pay for itself.
        (9, zlib.Z_DEFAULT_STRATEGY,
         bytes(rng.choice(b"abcdefgh \n.,0123") for _ in range(4000))),
    ]:
        trace = Decoder(raw_deflate(data, level, strategy)).stream()
        assert trace.status == "end"
        seen.update(b.btype for b in trace.blocks)
    assert {BTYPE_STORED, BTYPE_FIXED, BTYPE_DYNAMIC} <= seen


def test_symbol_bit_spans_are_contiguous_and_exact():
    """Every symbol's bits must tile the block body with no gaps or overlaps."""
    data = b"Example sentance for forensic analysis, which is an example sentance."
    for strategy in (zlib.Z_FIXED, zlib.Z_DEFAULT_STRATEGY):
        trace = Decoder(raw_deflate(data, 9, strategy)).stream()
        syms = [s for s in trace.symbols if trace.blocks[s.block].btype != BTYPE_STORED]
        for a, b in zip(syms, syms[1:]):
            assert a.bit_end == b.bit_start, "gap or overlap between symbols"
        for s in syms:
            if s.fields:
                assert s.fields[0].start == s.bit_start
                assert s.fields[-1].end == s.bit_end
                for f, g in zip(s.fields, s.fields[1:]):
                    assert f.end == g.start


def test_every_symbol_records_which_code_carried_it():
    """The alphabet entry that was read, not just what it meant.

    They are not the same question, and the value does not always determine the
    code: length 258 is legal as symbol 285 or as 284 with its five extra bits
    maxed. Anything counting per-symbol frequency has to be told which one was
    on the wire.
    """
    data = b"forensic analysis of forensic analysis of compressed evidence" * 4
    for strategy in (zlib.Z_FIXED, zlib.Z_DEFAULT_STRATEGY):
        trace = Decoder(raw_deflate(data, 9, strategy)).stream()
        saw = {"literal": 0, "match": 0, "eob": 0}
        for s in trace.symbols:
            if trace.blocks[s.block].btype == BTYPE_STORED:
                continue          # no Huffman code was involved
            saw[s.kind] += 1
            assert s.code_sym is not None, f"{s.kind} with no litlen symbol"
            if s.kind == "literal":
                assert s.code_sym == s.literal
                assert s.dist_sym is None
            elif s.kind == "eob":
                assert s.code_sym == 256
                assert s.dist_sym is None
            else:
                assert 257 <= s.code_sym <= 285
                assert s.dist_sym is not None and 0 <= s.dist_sym <= 29
                # The recorded code has to be the one whose range holds the
                # length that came out of it.
                base = LENGTH_BASE[s.code_sym - 257]
                span = 1 << LENGTH_EXTRA[s.code_sym - 257]
                assert base <= s.length <= base + span - 1
                dbase = DISTANCE_BASE[s.dist_sym]
                dspan = 1 << DISTANCE_EXTRA[s.dist_sym]
                assert dbase <= s.distance <= dbase + dspan - 1
        assert saw["literal"] and saw["match"] and saw["eob"]


def test_a_stored_block_records_no_huffman_symbol():
    """There was no code, so claiming one would be inventing it."""
    data = bytes(random.Random(4).getrandbits(8) for _ in range(4000))
    trace = Decoder(zlib.compress(data, 0)[2:-4]).stream()
    stored = [b for b in trace.blocks if b.btype == BTYPE_STORED]
    assert stored, "expected level 0 to store"
    for s in trace.symbols:
        if trace.blocks[s.block].btype == BTYPE_STORED:
            assert s.code_sym is None and s.dist_sym is None


def test_output_positions_track_the_tape():
    data = b"one two three one two three one two three four"
    trace = Decoder(raw_deflate(data, 9, zlib.Z_FIXED)).stream()
    for s in trace.symbols:
        if s.kind == "literal":
            assert trace.tape.out[s.out_pos] == s.literal
        elif s.kind == "match":
            src = s.out_pos - s.distance
            assert trace.tape.out[s.out_pos:s.out_pos + s.length] == \
                   data[s.out_pos:s.out_pos + s.length]
            assert src >= 0


def test_distance_code_29_decodes():
    """The original tool built only distance codes 0..28 and silently failed here."""
    # The filler has to be compressible: against 26K of pure entropy zlib's
    # hash chains saturate and it never looks far enough back to emit the match.
    filler = (b"the quick brown fox jumps over the lazy dog. " * 600)[:26000]
    tag = b"DISTANCE-CODE-29-CANARY"
    data = tag + filler + tag
    comp = raw_deflate(data, 9, zlib.Z_DEFAULT_STRATEGY)
    trace = Decoder(comp).stream()
    assert trace.status == "end"
    assert bytes(trace.tape.out) == data
    assert any(s.kind == "match" and s.distance > 24576 for s in trace.symbols), \
        "test did not actually produce a distance-code-29 match"
