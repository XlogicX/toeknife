"""Resynchronising a fragment whose own Huffman table is gone."""
import random
import zlib

import pytest

from toeknife.core.inflate import Decoder
from toeknife.core.scan import KIND_DYNAMIC, KIND_SYNC, scan
from toeknife.core.session import Session


def bits_to_bytes(bits):
    out = bytearray((len(bits) + 7) // 8)
    for i, b in enumerate(bits):
        if b:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def num(v, n):
    return [(v >> i) & 1 for i in range(n)]


def varied_source(size=250000):
    rng = random.Random(3)
    words = b"the quick brown fox jumps over lazy dogs while analysis proceeds ".split()
    body = b" ".join(rng.choice(words) for _ in range(size // 6))
    return (body + bytes(rng.randrange(256) for _ in range(2000)))[:size]


def streaming_deflate(src, every=8192, level=6):
    co = zlib.compressobj(level, zlib.DEFLATED, -15, 9)
    out = b""
    for i in range(0, len(src), every):
        out += co.compress(src[i:i + every]) + co.flush(zlib.Z_SYNC_FLUSH)
    return out + co.flush()


# ---- RFC 1951 field bounds -------------------------------------------------

@pytest.mark.parametrize("hlit_raw,hdist_raw,why", [
    (31, 0, "HLIT 288 exceeds the 286 codes the format allows"),
    (30, 0, "HLIT 287 exceeds the 286 codes the format allows"),
    (0, 31, "HDIST 32 exceeds the 30 codes the format allows"),
    (0, 30, "HDIST 31 exceeds the 30 codes the format allows"),
])
def test_out_of_range_counts_are_rejected_like_zlib(hlit_raw, hdist_raw, why):
    """zlib: "too many length or distance symbols" (puff.c: nlen>286 || ndist>30).

    The fields are five bits wide so they can encode counts the format forbids,
    and accepting them is a real source of false block headers when scanning
    damaged data.
    """
    bits = [1] + [0, 1] + num(hlit_raw, 5) + num(hdist_raw, 5) + num(0, 4)
    bits += num(1, 3) * 4
    data = bits_to_bytes(bits + [0] * 64)

    with pytest.raises(zlib.error):
        zlib.decompressobj(-15).decompress(data)
    trace = Decoder(data).stream(0)
    assert trace.status == "bad-table", why


def test_in_range_counts_still_work():
    src = b"forensic analysis of compressed data, repeatedly and at length. " * 60
    comp = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    blob = comp.compress(src) + comp.flush()
    trace = Decoder(blob).stream()
    assert trace.status == "end"
    assert bytes(trace.tape.out) == src
    for b in trace.blocks:
        if b.btype == 2:
            assert 257 <= b.hlit <= 286 and 1 <= b.hdist <= 30


# ---- finding a boundary ----------------------------------------------------

def test_sync_flush_marker_is_found_and_is_byte_aligned():
    src = varied_source(120000)
    comp = streaming_deflate(src)
    frag = comp[len(comp) // 3:]
    points = scan(frag, kinds=(KIND_SYNC,), max_bit=200000)
    assert points, "a streaming deflate should be full of flush markers"
    best = points[0]
    assert best.kind == KIND_SYNC
    assert best.bit_offset % 8 == 0, "a flush marker leaves the stream byte-aligned"


def test_resync_recovers_only_correct_bytes():
    """Everything the resync reports as known must actually be right.

    Coverage is data-dependent, and on back-reference-heavy input most of the
    output stays unknown because it copies from behind the resync point.  What
    must never happen is a byte confidently reported as known and wrong.
    """
    src = varied_source(200000)
    comp = streaming_deflate(src)
    frag = comp[len(comp) // 3:]
    points = scan(frag, kinds=(KIND_SYNC,), max_bit=200000)
    assert points
    session = Session(frag, mode="stream", bit_offset=points[0].bit_offset)
    buf, mask = session.recovered()
    known = [i for i, m in enumerate(mask) if m]
    assert len(known) > 1000, "should recover a meaningful amount"

    runs, run = [], []
    for i in known:
        if run and i == run[-1] + 1:
            run.append(i)
        else:
            runs.append(run)
            run = [i]
    runs.append(run)
    longest = max(runs, key=len)
    seg = bytes(buf[longest[0]:longest[-1] + 1])
    offset = src.find(seg)
    assert offset >= 0, "the longest recovered run must appear in the original"

    shift = offset - longest[0]
    wrong = sum(1 for i in known
                if 0 <= i + shift < len(src) and buf[i] != src[i + shift])
    assert wrong == 0, f"{wrong} known bytes disagree with the original"


def test_holes_compress_the_problem():
    """Unknowns are byte classes, not positions: that ratio is the payoff."""
    src = varied_source(200000)
    comp = streaming_deflate(src)
    frag = comp[len(comp) // 3:]
    points = scan(frag, kinds=(KIND_SYNC,), max_bit=200000)
    session = Session(frag, mode="stream", bit_offset=points[0].bit_offset)
    st = session.tape.stats()
    assert st["holes"] < st["hole_positions"], "classes should cover more than themselves"
    assert st["hole_positions"] / st["holes"] > 2


def test_resumed_streams_allow_references_behind_the_start():
    """A distance reaching behind the output is corruption at the head of a
    stream, and completely normal when resuming from a sync point."""
    src = varied_source(120000)
    comp = streaming_deflate(src)
    frag = comp[len(comp) // 3:]
    points = scan(frag, kinds=(KIND_SYNC,), max_bit=200000)
    session = Session(frag, mode="stream", bit_offset=points[0].bit_offset)
    assert session.resumed
    assert session.trace.status == "end"
    assert len(session.tape) > 1000

    whole = Session(comp, mode="stream")
    assert not whole.resumed
    assert bytes(whole.tape.out) == src
    assert not whole.tape.holes()


def test_noise_yields_no_usable_boundary():
    rng = random.Random(11)
    noise = bytes(rng.randrange(256) for _ in range(20000))
    points = scan(noise, max_bit=60000, min_out=256)
    assert not points, f"scanned noise should not decode to real output: {points[:2]}"


def test_validation_is_what_filters_dynamic_false_positives():
    rng = random.Random(5)
    noise = bytes(rng.randrange(256) for _ in range(20000))
    raw = scan(noise, kinds=(KIND_DYNAMIC,), max_bit=60000, validate=False)
    checked = scan(noise, kinds=(KIND_DYNAMIC,), max_bit=60000, min_out=256)
    assert len(raw) > len(checked), (
        "a dynamic header parses out of noise now and then; decode-validation "
        "is what removes it")


def test_resync_to_reopens_a_capture_as_a_stream():
    """The f3 path: scanning finds a block boundary, and resyncing there turns
    an unreadable mid-stream capture into a decoded one.

    Built the way a stream has to be written for this to pay off completely --
    Z_FIXED so the tables come from RFC 1951 rather than a lost header, and
    Z_FULL_FLUSH so the LZ77 window resets and no back-reference reaches behind
    the damage.  Both are the writer's choices, which is the point: the same cut
    on a sync-flushed stream leaves holes.
    """
    import zlib

    from toeknife.core.scan import scan
    from toeknife.core.session import Session

    raw = (b"the quick brown fox jumps over the lazy dog. " * 400 +
           b"pack my box with five dozen liquor jugs. " * 400)
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9, zlib.Z_FIXED)
    out = bytearray()
    for i in range(0, len(raw), 4096):
        out += co.compress(raw[i:i + 4096])
        out += co.flush(zlib.Z_FULL_FLUSH)
    out += co.flush()
    comp = bytes(out)
    assert zlib.decompress(comp, -15) == raw

    frag = comp[len(comp) // 3:]
    session = Session(frag)                    # as `toeknife FILE` opens it
    assert session.mode == "fragment"
    before = session.tape.stats()

    points = scan(frag)
    assert points, "no boundary found in a sync-marked stream"

    session.resync_to(points[0].bit_offset)
    after = session.tape.stats()

    assert session.mode == "stream"
    # Complete: every byte known, nothing left to fill, and nothing reaching
    # behind the start.
    assert after["holes"] == 0
    assert after["known"] == after["out_len"]
    assert after["min_pos"] == 0
    # Strictly better, and the size of the win is a property of the capture
    # rather than of resyncing: 4.9x on this small payload, 74x on
    # samples/fixed-fragment.bin.  Completeness above is the claim; this only
    # pins that the fragment path was the worse one.
    assert after["known"] > before["known"]
    buf, _mask = session.tape.recovered()
    assert bytes(buf) == raw[len(raw) - after["out_len"]:]


def test_resync_drops_guesses_that_no_longer_land_on_a_hole():
    """Output positions do not survive a change of starting point, so a guess
    made before a resync must not be silently re-seated onto a different byte."""
    import zlib

    from toeknife.core.scan import scan
    from toeknife.core.session import Session

    raw = b"the quick brown fox jumps over the lazy dog. " * 400
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9)
    out = bytearray()
    for i in range(0, len(raw), 4096):
        out += co.compress(raw[i:i + 4096])
        out += co.flush(zlib.Z_SYNC_FLUSH)
    out += co.flush()
    frag = bytes(out)[len(bytes(out)) // 3:]

    session = Session(frag)
    holes = session.tape.holes()
    if holes:
        session.set_guess(holes[0].anchor, ord("x"))
        assert session.guesses
    points = scan(frag)
    assert points
    session.resync_to(points[0].bit_offset)
    # Whatever survived must still be sitting on a real position, not a stale one.
    for pos, value in session.guesses.items():
        assert session.tape.value_at(pos) == value


def test_the_lost_table_specimen_recovers_almost_all_of_it():
    """samples/damaged/lost-table.bin: raw deflate, no container, and the first
    block's Huffman table destroyed.

    The point of the specimen is the contrast with a single-block stream, where
    the same damage is fatal: there the destroyed header is the only one and
    nothing downstream can be read. Here the encoder flushed, so every block
    carries its own table and losing the first costs its own 2,048 bytes and
    nothing else.
    """
    from pathlib import Path
    from toeknife.core.scan import scan
    from toeknife.core.session import Session
    spec = Path("samples/damaged/lost-table.bin")
    truth = Path("samples/damaged/lost-table.payload")
    if not spec.is_file() or not truth.is_file():
        pytest.skip("run samples/build/makedamaged.py to build the specimens")
    # The payload is read from disk, NOT rebuilt by calling makedamaged.docs().
    # docs() concatenates every file in docs/*.md, so it changes whenever the
    # documentation does -- adding one .md file shifted the payload and this
    # assertion failed at the offset where the new file landed, which reads
    # exactly like a decoder regression and is not one. The specimen is a fixed
    # artefact and its truth has to be fixed alongside it.
    payload = truth.read_bytes()

    data = spec.read_bytes()
    points = scan(data)
    assert points, "the flush markers have to be findable"
    best = points[0]
    s = Session(data)
    s.resync_to(best.bit_offset)
    got = bytes(s.tape.recovered()[0][:s.tape.length])

    assert s.tape.stats()["holes"] == 0, "a full flush leaves nothing dangling"
    assert got == payload[2048:], "everything after the dead block, exactly"
    assert len(got) == 45_952
    assert len(got) / len(payload) > 0.95


def test_the_lost_table_specimen_is_raw_with_nothing_to_check_against():
    """No container, so no CRC32 and no recorded length: every claim it makes
    rests on the bitstream alone. That is the case the specimen exists for."""
    from pathlib import Path
    from toeknife import containers
    spec = Path("samples/damaged/lost-table.bin")
    if not spec.is_file():
        pytest.skip("run samples/build/makedamaged.py to build the specimens")
    report = containers.parse(spec.read_bytes())
    assert report is None or not report.streams


def test_a_stored_header_found_in_noise_is_not_a_resync_point():
    """A chance LEN/NLEN pair inside carved-midblock.bin's dynamic block
    "decoded" 30,513 bytes of compressed data as a stored block, ranked first
    because it produced the most, and --resync put all of it on the tape as
    known (wrap-up #5). A stored candidate has to be followed by a block that
    decodes, or by the end of the stream."""
    from pathlib import Path
    from toeknife.core.scan import KIND_STORED, best_resync, scan
    data = Path("samples/damaged/carved-midblock.bin").read_bytes()
    assert not [p for p in scan(data) if p.kind == KIND_STORED
                and p.out_bytes > 1000], "no stored block survives validation"
    best = best_resync(data)
    assert best.kind != KIND_STORED and best.bit_offset == 234216, \
        "the real boundary: block 2 of the intact stream"
