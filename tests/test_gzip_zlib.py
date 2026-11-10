"""gzip (RFC 1952) and zlib (RFC 1950): the wrappers that carry a checksum."""
import gzip as gzip_mod
import io
import random
import struct
import zlib

import pytest

from toeknife.containers import detect, parse
from toeknife.containers.base import verify
from toeknife.containers.gzipfmt import MAGIC, read_header
from toeknife.containers.zlibfmt import looks_like_zlib
from toeknife.containers.zlibfmt import read_header as zlib_header
from toeknife.core.session import Session

BODY = b"forensic analysis of compressed data, repeatedly and at length. " * 300


def decode_and_verify(blob, stream):
    s = Session(stream.slice(bytes(blob)), mode="stream", stream=stream)
    return s, verify(stream, s.tape)


# ---- gzip ------------------------------------------------------------------

def test_gzip_round_trip_verifies_against_crc_and_isize():
    blob = gzip_mod.compress(BODY, 6)
    assert detect(blob) == "gzip"
    report = parse(blob)
    assert len(report.streams) == 1
    s = report.streams[0]
    assert s.uncompressed_size == len(BODY)
    assert s.crc32 == zlib.crc32(BODY) & 0xFFFFFFFF
    _, v = decode_and_verify(blob, s)
    assert v.ok is True, v.summary()


def test_gzip_filename_and_comment_are_recovered():
    buf = io.BytesIO()
    with gzip_mod.GzipFile(filename="evidence.txt", mode="wb", fileobj=buf,
                           mtime=1700000000) as f:
        f.write(BODY)
    blob = bytearray(buf.getvalue())
    report = parse(bytes(blob))
    assert report.streams[0].name == "evidence.txt"
    assert "mtime 1700000000" in " ".join(report.structure)
    _, v = decode_and_verify(blob, report.streams[0])
    assert v.ok is True


def test_multi_member_gzip_enumerates_every_member():
    """Concatenated members are a single valid gzip file, and each carries its
    own trailer."""
    a, b = BODY[:5000], BODY[5000:12000]
    blob = gzip_mod.compress(a) + gzip_mod.compress(b)
    report = parse(blob)
    assert len(report.streams) == 2
    assert [s.uncompressed_size for s in report.streams] == [len(a), len(b)]
    for stream, original in zip(report.streams, (a, b)):
        session, v = decode_and_verify(blob, stream)
        assert v.ok is True, v.summary()
        assert bytes(session.tape.out) == original


def test_gzip_with_an_extra_field():
    raw = gzip_mod.compress(BODY)
    extra = b"\x41\x42\x04\x00wxyz"          # one subfield: id 'AB', 4 bytes
    patched = bytearray(raw)
    patched[3] |= 0x04                        # set FEXTRA
    patched[10:10] = struct.pack("<H", len(extra)) + extra
    report = parse(bytes(patched))
    assert len(report.streams) == 1
    assert any("FEXTRA" in n for n in report.streams[0].notes)
    _, v = decode_and_verify(patched, report.streams[0])
    assert v.ok is True


def test_truncated_gzip_reports_rather_than_guesses():
    blob = gzip_mod.compress(BODY)[:-4]       # lose half the trailer
    report = parse(blob)
    assert report.streams
    s = report.streams[0]
    assert s.crc32 is None and s.uncompressed_size is None
    assert any("truncated" in n for n in s.notes)
    # with nothing recorded to check against, verification must say so
    _, v = decode_and_verify(blob, s)
    assert v.ok is None


def test_a_corrupt_member_does_not_hide_the_ones_after_it():
    """gzip records no length, so a member that will not decode takes the
    position of everything after it.  Falling back to the magic recovers them."""
    a, b = BODY[:4000], BODY[4000:9000]
    blob = bytearray(gzip_mod.compress(a) + gzip_mod.compress(b))
    first = parse(bytes(blob))
    assert len(first.streams) == 2
    start = first.streams[0].offset
    rng = random.Random(4)
    for i in range(start + 20, start + 60):
        blob[i] = rng.randrange(256)
    report = parse(bytes(blob))
    assert len(report.streams) == 2, "the second member must still be found"
    assert any("scanning" in w for w in report.warnings)
    # The second member decodes, so the walk picks up again there and it is
    # measured like any other member.
    second = report.streams[1]
    assert second.uncompressed_size == len(b)
    assert second.crc32 == zlib.crc32(b) & 0xFFFFFFFF


def test_a_damaged_member_takes_its_trailer_from_before_the_next_header():
    """The next member that decodes bounds the damaged one, and the 8 bytes
    before its header are the damaged member's trailer: its size and CRC32."""
    parts = [BODY[:3000], BODY[3000:7000], BODY[7000:9000]]
    blob = bytearray(b"".join(gzip_mod.compress(p, mtime=0) for p in parts))
    start = parse(bytes(blob)).streams[1].offset
    for i in range(start + 5, start + 40):
        blob[i] ^= 0xFF
    report = parse(bytes(blob))
    assert len(report.streams) == 3
    damaged, last = report.streams[1], report.streams[2]
    assert damaged.uncompressed_size == len(parts[1])
    assert damaged.crc32 == zlib.crc32(parts[1]) & 0xFFFFFFFF
    assert damaged.offset + damaged.length + 8 + 10 == last.offset
    assert any("before member 2's header" in n for n in damaged.notes)
    assert last.uncompressed_size == len(parts[2])


def test_a_damaged_first_member_does_not_take_the_last_members_trailer():
    """The last 8 bytes of a multi-member file belong to its last member."""
    parts = [BODY[:3000], BODY[3000:6000]]
    blob = bytearray(b"".join(gzip_mod.compress(p, mtime=0) for p in parts))
    start = parse(bytes(blob)).streams[0].offset
    for i in range(start + 5, start + 40):
        blob[i] ^= 0xFF
    damaged = parse(bytes(blob)).streams[0]
    assert damaged.uncompressed_size == len(parts[0])
    assert damaged.crc32 == zlib.crc32(parts[0]) & 0xFFFFFFFF


def test_gzip_header_reader_rejects_rubbish():
    assert read_header(b"", 0) is None
    assert read_header(b"\x1f\x8b", 0) is None            # too short
    assert read_header(b"not a gzip at all........", 0) is None
    assert read_header(MAGIC + b"\x08" + b"\x00" * 20, 0) is not None


def test_non_deflate_gzip_is_refused_with_a_reason():
    blob = bytearray(gzip_mod.compress(BODY))
    blob[2] = 12                                          # claim bzip2
    report = parse(bytes(blob))
    assert report.streams
    assert not report.streams[0].decodable
    assert "not DEFLATE" in report.streams[0].why_not


# ---- zlib ------------------------------------------------------------------

def test_zlib_round_trip_verifies_against_adler32():
    blob = zlib.compress(BODY, 6)
    assert detect(blob) == "zlib"
    report = parse(blob)
    s = report.streams[0]
    assert s.adler32 == zlib.adler32(BODY) & 0xFFFFFFFF
    assert s.crc32 is None
    session, v = decode_and_verify(blob, s)
    assert v.ok is True, v.summary()
    assert bytes(session.tape.out) == BODY


def test_zlib_preset_dictionary_becomes_holes_rather_than_a_refusal():
    """The dictionary is not in the file, so references into it are unknown
    bytes.  That is the same problem as a reference behind the start of a
    fragment, and the tape already models it."""
    zdict = b"forensic analysis of compressed data"
    co = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY,
                          zdict=zdict)
    blob = co.compress(BODY) + co.flush()
    report = parse(blob)
    s = report.streams[0]
    assert any("preset dictionary" in n for n in s.notes)
    session, v = decode_and_verify(blob, s)
    assert session.tape.holes(), "dictionary references should be holes"
    # a checksum over output with unknown bytes proves nothing, so do not claim it
    assert v.adler_got is None
    assert "unchecked" in v.summary()
    assert v.ok is not False


def test_zlib_header_signature_is_weak_and_confirmed_by_decoding():
    passing = sum(1 for cmf in range(256) for flg in range(256)
                  if zlib_header(bytes([cmf, flg]) + b"\x00" * 8) is not None)
    assert passing == 66, f"{passing} of 65536 byte pairs pass the header check"

    rng = random.Random(1)
    noise = bytes(rng.randrange(256) for _ in range(60000))
    headers = sum(1 for i in range(len(noise) - 64)
                  if zlib_header(noise[i:i + 64]) is not None)
    confirmed = sum(1 for i in range(len(noise) - 64)
                    if looks_like_zlib(noise[i:i + 64]))
    assert headers > confirmed, "decoding must reject most header false positives"


def test_random_data_is_not_mistaken_for_a_container():
    rng = random.Random(9)
    for _ in range(20):
        blob = bytes(rng.randrange(256) for _ in range(4000))
        assert detect(blob) in (None, "zlib", "zip")
        if detect(blob) is not None:
            report = parse(blob)
            assert not report.decodable or all(
                s.length is None or s.length < 64 for s in report.decodable)


# ---- the whole point -------------------------------------------------------

def test_corrupted_gzip_payload_fails_verification():
    """A damaged DEFLATE stream usually still decodes, to roughly the right
    length, with wrong bytes.  Only the checksum notices."""
    blob = bytearray(gzip_mod.compress(BODY))
    s = parse(bytes(blob)).streams[0]
    _, good = decode_and_verify(blob, s)
    assert good.ok is True
    rng = random.Random(2)
    mid = s.offset + s.length // 2
    for i in range(mid, mid + 40):
        blob[i] = rng.randrange(256)
    report = parse(bytes(blob))
    if report.streams and report.streams[0].crc32 is not None:
        _, bad = decode_and_verify(blob, report.streams[0])
        assert bad.ok is False
    else:
        assert report.warnings or report.streams[0].notes


def test_isize_gives_the_output_length_before_decoding():
    """The constraint that would pin a gap length: gzip records how long the
    original was, independently of anything we decode."""
    blob = gzip_mod.compress(BODY)
    s = parse(blob).streams[0]
    assert s.uncompressed_size == len(BODY)
    session, _ = decode_and_verify(blob, s)
    assert len(session.tape) == s.uncompressed_size


def test_a_single_member_that_will_not_decode_still_yields_its_trailer():
    """The end of a gzip member is found by decoding it, so damage that stops the
    decode used to take the TRAILER with it -- and the trailer is a CRC32 and a
    length of the original, the one thing that can confirm a repair of the very
    damage that stopped the decode.

    For the first member its position is not a guess: a gzip ends with it.
    """
    from toeknife.core import bitrot
    body = BODY[:9000]
    blob = bytearray(gzip_mod.compress(body))
    good = parse(bytes(blob)).streams[0]
    assert good.crc32 is not None

    # One inverted bit in the deflate header, which is what stops the walk.
    bit = good.offset * 8 + 3
    blob[bit >> 3] ^= 1 << (bit & 7)
    st = parse(bytes(blob)).streams[0]
    assert "did not decode" in " ".join(st.notes)
    assert st.crc32 == good.crc32, "the trailer is where a gzip always puts it"
    assert st.uncompressed_size == len(body)
    assert any("last 8 bytes" in n for n in st.notes), \
        "an inferred trailer position has to say that it was inferred"

    # And it is worth having: it turns the repair into a verified recovery.
    s = Session(st.slice(bytes(blob)), mode="stream", stream=st)
    assert s.verify().ok is False
    r = bitrot.repair(s, k=1)
    assert r.candidates
    s.data = bitrot.apply(s, r.candidates[0])
    s.trace = None
    s.select_alignment(s.bit_offset)
    assert s.verify().ok is True
    assert bytes(s.tape.out[:len(body)]) == body


def test_the_inferred_trailer_is_not_claimed_for_a_multi_member_file():
    """Reading the last 8 bytes is right for one member and wrong the moment
    another follows, so it is only done for the first and says so."""
    blob = bytearray(gzip_mod.compress(BODY[:3000])
                     + gzip_mod.compress(BODY[3000:6000]))
    first = parse(bytes(blob)).streams[0]
    bit = first.offset * 8 + 3
    blob[bit >> 3] ^= 1 << (bit & 7)
    streams = parse(bytes(blob)).streams
    # When the second member decodes, the damaged one's trailer is the 8 bytes
    # before it, labelled as read from there; it is never the file's last 8.
    damaged = streams[0]
    if damaged.crc32 is not None:
        assert damaged.crc32 == zlib.crc32(BODY[:3000]) & 0xFFFFFFFF
        assert any("before member 1's header" in n for n in damaged.notes)


def _three_members(bad=1):
    parts = [BODY[:3000], BODY[3000:7000], BODY[7000:9000]]
    blob = bytearray(b"".join(gzip_mod.compress(p, mtime=0) for p in parts))
    start = parse(bytes(blob)).streams[bad].offset
    for i in range(start + 5, start + 40):
        blob[i] ^= 0xFF
    return parts, bytes(blob)


def test_the_members_join_into_the_file_gunzip_would_write():
    """A member that decodes to nothing still takes its recorded size, so the
    members after it land where they belong."""
    from toeknife.core import members
    parts, blob = _three_members()
    report = parse(blob)
    buf, mask, got = members.join(report, members.open_all(blob, report))
    assert len(buf) == sum(len(p) for p in parts)
    assert buf[:3000] == parts[0] and buf[7000:] == parts[2]
    assert buf[3000:7000] == b"?" * 4000 and not any(mask[3000:7000])
    assert [p.start for p in got] == [0, 3000, 7000]
    assert [p.verdict.ok for p in got] == [True, False, True]


def test_entry_all_with_a_copy_writes_the_whole_file(tmp_path, capsys):
    from toeknife import cli
    parts, blob = _three_members()
    gz, copy, out = tmp_path / "m.gz", tmp_path / "copy.txt", tmp_path / "o"
    gz.write_bytes(blob)
    copy.write_bytes(b"".join(parts))
    rc = cli.main([str(gz), "--entry", "all", "--fill-from", str(copy),
                   "--export", str(out), "--verify"])
    assert rc == 0
    assert out.read_bytes() == b"".join(parts)
    assert "3 of 3 members match their trailers" in capsys.readouterr().out


def test_entry_all_refuses_a_single_member(tmp_path):
    from toeknife import cli
    gz = tmp_path / "one.gz"
    gz.write_bytes(gzip_mod.compress(BODY[:3000]))
    assert cli.main([str(gz), "--entry", "all"]) == 1


def test_two_damaged_members_in_a_row_are_both_listed_with_their_own_trailers():
    """The next member that decodes is past both; each damaged one ends where
    the next header starts."""
    parts = [BODY[:2000], BODY[2000:4000], BODY[4000:6000], BODY[6000:9000]]
    blob = bytearray(b"".join(gzip_mod.compress(p, mtime=0) for p in parts))
    offsets = [s.offset for s in parse(bytes(blob)).streams]
    for k in (1, 2):
        for i in range(offsets[k] + 5, offsets[k] + 40):
            blob[i] ^= 0xFF
    streams = parse(bytes(blob)).streams
    assert [s.offset for s in streams] == offsets
    for s, p in zip(streams, parts):
        assert s.crc32 == zlib.crc32(p) & 0xFFFFFFFF
        assert s.uncompressed_size == len(p)
