"""ZIP container parsing, including the archives that are damaged on purpose."""
import io
import random
import struct
import zipfile

import pytest

from toeknife.containers import detect, parse
from toeknife.containers.base import verify
from toeknife.containers.zip import (CD_SIG, EOCD_SIG, LOCAL_SIG, find_eocd)
from toeknife.core.session import Session

BODY = (b"forensic analysis of compressed data, at some length, repeatedly. " * 60)
SMALL = b"a short entry\n"


def varied(size=120000):
    """Content that does not compress to almost nothing, so a corrupted stream
    is long enough to corrupt in the middle of."""
    rng = random.Random(17)
    words = b"the quick brown fox jumps over lazy dogs forensic analysis".split()
    text = b" ".join(rng.choice(words) for _ in range(size // 5))
    return (text + bytes(rng.randrange(256) for _ in range(size // 4)))[:size]


def build(entries, comment=b"", level=6):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as z:
        for name, payload, method in entries:
            z.writestr(name, payload, compress_type=method)
        if comment:
            z.comment = comment
    return bytearray(buf.getvalue())


def normal():
    return build([("hello.txt", BODY, zipfile.ZIP_DEFLATED),
                  ("stored.bin", SMALL, zipfile.ZIP_STORED),
                  ("nested/deep.txt", BODY * 2, zipfile.ZIP_DEFLATED)])


def decode_and_verify(blob, stream):
    session = Session(stream.slice(bytes(blob)), mode="stream", stream=stream)
    return session, verify(stream, session.tape)


# ---- the happy path -------------------------------------------------------

def test_detects_and_enumerates():
    blob = normal()
    assert detect(bytes(blob)) == "zip"
    report = parse(bytes(blob))
    assert [s.name for s in report.streams] == [
        "hello.txt", "stored.bin", "nested/deep.txt"]
    assert not report.warnings
    assert len(report.decodable) == 2


def test_every_deflate_entry_verifies_against_its_crc():
    blob = normal()
    report = parse(bytes(blob))
    for stream in report.decodable:
        _, v = decode_and_verify(blob, stream)
        assert v.ok is True, f"{stream.name}: {v.summary()}"
        assert v.complete


def test_stored_entries_are_reported_not_attempted():
    report = parse(bytes(normal()))
    stored = next(s for s in report.streams if s.name == "stored.bin")
    assert not stored.decodable
    assert stored.why_not == "stored, not compressed"
    assert stored.uncompressed_size == len(SMALL)


def test_uncompressed_size_and_crc_are_carried_through():
    report = parse(bytes(normal()))
    s = next(x for x in report.streams if x.name == "hello.txt")
    assert s.uncompressed_size == len(BODY)
    assert s.crc32 is not None
    assert s.encoding == "raw"          # zip entries are bare DEFLATE


# ---- shapes that break naive parsers --------------------------------------

def test_streamed_entry_with_data_descriptor():
    """Written to a non-seekable sink, sizes go in a trailing descriptor and the
    local header keeps the zeros the spec mandates."""
    class NoSeek(io.RawIOBase):
        def __init__(self):
            self.b = bytearray()

        def write(self, d):
            self.b += d
            return len(d)

        def writable(self):
            return True

        def seekable(self):
            return False

    sink = NoSeek()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("streamed.txt", BODY)
    blob = bytearray(sink.b)
    report = parse(bytes(blob))
    s = report.streams[0]
    assert s.name == "streamed.txt"
    _, v = decode_and_verify(blob, s)
    assert v.ok is True, v.summary()


def test_forced_zip64_extra_fields():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        with z.open("big.txt", "w", force_zip64=True) as f:
            f.write(BODY * 4)
    blob = bytearray(buf.getvalue())
    report = parse(bytes(blob))
    assert len(report.streams) == 1
    s = report.streams[0]
    assert s.uncompressed_size == len(BODY) * 4
    _, v = decode_and_verify(blob, s)
    assert v.ok is True, v.summary()


def test_prepended_stub_shifts_every_offset():
    """A self-extracting stub, or a carve that lost the front, moves the data
    without updating the recorded offsets.  The shift is derivable because the
    EOCD says how big the directory is and where the file ends."""
    blob = bytearray(b"MZ" + b"\x90" * 4094) + normal()
    report = parse(bytes(blob))
    assert any("shifted by +4096" in line for line in report.structure)
    for stream in report.decodable:
        _, v = decode_and_verify(blob, stream)
        assert v.ok is True, f"{stream.name}: {v.summary()}"


def test_missing_central_directory_falls_back_to_scanning():
    blob = normal()
    truncated = blob[:len(blob) - 140]        # lose the EOCD and part of the CD
    report = parse(bytes(truncated))
    assert any("no end-of-central-directory" in w for w in report.warnings)
    assert any("scanning for local headers" in w for w in report.warnings)
    names = [s.name for s in report.streams]
    assert "hello.txt" in names and "nested/deep.txt" in names
    for stream in report.decodable:
        _, v = decode_and_verify(truncated, stream)
        assert v.ok is True, f"{stream.name}: {v.summary()}"


def test_eocd_signature_inside_a_comment_is_not_mistaken_for_the_record():
    """The comment length has to account for exactly the trailing bytes, or a
    PK\\x05\\x06 in a comment hijacks the parse."""
    blob = build([("a.txt", BODY, zipfile.ZIP_DEFLATED)],
                 comment=EOCD_SIG + b"\x00" * 18 + b"decoy")
    real = find_eocd(bytes(blob))
    assert real is not None
    assert bytes(blob[real:real + 4]) == EOCD_SIG
    report = parse(bytes(blob))
    assert [s.name for s in report.streams] == ["a.txt"]
    assert not report.warnings


def test_encrypted_entry_is_flagged_and_refused():
    blob = normal()
    pos = blob.find(LOCAL_SIG)
    flags = struct.unpack_from("<H", blob, pos + 6)[0]
    struct.pack_into("<H", blob, pos + 6, flags | 0x0001)
    cd = blob.find(CD_SIG)
    cflags = struct.unpack_from("<H", blob, cd + 8)[0]
    struct.pack_into("<H", blob, cd + 8, cflags | 0x0001)
    report = parse(bytes(blob))
    s = report.streams[0]
    assert s.encrypted
    assert not s.decodable
    assert s.why_not == "entry is encrypted"


def test_local_and_central_disagreement_is_reported():
    """Not a corner case to paper over: a local header contradicting the central
    directory is damage or hand-editing, and worth saying out loud."""
    blob = normal()
    pos = blob.find(LOCAL_SIG)
    struct.pack_into("<I", blob, pos + 14, 0xDEADBEEF)     # bogus local CRC
    report = parse(bytes(blob))
    s = report.streams[0]
    assert any("crc32 disagrees" in n for n in s.notes), s.notes


# ---- damage detection ------------------------------------------------------

def test_corrupt_entry_fails_verification():
    """The important one.  A corrupted DEFLATE stream often still decodes, to
    roughly the right length, with wrong bytes.  Nothing but the checksum
    notices, which is the whole reason for reading the container."""
    blob = build([("evidence.txt", varied(), zipfile.ZIP_DEFLATED)])
    report = parse(bytes(blob))
    s = report.streams[0]
    _, good = decode_and_verify(blob, s)
    assert good.ok is True

    rng = random.Random(9)
    mid = s.offset + s.length // 2
    assert mid + 200 < s.offset + s.length, "corruption must land inside the entry"
    for i in range(mid, mid + 200):
        blob[i] = rng.randrange(256)
    _, bad = decode_and_verify(blob, parse(bytes(blob)).streams[0])
    assert bad.ok is False, "corruption must not pass verification"


def test_partial_recovery_leaves_the_crc_unchecked_rather_than_failing():
    """A decode with holes cannot be checksummed.  Reporting a CRC failure there
    would be reporting a fact we do not have."""
    from toeknife.core.tape import Tape
    from toeknife.containers.base import Stream
    tape = Tape()
    for b in b"abc":
        tape.emit_literal(b)
    tape.emit_copy(distance=5, length=4)      # reaches behind the start
    s = Stream(name="x", container="zip", offset=0, crc32=0x12345678,
               uncompressed_size=len(tape))
    v = verify(s, tape)
    assert not v.complete
    assert v.crc_got is None
    assert "unchecked" in v.summary()
    assert v.ok is True or v.ok is None      # size may still match; crc is silent


# ---- selection -------------------------------------------------------------

def test_entry_selection_by_index_and_name():
    from toeknife.cli import pick_entry
    report = parse(bytes(normal()))
    assert pick_entry(report, "0").name == "hello.txt"
    assert pick_entry(report, "-1").name == "nested/deep.txt"
    assert pick_entry(report, "nested/deep.txt").name == "nested/deep.txt"
    assert pick_entry(report, "deep").name == "nested/deep.txt"
    with pytest.raises(ValueError):
        pick_entry(report, "nope")
    with pytest.raises(ValueError):
        pick_entry(report, "99")
    with pytest.raises(ValueError):
        pick_entry(report, "txt")            # ambiguous, must not guess


def test_empty_and_junk_inputs_do_not_crash():
    assert parse(b"") is None
    assert parse(b"not a zip at all, just some bytes") is None
    assert detect(b"PK\x03\x04") == "zip"
    report = parse(b"PK\x03\x04" + b"\x00" * 26)
    assert report is not None
    assert report.warnings
