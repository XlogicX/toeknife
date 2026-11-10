"""Flipped bits found by a PNG chunk's CRC, and the PDF scanner's object
names and cut-short streams (manual workflow I)."""
import random
import struct
import zlib

from toeknife.containers import parse
from toeknife.core import chunkcrc
from toeknife.core.session import Session


def chunk(ctype, body):
    return (struct.pack(">I", len(body)) + ctype + body
            + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF))


def png(size=4096):
    rng = random.Random(7)
    raw = b"".join(b"\x00" + bytes((x * y + rng.randrange(4)) & 0xFF
                                   for x in range(200))
                   for y in range(120))
    z = zlib.compress(raw, 9)
    out = bytearray(b"\x89PNG\r\n\x1a\n" + chunk(
        b"IHDR", struct.pack(">IIBBBBB", 200, 120, 8, 0, 0, 0, 0)))
    starts = []
    for i in range(0, len(z), size):
        starts.append(len(out) + 8)
        out += chunk(b"IDAT", z[i:i + size])
    out += chunk(b"IEND", b"")
    return bytes(out), starts


def session(blob):
    st = parse(blob).streams[0]
    s = Session(st.slice(blob), stream=st)
    s.file_data = blob
    return s


def test_one_flipped_bit_is_found_proven_and_written_back():
    good, starts = png()
    bad = bytearray(good)
    bad[starts[1] + 1000] ^= 0x08
    s = session(bytes(bad))
    assert chunkcrc.bad_segments(s) == [1]
    got = chunkcrc.repair(s, bytes(bad))
    proven = [f for f in got.fixes if f.proven]
    assert [(f.segment, f.bits) for f in proven] == [(1, (1000 * 8 + 3,))]
    chunkcrc.apply(s, proven[0])
    assert s.verify().ok is True
    assert chunkcrc.repaired_file(s) == good
    assert chunkcrc.bad_segments(s) == []


def test_two_flipped_bits_in_one_chunk():
    good, starts = png()
    bad = bytearray(good)
    bad[starts[0] + 300] ^= 0x01
    bad[starts[0] + 2900] ^= 0x40
    got = chunkcrc.repair(session(bytes(bad)), bytes(bad))
    assert [f.bits for f in got.fixes if f.proven] == [(300 * 8, 2900 * 8 + 6)]


def test_a_hit_on_the_stored_crc_leaves_the_data_alone():
    good, starts = png()
    bad = bytearray(good)
    length = struct.unpack_from(">I", good, starts[1] - 8)[0]
    bad[starts[1] + length] ^= 0x02          # first byte of the stored CRC
    got = chunkcrc.repair(session(bytes(bad)), bytes(bad))
    assert [f.where for f in got.fixes] == ["crc"]


def test_three_bits_are_out_of_reach_and_say_so():
    good, starts = png()
    bad = bytearray(good)
    for at in (100, 900, 2000):
        bad[starts[1] + at] ^= 0x10
    got = chunkcrc.repair(session(bytes(bad)), bytes(bad))
    assert not [f for f in got.fixes if f.proven]
    assert "fail" in got.summary(len(starts))


def test_cli_writes_the_repaired_png(tmp_path):
    from toeknife import cli
    good, starts = png()
    bad = bytearray(good)
    bad[starts[2] + 50] ^= 0x80
    src, out = tmp_path / "bad.png", tmp_path / "fixed.png"
    src.write_bytes(bytes(bad))
    assert cli.main([str(src), "--chunk-repair", "--write-repaired",
                     str(out), "--verify"]) == 0
    assert out.read_bytes() == good


# ---- PDF --------------------------------------------------------------------

def pdf(n=4):
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i in range(n):
        body = zlib.compress(f"BT (page {i}) Tj ET\n".encode() * 40)
        offsets.append(len(out))
        out += (f"{10 + i} 0 obj\n<< /Length {len(body)} /Filter "
                f"/FlateDecode >>\nstream\n").encode() + body
        out += b"\nendstream\nendobj\n"
    return bytes(out)


def test_each_stream_is_named_by_its_own_object():
    names = [s.name for s in parse(pdf()).streams]
    assert names == ["obj 10", "obj 11", "obj 12", "obj 13"]


def test_a_stream_the_file_cuts_short_claims_no_adler():
    blob = pdf()
    cut = blob[:blob.rfind(b">>\nstream\n") + len(b">>\nstream\n") + 10]
    last = parse(cut).streams[-1]
    assert last.name == "obj 13"
    assert last.adler32 is None
    assert any("ends inside this stream" in n for n in last.notes)
