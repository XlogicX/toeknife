"""PNG, SWF, WOFF, TIFF, Anvil and PDF."""
import gzip as gzip_mod
import struct
import zlib

import pytest

from toeknife.containers import detect, parse
from toeknife.containers.base import Stream
from toeknife.core.session import Session

FILLER = b"forensic analysis of compressed data, at some length. " * 200


def roundtrip(data: bytes, stream: Stream) -> bytes:
    session = Session(stream.slice(data), mode="stream", stream=stream)
    assert session.trace.status == "end", session.trace.status
    assert not session.tape.holes()
    return bytes(session.tape.out)


# ---- PNG -------------------------------------------------------------------

def png_chunk(ctype: bytes, body: bytes) -> bytes:
    return (struct.pack(">I", len(body)) + ctype + body
            + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF))


def make_png(idat_parts: int = 3, extra=b"", scramble=False) -> tuple[bytes, bytes]:
    w = h = 48
    raw = b"".join(b"\x00" + bytes(((x * 3 + y * 7) % 251) for x in range(w))
                   for y in range(h))
    z = zlib.compress(raw, 6)
    step = len(z) // idat_parts + 1
    idats = [png_chunk(b"IDAT", z[i:i + step]) for i in range(0, len(z), step)]
    head = (b"\x89PNG\r\n\x1a\n"
            + png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)))
    if scramble:                      # a non-IDAT chunk in the middle of the run
        idats.insert(1, png_chunk(b"tEXt", b"Note\x00interrupted"))
    return head + b"".join(idats) + extra + png_chunk(b"IEND", b""), raw


def test_png_pixel_data_spans_several_idat_chunks():
    """One zlib stream cut across chunks, each with a length, type and CRC
    between: it is not a contiguous run and cannot be sliced as one."""
    blob, raw = make_png(idat_parts=4)
    assert detect(blob) == "png"
    report = parse(blob)
    idat = report.streams[0]
    assert idat.segments is not None and len(idat.segments) == 4
    assert roundtrip(blob, idat) == raw


def test_png_single_idat_still_works():
    blob, raw = make_png(idat_parts=1)
    report = parse(blob)
    assert roundtrip(blob, report.streams[0]) == raw


def test_png_reports_a_broken_idat_run_rather_than_silently_joining_it():
    """A run split by another chunk is missing bytes in the middle, and will
    decode to nonsense somewhere without failing outright."""
    blob, _ = make_png(idat_parts=3, scramble=True)
    report = parse(blob)
    assert any("not consecutive" in w for w in report.warnings)


def test_png_compressed_text_chunks_are_found():
    payload = b"forensic notes about this image. " * 30
    ztxt = png_chunk(b"zTXt", b"Comment\x00\x00" + zlib.compress(payload))
    itxt = png_chunk(b"iTXt", b"Desc\x00\x01\x00en\x00Beschreibung\x00"
                     + zlib.compress(payload))
    blob, _ = make_png(extra=ztxt + itxt)
    report = parse(blob)
    names = [s.name for s in report.streams]
    assert any(n.startswith("zTXt") for n in names), names
    assert any(n.startswith("iTXt") for n in names), names
    for s in report.streams[1:]:
        assert roundtrip(blob, s) == payload


def test_png_uncompressed_itxt_is_not_mistaken_for_a_stream():
    plain = png_chunk(b"iTXt", b"Desc\x00\x00\x00en\x00Beschreibung\x00plain text")
    blob, _ = make_png(extra=plain)
    report = parse(blob)
    assert all(not s.name.startswith("iTXt") for s in report.streams)


def test_png_chunk_crc_damage_is_localised():
    blob, _ = make_png()
    broken = bytearray(blob)
    at = blob.find(b"IDAT") + 6
    broken[at] ^= 0xFF
    report = parse(bytes(broken))
    assert any("CRC mismatch" in w for w in report.warnings)


# ---- SWF -------------------------------------------------------------------

def test_swf_cws_body_decodes():
    body = FILLER[:9000]
    blob = (b"CWS" + bytes([13]) + struct.pack("<I", 8 + len(body))
            + zlib.compress(body, 6))
    assert detect(blob) == "swf"
    report = parse(blob)
    assert len(report.streams) == 1
    assert report.streams[0].uncompressed_size == len(body)
    assert roundtrip(blob, report.streams[0]) == body


@pytest.mark.parametrize("sig,why", [(b"FWS", "not compressed"), (b"ZWS", "LZMA")])
def test_swf_other_signatures_are_explained_not_attempted(sig, why):
    blob = sig + bytes([13]) + struct.pack("<I", 1000) + b"\x00" * 40
    report = parse(blob)
    assert not report.streams
    assert any(why in w for w in report.warnings)


# ---- WOFF ------------------------------------------------------------------

def make_woff(tables):
    header_len = 44 + len(tables) * 20
    bodies, directory, offset = b"", b"", header_len
    for tag, payload, compress in tables:
        blob = zlib.compress(payload, 6) if compress else payload
        directory += struct.pack(">4sIIII", tag, offset, len(blob), len(payload),
                                 0xAABBCCDD)
        bodies += blob
        offset += len(blob)
    head = (b"wOFF" + b"\x00\x01\x00\x00" + struct.pack(">I", offset)
            + struct.pack(">HH", len(tables), 0) + struct.pack(">I", 4000)
            + struct.pack(">HH", 1, 0) + struct.pack(">III", 0, 0, 0)
            + struct.pack(">II", 0, 0))
    assert len(head) == 44
    return head + directory + bodies


def test_woff_compressed_tables_decode_and_stored_ones_are_skipped():
    glyf = FILLER[:6000]
    head = b"\x00\x01\x02\x03"
    blob = make_woff([(b"glyf", glyf, True), (b"head", head, False)])
    assert detect(blob) == "woff"
    report = parse(blob)
    assert [s.name for s in report.streams] == ["table glyf"]
    assert report.streams[0].uncompressed_size == len(glyf)
    assert roundtrip(blob, report.streams[0]) == glyf
    assert any("head: stored" in line for line in report.structure)


def test_woff_length_disagreement_is_reported():
    blob = bytearray(make_woff([(b"glyf", FILLER[:3000], True)]))
    struct.pack_into(">I", blob, 8, 999999)
    report = parse(bytes(blob))
    assert any("declares 999999" in w for w in report.warnings)


# ---- TIFF ------------------------------------------------------------------

def make_tiff(strips, little=True, compression=8):
    end = "<" if little else ">"
    blobs = [zlib.compress(s, 6) for s in strips]
    magic = b"II\x2a\x00" if little else b"MM\x00\x2a"
    data_start = 8
    offsets, at = [], data_start
    for b in blobs:
        offsets.append(at)
        at += len(b)
    ifd_off = at
    n = 6
    arrays = ifd_off + 2 + n * 12 + 4
    entries = [(256, 3, 1, 64), (257, 3, 1, 64), (259, 3, 1, compression),
               (278, 3, 1, 32),
               (273, 4, len(blobs), arrays),
               (279, 4, len(blobs), arrays + 4 * len(blobs))]
    out = magic + struct.pack(end + "I", ifd_off) + b"".join(blobs)
    out += struct.pack(end + "H", n)
    for tag, typ, cnt, val in entries:
        # A value short enough to fit in the four-byte field is left-justified
        # in the lower-numbered bytes, so a SHORT sits in bytes 0..1 whichever
        # byte order the file is in.  Packing it as a four-byte integer puts it
        # in the wrong half under big-endian.
        field = (struct.pack(end + "H", val) + b"\x00\x00" if typ == 3
                 else struct.pack(end + "I", val))
        out += struct.pack(end + "HHI", tag, typ, cnt) + field
    out += struct.pack(end + "I", 0)
    for o in offsets:
        out += struct.pack(end + "I", o)
    for b in blobs:
        out += struct.pack(end + "I", len(b))
    return out


@pytest.mark.parametrize("little", [True, False])
@pytest.mark.parametrize("compression", [8, 32946])
def test_tiff_strips_decode_in_both_byte_orders(little, compression):
    """8 is Adobe's registration of Deflate and 32946 the original extension;
    both occur and neither is more correct."""
    strips = [b"strip one " * 300, b"strip two " * 200]
    blob = make_tiff(strips, little=little, compression=compression)
    assert detect(blob) == "tiff"
    report = parse(blob)
    assert len(report.streams) == 2
    for s, original in zip(report.streams, strips):
        assert roundtrip(blob, s) == original


def test_tiff_with_other_compression_is_reported_not_attempted():
    blob = make_tiff([b"data" * 500], compression=5)     # LZW
    report = parse(blob)
    assert not report.streams
    assert any("not DEFLATE" in w for w in report.warnings)


# ---- Minecraft Anvil -------------------------------------------------------

def make_mca(chunks):
    SECTOR = 4096
    loc, ts, body = bytearray(SECTOR), bytearray(SECTOR), bytearray()
    sector = 2
    for i, (payload, scheme) in enumerate(chunks):
        blob = (zlib.compress(payload, 6) if scheme == 2
                else gzip_mod.compress(payload) if scheme == 1 else payload)
        rec = struct.pack(">I", len(blob) + 1) + bytes([scheme]) + blob
        rec += b"\x00" * ((-len(rec)) % SECTOR)
        struct.pack_into(">I", loc, i * 4, (sector << 8) | (len(rec) // SECTOR))
        body += rec
        sector += len(rec) // SECTOR
    return bytes(loc) + bytes(ts) + bytes(body)


def test_anvil_chunks_decode_whichever_scheme_each_uses():
    """Compression is per chunk, so one region file can mix gzip and zlib."""
    payloads = [b"chunk zero " * 200, b"chunk one " * 300, b"chunk two " * 150]
    blob = make_mca([(payloads[0], 2), (payloads[1], 2), (payloads[2], 1)])
    assert detect(blob) == "mca"
    report = parse(blob)
    assert len(report.streams) == 3
    for s, original in zip(report.streams, payloads):
        assert roundtrip(blob, s) == original


def test_anvil_is_recognised_without_a_signature():
    """Anvil has no magic, so recognition rests on the location table pointing
    only inside the file and past the header sectors."""
    blob = make_mca([(b"only chunk " * 100, 2)])
    assert detect(blob) == "mca"
    assert detect(b"\x00" * 12288) != "mca", "an empty table is not a region file"
    broken = bytearray(blob)
    struct.pack_into(">I", broken, 0, (99999 << 8) | 1)      # points past the end
    assert detect(bytes(broken)) != "mca"


def test_anvil_stored_chunks_are_reported_not_attempted():
    blob = make_mca([(b"uncompressed nbt " * 100, 3)])
    report = parse(blob)
    assert len(report.streams) == 1
    assert not report.streams[0].decodable


# ---- PDF -------------------------------------------------------------------

def pdf_object(num, payload):
    z = zlib.compress(payload, 6)
    return (f"{num} 0 obj\n<< /Length {len(z)} /Filter /FlateDecode >>\n"
            f"stream\n".encode() + z + b"\nendstream\nendobj\n")


def test_pdf_flatedecode_streams_are_found_by_scanning():
    """Locating them via the xref would be correct for a valid file and useless
    here, since the xref is at the end and the end is what damage takes."""
    a, b = b"first object " * 200, b"second object " * 150
    blob = (b"%PDF-1.7\n" + pdf_object(1, a) + pdf_object(2, b)
            + b"trailer\n<< /Root 1 0 R >>\nstartxref\n0\n%%EOF\n")
    assert detect(blob) == "pdf"
    report = parse(blob)
    assert len(report.streams) == 2
    assert roundtrip(blob, report.streams[0]) == a
    assert roundtrip(blob, report.streams[1]) == b


def test_the_word_stream_inside_compressed_data_is_not_reported_as_an_object():
    """Compressed bytes spell anything given enough of them.  The keyword only
    counts when it directly follows the dictionary that describes it."""
    payload = b"stream\nstream\n" * 400 + b"endstream\n" * 50
    blob = (b"%PDF-1.7\n" + pdf_object(1, payload)
            + b"trailer\n<< >>\nstartxref\n0\n%%EOF\n")
    report = parse(blob)
    assert len(report.streams) == 1, [s.name for s in report.streams]
    assert roundtrip(blob, report.streams[0]) == payload


def test_pdf_encryption_is_flagged():
    blob = (b"%PDF-1.7\n" + pdf_object(1, b"x" * 500)
            + b"trailer\n<< /Encrypt 9 0 R >>\nstartxref\n0\n%%EOF\n")
    report = parse(blob)
    assert any("encrypted" in w for w in report.warnings)


# ---- the segment machinery -------------------------------------------------

def test_slice_joins_segments_and_strips_the_wrapper():
    #        0123 4567 89A  BC  DEF  GHIJ
    data = b"XXXX" + b"HEAD" + b"one" + b"YY" + b"two" + b"TAIL"
    # two segments joining to HEAD|one|two|TAIL, with the wrapper stripped off
    s = Stream(name="t", container="test", offset=4,
               segments=[(4, 7), (13, 7)], header_skip=4, trailer_skip=4)
    assert s.slice(data) == b"onetwo"


def test_slice_without_segments_is_a_plain_range():
    data = bytes(range(40))
    assert Stream(name="t", container="test", offset=5, length=10).slice(data) \
           == data[5:15]
    assert Stream(name="t", container="test", offset=5).slice(data) == data[5:]
