"""PNG (RFC 2083), and the compressed text chunks that ride along with it.

The pixel data is one zlib stream split across however many IDAT chunks the
encoder chose, with a length, a type and a CRC wedged between each pair. So it
is not a contiguous run of bytes and cannot be handled by an offset and a
length: the payloads have to be gathered.

The spec requires the IDAT run to be consecutive. When it is not, that is worth
saying rather than quietly gathering the pieces, because a stream reassembled
out of a broken run is missing bytes in the middle and will decode to nonsense
somewhere without ever failing outright.

Each chunk carries a CRC32 over its own type and data, which is a checksum over
the *compressed* bytes rather than the output. It cannot verify a recovery, but
it does say which chunk the damage is in.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

from .base import ContainerReport, Stream
from .zlibfmt import stream_at

SIGNATURE = b"\x89PNG\r\n\x1a\n"
COMPRESSED_TEXT = {b"zTXt", b"iTXt", b"iCCP"}


@dataclass
class Chunk:
    offset: int
    length: int
    ctype: bytes
    data_offset: int
    crc_ok: bool
    stored_crc: int = 0


def chunks(data: bytes):
    pos = len(SIGNATURE)
    while pos + 12 <= len(data):
        length = struct.unpack_from(">I", data, pos)[0]
        ctype = data[pos + 4:pos + 8]
        body = pos + 8
        end = body + length
        if end + 4 > len(data):
            return
        stored = struct.unpack_from(">I", data, end)[0]
        actual = zlib.crc32(data[pos + 4:end]) & 0xFFFFFFFF
        yield Chunk(pos, length, ctype, body, stored == actual, stored)
        pos = end + 4
        if ctype == b"IEND":
            return


def _text_payload(data: bytes, chunk: Chunk):
    """Where the zlib data starts inside a compressed text chunk, or None."""
    at, end = chunk.data_offset, chunk.data_offset + chunk.length
    nul = data.find(b"\x00", at, end)
    if nul < 0:
        return None
    if chunk.ctype == b"zTXt":
        return nul + 2 if data[nul + 1:nul + 2] == b"\x00" else None
    if chunk.ctype == b"iCCP":
        return nul + 2 if data[nul + 1:nul + 2] == b"\x00" else None
    # iTXt: keyword\0, compression flag, compression method, lang\0, translated\0
    if data[nul + 1:nul + 2] != b"\x01":
        return None                      # flag 0 means the text is not compressed
    lang = data.find(b"\x00", nul + 3, end)
    if lang < 0:
        return None
    trans = data.find(b"\x00", lang + 1, end)
    return trans + 1 if trans >= 0 else None


def parse(data: bytes) -> ContainerReport:
    report = ContainerReport(kind="png")
    found = list(chunks(data))
    if not found:
        report.warnings.append("no PNG chunks found after the signature")
        return report

    bad = [c for c in found if not c.crc_ok]
    if bad:
        report.warnings.append(
            "chunk CRC mismatch at " + ", ".join(str(c.offset) for c in bad)
            + " (a chunk CRC covers the compressed bytes, so it says which "
              "chunk holds the damage, and repairs one or two flipped bits)")

    idat = [c for c in found if c.ctype == b"IDAT"]
    types = [c.ctype for c in found]
    report.structure.append(
        f"{len(found)} chunks: " + " ".join(dict.fromkeys(
            t.decode("latin-1") for t in types)))

    if idat:
        first = types.index(b"IDAT")
        last = len(types) - 1 - types[::-1].index(b"IDAT")
        if types[first:last + 1].count(b"IDAT") != last - first + 1:
            report.warnings.append(
                "the IDAT run is not consecutive, which the format forbids; the "
                "gathered pixel stream is missing bytes from the middle")
        segments = [(c.data_offset, c.length) for c in idat]
        total = sum(ln for _, ln in segments)
        report.structure.append(
            f"IDAT: {len(idat)} chunk(s), {total} bytes of zlib stream")
        notes = ([f"gathered from {len(idat)} IDAT chunk(s)"]
                 if len(idat) > 1 else [])
        at = 0
        for k, c in enumerate(idat):
            if not c.crc_ok:
                notes.append(
                    f"IDAT chunk {k + 1} of {len(idat)} (file offset "
                    f"{c.offset:,}) fails its CRC: the damage is in its "
                    f"{c.length:,} bytes, stream bytes {at:,}-"
                    f"{at + c.length - 1:,}")
            at += c.length
        s = stream_at(data, idat[0].data_offset, total, "IDAT pixel data", "png",
                      segments=segments, notes=notes or None,
                      segment_crcs=[(c.ctype, c.stored_crc) for c in idat])
        if s is None:
            report.warnings.append("the IDAT run does not start with a zlib header")
        else:
            report.streams.append(s)
    else:
        report.warnings.append("no IDAT chunks: no pixel data in this file")

    for c in found:
        if c.ctype not in COMPRESSED_TEXT:
            continue
        at = _text_payload(data, c)
        if at is None:
            continue
        name = data[c.data_offset:data.find(b"\x00", c.data_offset)].decode(
            "latin-1", "replace")
        length = c.data_offset + c.length - at
        s = stream_at(data, at, length,
                      f"{c.ctype.decode()} {name}".strip(), "png")
        if s is not None:
            report.streams.append(s)
    return report


def looks_like_png(data: bytes) -> bool:
    return data[:8] == SIGNATURE
