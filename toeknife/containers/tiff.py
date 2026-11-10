"""TIFF with Deflate compression: one zlib stream per strip or tile.

Two compression tag values mean DEFLATE. 32946 is the original Deflate
extension; 8 is the later Adobe registration of the same thing. Both appear in
the wild and neither is more correct than the other.

An image is stored either in strips or in tiles, never both, so whichever pair
of offset and byte-count tags is present decides how many streams there are.
"""
from __future__ import annotations

import struct

from .base import ContainerReport
from .zlibfmt import stream_at

TAG_COMPRESSION = 259
TAG_STRIP_OFFSETS = 273
TAG_STRIP_COUNTS = 279
TAG_TILE_OFFSETS = 324
TAG_TILE_COUNTS = 325
DEFLATE_TAGS = {8: "Adobe Deflate", 32946: "Deflate"}

TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8,
             11: 4, 12: 8, 16: 8, 17: 8, 18: 8}


def _u16(data, pos, le):
    return struct.unpack_from("<H" if le else ">H", data, pos)[0]


def _u32(data, pos, le):
    return struct.unpack_from("<I" if le else ">I", data, pos)[0]


def _values(data, typ, count, value_field, le):
    """An IFD entry's values, which live inline when they fit in four bytes."""
    size = TYPE_SIZE.get(typ, 0)
    if not size:
        return []
    total = size * count
    if total > 4 and value_field + 4 > len(data):
        return []
    at = value_field if total <= 4 else _u32(data, value_field, le)
    out = []
    for i in range(count):
        pos = at + i * size
        if pos + size > len(data):
            break
        if size == 2:
            out.append(_u16(data, pos, le))
        elif size == 4:
            out.append(_u32(data, pos, le))
        elif size == 8:
            out.append(struct.unpack_from("<Q" if le else ">Q", data, pos)[0])
        else:
            out.append(data[pos])
    return out


def parse(data: bytes) -> ContainerReport:
    report = ContainerReport(kind="tiff")
    le = data[:4] == b"II\x2a\x00"
    report.structure.append(
        f"TIFF, {'little' if le else 'big'}-endian")
    if len(data) < 8:
        # A fuzzed TIFF cut inside its 8-byte header crashed here (wrap-up #2).
        report.warnings.append(f"the header is cut short: {len(data)} of 8 "
                               f"bytes, so there is no IFD to read")
        return report
    ifd = _u32(data, 4, le)
    index = 0
    seen = set()

    while ifd and ifd + 2 <= len(data) and ifd not in seen:
        seen.add(ifd)
        count = _u16(data, ifd, le)
        entries = {}
        for i in range(count):
            entry = ifd + 2 + i * 12
            if entry + 12 > len(data):
                report.warnings.append(f"IFD {index} truncated")
                break
            tag = _u16(data, entry, le)
            typ = _u16(data, entry + 2, le)
            n = _u32(data, entry + 4, le)
            entries[tag] = (typ, n, entry + 8)

        comp = 1
        if TAG_COMPRESSION in entries:
            got = _values(data, *entries[TAG_COMPRESSION], le)
            comp = got[0] if got else 1
        name = DEFLATE_TAGS.get(comp)
        report.structure.append(
            f"IFD {index}: {count} entries, compression {comp}"
            + (f" ({name})" if name else ""))

        if name:
            if TAG_TILE_OFFSETS in entries and TAG_TILE_COUNTS in entries:
                offs = _values(data, *entries[TAG_TILE_OFFSETS], le)
                lens = _values(data, *entries[TAG_TILE_COUNTS], le)
                label = "tile"
            else:
                offs = _values(data, *entries.get(TAG_STRIP_OFFSETS, (0, 0, 0)), le)
                lens = _values(data, *entries.get(TAG_STRIP_COUNTS, (0, 0, 0)), le)
                label = "strip"
            for k, (off, ln) in enumerate(zip(offs, lens)):
                if not ln or off + ln > len(data):
                    report.warnings.append(
                        f"IFD {index} {label} {k} runs past the end of the file")
                    continue
                s = stream_at(data, off, ln, f"ifd{index}.{label}{k}", "tiff")
                if s is not None:
                    report.streams.append(s)
            if not offs:
                report.warnings.append(
                    f"IFD {index} claims Deflate but records no strips or tiles")
        elif comp != 1:
            report.warnings.append(
                f"IFD {index} uses compression {comp}, which is not DEFLATE")

        nxt = ifd + 2 + count * 12
        ifd = _u32(data, nxt, le) if nxt + 4 <= len(data) else 0
        index += 1

    if not report.streams and not report.warnings:
        report.warnings.append("no Deflate-compressed image data in this TIFF")
    return report


def looks_like_tiff(data: bytes) -> bool:
    return data[:4] in (b"II\x2a\x00", b"MM\x00\x2a")
