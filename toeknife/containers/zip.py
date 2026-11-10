"""ZIP (PKWARE APPNOTE), and everything built on it: jar, apk, docx, xlsx, epub.

A ZIP describes its contents twice: once per entry in a local file header, and
once centrally in the directory at the end.  Ordinary readers trust the central
directory.  A forensic reader cannot, because the tail of a file is exactly what
truncation removes, so this parses both, reconciles them, and reports where they
disagree.  Disagreement is not noise: a local header that contradicts the
central directory is either damage or someone editing an archive by hand.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from .base import RAW, ContainerReport, Stream

LOCAL_SIG = b"PK\x03\x04"
CD_SIG = b"PK\x01\x02"
EOCD_SIG = b"PK\x05\x06"
DD_SIG = b"PK\x07\x08"
ZIP64_EOCD_SIG = b"PK\x06\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"

LOCAL_HEADER_LEN = 30
CD_HEADER_LEN = 46
EOCD_LEN = 22
ZIP64_LOCATOR_LEN = 20
ZIP64_SENTINEL = 0xFFFFFFFF
ZIP64_SENTINEL16 = 0xFFFF
# An EOCD comment is a 16-bit length, so the record can start at most that far
# from the end of the file.
EOCD_SEARCH_SPAN = EOCD_LEN + 0xFFFF

FLAG_ENCRYPTED = 0x0001
FLAG_DATA_DESCRIPTOR = 0x0008
FLAG_STRONG_ENCRYPTION = 0x0040
FLAG_UTF8_NAME = 0x0800

METHODS = {
    0: "stored", 1: "shrunk", 2: "reduce1", 3: "reduce2", 4: "reduce3",
    5: "reduce4", 6: "imploded", 8: "deflate", 9: "deflate64", 10: "dclimplode",
    12: "bzip2", 14: "lzma", 16: "cmpsc", 18: "terse", 19: "lz77",
    93: "zstd", 94: "mp3", 95: "xz", 96: "jpeg", 97: "wavpack", 98: "ppmd",
    99: "aes",
}


def method_name(m: int) -> str:
    return METHODS.get(m, f"method{m}")


@dataclass
class Entry:
    """One entry, as described by one of the two places ZIP describes it."""
    name: str
    source: str                    # 'central' | 'local' | 'scan'
    header_offset: int
    data_offset: int
    method: int
    flags: int
    crc32: int
    compressed_size: int | None
    uncompressed_size: int | None
    zip64: bool = False

    @property
    def encrypted(self) -> bool:
        return bool(self.flags & (FLAG_ENCRYPTED | FLAG_STRONG_ENCRYPTION))

    @property
    def streamed(self) -> bool:
        """Sizes live in a trailing data descriptor, not the local header."""
        return bool(self.flags & FLAG_DATA_DESCRIPTOR)


def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def _name(raw: bytes, flags: int) -> str:
    enc = "utf-8" if flags & FLAG_UTF8_NAME else "cp437"
    try:
        return raw.decode(enc)
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def find_eocd(data: bytes) -> int | None:
    """Locate the end-of-central-directory record.

    Scans backward and accepts a candidate only when its declared comment length
    accounts for exactly the bytes that follow it.  Without that check, a
    `PK\\x05\\x06` sitting inside a file comment or inside compressed data will
    be mistaken for the real record.
    """
    start = max(0, len(data) - EOCD_SEARCH_SPAN)
    pos = data.rfind(EOCD_SIG, start)
    while pos >= 0:
        if pos + EOCD_LEN <= len(data):
            comment_len = _u16(data, pos + 20)
            if pos + EOCD_LEN + comment_len == len(data):
                return pos
        nxt = data.rfind(EOCD_SIG, start, pos)
        if nxt == pos:
            break
        pos = nxt
    return None


def _zip64_eocd(data: bytes, eocd_pos: int):
    """Follow the ZIP64 locator, if the archive has one."""
    loc = eocd_pos - ZIP64_LOCATOR_LEN
    if loc < 0 or data[loc:loc + 4] != ZIP64_LOCATOR_SIG:
        return None
    z64 = _u64(data, loc + 8)
    if z64 + 56 > len(data) or data[z64:z64 + 4] != ZIP64_EOCD_SIG:
        return None
    return {"entries": _u64(data, z64 + 32),
            "cd_size": _u64(data, z64 + 40),
            "cd_offset": _u64(data, z64 + 48)}


def _zip64_extra(extra: bytes, want_sizes: bool, want_offset: bool):
    """Pull the real values out of a ZIP64 extended information field (0x0001).

    Fields appear only when the 32-bit slot held the sentinel, in a fixed order,
    so which ones are present depends on the header that referenced them.
    """
    pos = 0
    while pos + 4 <= len(extra):
        hid = _u16(extra, pos)
        size = _u16(extra, pos + 2)
        body = extra[pos + 4:pos + 4 + size]
        pos += 4 + size
        if hid != 0x0001:
            continue
        out, at = {}, 0
        if want_sizes and at + 16 <= len(body):
            out["uncompressed_size"] = _u64(body, at)
            out["compressed_size"] = _u64(body, at + 8)
            at += 16
        if want_offset and at + 8 <= len(body):
            out["header_offset"] = _u64(body, at)
        return out
    return {}


def walk_central(data: bytes, cd_offset: int, count: int | None,
                 warnings: list[str]):
    """Read the central directory.  Yields Entry objects with no data offset yet."""
    pos, seen = cd_offset, 0
    while pos + CD_HEADER_LEN <= len(data) and data[pos:pos + 4] == CD_SIG:
        flags = _u16(data, pos + 8)
        method = _u16(data, pos + 10)
        crc = _u32(data, pos + 16)
        csize = _u32(data, pos + 20)
        usize = _u32(data, pos + 24)
        nlen = _u16(data, pos + 28)
        elen = _u16(data, pos + 30)
        clen = _u16(data, pos + 32)
        hoff = _u32(data, pos + 42)
        name = _name(data[pos + CD_HEADER_LEN:pos + CD_HEADER_LEN + nlen], flags)
        extra = data[pos + CD_HEADER_LEN + nlen:pos + CD_HEADER_LEN + nlen + elen]
        z64 = (csize == ZIP64_SENTINEL or usize == ZIP64_SENTINEL
               or hoff == ZIP64_SENTINEL)
        if z64:
            got = _zip64_extra(extra,
                               csize == ZIP64_SENTINEL or usize == ZIP64_SENTINEL,
                               hoff == ZIP64_SENTINEL)
            usize = got.get("uncompressed_size", usize)
            csize = got.get("compressed_size", csize)
            hoff = got.get("header_offset", hoff)
        yield Entry(name, "central", hoff, -1, method, flags, crc, csize, usize, z64)
        pos += CD_HEADER_LEN + nlen + elen + clen
        seen += 1
        if count is not None and seen >= count:
            break
    if count is not None and seen < count:
        warnings.append(
            f"central directory declares {count} entries but only {seen} parsed; "
            f"the directory is truncated or damaged")


def read_local(data: bytes, pos: int, source: str = "local") -> Entry | None:
    """Read one local file header at pos."""
    if pos + LOCAL_HEADER_LEN > len(data) or data[pos:pos + 4] != LOCAL_SIG:
        return None
    flags = _u16(data, pos + 6)
    method = _u16(data, pos + 8)
    crc = _u32(data, pos + 14)
    csize = _u32(data, pos + 18)
    usize = _u32(data, pos + 22)
    nlen = _u16(data, pos + 26)
    elen = _u16(data, pos + 28)
    if pos + LOCAL_HEADER_LEN + nlen + elen > len(data):
        return None
    name = _name(data[pos + LOCAL_HEADER_LEN:pos + LOCAL_HEADER_LEN + nlen], flags)
    extra = data[pos + LOCAL_HEADER_LEN + nlen:pos + LOCAL_HEADER_LEN + nlen + elen]
    z64 = csize == ZIP64_SENTINEL or usize == ZIP64_SENTINEL
    if z64:
        got = _zip64_extra(extra, True, False)
        usize = got.get("uncompressed_size", usize)
        csize = got.get("compressed_size", csize)
    data_at = pos + LOCAL_HEADER_LEN + nlen + elen
    e = Entry(name, source, pos, data_at, method, flags, crc,
              None if csize == 0 else csize, None if usize == 0 else usize, z64)
    if e.streamed:
        # Spec says the local header carries zeros here; the truth is in the
        # data descriptor after the data, or in the central directory.
        e.compressed_size = None if csize == 0 else csize
        e.uncompressed_size = None if usize == 0 else usize
    return e


def scan_local_headers(data: bytes, warnings: list[str]) -> list[Entry]:
    """Find entries by signature when the directory cannot be trusted.

    A four-byte signature alone is weak, so each hit is only kept when the rest
    of the header is self-consistent: the name and extra field must fit, and the
    method must be one PKWARE actually defines.
    """
    out, pos = [], data.find(LOCAL_SIG)
    while pos >= 0:
        e = read_local(data, pos, source="scan")
        if e is not None and (e.method in METHODS or e.method == 8):
            out.append(e)
        pos = data.find(LOCAL_SIG, pos + 1)
    if out:
        warnings.append(f"recovered {len(out)} entries by scanning for local "
                        f"headers; sizes may be missing for streamed entries")
    return out


def read_data_descriptor(data: bytes, at: int, zip64: bool):
    """Read a data descriptor, whose signature is optional."""
    pos = at
    if data[pos:pos + 4] == DD_SIG:
        pos += 4
    width = 8 if zip64 else 4
    if pos + 4 + 2 * width > len(data):
        return None
    crc = _u32(data, pos)
    csize = _u64(data, pos + 4) if zip64 else _u32(data, pos + 4)
    usize = _u64(data, pos + 4 + width) if zip64 else _u32(data, pos + 8)
    return {"crc32": crc, "compressed_size": csize, "uncompressed_size": usize}


def parse(data: bytes) -> ContainerReport:
    report = ContainerReport(kind="zip")
    warnings, structure = report.warnings, report.structure

    eocd = find_eocd(data)
    central: list[Entry] = []
    if eocd is None:
        warnings.append("no end-of-central-directory record: the archive is "
                        "truncated, or this is only a piece of one")
    else:
        total = _u16(data, eocd + 10)
        cd_size = _u32(data, eocd + 12)
        cd_offset = _u32(data, eocd + 16)
        comment_len = _u16(data, eocd + 20)
        structure.append(f"EOCD at {eocd}: {total} entries, "
                         f"central directory {cd_size} B at {cd_offset}"
                         + (f", {comment_len} B comment" if comment_len else ""))
        z64 = _zip64_eocd(data, eocd)
        if z64:
            total, cd_size, cd_offset = z64["entries"], z64["cd_size"], z64["cd_offset"]
            structure.append(f"ZIP64 EOCD: {total} entries, "
                             f"central directory {cd_size} B at {cd_offset}")

        # A self-extracting stub, or a carve that lost the front, shifts every
        # recorded offset by a constant.  The directory's true position is
        # derivable from the EOCD, so the shift is measurable.
        actual = eocd - cd_size
        shift = 0
        if actual >= 0 and data[actual:actual + 4] == CD_SIG and actual != cd_offset:
            shift = actual - cd_offset
            structure.append(f"offsets are shifted by {shift:+} bytes "
                             f"(prepended data, or a truncated front)")
            cd_offset = actual
        if cd_offset < len(data) and data[cd_offset:cd_offset + 4] == CD_SIG:
            central = list(walk_central(data, cd_offset, total, warnings))
            for e in central:
                e.header_offset += shift
        else:
            warnings.append(f"no central directory header at {cd_offset}")

    locals_: list[Entry] = []
    for e in central:
        got = read_local(data, e.header_offset)
        if got is not None:
            locals_.append(got)
    if not central or len(locals_) < len(central):
        found = scan_local_headers(data, warnings)
        known = {e.header_offset for e in locals_}
        locals_.extend(e for e in found if e.header_offset not in known)

    by_offset = {e.header_offset: e for e in locals_}
    seen_offsets = set()
    for e in central:
        local = by_offset.get(e.header_offset)
        seen_offsets.add(e.header_offset)
        report.streams.append(_to_stream(data, e, local, warnings))
    for e in locals_:
        if e.header_offset not in seen_offsets:
            report.streams.append(_to_stream(data, e, e, warnings))

    report.streams.sort(key=lambda s: s.offset)
    if not report.streams:
        warnings.append("no ZIP entries found")
    return report


def _to_stream(data: bytes, entry: Entry, local: Entry | None,
               warnings: list[str]) -> Stream:
    """Combine the two descriptions of an entry into one located stream."""
    notes = []
    data_offset = local.data_offset if local is not None else -1
    if data_offset < 0:
        # No readable local header: the data offset cannot be known exactly,
        # because it depends on the name and extra-field lengths stored there.
        notes.append("local header unreadable; data offset is a guess")
        data_offset = entry.header_offset + LOCAL_HEADER_LEN

    csize = entry.compressed_size
    usize = entry.uncompressed_size
    crc = entry.crc32
    source = entry.source

    if local is not None and local is not entry:
        for field_, a, b in (("method", entry.method, local.method),
                             ("crc32", entry.crc32, local.crc32)):
            if b and a != b:
                notes.append(f"{field_} disagrees: central {a}, local {b}")
        if local.compressed_size and csize and local.compressed_size != csize:
            notes.append(f"compressed size disagrees: central {csize}, "
                         f"local {local.compressed_size}")
        if csize is None:
            csize = local.compressed_size
        if usize is None:
            usize = local.uncompressed_size
        if not crc:
            crc = local.crc32

    if entry.streamed and csize is None:
        dd = None
        if local is not None:
            nxt = data.find(LOCAL_SIG, data_offset)
            hunt = nxt if nxt >= 0 else len(data)
            for back in (16, 20, 24, 12):
                at = hunt - back
                if at > data_offset:
                    cand = read_data_descriptor(data, at, entry.zip64)
                    if cand and cand["compressed_size"] == at - data_offset:
                        dd = cand
                        break
        if dd:
            csize, usize, crc = (dd["compressed_size"], dd["uncompressed_size"],
                                 dd["crc32"])
            notes.append("sizes recovered from the trailing data descriptor")
        else:
            notes.append("streamed entry: no size recorded, decoding to end of "
                         "final block")

    if entry.streamed and not crc:
        crc = None
    return Stream(
        name=entry.name or f"<entry at {entry.header_offset}>",
        container="zip", offset=data_offset, length=csize, encoding=RAW,
        method=entry.method, method_name=method_name(entry.method),
        crc32=crc or None, uncompressed_size=usize,
        encrypted=entry.encrypted, source=source, notes=notes)


def looks_like_zip(data: bytes) -> bool:
    return data[:4] in (LOCAL_SIG, CD_SIG, EOCD_SIG) or find_eocd(data) is not None
