"""gzip (RFC 1952).

Neither gzip nor zlib records how long its compressed data is, so the end of a
member can only be found by decoding to it. That matters more here than in a
normal reader: a member that fails to decode takes the position of every member
after it with it, so a damaged file falls back to scanning for the magic instead
of walking.

What makes gzip worth reading carefully is the trailer. Eight bytes: a CRC32 of
the original data and its length mod 2^32. That is a known output length and a
checksum, which is what turns "does this decode look right" into arithmetic, and
what would pin the length of a gap if the rest of a stream were missing.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from ..core.inflate import Decoder
from .base import RAW, ContainerReport, Stream

MAGIC = b"\x1f\x8b"
HEADER_LEN = 10
TRAILER_LEN = 8
CM_DEFLATE = 8

FTEXT = 0x01
FHCRC = 0x02
FEXTRA = 0x04
FNAME = 0x08
FCOMMENT = 0x10

OS_NAMES = {
    0: "FAT", 1: "Amiga", 2: "VMS", 3: "Unix", 4: "VM/CMS", 5: "Atari TOS",
    6: "HPFS", 7: "Macintosh", 8: "Z-System", 9: "CP/M", 10: "TOPS-20",
    11: "NTFS", 12: "QDOS", 13: "Acorn RISCOS", 255: "unknown",
}


@dataclass
class Member:
    index: int
    header_offset: int
    data_offset: int
    data_length: int | None
    flags: int
    method: int
    mtime: int
    xfl: int
    os: int
    name: str = ""
    comment: str = ""
    crc32: int | None = None
    isize: int | None = None
    status: str = "ok"
    # The trailer was taken from the end of the file rather than walked to,
    # because the member would not decode. Right for a single-member gzip, wrong
    # if anything follows it, so it travels as a note rather than silently.
    trailer_inferred: bool = False
    trailer_where: str = ""
    # The walk reached this member by scanning past member `found_after`,
    # which did not decode.
    found_after: int | None = None



# The most output DEFLATE can produce per input byte: a 258-byte match costs
# at least two bits, which is 1,032 bytes out per byte in.
MAX_RATIO = 1032

def _cstring(data: bytes, pos: int) -> tuple[str, int]:
    end = data.find(b"\x00", pos)
    if end < 0:
        return data[pos:].decode("latin-1"), len(data)
    return data[pos:end].decode("latin-1"), end + 1


def read_header(data: bytes, pos: int) -> Member | None:
    if pos + HEADER_LEN > len(data) or data[pos:pos + 2] != MAGIC:
        return None
    method = data[pos + 2]
    flags = data[pos + 3]
    mtime = struct.unpack_from("<I", data, pos + 4)[0]
    xfl, os_id = data[pos + 8], data[pos + 9]
    at = pos + HEADER_LEN
    name = comment = ""
    if flags & FEXTRA:
        if at + 2 > len(data):
            return None
        xlen = struct.unpack_from("<H", data, at)[0]
        at += 2 + xlen
    if flags & FNAME:
        name, at = _cstring(data, at)
    if flags & FCOMMENT:
        comment, at = _cstring(data, at)
    if flags & FHCRC:
        at += 2
    if at > len(data):
        return None
    m = Member(0, pos, at, None, flags, method, mtime, xfl, os_id, name, comment)
    if method != CM_DEFLATE:
        m.status = f"compression method {method}, not deflate"
    return m


def _find_end(data: bytes, data_offset: int) -> tuple[int | None, str]:
    """Decode far enough to learn where the member's deflate data stops."""
    try:
        trace = Decoder(data).stream(data_offset * 8)
    except Exception as e:                       # pragma: no cover - defensive
        return None, f"decode failed: {e}"
    if trace.status != "end":
        return None, trace.status
    return (trace.bit_end + 7) // 8, "ok"


def _next_member(data: bytes, start: int) -> int | None:
    """The first gzip header at or after `start` whose deflate data decodes to
    its end. The magic alone is two bytes and turns up inside compressed data;
    a member that decodes to an end-of-stream is one."""
    at = data.find(MAGIC, start)
    while at >= 0:
        m = read_header(data, at)
        if m is not None and m.method == CM_DEFLATE:
            if _find_end(data, m.data_offset)[0] is not None:
                return at
        at = data.find(MAGIC, at + 1)
    return None


def parse(data: bytes) -> ContainerReport:
    report = ContainerReport(kind="gzip")
    members: list[Member] = []
    pos, index = 0, 0
    walked_cleanly = True
    resumed_after: Member | None = None

    while pos < len(data):
        m = read_header(data, pos)
        if m is None:
            break
        m.index = index
        index += 1
        if resumed_after is not None:
            m.found_after = resumed_after.index
            resumed_after = None
        if m.status != "ok":
            members.append(m)
            walked_cleanly = False
            break
        end, why = _find_end(data, m.data_offset)
        if end is None:
            m.status = f"deflate data did not decode ({why})"
            walked_cleanly = False
            # The end of a member is found by decoding it, so a member that
            # will not decode has no measured end. The next member that does
            # decode gives it one: a member's trailer is the 8 bytes before the
            # next member's header, and that trailer is a CRC32 and a length of
            # the original, the one thing that can confirm a repair of the
            # damage. Headers in between whose data does not decode either are
            # damaged members too; each ends where the next one starts.
            nxt = _next_member(data, m.header_offset + 1)
            chain = [m]
            for at in _later_headers(data, m.header_offset + 1):
                if nxt is not None and at >= nxt:
                    break
                e = read_header(data, at)
                e.index = index
                index += 1
                e.status = (f"deflate data did not decode "
                            f"({_find_end(data, e.data_offset)[1]})")
                chain.append(e)
            for k, c in enumerate(chain):
                if k + 1 < len(chain):
                    bound, who = chain[k + 1].header_offset, chain[k + 1].index
                else:
                    bound, who = nxt, index
                if bound is None:
                    # Nothing after it even looks like a member: it is the
                    # last one, and the file ends with its trailer.
                    if len(data) - c.data_offset > TRAILER_LEN:
                        c.crc32, c.isize = struct.unpack_from(
                            "<II", data, len(data) - TRAILER_LEN)
                        c.data_length = len(data) - TRAILER_LEN - c.data_offset
                        c.trailer_inferred = True
                        c.trailer_where = "the last 8 bytes of the file"
                elif bound - TRAILER_LEN > c.data_offset:
                    c.crc32, c.isize = struct.unpack_from(
                        "<II", data, bound - TRAILER_LEN)
                    c.data_length = bound - TRAILER_LEN - c.data_offset
                    c.trailer_inferred = True
                    c.trailer_where = f"the 8 bytes before member {who}'s header"
            members.extend(chain)
            if nxt is None:
                break
            resumed_after = chain[-1]
            pos = nxt
            continue
        m.data_length = end - m.data_offset
        if end + TRAILER_LEN <= len(data):
            m.crc32, m.isize = struct.unpack_from("<II", data, end)
        else:
            m.status = "truncated: no trailer"
            walked_cleanly = False
        members.append(m)
        pos = end + TRAILER_LEN
        if m.status != "ok":
            break

    if not members:
        report.warnings.append("no gzip member header found")
        return report

    resumed = [m for m in members if m.found_after is not None]
    for m in resumed:
        report.warnings.append(
            f"member {m.found_after} did not decode; the walk picked up again "
            f"at member {m.index} (offset {m.header_offset}), the next gzip "
            f"header found by scanning whose data decodes to its end")

    # What is left after the walk stopped for good: headers that parse, whose
    # data does not decode, so nothing measures where they end.
    if not walked_cleanly and members[-1].status != "ok":
        known = {m.header_offset for m in members}
        found = 0
        for at in _later_headers(data, members[-1].header_offset + 1):
            if at not in known:
                extra = read_header(data, at)
                extra.index = index
                index += 1
                extra.status = "found by scanning; length unknown"
                members.append(extra)
                found += 1
        if found:
            report.warnings.append(
                f"walk stopped early; {found} further member(s) found by scanning "
                f"for the magic, with no reliable length")

    for m in members:
        report.structure.append(
            f"member {m.index} at {m.header_offset}: "
            + (f"{m.name!r} " if m.name else "")
            + f"OS {OS_NAMES.get(m.os, m.os)}, mtime {m.mtime}"
            + (f", {m.comment!r}" if m.comment else "")
            + (f", {m.status}" if m.status != "ok" else ""))
        notes = []
        if m.status != "ok":
            notes.append(m.status)
        # Wherever the trailer came from. A walked one can be damaged too: a
        # burst over member 1's trailer read as 4,185,609,946 bytes, was
        # trusted, and --entry all wrote a 4.2 GB export (wrap-up #2).
        impossible = (m.isize is not None and m.data_length
                      and m.isize > MAX_RATIO * m.data_length)
        if m.trailer_inferred and not impossible:
            if m.trailer_where.startswith("the last"):
                notes.append("CRC32 and length read from the last 8 bytes of "
                             "the file, since the member could not be walked "
                             "to its end -- correct for a single-member gzip, "
                             "and not a claim that this is one")
            else:
                notes.append(f"CRC32 and length read from {m.trailer_where}, "
                             f"where its trailer sits if nothing lies between "
                             f"the two members")
        if m.found_after is not None:
            notes.append(f"found by scanning past member {m.found_after}; "
                         f"decodes to its end")
        if m.flags & FEXTRA:
            notes.append("has an FEXTRA field")
        crc32, isize = m.crc32, m.isize
        if impossible:
            # DEFLATE cannot produce more than about 1,032 bytes per byte, so
            # these 8 bytes are not a trailer: the file was cut short and they
            # are the middle of the stream. Trusted, one such "size" became a
            # 2.8 GB output gap and the process was killed.
            if not m.trailer_inferred:
                notes.append(f"the trailer gives a length of {isize:,}, more "
                             f"than DEFLATE can produce from "
                             f"{m.data_length:,} bytes; it is damaged, and "
                             f"nothing records the size or CRC")
            elif m.trailer_where.startswith("the last"):
                notes.append(f"the last 8 bytes of the file give a length of "
                             f"{isize:,}, more than DEFLATE can produce from "
                             f"{m.data_length:,} bytes; the file is cut short, "
                             f"its trailer is gone, and nothing records the "
                             f"size or CRC")
            else:
                notes.append(f"{m.trailer_where} give a length of {isize:,}, "
                             f"more than DEFLATE can produce from "
                             f"{m.data_length:,} bytes; something lies between "
                             f"the members, and nothing records the size or "
                             f"CRC")
            crc32 = isize = None
        elif isize is not None and m.data_length:
            ratio = isize / max(m.data_length, 1)
            if ratio > 2000:
                notes.append(f"declared output is {ratio:.0f}x the compressed "
                             f"size, which is suspicious")
        report.streams.append(Stream(
            name=m.name or f"member{m.index}",
            container="gzip", offset=m.data_offset, length=m.data_length,
            encoding=RAW, method=8 if m.method == CM_DEFLATE else m.method,
            method_name="deflate" if m.method == CM_DEFLATE else f"method{m.method}",
            crc32=crc32, uncompressed_size=isize,
            source=(m.trailer_where if m.trailer_inferred and crc32 is not None
                    else "header"),
            notes=notes))
    return report


def _later_headers(data: bytes, start: int) -> list[int]:
    """Offsets of every parseable deflate member header at or after `start`."""
    out, at = [], data.find(MAGIC, start)
    while at >= 0:
        m = read_header(data, at)
        if m is not None and m.method == CM_DEFLATE:
            out.append(at)
        at = data.find(MAGIC, at + 1)
    return out


def looks_like_gzip(data: bytes) -> bool:
    """Magic plus a header that parses.

    Deliberately does not require deflate.  A gzip carrying bzip2 is still a
    gzip, and saying "this is a gzip using a method I cannot decode" is more use
    than failing to recognise the file.
    """
    return data[:2] == MAGIC and read_header(data, 0) is not None
