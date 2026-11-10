"""zlib (RFC 1950).

Two header bytes, the deflate data, and a four-byte Adler-32 of the original.
The signature is weak: exhaustively, 66 of the 65536 possible byte pairs satisfy
the method, window and checksum-multiple constraints, so random data offers a
false header about once every 1100 bytes. Requiring the deflate behind it to
actually decode takes that to about once every 15000. Detection therefore always
confirms by decoding.

A stream with FDICT set compressed against a preset dictionary that is not in
the file. Other tools refuse those. ToeKnife decodes them anyway and lets the
references that reach into the missing dictionary become holes, which is what
the constraint tape is for: a back-reference to a dictionary you do not have is
the same problem as a back-reference behind the start of a fragment.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from ..core.inflate import Decoder
from .base import RAW, ContainerReport, Stream

HEADER_LEN = 2
DICTID_LEN = 4
TRAILER_LEN = 4
CM_DEFLATE = 8
FDICT = 0x20

LEVELS = {0: "fastest", 1: "fast", 2: "default", 3: "best"}


@dataclass
class Header:
    offset: int
    cm: int
    cinfo: int
    flevel: int
    fdict: bool
    dictid: int | None
    data_offset: int

    @property
    def window(self) -> int:
        return 1 << (self.cinfo + 8)


def read_header(data: bytes, pos: int = 0) -> Header | None:
    if pos + HEADER_LEN > len(data):
        return None
    cmf, flg = data[pos], data[pos + 1]
    if (cmf & 0x0F) != CM_DEFLATE:
        return None
    if ((cmf << 8) | flg) % 31 != 0:
        return None
    cinfo = cmf >> 4
    if cinfo > 7:                      # window above 32K is not valid deflate
        return None
    at = pos + HEADER_LEN
    dictid = None
    if flg & FDICT:
        if at + DICTID_LEN > len(data):
            return None
        dictid = struct.unpack_from(">I", data, at)[0]
        at += DICTID_LEN
    return Header(pos, cmf & 0x0F, cinfo, (flg >> 6) & 3, bool(flg & FDICT),
                  dictid, at)


def parse(data: bytes) -> ContainerReport:
    report = ContainerReport(kind="zlib")
    head = read_header(data)
    if head is None:
        report.warnings.append("no zlib header at offset 0")
        return report

    length = None
    status = "ok"
    try:
        trace = Decoder(data, strict_window=not head.fdict).stream(head.data_offset * 8)
        if trace.status == "end":
            length = (trace.bit_end + 7) // 8 - head.data_offset
        else:
            status = trace.status
    except Exception as e:                       # pragma: no cover - defensive
        status = f"decode failed: {e}"

    adler = None
    if length is not None:
        end = head.data_offset + length
        if end + TRAILER_LEN <= len(data):
            adler = struct.unpack_from(">I", data, end)[0]
        else:
            status = "truncated: no Adler-32 trailer"

    report.structure.append(
        f"zlib header at 0: window {head.window} bytes, "
        f"{LEVELS.get(head.flevel, head.flevel)} compression level"
        + (f", preset dictionary {head.dictid:08x}" if head.fdict else ""))

    notes = []
    if status != "ok":
        notes.append(status)
    if head.fdict:
        notes.append(f"compressed against preset dictionary {head.dictid:08x}, "
                     f"which is not in this file: references reaching into it "
                     f"become holes")
    report.streams.append(Stream(
        name="zlib stream", container="zlib", offset=head.data_offset,
        length=length, encoding=RAW, method=8, method_name="deflate",
        adler32=adler, source="header", notes=notes,
        references_outside=head.fdict, dictid=head.dictid))
    return report


def stream_at(data: bytes, offset: int, length: int, name: str, container: str,
              segments=None, notes=None, segment_crcs=None,
              cut_short: bool = False) -> Stream | None:
    """A Stream for a zlib-wrapped run embedded in some larger format.

    PNG, SWF, WOFF, TIFF and Anvil all store their DEFLATE inside zlib rather
    than raw, so they all want the same thing: strip the two header bytes, keep
    the Adler-32 off the end as a constraint, and point at what is between.
    """
    body = (b"".join(data[o:o + l] for o, l in segments) if segments
            else data[offset:offset + length])
    head = read_header(body, 0)
    if head is None:
        return None
    adler = None
    if not cut_short and len(body) >= head.data_offset + TRAILER_LEN:
        adler = struct.unpack_from(">I", body, len(body) - TRAILER_LEN)[0]
    notes = list(notes or [])
    if cut_short:
        # The file ends inside the stream: its last 4 bytes are compressed
        # data, and the Adler-32 went with the rest.
        notes.append("the file ends inside this stream; its Adler-32 is gone")
    if head.fdict:
        notes.append(f"preset dictionary {head.dictid:08x}, not in this file")
    return Stream(
        name=name, container=container,
        offset=offset + (head.data_offset if segments is None else 0),
        length=(None if segments else max(0, length - head.data_offset
                                          - (0 if cut_short else TRAILER_LEN))),
        encoding=RAW, method=8, method_name="deflate", adler32=adler,
        source="header", notes=notes, references_outside=head.fdict,
        dictid=head.dictid, segments=segments, segment_crcs=segment_crcs,
        header_skip=head.data_offset if segments else 0,
        trailer_skip=TRAILER_LEN if segments else 0)


def looks_like_zlib(data: bytes) -> bool:
    """A weak signature, so confirm it by decoding.

    Measured on random data: the header check alone fires about once per 1100
    bytes, and once per 15000 after decode-confirmation.
    """
    head = read_header(data)
    if head is None:
        return False
    try:
        trace = Decoder(data, strict_window=not head.fdict).stream(
            head.data_offset * 8, max_symbols=64)
    except Exception:
        return False
    return trace.status in ("end", "limit", "eob", "truncated") and bool(trace.blocks)
