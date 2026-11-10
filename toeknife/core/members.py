"""A multi-member gzip written back as one file.

gunzip writes every member's output end to end, so the file a multi-member
gzip holds is the members' outputs joined. Each member is its own session
here, because each has its own trailer to prove it by; this puts them back
together. A member that falls short of its recorded size is padded to that
size with the fill byte, so everything after it lands at its true offset.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Part:
    name: str
    start: int              # offset of this member's first byte in the file
    length: int             # bytes it contributes, padding included
    known: int
    verdict: object         # Session.verify(), or None


def multi(report) -> bool:
    """Whether "all" means anything: a gzip with more than one member."""
    return (report is not None and report.kind == "gzip"
            and len(report.streams) > 1)


def open_all(data: bytes, report, model=None) -> list:
    """A session for every member, in file order."""
    from .session import Session
    out = []
    for stream in report.streams:
        if not stream.decodable or stream.length is None:
            out.append(None)
            continue
        out.append(Session(stream.slice(data), mode="stream", model=model,
                           stream=stream))
    return out


def join(report, sessions, fill: int = 0x3F) -> tuple[bytes, bytes, list]:
    """(buffer, known mask, parts) for the whole file."""
    buf, mask, parts = bytearray(), bytearray(), []
    for stream, s in zip(report.streams, sessions):
        start = len(buf)
        if s is not None:
            b, m, _ = s.full(fill)
            buf += b
            mask += m
        want = stream.uncompressed_size
        if want is not None and len(buf) - start < want:
            pad = want - (len(buf) - start)
            buf += bytes([fill]) * pad
            mask += bytes(pad)
        parts.append(Part(stream.name, start, len(buf) - start,
                          sum(mask[start:]),
                          s.verify() if s is not None else None))
    return bytes(buf), bytes(mask), parts
