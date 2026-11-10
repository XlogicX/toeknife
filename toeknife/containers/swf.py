"""Compressed SWF: an eight-byte header then a zlib stream of the body."""
from __future__ import annotations

import struct

from .base import ContainerReport
from .zlibfmt import stream_at

HEADER_LEN = 8
SIGNATURES = {b"CWS": "zlib", b"FWS": "uncompressed", b"ZWS": "lzma"}


def parse(data: bytes) -> ContainerReport:
    report = ContainerReport(kind="swf")
    sig = data[:3]
    kind = SIGNATURES.get(sig)
    version = data[3] if len(data) > 3 else 0
    declared = struct.unpack_from("<I", data, 4)[0] if len(data) >= 8 else 0
    report.structure.append(
        f"SWF v{version}, {kind or 'unknown'} body, header declares "
        f"{declared} bytes uncompressed")
    if declared and declared != len(data) and kind == "uncompressed":
        report.warnings.append(
            f"header declares {declared} bytes but the file is {len(data)}")

    if kind == "uncompressed":
        report.warnings.append("FWS: this SWF is not compressed")
        return report
    if kind == "lzma":
        report.warnings.append("ZWS: this SWF uses LZMA, not DEFLATE")
        return report

    s = stream_at(data, HEADER_LEN, len(data) - HEADER_LEN, "SWF body", "swf")
    if s is None:
        report.warnings.append("no zlib header after the SWF header")
        return report
    # The SWF header's length field counts the decompressed file including the
    # eight-byte header, so the body is that minus the header.
    if declared > HEADER_LEN:
        s.uncompressed_size = declared - HEADER_LEN
    report.streams.append(s)
    return report


def looks_like_swf(data: bytes) -> bool:
    return data[:3] in SIGNATURES and len(data) > HEADER_LEN
