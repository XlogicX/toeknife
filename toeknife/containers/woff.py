"""WOFF 1.0: a table directory, each table optionally zlib-compressed.

A table is compressed when its compLength is less than its origLength, and
stored verbatim when they are equal. The checksums in the directory are over the
*uncompressed* table, so they are a genuine constraint on a recovery, unlike a
CRC over the compressed bytes.
"""
from __future__ import annotations

import struct

from .base import ContainerReport
from .zlibfmt import stream_at

SIGNATURE = b"wOFF"
HEADER_LEN = 44
ENTRY_LEN = 20


def parse(data: bytes) -> ContainerReport:
    report = ContainerReport(kind="woff")
    if len(data) < HEADER_LEN:
        report.warnings.append("truncated: shorter than a WOFF header")
        return report
    flavor = data[4:8]
    declared, num_tables = struct.unpack_from(">IH", data, 8)
    meta_off, meta_len = struct.unpack_from(">II", data, 24)
    priv_off, priv_len = struct.unpack_from(">II", data, 36)
    report.structure.append(
        f"WOFF wrapping {_flavor(flavor)}, {num_tables} tables, "
        f"declares {declared} bytes")
    if declared != len(data):
        report.warnings.append(
            f"header declares {declared} bytes but the file is {len(data)}")

    for i in range(num_tables):
        entry = HEADER_LEN + i * ENTRY_LEN
        if entry + ENTRY_LEN > len(data):
            report.warnings.append(
                f"directory truncated after {i} of {num_tables} entries")
            break
        tag = data[entry:entry + 4]
        offset, comp_len, orig_len, checksum = struct.unpack_from(">IIII", data,
                                                                  entry + 4)
        name = tag.decode("latin-1", "replace").strip()
        if offset + comp_len > len(data):
            report.warnings.append(f"table {name} runs past the end of the file")
            continue
        if comp_len >= orig_len:
            report.structure.append(f"  {name}: stored, {orig_len} bytes")
            continue
        s = stream_at(data, offset, comp_len, f"table {name}", "woff")
        if s is None:
            report.warnings.append(f"table {name} is marked compressed but has "
                                   f"no zlib header")
            continue
        s.uncompressed_size = orig_len
        s.notes.append(f"directory checksum {checksum:08x} is over the "
                       f"uncompressed table")
        report.streams.append(s)

    for label, off, length in (("metadata", meta_off, meta_len),
                               ("private", priv_off, priv_len)):
        if not (off and length and off + length <= len(data)):
            continue
        s = stream_at(data, off, length, label, "woff")
        if s is not None:
            report.streams.append(s)
    if not report.streams:
        report.warnings.append("no compressed tables in this WOFF")
    return report


def looks_like_woff(data: bytes) -> bool:
    return data[:4] == SIGNATURE and len(data) >= HEADER_LEN


def _flavor(raw: bytes) -> str:
    """The wrapped font's sfnt version, named. It is a number for TrueType
    (00 01 00 00), and printed as text it came out blank."""
    named = {b"\x00\x01\x00\x00": "TrueType", b"OTTO": "OpenType (CFF)",
             b"true": "TrueType (Apple)", b"typ1": "PostScript Type 1"}
    return named.get(raw, f"flavour {raw.hex()}")
