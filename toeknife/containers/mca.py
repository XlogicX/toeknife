"""Minecraft Anvil region files (.mca, .mcr): up to 1024 chunks in 4 KiB sectors.

Two 4 KiB tables at the front: a location table of 1024 entries, each a 3-byte
sector offset and a 1-byte sector count, then 1024 timestamps. A zero entry
means that chunk was never generated.

Each chunk begins with a 4-byte length and a 1-byte compression scheme, so
unlike most containers here the compression is per chunk and a single file can
mix gzip and zlib.
"""
from __future__ import annotations

import struct

from .base import ContainerReport, Stream
from .zlibfmt import stream_at

SECTOR = 4096
LOCATIONS = 1024
HEADER_SECTORS = 2
SCHEMES = {1: "gzip", 2: "zlib", 3: "stored", 4: "lz4", 127: "custom"}


def parse(data: bytes) -> ContainerReport:
    report = ContainerReport(kind="mca")
    if len(data) < SECTOR * HEADER_SECTORS:
        report.warnings.append("shorter than the two header sectors")
        return report

    present = 0
    schemes: dict[str, int] = {}
    for i in range(LOCATIONS):
        off = struct.unpack_from(">I", data, i * 4)[0]
        sector, count = off >> 8, off & 0xFF
        if not sector or not count:
            continue
        present += 1
        start = sector * SECTOR
        if start + 5 > len(data):
            report.warnings.append(
                f"chunk {i} points at sector {sector}, past the end of the file")
            continue
        length = struct.unpack_from(">I", data, start)[0]
        scheme = data[start + 4]
        payload, plen = start + 5, max(0, length - 1)
        if payload + plen > len(data):
            report.warnings.append(f"chunk {i} declares {plen} bytes but the "
                                   f"file ends first")
            plen = len(data) - payload
        label = SCHEMES.get(scheme, f"scheme{scheme}")
        schemes[label] = schemes.get(label, 0) + 1
        cx, cz = i % 32, i // 32

        if scheme == 2:
            s = stream_at(data, payload, plen, f"chunk {cx},{cz}", "mca")
            if s is None:
                report.warnings.append(f"chunk {i} claims zlib but has no header")
                continue
            report.streams.append(s)
        elif scheme == 1:
            from .gzipfmt import read_header
            member = read_header(data, payload)
            if member is None:
                report.warnings.append(f"chunk {i} claims gzip but has no header")
                continue
            report.streams.append(Stream(
                name=f"chunk {cx},{cz}", container="mca",
                offset=member.data_offset,
                length=payload + plen - member.data_offset - 8,
                method=8, method_name="deflate", source="header",
                notes=["gzip-framed chunk"]))
        else:
            report.streams.append(Stream(
                name=f"chunk {cx},{cz}", container="mca", offset=payload,
                length=plen, method=0 if scheme == 3 else scheme,
                method_name=label, source="header",
                notes=[f"compression scheme {scheme} ({label})"]))

    report.structure.append(
        f"{present} of {LOCATIONS} chunk slots in use"
        + (": " + ", ".join(f"{n}x {k}" for k, n in sorted(schemes.items()))
           if schemes else ""))
    if not present:
        report.warnings.append("the location table is empty: no chunks generated")
    return report


def looks_like_mca(data: bytes) -> bool:
    """Anvil has no magic, so judge it by the shape of its location table.

    A region file is a whole number of 4 KiB sectors, and every entry in the
    location table must point inside the file and past the two header sectors.
    That is enough structure to be safe, and it is checked strictly because
    there is no signature to fall back on.
    """
    if len(data) < SECTOR * (HEADER_SECTORS + 1) or len(data) % SECTOR:
        return False
    used = 0
    for i in range(LOCATIONS):
        off = struct.unpack_from(">I", data, i * 4)[0]
        sector, count = off >> 8, off & 0xFF
        if not sector and not count:
            continue
        if sector < HEADER_SECTORS or not count:
            return False
        if (sector + count) * SECTOR > len(data):
            return False
        used += 1
    return used > 0
