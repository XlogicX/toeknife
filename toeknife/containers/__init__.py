"""Locating DEFLATE streams inside the wrappers people actually store them in.

ZIP, gzip and zlib so far.  The rest of the family (PNG, PDF, SWF, WOFF, TIFF,
Minecraft Anvil, pcap) plugs in behind the same `parse(data) -> ContainerReport`
interface.

pcap is the odd one: it holds a stream that was carried rather than stored, so it
is the only container here that can report bytes as MISSING rather than merely
absent -- TCP numbers them, so a lost packet leaves a hole of measured width.
"""
from __future__ import annotations

from .base import ContainerReport, GZIP, RAW, Stream, Verdict, ZLIB, verify
from . import gzipfmt, mca, pcap, pdf, png, swf, tiff, woff, zlibfmt
from . import zip as zip_container

__all__ = ["ContainerReport", "Stream", "Verdict", "verify", "detect", "parse",
           "RAW", "ZLIB", "GZIP"]

# Order matters: strong signatures first.  Anvil has no magic at all and is
# recognised by the shape of its location table, and a zlib header is two bytes
# with only a handful of constrained bits, so both are tried last and only after
# everything with a real signature has declined.
PARSERS = {
    "pcap": (pcap.looks_like_pcap, pcap.parse),
    "gzip": (gzipfmt.looks_like_gzip, gzipfmt.parse),
    "png": (png.looks_like_png, png.parse),
    "zip": (zip_container.looks_like_zip, zip_container.parse),
    "pdf": (pdf.looks_like_pdf, pdf.parse),
    "swf": (swf.looks_like_swf, swf.parse),
    "woff": (woff.looks_like_woff, woff.parse),
    "tiff": (tiff.looks_like_tiff, tiff.parse),
    "mca": (mca.looks_like_mca, mca.parse),
    "zlib": (zlibfmt.looks_like_zlib, zlibfmt.parse),
}


def detect(data: bytes) -> str | None:
    for kind, (sniff, _) in PARSERS.items():
        if sniff(data):
            return kind
    return None


def parse(data: bytes, kind: str | None = None) -> ContainerReport | None:
    kind = kind or detect(data)
    if kind is None or kind not in PARSERS:
        return None
    try:
        return PARSERS[kind][1](data)
    except Exception as e:
        # Damaged input is the whole point, and a parser that meets a layout
        # it did not foresee says so rather than ending the run in a
        # traceback (wrap-up #2's fuzzing found one, in TIFF).
        report = ContainerReport(kind=kind)
        report.warnings.append(f"the {kind} parser stopped on this file: "
                               f"{type(e).__name__}: {e}")
        return report
