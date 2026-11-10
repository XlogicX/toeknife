"""PDF: object streams filtered through /FlateDecode.

Locating them by parsing the cross-reference table is the correct approach for a
valid file and the wrong one here, because the xref is at the end and the end is
what damage takes. So this scans for `stream` keywords and reads the dictionary
immediately before each, which works on a file whose trailer is gone.

/Length is often an indirect reference, and resolving it needs the xref we are
refusing to depend on, so the extent comes from the matching `endstream` instead.
"""
from __future__ import annotations

import re

from .base import RAW, ZLIB, ContainerReport, Stream
from .zlibfmt import read_header, stream_at

SIGNATURE = b"%PDF-"
DICT_WINDOW = 900
STREAM_KW = re.compile(rb"stream(\r\n|\r|\n)")


def _filters(dictionary: bytes) -> list[str]:
    return [m.decode() for m in re.findall(rb"/(\w+Decode)", dictionary)]


def parse(data: bytes) -> ContainerReport:
    report = ContainerReport(kind="pdf")
    version = data[5:8].decode("latin-1", "replace") if len(data) > 8 else "?"
    report.structure.append(f"PDF {version}")
    if re.search(rb"/Encrypt\b", data):
        report.warnings.append(
            "the document is encrypted; its streams cannot be decompressed "
            "without the key")

    index = 0
    skip_until = 0
    for m in STREAM_KW.finditer(data):
        start = m.end()
        # A `stream` keyword inside an already-identified stream body is bytes
        # of compressed data that happen to spell it, not a new object.
        if m.start() < skip_until:
            continue
        head = data[max(0, m.start() - DICT_WINDOW):m.start()]
        if b"<<" not in head:
            continue
        # The keyword has to directly follow the dictionary that describes it.
        # Without this, any occurrence of the word inside compressed data picks
        # up whatever dictionary happens to precede it and is reported as an
        # object that is not there.
        if head.rstrip(b" \t\r\n")[-2:] != b">>":
            continue
        dictionary = head[head.rfind(b"<<"):]
        filters = _filters(dictionary)
        if "FlateDecode" not in filters:
            continue
        end = data.find(b"endstream", start)
        cut = end < 0
        if end < 0:
            end = len(data)
        # endstream is preceded by an EOL that is not part of the data.
        body_end = end
        while body_end > start and data[body_end - 1] in (0x0A, 0x0D):
            body_end -= 1

        # The object this stream belongs to is the LAST header in the window;
        # the window reaches back past earlier, shorter objects.
        objs = list(re.finditer(rb"(\d+)\s+(\d+)\s+obj\b", head))
        obj = objs[-1] if objs else None
        name = f"obj {obj.group(1).decode()}" if obj else f"stream@{start}"
        if cut:
            report.warnings.append(
                f"{name} has no endstream: the file ends inside it")
        notes = []
        if len(filters) > 1:
            notes.append("filter chain " + " ".join(filters)
                         + "; the other filters still have to be undone by hand")

        length = body_end - start
        skip_until = end
        if read_header(data[start:start + 2]) is not None:
            s = stream_at(data, start, length, name, "pdf", notes=notes,
                          cut_short=cut)
            if s is not None:
                report.streams.append(s)
                index += 1
                continue
        # Some producers write raw DEFLATE under /FlateDecode despite the spec
        # calling for a zlib wrapper, so fall back rather than skipping it.
        notes.append("no zlib header; treating the body as raw DEFLATE")
        report.streams.append(Stream(
            name=name, container="pdf", offset=start, length=length,
            encoding=RAW, method=8, method_name="deflate", source="scan",
            notes=notes))
        index += 1

    report.structure.append(f"{index} /FlateDecode stream(s) found by scanning")
    if not report.streams:
        report.warnings.append("no /FlateDecode streams found")
    return report


def looks_like_pdf(data: bytes) -> bool:
    return data[:5] == SIGNATURE
