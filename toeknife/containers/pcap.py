"""Packet captures: the container that knows exactly what is missing.

Every other container here describes a file that is all present. A capture is
different in kind. It records a byte stream as it was carried, in whatever pieces
the path's MTU dictated, and when one of those pieces is lost the capture still
knows precisely where the hole is and how wide -- because TCP numbers every byte,
and the segments either side of the hole carry those numbers.

That is worth more than it sounds. Everywhere else in this tool, a fragment that
resumes after damage has to be *found*: scan for a block boundary, rank the
candidates, hope the encoder flushed. Here the resumption point is not searched
for, it is read off the wire format. And the width of the hole is not estimated,
it is arithmetic on two sequence numbers.

What the capture does not know is how many bytes the missing compressed bytes
would have produced. That is the output gap, and it has to come from elsewhere --
a gzip trailer's ISIZE, a Content-Length, or a search. Saying which of those
supplied it is the difference between a measurement and a guess, so the stream
records it.

Scope, stated rather than discovered: classic pcap only (pcapng is recognised and
refused), Ethernet, IPv4, TCP. Enough for a captured HTTP response, which is what
DEFLATE on a wire almost always is.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from .base import GZIP, RAW, ZLIB, ContainerReport, Stream

# Classic pcap, in the four spellings that exist: two byte orders x two
# timestamp resolutions. The nanosecond variants differ only in what the second
# field means, which nothing here reads.
MAGICS = {
    b"\xd4\xc3\xb2\xa1": ("<", "us"), b"\xa1\xb2\xc3\xd4": (">", "us"),
    b"\x4d\x3c\xb2\xa1": ("<", "ns"), b"\xa1\xb2\x3c\x4d": (">", "ns"),
}
PCAPNG_MAGIC = b"\x0a\x0d\x0d\x0a"

LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LINUX_SLL = 113
ETHERTYPE_IPV4 = 0x0800
ETHERTYPE_VLAN = 0x8100
PROTO_TCP = 6


@dataclass
class Segment:
    """One run of TCP payload, and where in the FILE it lives."""
    seq: int
    offset: int
    length: int


@dataclass
class Flow:
    """One direction of one TCP connection."""
    src: str
    dst: str
    sport: int
    dport: int
    segments: list[Segment] = field(default_factory=list)

    @property
    def key(self) -> tuple:
        return (self.src, self.sport, self.dst, self.dport)

    @property
    def name(self) -> str:
        return f"{self.src}:{self.sport}->{self.dst}:{self.dport}"


def _ip(raw: bytes) -> str:
    return ".".join(str(b) for b in raw)


def _walk_ethernet(frame: bytes) -> bytes | None:
    """The IPv4 payload of an Ethernet frame, or None."""
    if len(frame) < 14:
        return None
    kind = struct.unpack_from(">H", frame, 12)[0]
    at = 14
    while kind == ETHERTYPE_VLAN and len(frame) >= at + 4:
        kind = struct.unpack_from(">H", frame, at + 2)[0]
        at += 4
    return frame[at:] if kind == ETHERTYPE_IPV4 else None


def _walk_link(frame: bytes, linktype: int) -> bytes | None:
    if linktype == LINKTYPE_ETHERNET:
        return _walk_ethernet(frame)
    if linktype == LINKTYPE_RAW:
        return frame
    if linktype == LINKTYPE_LINUX_SLL and len(frame) > 16:
        return frame[16:] if struct.unpack_from(">H", frame, 14)[0] == \
            ETHERTYPE_IPV4 else None
    return None


def _walk_tcp(ip: bytes, base: int):
    """(flowparts, seq, payload_offset, payload_length) for an IPv4/TCP packet.

    `base` is where `ip` starts in the file, so the returned offset points at the
    payload's real position -- which is what lets a reassembled stream still say
    which bytes of the capture each symbol came from.
    """
    if len(ip) < 20 or (ip[0] >> 4) != 4:
        return None
    ihl = (ip[0] & 0x0F) * 4
    if ihl < 20 or len(ip) < ihl or ip[9] != PROTO_TCP:
        return None
    total = struct.unpack_from(">H", ip, 2)[0]
    # Trust the IP length over the frame length: Ethernet pads short frames, and
    # counting the padding as payload would invent bytes that were never sent.
    end = min(len(ip), total if total else len(ip))
    tcp = ip[ihl:end]
    if len(tcp) < 20:
        return None
    sport, dport, seq = struct.unpack_from(">HHI", tcp, 0)
    off = (tcp[12] >> 4) * 4
    if off < 20 or len(tcp) < off:
        return None
    return ((_ip(ip[12:16]), _ip(ip[16:20]), sport, dport), seq,
            base + ihl + off, len(tcp) - off)


def packets(data: bytes):
    """Every record in a classic pcap: (offset, length, incl, orig, linktype)."""
    if len(data) < 24 or data[:4] not in MAGICS:
        return
    endian, _res = MAGICS[data[:4]]
    linktype = struct.unpack_from(endian + "I", data, 20)[0]
    at = 24
    while at + 16 <= len(data):
        _s, _u, incl, orig = struct.unpack_from(endian + "IIII", data, at)
        at += 16
        if incl > len(data) - at:
            break                       # truncated file: stop, do not guess
        yield at, incl, incl, orig, linktype
        at += incl


def flows(data: bytes) -> dict[tuple, Flow]:
    out: dict[tuple, Flow] = {}
    for off, length, incl, orig, linktype in packets(data):
        if incl < orig:
            continue                    # snapped short; handled by the caller
        ip = _walk_link(data[off:off + length], linktype)
        if ip is None:
            continue
        got = _walk_tcp(ip, off + (length - len(ip)))
        if got is None:
            continue
        (src, dst, sport, dport), seq, poff, plen = got
        if not plen:
            continue                    # pure ACK, SYN, FIN: no bytes carried
        key = (src, sport, dst, dport)
        flow = out.get(key)
        if flow is None:
            flow = out[key] = Flow(src, dst, sport, dport)
        flow.segments.append(Segment(seq, poff, plen))
    return out


def reassemble(flow: Flow) -> tuple[list[tuple[int, int]], list[tuple[int, int]],
                                    list[str]]:
    """(segments, missing, notes) for one direction, in sequence order.

    `segments` are (file offset, length) to be concatenated. `missing` are
    (position in the concatenation, bytes absent) -- the holes, measured rather
    than inferred, because the sequence numbers either side of one say how wide
    it is.

    Retransmissions and overlaps are resolved by keeping what arrived first,
    which is what the receiving stack did.
    """
    segs = sorted(flow.segments, key=lambda s: (s.seq, s.offset))
    out: list[tuple[int, int]] = []
    missing: list[tuple[int, int]] = []
    notes: list[str] = []
    if not segs:
        return out, missing, notes
    expect = segs[0].seq
    produced = 0
    for s in segs:
        if s.seq < expect:
            skip = expect - s.seq
            if skip >= s.length:
                continue                # a pure retransmission of known bytes
            out.append((s.offset + skip, s.length - skip))
            produced += s.length - skip
            notes.append(f"{skip} B of overlap at seq {s.seq} kept from the "
                         f"earlier copy")
            expect = s.seq + s.length
            continue
        if s.seq > expect:
            # Positions are into the CONCATENATION of what survives, because
            # that is the only byte string a caller has: the hole is where the
            # two sides meet, not an offset in a stream that includes it. With
            # one hole the two readings coincide, which is exactly the sort of
            # coincidence that hides a bug until the second hole shows up.
            missing.append((produced, s.seq - expect))
        out.append((s.offset, s.length))
        produced += s.length
        expect = s.seq + s.length
    return out, missing, notes


def _http_body(head: bytes) -> tuple[int, dict]:
    """(offset of the body, headers) for an HTTP response, or (0, {})."""
    if not head.startswith(b"HTTP/"):
        return 0, {}
    end = head.find(b"\r\n\r\n")
    if end < 0:
        return 0, {}
    fields = {}
    for line in head[:end].split(b"\r\n")[1:]:
        if b":" in line:
            k, v = line.split(b":", 1)
            fields[k.strip().lower().decode("latin-1")] = \
                v.strip().decode("latin-1")
    return end + 4, fields


def _encoding_of(body: bytes, fields: dict) -> tuple[str, str]:
    """(encoding, why). The bytes decide; the header is corroboration."""
    enc = fields.get("content-encoding", "").lower()
    if body[:2] == b"\x1f\x8b":
        return GZIP, "gzip magic"
    if len(body) >= 2 and body[0] & 0x0F == 8 and \
            ((body[0] << 8) | body[1]) % 31 == 0:
        return ZLIB, "zlib header"
    if "gzip" in enc:
        return GZIP, "Content-Encoding, but the magic is absent"
    if "deflate" in enc:
        return RAW, "Content-Encoding: deflate"
    return "", ""


def looks_like_pcap(data: bytes) -> bool:
    return data[:4] in MAGICS or data[:4] == PCAPNG_MAGIC


def parse(data: bytes) -> ContainerReport | None:
    """Every reassembled TCP stream that carries something DEFLATE-shaped."""
    if data[:4] == PCAPNG_MAGIC:
        report = ContainerReport(kind="pcapng")
        report.warnings.append(
            "pcapng, which this does not read. Convert it: "
            "`editcap -F pcap in.pcapng out.pcap`")
        return report
    if data[:4] not in MAGICS:
        return None
    report = ContainerReport(kind="pcap")

    total = snapped = 0
    for _off, _l, incl, orig, _lt in packets(data):
        total += 1
        snapped += incl < orig
    report.structure.append(f"{total} packet(s)")
    if snapped:
        # A different loss from a missing packet, and a worse one: the bytes are
        # gone with no record of their content and no gap in the sequence
        # numbers to mark them, because they were captured and then discarded.
        report.warnings.append(
            f"{snapped} packet(s) were captured short (incl_len < orig_len) and "
            f"are skipped: a snaplen truncation leaves no sequence gap, so the "
            f"loss cannot be measured from the file")

    for flow in flows(data).values():
        segs, missing, notes = reassemble(flow)
        if not segs:
            continue
        joined = b"".join(data[o:o + n] for o, n in segs)
        body_at, fields = _http_body(joined[:4096])
        body = joined[body_at:]
        encoding, why = _encoding_of(body, fields)
        if not encoding:
            continue
        lost = sum(n for _p, n in missing)
        notes = list(notes)
        notes.append(f"reassembled from {len(segs)} TCP segment(s); {why}")
        if missing:
            # Both coordinates: the TCP stream counts the HTTP head, the body
            # does not, and a reader comparing this with the interface (which
            # counts in the body) saw two different numbers for one gap.
            where = ", ".join(f"{n:,} B at stream offset {p:,} (HTTP body "
                              f"byte {p - body_at:,})" for p, n in missing)
            notes.append(f"MISSING {lost:,} B in {len(missing)} hole(s): {where}"
                         f", measured from the TCP sequence numbers")
        # Offsets are into the file, and the body may start part way into the
        # first segment: trim it there so the stream begins at the payload.
        trimmed, skip = [], body_at
        for off, n in segs:
            if skip >= n:
                skip -= n
                continue
            trimmed.append((off + skip, n - skip))
            skip = 0
        # The gzip trailer, when the end of the stream survived. It is the only
        # thing here that can check a recovery: fill the holes the loss left and
        # this says whether the answer is right. Read only when the last 8 bytes
        # are behind every hole, because a trailer with a hole in it is not a
        # trailer, and a checksum nobody can trust is worse than none.
        crc = size = None
        holes_after = [p for p, n in missing if p + n > len(joined) - 8]
        if encoding == GZIP and len(body) >= 8 and not holes_after:
            crc, size = struct.unpack_from("<II", joined, len(joined) - 8)
            notes.append(f"the gzip trailer survived: CRC32 {crc:08x} over "
                         f"{size:,} bytes, which is what a filled-in recovery "
                         f"can be checked against")
        elif encoding == GZIP:
            notes.append("the gzip trailer is inside the loss, so there is "
                         "nothing to check a recovery against")
        report.streams.append(Stream(
            name=flow.name, container="pcap", offset=trimmed[0][0],
            length=None, encoding=encoding, segments=trimmed,
            crc32=crc, uncompressed_size=size,
            missing=[(p - body_at, n) for p, n in missing if p >= body_at],
            source="tcp", notes=notes))
    if not report.streams and not report.warnings:
        report.warnings.append("no TCP stream in this capture carries anything "
                               "DEFLATE-shaped")
    return report
