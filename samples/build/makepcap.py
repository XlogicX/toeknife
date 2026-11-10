#!/usr/bin/env python3
"""Build a pcap carrying a gzip'd HTTP response, and one with a packet missing.

The case this exists for: a DEFLATE stream captured off the wire, spread over
however many TCP segments the path's MTU dictated, with one segment lost. That is
a different kind of damage from everything in `samples/damaged/`, and a better
kind, because the container knows exactly what is gone. TCP sequence numbers give
the byte offset of the hole and its length -- no searching for where the fragment
resumes, and no guessing how much is missing.

What the capture still does NOT know is how many bytes those missing compressed
bytes would have produced. That is the output gap, and it comes from somewhere
else: the gzip trailer's ISIZE, which records the original length, so the gap is
arithmetic rather than a search.

Deterministic: fixed timestamps, fixed addresses, seeded nothing.

    .venv/bin/python samples/build/makepcap.py
"""
import glob
import os
import struct
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "samples", "damaged")

# Classic pcap, microsecond resolution, little-endian, Ethernet.
PCAP_MAGIC = 0xA1B2C3D4
LINKTYPE_ETHERNET = 1
MTU_PAYLOAD = 1460              # a plain 1500-byte path

CLIENT_IP, SERVER_IP = "10.0.0.2", "93.184.216.34"
CLIENT_PORT, SERVER_PORT = 51000, 80
CLIENT_MAC = bytes.fromhex("020000000001")
SERVER_MAC = bytes.fromhex("020000000002")


def ip_to_bytes(s):
    return bytes(int(p) for p in s.split("."))


def checksum(data: bytes) -> int:
    if len(data) % 2:
        data += b"\x00"
    total = 0
    for i in range(0, len(data), 2):
        total += (data[i] << 8) | data[i + 1]
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def ethernet(src, dst, payload):
    return dst + src + b"\x08\x00" + payload


def ipv4(src, dst, payload, ident):
    header = struct.pack(
        ">BBHHHBBH4s4s", 0x45, 0, 20 + len(payload), ident, 0x4000, 64, 6, 0,
        ip_to_bytes(src), ip_to_bytes(dst))
    header = header[:10] + struct.pack(">H", checksum(header)) + header[12:]
    return header + payload


def tcp(sport, dport, seq, ack, payload, flags=0x18, src=None, dst=None):
    """flags: 0x02 SYN, 0x10 ACK, 0x18 PSH+ACK, 0x11 FIN+ACK."""
    header = struct.pack(">HHIIBBHHH", sport, dport, seq, ack, 0x50, flags,
                         65535, 0, 0)
    # The pseudo-header checksum, because a capture with wrong checksums invites
    # the reader to wonder whether the tool or the file is at fault.
    pseudo = (ip_to_bytes(src) + ip_to_bytes(dst) + b"\x00\x06"
              + struct.pack(">H", len(header) + len(payload)))
    ck = checksum(pseudo + header + payload)
    header = header[:16] + struct.pack(">H", ck) + header[18:]
    return header + payload


def packet(ts, blob):
    """One pcap record. incl_len == orig_len: nothing was snapped short."""
    sec, usec = ts
    return struct.pack("<IIII", sec, usec, len(blob), len(blob)) + blob


def docs_body():
    """The default payload: the pinned samples/damaged/payload.txt, so a clean
    checkout rebuilds the captures byte for byte. It was docs/*.md, read live,
    which broke that every time a doc changed."""
    with open(os.path.join(OUT, "payload.txt"), "rb") as fh:
        return fh.read()[:120_000]


def build(drop_index=None, body=None, mss=MTU_PAYLOAD,
          content_type=b"text/markdown", path=b"/docs.md"):
    """The whole capture. `drop_index` removes one response data packet.

    `body` and `mss` build captures with other payloads and smaller
    segments; the defaults build the shipped samples.

    Removing the RECORD entirely is what a lost packet looks like to whoever
    reads the file afterwards: the sequence numbers of everything around it are
    untouched, so the hole is exactly measurable and nothing else moves.
    """
    if body is None:
        body = docs_body()
    gz = zlib.compressobj(9, zlib.DEFLATED, 16 + 15)     # gzip wrapper
    payload = gz.compress(body) + gz.flush()
    head = (b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: " + content_type + b"\r\n"
            b"Content-Encoding: gzip\r\n"
            + f"Content-Length: {len(payload)}\r\n".encode()
            + b"Connection: close\r\n\r\n")
    response = head + payload

    records = []
    cseq, sseq = 1000, 5000
    ts = 1700000000

    def stamp(n):
        return (ts + n // 1000, (n % 1000) * 1000)

    def c2s(seq, ack, data, flags=0x18):
        return ethernet(CLIENT_MAC, SERVER_MAC,
                        ipv4(CLIENT_IP, SERVER_IP,
                             tcp(CLIENT_PORT, SERVER_PORT, seq, ack, data,
                                 flags, CLIENT_IP, SERVER_IP), len(records)))

    def s2c(seq, ack, data, flags=0x18):
        return ethernet(SERVER_MAC, CLIENT_MAC,
                        ipv4(SERVER_IP, CLIENT_IP,
                             tcp(SERVER_PORT, CLIENT_PORT, seq, ack, data,
                                 flags, SERVER_IP, CLIENT_IP), len(records)))

    records.append(packet(stamp(0), c2s(cseq, 0, b"", 0x02)))          # SYN
    records.append(packet(stamp(1), s2c(sseq, cseq + 1, b"", 0x12)))   # SYN/ACK
    cseq += 1
    sseq += 1
    records.append(packet(stamp(2), c2s(cseq, sseq, b"", 0x10)))       # ACK

    request = (b"GET " + path + b" HTTP/1.1\r\nHost: example.com\r\n"
               b"Accept-Encoding: gzip\r\n\r\n")
    records.append(packet(stamp(3), c2s(cseq, sseq, request)))
    cseq += len(request)

    data_packets = []
    for i in range(0, len(response), mss):
        chunk = response[i:i + mss]
        data_packets.append((sseq, chunk))
        sseq += len(chunk)

    dropped = None
    for n, (seq, chunk) in enumerate(data_packets):
        if drop_index is not None and n == drop_index:
            dropped = (seq, len(chunk))
            continue                       # the packet simply is not in the file
        records.append(packet(stamp(4 + n), s2c(seq, cseq, chunk)))

    records.append(packet(stamp(4 + len(data_packets)),
                          s2c(sseq, cseq, b"", 0x11)))                 # FIN/ACK

    header = struct.pack("<IHHiIII", PCAP_MAGIC, 2, 4, 0, 0, 65535,
                         LINKTYPE_ETHERNET)
    return (header + b"".join(records), len(response), len(payload),
            len(body), len(data_packets), dropped, len(head))


def write(name, blob, note):
    path = os.path.join(OUT, name)
    with open(path, "wb") as fh:
        fh.write(blob)
    print(f"{name:26} {len(blob):8} B   {note}")


def main():
    os.makedirs(OUT, exist_ok=True)
    # The samples' own head. build()'s defaults are kept as they are because
    # the head's length moves every segment boundary after it, and other
    # captures were measured with those defaults.
    head = dict(content_type=b"text/plain; charset=utf-8", path=b"/holmes.txt")
    whole, resp_len, payload_len, body_len, n_packets, _d, head_len = \
        build(**head)
    write("http-gzip.pcap", whole,
          f"{n_packets} data packets carrying a {payload_len:,} B gzip of "
          f"{body_len:,} B, intact")

    # Somewhere in the middle, well clear of the gzip header and the trailer, so
    # the table survives and the ISIZE that measures the gap survives with it.
    drop = n_packets // 2
    lossy, _r, _p, _b, _n, dropped, _h = build(drop_index=drop, **head)
    seq, length = dropped
    write("http-gzip-loss.pcap", lossy,
          f"same capture, data packet {drop} of {n_packets} missing: "
          f"{length} B gone at stream offset {seq - 5001 - 0:,}")
    print(f"\n  response head {head_len} B, gzip payload {payload_len:,} B, "
          f"original {body_len:,} B")
    print(f"  the hole is {length} B of compressed data, and its position is "
          f"exact: TCP said so")
    return 0


if __name__ == "__main__":
    sys.exit(main())
