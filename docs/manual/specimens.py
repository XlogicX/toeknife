#!/usr/bin/env python3
"""Build the manual's specimens from any text.

    python3 docs/manual/specimens.py article.txt [OUTDIR]

Writes, into OUTDIR (default: the current directory):

    stack-http.pcap          B: a gzip'd HTTP response, every packet present
    stack-http-drop17.pcap   B: the same capture without response packet 17
    edited-copy.txt          B: the text with "One" changed to "Two" at 517
    stack.zip                C: five entries, two of them damaged
    stack-noeocd.zip         C: stack.zip with the directory and end record cut
    stack-sfx.zip            C: stack.zip behind a 4,096-byte stub
    stack-rot.gz             D: gzip with bit 262 of the compressed data flipped
    stack-carve.bin          E: GNU gzip's DEFLATE, the first 3,000 bytes gone
    mirror.txt.gz            E: the text, gzip'd again (a copy from a mirror)
    edited-copy.txt.gz       E: edited-copy.txt, gzip'd the same way
    stack-part1.gz           F: the first 6,000 bytes of a gzip file
    stack-part2.bin          F: the same file from byte 9,000 to its end
    stack-members.gz         G: three gzip members, the middle one's table gone
    stack-dict.zlib          H: zlib against a preset dictionary not in the file
    stackpack.c              H: the program that wrote it, dictionary inside
    stack.png                I: a 320x240 greyscale PNG, one IDAT bit flipped
    stack.pdf                I: the text as a 30-page PDF
    stack-cut.pdf            I: stack.pdf carved short at byte 19,500

Standard library only, plus the `gzip` command for E, and deterministic: fixed timestamps, fixed addresses,
seeded randomness. The capture fetches the input's file name. Run on the
Phrack article saved as p49-14.txt (60,947 bytes) it reproduces the files the
screenshots were taken from, byte for byte.
"""
import os
import random
import struct
import sys
import zlib

# ---- B: a packet capture -------------------------------------------------

CLIENT_IP, SERVER_IP = "10.0.0.2", "93.184.216.34"
CLIENT_PORT, SERVER_PORT = 51000, 80
CLIENT_MAC = bytes.fromhex("020000000001")
SERVER_MAC = bytes.fromhex("020000000002")


def _ip(s):
    return bytes(int(p) for p in s.split("."))


def _checksum(data):
    if len(data) % 2:
        data += b"\x00"
    total = 0
    for i in range(0, len(data), 2):
        total += (data[i] << 8) | data[i + 1]
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _tcp(sport, dport, seq, ack, payload, flags, src, dst):
    header = struct.pack(">HHIIBBHHH", sport, dport, seq, ack, 0x50, flags,
                         65535, 0, 0)
    pseudo = (_ip(src) + _ip(dst) + b"\x00\x06"
              + struct.pack(">H", len(header) + len(payload)))
    ck = _checksum(pseudo + header + payload)
    return header[:16] + struct.pack(">H", ck) + header[18:] + payload


def _ipv4(src, dst, payload, ident):
    header = struct.pack(">BBHHHBBH4s4s", 0x45, 0, 20 + len(payload), ident,
                         0x4000, 64, 6, 0, _ip(src), _ip(dst))
    header = header[:10] + struct.pack(">H", _checksum(header)) + header[12:]
    return header + payload


def _record(n, frame):
    sec, usec = 1700000000 + n // 1000, (n % 1000) * 1000
    return struct.pack("<IIII", sec, usec, len(frame), len(frame)) + frame


def pcap(body, drop=None, mss=536, path=b"/article.txt"):
    """A client fetching `body` gzip'd over HTTP, cut into `mss`-byte
    segments; `drop` leaves one response segment out of the file."""
    gz = zlib.compressobj(9, zlib.DEFLATED, 16 + 15)
    payload = gz.compress(body) + gz.flush()
    response = (b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n"
                b"Content-Encoding: gzip\r\n"
                + f"Content-Length: {len(payload)}\r\n".encode()
                + b"Connection: close\r\n\r\n" + payload)
    records = []

    def frame(to_server, seq, ack, data, flags):
        src, dst = (CLIENT_IP, SERVER_IP) if to_server else (SERVER_IP, CLIENT_IP)
        sp, dp = ((CLIENT_PORT, SERVER_PORT) if to_server
                  else (SERVER_PORT, CLIENT_PORT))
        macs = (SERVER_MAC + CLIENT_MAC if to_server
                else CLIENT_MAC + SERVER_MAC)
        return macs + b"\x08\x00" + _ipv4(
            src, dst, _tcp(sp, dp, seq, ack, data, flags, src, dst),
            len(records))

    cseq, sseq = 1000, 5000
    records.append(_record(0, frame(True, cseq, 0, b"", 0x02)))
    records.append(_record(1, frame(False, sseq, cseq + 1, b"", 0x12)))
    cseq, sseq = cseq + 1, sseq + 1
    records.append(_record(2, frame(True, cseq, sseq, b"", 0x10)))
    request = (b"GET " + path + b" HTTP/1.1\r\nHost: example.com\r\n"
               b"Accept-Encoding: gzip\r\n\r\n")
    records.append(_record(3, frame(True, cseq, sseq, request, 0x18)))
    cseq += len(request)
    segments = []
    for i in range(0, len(response), mss):
        segments.append((sseq, response[i:i + mss]))
        sseq += len(segments[-1][1])
    for n, (seq, chunk) in enumerate(segments):
        if n != drop:
            records.append(_record(4 + n, frame(False, seq, cseq, chunk, 0x18)))
    records.append(_record(4 + len(segments), frame(False, sseq, cseq, b"", 0x11)))
    header = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    return header + b"".join(records)


# ---- C: a damaged zip ----------------------------------------------------

def deflate(raw, level=6):
    co = zlib.compressobj(level, zlib.DEFLATED, -15)
    return co.compress(raw) + co.flush()


def silent_flip(raw):
    """The first bit, from a third of the way in, whose flip still inflates
    to full length: a decode that completes and is quietly wrong."""
    blob = deflate(raw)
    for pos in range(len(blob) // 3, 2 * len(blob) // 3):
        for bit in range(8):
            cand = bytearray(blob)
            cand[pos] ^= 1 << bit
            try:
                got = zlib.decompressobj(-15).decompress(bytes(cand))
            except zlib.error:
                continue
            if len(got) == len(raw) and got != raw:
                return bytes(cand)
    raise SystemExit("no silent flip found")


def burst(raw, count=200, seed=1234):
    """`count` random bytes over the compressed data, starting halfway in."""
    rng = random.Random(seed)
    blob = bytearray(deflate(raw))
    start = len(blob) // 2
    for i in range(count):
        blob[start + i] = rng.randrange(256)
    return bytes(blob)


class ZipCrypto:
    """PKWARE's traditional stream cipher, so the encrypted entry is real."""

    def __init__(self, password):
        self.k = [0x12345678, 0x23456789, 0x34567890]
        for ch in password:
            self.update(ch)

    @staticmethod
    def _crc(crc, b):
        return zlib.crc32(bytes([b]), crc ^ 0xFFFFFFFF) ^ 0xFFFFFFFF

    def update(self, b):
        k = self.k
        k[0] = self._crc(k[0], b)
        k[1] = (k[1] + (k[0] & 0xFF)) & 0xFFFFFFFF
        k[1] = (k[1] * 134775813 + 1) & 0xFFFFFFFF
        k[2] = self._crc(k[2], (k[1] >> 24) & 0xFF)

    def encrypt(self, data):
        out = bytearray()
        for p in data:
            t = (self.k[2] | 2) & 0xFFFF
            out.append(p ^ (((t * (t ^ 1)) >> 8) & 0xFF))
            self.update(p)
        return bytes(out)


DOS_TIME, DOS_DATE = 0x6000, 0x5901          # 2024-08-01 12:00:00


class Entry:
    def __init__(self, name, raw, method=8, blob=None, flags=0):
        self.name = name.encode()
        self.crc = zlib.crc32(raw) & 0xFFFFFFFF
        self.usize = len(raw)
        self.method, self.flags = method, flags
        self.blob = blob if blob is not None else deflate(raw)

    def local(self):
        return (struct.pack("<IHHHHHIIIHH", 0x04034B50, 20, self.flags,
                            self.method, DOS_TIME, DOS_DATE, self.crc,
                            len(self.blob), self.usize, len(self.name), 0)
                + self.name + self.blob)

    def central(self, offset):
        return (struct.pack("<IHHHHHHIIIHHHHHII", 0x02014B50, 20, 20,
                            self.flags, self.method, DOS_TIME, DOS_DATE,
                            self.crc, len(self.blob), self.usize,
                            len(self.name), 0, 0, 0, 0, 0, offset)
                + self.name)


def build_zip(entries, stub=b"", with_directory=True):
    """`stub` goes in front after the offsets are recorded, so every one of
    them is short by its length, as in a self-extracting archive."""
    out, offsets = bytearray(), []
    for e in entries:
        offsets.append(len(out))
        out += e.local()
    if with_directory:
        cd_start = len(out)
        for e, off in zip(entries, offsets):
            out += e.central(off)
        out += struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(entries),
                           len(entries), len(out) - cd_start, cd_start, 0)
    return stub + bytes(out)


def zip_entries(art):
    secret = art[:4000]
    cipher = ZipCrypto(b"forensics")
    enc = Entry("secret.txt", secret, flags=1)
    enc.blob = (cipher.encrypt(bytes(11) + bytes([zlib.crc32(secret) >> 24]))
                + cipher.encrypt(enc.blob))
    return [Entry("intact.txt", art),
            Entry("rotted.txt", art, blob=silent_flip(art)),
            Entry("shredded.txt", art, blob=burst(art)),
            Entry("stored.bin", b"\x00\x01\x02\x03" * 64, method=0,
                  blob=b"\x00\x01\x02\x03" * 64),
            enc]


# ---- D: bit rot in a header ----------------------------------------------

def rotted_gzip(art, bit=262):
    blob = bytearray(deflate(art))
    blob[bit >> 3] ^= 1 << (bit & 7)
    return (b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + bytes(blob)
            + struct.pack("<II", zlib.crc32(art) & 0xFFFFFFFF, len(art)))


# ---- E: a carve whose table is gone ---------------------------------------

def gnu_gzip(raw):
    """GNU gzip -6, no name or time stamp. Its blocks are what E is about:
    it splits the Phrack article at output 17,724, where zlib keeps one."""
    import subprocess
    return subprocess.run(["gzip", "-6", "-n", "-c"], input=raw,
                          capture_output=True, check=True).stdout


def carve(gz, lose=3000):
    """The member's DEFLATE data with the first `lose` bytes gone, table and
    all."""
    return gz[10:-8][lose:]


# ---- G: a multi-member gzip with one bad member ----------------------------

GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03"


def members(art, cuts=(20000, 40000), bad=1, at=40, count=60):
    """The text in three gzip members, as `cat a.gz b.gz c.gz` makes. Member
    `bad` has `count` bytes of its DEFLATE data inverted from byte `at`:
    the back half of its Huffman table and the first body bytes."""
    out = b""
    bounds = (0,) + tuple(cuts) + (len(art),)
    for i in range(len(bounds) - 1):
        part = art[bounds[i]:bounds[i + 1]]
        blob = bytearray(deflate(part))
        if i == bad:
            for j in range(at, at + count):
                blob[j] ^= 0xFF
        out += (GZIP_HEADER + bytes(blob)
                + struct.pack("<II", zlib.crc32(part) & 0xFFFFFFFF, len(part)))
    return out


# ---- H: a zlib stream that needs a preset dictionary ----------------------

def dictionary(art, words=200):
    """What zlib's documentation suggests a preset dictionary be: strings
    common in the data, the commonest last, where copies from it are
    shortest. Here, the article's 200 commonest words of four letters or
    more."""
    import re
    from collections import Counter
    count = Counter(re.findall(rb"[A-Za-z_%$]{4,}", art))
    common = [w for w, _ in sorted(count.items(),
                                   key=lambda kv: (-kv[1], kv[0]))[:words]]
    return b" ".join(reversed(common))


def with_dictionary(art, zdict):
    co = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY,
                          zdict=zdict)
    return co.compress(art) + co.flush()


def dictionary_program(zdict):
    """A C source that compresses with the dictionary, as the program that
    wrote the stream would; the dictionary is one string constant in it."""
    return (b"/* stackpack.c: store articles compressed */\n"
            b"#include <stdio.h>\n#include <string.h>\n#include <zlib.h>\n\n"
            b"static const char dict[] =\n    \"" + zdict + b"\";\n\n"
            b"int pack(z_stream *zs)\n{\n"
            b"    if (deflateInit(zs, 6) != Z_OK)\n        return -1;\n"
            b"    return deflateSetDictionary(zs, (const Bytef *)dict,\n"
            b"                                sizeof dict - 1);\n}\n")


# ---- I: DEFLATE inside a PNG and a PDF -----------------------------------

def _chunk(ctype, body):
    return (struct.pack(">I", len(body)) + ctype + body
            + struct.pack(">I", zlib.crc32(ctype + body) & 0xFFFFFFFF))


def png(width=320, height=240, idat=8192, flip=(1, 3000, 4)):
    """A greyscale picture of interfering waves with a little seeded noise,
    its zlib stream cut into 8 KB IDAT chunks as libpng does. `flip` is
    (chunk, byte, bit) of IDAT data to invert after the CRCs are written."""
    import math
    rng = random.Random(49)
    raw = bytearray()
    for y in range(height):
        raw.append(0)                               # filter type: none
        for x in range(width):
            v = (128 + 60 * math.sin(x / 23.0) * math.cos(y / 17.0)
                 + 40 * math.sin((x + y) / 41.0) + rng.randrange(-2, 3))
            raw.append(max(0, min(255, int(v))))
    z = zlib.compress(bytes(raw), 9)
    out = bytearray(b"\x89PNG\r\n\x1a\n" + _chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)))
    starts = []
    for i in range(0, len(z), idat):
        starts.append(len(out) + 8)
        out += _chunk(b"IDAT", z[i:i + idat])
    out += _chunk(b"IEND", b"")
    if flip is not None:
        k, byte, bit = flip
        out[starts[k] + byte] ^= 1 << bit
    return bytes(out)


def pdf(art, lines_per_page=60):
    """The text set in Courier, one /FlateDecode content stream per page."""
    lines = art.split(b"\n")

    def esc(line):
        return (line.replace(b"\\", b"\\\\").replace(b"(", b"\\(")
                .replace(b")", b"\\)"))
    pages = [b"BT /F1 9 Tf 11 TL 40 760 Td\n"
             + b"".join(b"(" + esc(l) + b") '\n"
                        for l in lines[i:i + lines_per_page]) + b"ET\n"
             for i in range(0, len(lines), lines_per_page)]
    font = 3 + 2 * len(pages)
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids ["
            + " ".join(f"{3 + 2 * i} 0 R" for i in range(len(pages))).encode()
            + b"] /Count " + str(len(pages)).encode() + b" >>"]
    for i, page in enumerate(pages):
        objs.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    b"/Resources << /Font << /F1 " + str(font).encode()
                    + b" 0 R >> >> /Contents " + str(4 + 2 * i).encode()
                    + b" 0 R >>")
        z = zlib.compress(page, 6)
        objs.append(b"<< /Length " + str(len(z)).encode()
                    + b" /Filter /FlateDecode >>\nstream\n" + z
                    + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
    out, offsets = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), []
    for i, o in enumerate(objs):
        offsets.append(len(out))
        out += f"{i + 1} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += (f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
            + b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
            + f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
              f"startxref\n{xref}\n%%EOF\n".encode())
    return bytes(out)


def main(argv):
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    art = open(argv[0], "rb").read()
    out = argv[1] if len(argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    entries = zip_entries(art)
    path = b"/" + os.path.basename(argv[0]).encode()     # the URL fetched
    edited = art[:517] + b"Two" + art[520:] if art[517:520] == b"One" else None
    files = {
        "stack-http.pcap": pcap(art, path=path),
        "stack-http-drop17.pcap": pcap(art, drop=17, path=path),
        "stack.zip": build_zip(entries),
        "stack-noeocd.zip": build_zip(entries, with_directory=False),
        "stack-sfx.zip": build_zip(entries, stub=b"MZ" + b"\x90" * 4094),
        "stack-rot.gz": rotted_gzip(art),
    }
    whole = zlib.compressobj(6, zlib.DEFLATED, 16 + 15)
    whole = whole.compress(art) + whole.flush()
    files["stack-part1.gz"] = whole[:6000]
    files["stack-part2.bin"] = whole[9000:]
    files["stack-members.gz"] = members(art)
    zdict = dictionary(art)
    files["stack-dict.zlib"] = with_dictionary(art, zdict)
    files["stackpack.c"] = dictionary_program(zdict)
    files["stack.png"] = png()
    files["stack.pdf"] = pdf(art)
    files["stack-cut.pdf"] = files["stack.pdf"][:19500]
    gz = gnu_gzip(art)
    files["stack-carve.bin"] = carve(gz)
    files["mirror.txt.gz"] = gz
    if edited is not None:
        files["edited-copy.txt"] = edited
        files["edited-copy.txt.gz"] = gnu_gzip(edited)
    for name, blob in files.items():
        with open(os.path.join(out, name), "wb") as fh:
            fh.write(blob)
        print(f"{name:24} {len(blob):>7,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
