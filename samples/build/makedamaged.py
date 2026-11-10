"""Build `samples/damaged/`: one file per way a DEFLATE-bearing file goes wrong.

The samples that already exist are about *carving* -- a fragment that begins
mid-stream, with the table behind the damage.  These are about the other half of
the tool: containers, and what the wrapper knows that the compressed bytes do
not.  Each file here is built to make exactly one behaviour visible, and the
commands that show it are in `samples/damaged/README.md`.

The payload is `samples/damaged/payload.txt`, 256 KB of The Adventures of
Sherlock Holmes, pinned, so a recovered tape reads as prose and a clean
checkout regenerates these byte for byte.

    .venv/bin/python samples/build/makedamaged.py

Deterministic: no randomness that is not seeded, no timestamps in any header.
"""
import glob
import os
import random
import struct
import sys
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "samples", "damaged")

DOS_TIME = 0x6000        # 12:00:00, fixed so the files are byte-reproducible
DOS_DATE = 0x5901        # 2024-08-01


PAYLOAD = os.path.join(ROOT, "samples", "damaged", "payload.txt")


def docs():
    """The pinned payload: 256 KB of The Adventures of Sherlock Holmes.

    It was the project's own docs/*.md, read live, so any edit to a doc changed
    every sample and the README's byte-identical promise broke, and the
    samples carried the docs of whatever day they were built.
    A public-domain book that is not in any corpus the manual trains on keeps
    the recovered bytes readable without the corpus holding the answer.
    """
    return open(PAYLOAD, "rb").read()


# ---- zip --------------------------------------------------------------------

def crc_update(crc, b):
    return zlib.crc32(bytes([b]), crc ^ 0xFFFFFFFF) ^ 0xFFFFFFFF


class ZipCrypto:
    """The original PKWARE stream cipher, so the encrypted entry is genuinely
    encrypted rather than an entry with the flag bit set as a prop."""

    def __init__(self, password):
        self.k = [0x12345678, 0x23456789, 0x34567890]
        for ch in password:
            self.update(ch)

    def update(self, b):
        k = self.k
        k[0] = crc_update(k[0], b)
        k[1] = (k[1] + (k[0] & 0xFF)) & 0xFFFFFFFF
        k[1] = (k[1] * 134775813 + 1) & 0xFFFFFFFF
        k[2] = crc_update(k[2], (k[1] >> 24) & 0xFF)

    def byte(self):
        t = (self.k[2] | 2) & 0xFFFF
        return ((t * (t ^ 1)) >> 8) & 0xFF

    def encrypt(self, data):
        out = bytearray()
        for p in data:
            out.append(p ^ self.byte())
            self.update(p)
        return bytes(out)


class Entry:
    def __init__(self, name, raw, method=8, blob=None, flags=0):
        self.name = name.encode()
        self.crc = zlib.crc32(raw) & 0xFFFFFFFF
        self.usize = len(raw)
        self.method = method
        self.flags = flags
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


def deflate(raw, level=6):
    co = zlib.compressobj(level, zlib.DEFLATED, -15)
    return co.compress(raw) + co.flush()


def build_zip(entries, stub=b"", with_directory=True):
    """`stub` is prepended *after* the offsets are recorded, which is what a
    self-extracting header or a carve that lost the front looks like: every
    recorded offset is short by a constant that has to be measured."""
    out = bytearray()
    offsets = []
    for e in entries:
        offsets.append(len(out))
        out += e.local()
    if not with_directory:
        return stub + bytes(out)
    cd_start = len(out)
    for e, off in zip(entries, offsets):
        out += e.central(off)
    cd_size = len(out) - cd_start
    out += struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, len(entries),
                       len(entries), cd_size, cd_start, 0)
    return stub + bytes(out)


# ---- damage -----------------------------------------------------------------

def silent_flip(raw):
    """A single flipped bit that keeps every Huffman code legal, so the stream
    still inflates to a full-length output that is quietly wrong.

    This is the case a checksum exists for.  Searched rather than chosen: most
    flips hit a code and the decode dies loudly, which is the easy outcome.
    """
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
                differ = sum(a != b for a, b in zip(got, raw))
                return bytes(cand), (pos, bit, differ)
    raise SystemExit("no silent flip found")


def burst(raw, count=200, seed=1234):
    """A burst of damage in the middle: the loud case, where the decode stops
    early and the recorded size is what notices."""
    import random
    rng = random.Random(seed)
    blob = bytearray(deflate(raw))
    start = len(blob) // 2
    for i in range(count):
        blob[start + i] = rng.randrange(256)
    try:
        got = zlib.decompressobj(-15).decompress(bytes(blob))
    except zlib.error as exc:
        got = getattr(exc, "partial", b"")
    return bytes(blob), len(got)


# ---- png --------------------------------------------------------------------

def chunk(kind, body):
    return (struct.pack(">I", len(body)) + kind + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))


def build_png(w, h, damage=False, split_run=False):
    rows = bytearray()
    for y in range(h):
        rows.append(0)                                    # filter: none
        for x in range(w):
            rows += bytes(((x * 3 + y) % 256, (y * 5) % 256, (x ^ y) % 256))
    blob = zlib.compress(bytes(rows), 6)
    half = len(blob) // 2
    out = bytearray(b"\x89PNG\r\n\x1a\n"
                    + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
    out += chunk(b"IDAT", blob[:half])
    if split_run:
        out += chunk(b"tEXt", b"Comment\x00interrupts the IDAT run")
    second = len(out) + 8                                 # payload of IDAT #2
    out += chunk(b"IDAT", blob[half:])
    # A compressed text chunk as well as the pixels, so the file holds two
    # streams and the listing (with its warnings) prints instead of the single
    # decodable stream opening straight into the view.
    out += chunk(b"zTXt", b"Comment\x00\x00"
                 + zlib.compress(b"exported by the thing that damaged it; "
                                 b"see the chunk CRCs\n" * 8, 6))
    out += chunk(b"IEND", b"")
    if damage:
        # After the chunk CRC is computed, not before: the whole point is that
        # the CRC no longer matches, which says WHICH chunk holds the damage
        # while saying nothing about whether a recovery is right.
        out[second + 200] ^= 0x40
    return bytes(out)


# ---- pdf --------------------------------------------------------------------

def build_pdf(streams, keep_xref=True):
    out = bytearray(b"%PDF-1.5\n")
    offsets = []
    for i, payload in enumerate(streams, start=1):
        blob = zlib.compress(payload, 6)     # FlateDecode is zlib-wrapped
        offsets.append(len(out))
        out += (f"{i} 0 obj\n<< /Length {len(blob)} /Filter /FlateDecode >>\n"
                "stream\n".encode() + blob + b"\nendstream\nendobj\n")
    start = len(out)
    if keep_xref:
        out += f"xref\n0 {len(streams) + 1}\n0000000000 65535 f \n".encode()
        for off in offsets:
            out += f"{off:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(streams) + 1} >>\nstartxref\n"
                f"{start}\n%%EOF\n").encode()
    else:
        out += b"%%EOF\n"          # the tail, and the map with it, is gone
    return bytes(out)


# ---- main -------------------------------------------------------------------

def write(name, blob, note):
    path = os.path.join(OUT, name)
    with open(path, "wb") as fh:
        fh.write(blob)
    print(f"{name:26} {len(blob):8} B   {note}")


def main():
    os.makedirs(OUT, exist_ok=True)
    src = docs()
    clean = src[:24000]
    rot = src[24000:120000]
    shred = src[120000:250000]

    rotted, (pos, bit, differ) = silent_flip(rot)
    shredded, got = burst(shred)

    entries = [
        Entry("notes.txt", clean),
        Entry("rotted.md", rot, blob=rotted),
        Entry("shredded.md", shred, blob=shredded),
        Entry("stored.bin", b"\x00\x01\x02\x03" * 64, method=0,
              blob=b"\x00\x01\x02\x03" * 64),
    ]
    secret = clean[:4000]
    cipher = ZipCrypto(b"forensics")
    header = cipher.encrypt(bytes(11) + bytes([(zlib.crc32(secret) >> 24) & 0xFF]))
    enc = Entry("secret.txt", secret, flags=1)
    enc.blob = header + cipher.encrypt(enc.blob)
    entries.append(enc)

    zipped = build_zip(entries)
    write("evidence.zip", zipped,
          f"5 entries; rotted.md = 1 flipped bit at byte {pos} bit {bit}, "
          f"{differ} output bytes wrong; shredded.md loses {len(shred) - got} B")
    write("evidence-noeocd.zip", build_zip(entries, with_directory=False),
          "same entries, central directory and EOCD cut off")
    write("evidence-sfx.zip", build_zip(entries, stub=b"MZ" + b"\x90" * 4094),
          "4096-byte stub prepended: every recorded offset is off by 4096")

    # gzip: three members, the middle one damaged
    members = []
    for i, part in enumerate([src[:40000], src[40000:90000], src[90000:150000]]):
        blob = deflate(part)
        if i == 1:
            # Early damage, so the member does not merely decode wrongly, it
            # stops.  gzip records no compressed length, so the end of a member
            # is found by decoding to it -- and a member that will not decode
            # takes the position of every member after it with it.
            blob = bytearray(blob)
            for j in range(60):
                blob[40 + j] ^= 0xFF
            blob = bytes(blob)
        members.append(b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + blob
                       + struct.pack("<II", zlib.crc32(part) & 0xFFFFFFFF,
                                     len(part) & 0xFFFFFFFF))
    write("capture.gz", b"".join(members),
          "3 members, member 1 corrupted: it takes the position of members 2-3")

    # zlib with a preset dictionary the file does not carry
    zdict = b"forensic analysis of compressed evidence"
    co = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY,
                          zdict=zdict)
    body = src[:19200]
    write("preset-dict.zlib", co.compress(body) + co.flush(),
          f"FDICT set, {len(zdict)}-byte dictionary absent from the file")

    # two fragments with a hole between them
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9)
    comp, marks = b"", [(0, 0)]
    payload = src[:60000]
    for i in range(0, len(payload), 1024):
        comp += co.compress(payload[i:i + 1024]) + co.flush(zlib.Z_SYNC_FLUSH)
        marks.append((len(comp), min(i + 1024, len(payload))))
    comp += co.flush()
    a_end, a_out = marks[8]
    b_start, b_out = marks[10]
    write("capture-head.bin", comp[:a_end], "head of a stream, with its tables")
    write("capture-tail.bin", comp[b_start:],
          f"the same stream later on; {b_out - a_out} output bytes missing "
          f"between them, total decompressed size {len(payload)}")

    # A carve out of the middle of a static archive: no flushing, so blocks are
    # tens of KB apart and the fragment begins inside a dynamic block whose
    # table is behind the damage.  The bad case on purpose, and the one
    # `--routes` exists to talk about.
    whole = deflate(src, level=9)
    write("carved-midblock.bin", whole[30000:90000],
          f"60 KB carved out of the middle of a {len(whole)} B unflushed "
          "stream: starts inside a dynamic block, table gone")

    # A table that is GONE rather than wrong, in a stream that survives it.
    #
    # The single-block case is hopeless and the tool says so: one dynamic header,
    # destroyed, and there is no parse and nothing to resync to. What makes this
    # one recoverable is that the encoder flushed, so the stream is 24 dynamic
    # blocks each carrying its own table -- destroying the first costs its own
    # 2,048 bytes and nothing else.
    #
    # Z_FULL_FLUSH rather than Z_SYNC_FLUSH, and the difference is the whole
    # result. A full flush resets the window, so no later block can reference the
    # destroyed one and every byte after the damage decodes exactly. With a sync
    # flush the window carries over, back-references reach into the lost 2 KB,
    # and the unknowns propagate forward through the match chains -- the same
    # bytes recovered, but 15,555 known against 45,952 rather than all of them.
    #
    # Raw deflate on purpose: no container, so no CRC32 and no recorded length.
    # Nothing outside the bitstream confirms any of this.
    chunk = 2048
    payload = src[:48000]
    co = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    parts = []
    for i in range(0, len(payload), chunk):
        parts.append(co.compress(payload[i:i + chunk]))
        parts.append(co.flush(zlib.Z_FULL_FLUSH))
    parts.append(co.flush())
    intact = b"".join(parts)
    from toeknife.core.inflate import Decoder
    tr = Decoder(intact).stream(0)
    first_body = min(s.bit_start for s in tr.symbols if s.block == 0)
    rng = random.Random(1951)
    lost = bytearray(intact)
    # Everything between BTYPE and the first symbol: the table, and only it. The
    # block still announces itself as dynamic, which is what makes the failure
    # 'bad-table' rather than a fragment the decoder walks straight past.
    for bit in range(tr.blocks[0].bit_start + 3, first_body):
        if rng.random() < 0.5:
            lost[bit >> 3] ^= 1 << (bit & 7)
    write("lost-table.bin", bytes(lost),
          f"raw deflate, {len(tr.blocks)} blocks, first table destroyed: 0 bytes "
          f"out until f3 resyncs, then {len(payload) - chunk:,} of "
          f"{len(payload):,} exactly ({100 * (len(payload) - chunk) / len(payload):.1f}%)")
    # The exact bytes this specimen was built from, beside it, so
    # tests/test_scan.py compares the recovered bytes against a file rather
    # than re-deriving them from the payload.
    write("lost-table.payload", payload,
          f"the {len(payload):,} bytes lost-table.bin was built from")

    # One inverted bit inside a dynamic block's header. The distinction from
    # every other sample here is that nothing is MISSING: the table is present
    # and wrong, so recovery is a correction problem and the search is the flips
    # over the header rather than the tables over the alphabet.
    #
    # Gzip rather than raw, for two reasons. A raw .bin has no container, so the
    # tool opens it as a fragment and aligns past the broken header -- there is
    # then no dynamic header in view to repair. And gzip records a CRC32 of the
    # original, which turns the repair from a candidate into a verified recovery:
    # f7 undoes the flip, f8 confirms it was the right flip.
    body = src[:48000]
    intact = deflate(body, level=9)
    # The bit is searched for rather than picked: which bit breaks the table that
    # decodes the table depends on the header's own layout, and a hardcoded
    # position quietly stopped landing there when the payload changed. Wanted is
    # the loudest failure available -- nothing decoded at all.
    from toeknife.core.inflate import Decoder
    HEAD = 10                       # the fixed gzip header this writes
    chosen = None
    for bit in range(3, 400):
        trial = bytearray(intact)
        trial[bit >> 3] ^= 1 << (bit & 7)
        try:
            tr = Decoder(bytes(trial)).stream(0)
            out = tr.tape.stats()["out_len"]
        except Exception:
            continue
        if out == 0 and tr.status.startswith("bad"):
            chosen = bit
            break
    if chosen is None:
        raise SystemExit("no single bit produced a total header failure")
    rotted = bytearray(intact)
    rotted[chosen >> 3] ^= 1 << (chosen & 7)
    write("bitrot-header.gz",
          b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03" + bytes(rotted)
          + struct.pack("<II", zlib.crc32(body) & 0xFFFFFFFF,
                        len(body) & 0xFFFFFFFF),
          f"one flipped bit (stream bit {chosen}, {HEAD * 8 + chosen} in the "
          f"file) in a dynamic header: 0 bytes out until f7 repairs it, then "
          f"all {len(body):,} with the gzip CRC to confirm it")

    write("plot-damaged.png", build_png(120, 80, damage=True),
          "bit rot inside IDAT: the chunk CRC localises it, and cannot verify "
          "the recovery")
    write("plot-splitrun.png", build_png(120, 80, split_run=True),
          "a tEXt chunk interrupting the IDAT run, which the spec forbids")
    write("report-noxref.pdf", build_pdf([src[:8000], src[8000:20000]],
                                         keep_xref=False),
          "two FlateDecode objects, cross-reference table destroyed")


if __name__ == "__main__":
    sys.exit(main())
