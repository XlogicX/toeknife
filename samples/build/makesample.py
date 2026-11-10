"""Build `samples/streaming-fragment.bin`: a truncated streaming capture.

The tool's hardest and most common case is a capture that begins mid-stream, so
the Huffman table defining its first symbols is behind the damage.  How much
comes back depends entirely on how the stream was written, and this builds the
good case on purpose: `Z_SYNC_FLUSH` every 8 KB, which is what an HTTP server or
a chunked response emits.  Each flush leaves an empty stored block -- the
`00 00 FF FF` marker -- so a fresh header starts byte-aligned right after it, and
everything from the next marker onward reads without outside help.

The payload is `samples/damaged/payload.txt`, 256 KB of The Adventures of
Sherlock Holmes, pinned, so the recovered tape reads as prose and a clean
checkout rebuilds both samples byte for byte.

    .venv/bin/python samples/build/makesample.py
"""
import hashlib
import os
import sys
import zlib

CHUNK = 8192
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "samples", "streaming-fragment.bin")
OUT_FIXED = os.path.join(ROOT, "samples", "fixed-fragment.bin")


def payload():
    with open(os.path.join(ROOT, "samples", "damaged", "payload.txt"), "rb") as fh:
        return fh.read()


def build_fixed():
    """The other sample: a fragment that decodes COMPLETELY, with no holes.

    Two properties do it, and both are choices the writer made rather than luck.

    `Z_FIXED` forces fixed-Huffman blocks, whose tables are defined by RFC 1951
    rather than carried in a header -- so losing the head of the stream loses no
    table, and the only thing needed is the right bit alignment.

    `Z_FULL_FLUSH` resets the LZ77 window at each flush, so no back-reference
    after a marker can point before it.  That is what removes the holes: the
    usual reason a fragment has unknowns is copies reaching into the window that
    went missing with the head, and after a full flush there is nothing back
    there to reach for.

    A plain `Z_SYNC_FLUSH` stream gets the first property and not the second,
    which is exactly the difference between this sample and the other one.
    """
    raw = payload()
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9, zlib.Z_FIXED)
    out = bytearray()
    for i in range(0, len(raw), 16384):
        out += co.compress(raw[i:i + 16384])
        out += co.flush(zlib.Z_FULL_FLUSH)
    out += co.flush()
    comp = bytes(out)
    assert zlib.decompress(comp, -15) == raw, "stream does not round-trip"
    cut = len(comp) // 3
    frag = comp[cut:]
    with open(OUT_FIXED, "wb") as fh:
        fh.write(frag)
    marker = b"\x00\x00\xff\xff"
    print(f"\n  [fixed] payload {len(raw):,} B -> compressed {len(comp):,} B")
    print(f"  [fixed] discarded the first {cut:,} B; fragment {len(frag):,} B, "
          f"{frag.count(marker)} markers, first at {frag.find(marker):,}")
    print(f"  [fixed] sha256 {hashlib.sha256(frag).hexdigest()[:32]}")
    print(f"  [fixed] written to {OUT_FIXED}")


def main():
    raw = payload()
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9)
    out = bytearray()
    for i in range(0, len(raw), CHUNK):
        out += co.compress(raw[i:i + CHUNK])
        out += co.flush(zlib.Z_SYNC_FLUSH)
    out += co.flush()
    comp = bytes(out)
    # If this ever fails the sample is not what the README says it is.
    assert zlib.decompress(comp, -15) == raw, "stream does not round-trip"
    marker = b"\x00\x00\xff\xff"

    cut = len(comp) // 3          # discard the head: the table goes with it
    frag = comp[cut:]
    with open(OUT, "wb") as fh:
        fh.write(frag)
    print(f"  payload    {len(raw):,} bytes (samples/damaged/payload.txt)")
    print(f"  compressed {len(comp):,} bytes, {comp.count(marker)} sync markers")
    print(f"  discarded  the first {cut:,} bytes")
    print(f"  fragment   {len(frag):,} bytes, {frag.count(marker)} markers left, "
          f"first at byte {frag.find(marker):,}")
    print(f"  sha256     {hashlib.sha256(frag).hexdigest()[:32]}")
    print(f"  written to {OUT}")
    build_fixed()
    return 0


if __name__ == "__main__":
    sys.exit(main())
