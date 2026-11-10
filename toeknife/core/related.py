"""Take the table from a related capture, and say up front what the odds are.

The highest-yield route measured in this project does not search for anything: if
the same content can be obtained compressed from somewhere else -- a mirror, a
CDN sibling, a cache, a later fetch -- the table comes back exactly and no
plaintext is needed. It was measured at 39.5% on sibling domains.

What was established next is the part that makes this honest to
offer: **what predicts success is token overlap, and nothing else.** A re-fetch of
the same url four weeks later succeeds 10.1% of the time and a sibling domain
39.5%, but conditioned on overlap the two lie on one curve -- elapsed time is
only a proxy for how far the content moved. So the operator can be told the odds
*before* the attempt, from the two captures alone.

One caveat keeps it from over-promising: the curve is
calibrated on HTML. On css/js at the same measured overlap the table survives
less often, so the numbers below say which content they came from.
"""
from __future__ import annotations

import gzip
import re
import zlib

from .inflate import Decoder

WORD = re.compile(rb"[A-Za-z0-9_-]+")

# The level the whole project compares tables at. Both sides must be compressed
# the same way or the comparison is about the encoder rather than the content.
LEVEL = 6

# Measured over 1,621 changed HTML pairs across two gaps
# (thirteen hours and four weeks) that agree once binned by overlap.
HTML_CURVE = (
    (0.00, 0.90, "almost never", "0.0-0.2% over 573 pairs"),
    (0.90, 0.95, "rarely", "2-4% over 210 pairs"),
    (0.95, 0.99, "sometimes", "9-11% over 353 pairs"),
    (0.99, 0.999, "often", "44-56% over 404 pairs"),
    (0.999, 1.01, "usually", "70-71% over 81 pairs"),
)

# Measured over 652 changed css/js pairs: lower at matched overlap, and the
# samples in the top bands are small, so this is stated as a direction.
ASSET_NOTE = ("measured lower than HTML at the same overlap -- 1 of 46 where "
              "HTML gives about 10%, 3 of 15 where it gives 44-56%")


def overlap(a: bytes, b: bytes) -> float:
    """Token-set overlap, the same measure the calibration was built on.

    Keep it exactly as it is: if this measure drifts from the one the curve
    was built on, the curve above stops applying to it.
    """
    sa, sb = set(WORD.findall(a)), set(WORD.findall(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def expectation(value: float, kind: str = "html") -> str:
    """What that overlap has historically been worth, with its sample size."""
    for lo, hi, word, evidence in HTML_CURVE:
        if lo <= value < hi:
            base = (f"{word} -- on HTML pairs at this overlap the table "
                    f"carried over {evidence}")
            if kind != "html":
                return base + f".\nFor {kind}, {ASSET_NOTE}"
            return base
    return "no measurement covers that overlap"


def plain_bytes(blob: bytes) -> bytes:
    """The content, whichever wrapper it arrived in."""
    if blob[:2] == b"\x1f\x8b":
        return gzip.decompress(blob)
    if blob and (blob[0] & 0x0F) == 8:
        try:
            return zlib.decompress(blob)
        except zlib.error:
            pass
    try:                                    # already a raw DEFLATE stream?
        return zlib.decompress(blob, -15)
    except zlib.error:
        return blob                         # take it as plain content


def tables_from(content: bytes, level: int = LEVEL):
    """Compress the related content and lift its first dynamic block's tables.

    The table is a function of the content AND the encoder, so this fixes the
    encoder at the level everything else in the project uses. A capture whose
    server used different settings will not match here even when the content is
    identical -- measured at about 4% of byte-identical assets
    and 7% of byte-identical HTML.
    """
    co = zlib.compressobj(level, zlib.DEFLATED, -15, 9)
    comp = co.compress(content) + co.flush()
    try:
        tr = Decoder(comp).stream()
    except Exception:
        return None, None
    for blk in tr.blocks:
        if blk.btype == 2 and blk.litlen is not None:
            return blk.litlen, blk.dist
    return None, None


# Below this many distinct tokens on the target side, an overlap figure is not
# measuring similarity, it is measuring how little has been recovered. Quoting a
# calibrated rate off it would dress up noise as a prediction, which is the same
# failure as a check that refuses silently -- so this abstains out loud instead.
MIN_TOKENS = 40


def report(target_content: bytes, related_blob: bytes,
           kind: str = "html") -> dict:
    """Everything the operator should see before deciding to try the transplant."""
    other = plain_bytes(related_blob)
    have_tokens = len(set(WORD.findall(target_content or b"")))
    if have_tokens < MIN_TOKENS:
        ov = None
        exp = (f"not predictable: only {have_tokens} distinct token(s) have "
               f"been recovered here, and the overlap curve needs something to "
               f"compare. The transplant can still be tried -- it just cannot "
               f"be priced first.")
    else:
        ov = overlap(target_content, other)
        exp = expectation(ov, kind)
    litlen, dist = tables_from(other)
    return {
        "overlap": ov,
        "have_tokens": have_tokens,
        "expectation": exp,
        "related_bytes": len(other),
        "litlen": litlen,
        "dist": dist,
        "usable": litlen is not None,
        # The common cause is size rather than anything exotic: zlib emits a
        # FIXED block for small inputs, which carries no table at all. Around
        # 1.7 KB of repetitive text still comes out fixed; 8.8 KB comes out
        # dynamic. A related capture smaller than a few KB is usually no use.
        "why_not": ("" if litlen is not None else
                    "the related capture has no dynamic block to lift a table "
                    "from -- most often because it is small enough that zlib "
                    "used a fixed table, which carries none. A few KB or more "
                    "is usually needed"),
    }
