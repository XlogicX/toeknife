"""Where the damage starts, in a decode that stopped on it.

A burst stops a decode some way past where it starts: the decoder reads
garbage as symbols until one of them is invalid. On shredded.txt that is 608
bytes, 574 of them wrong; on one synthetic burst the garbage decoded as valid
symbols for about 5,000 bytes, 4,780 of them wrong. They went on the
tape as literals and copies, in the colour of facts.

Nothing in the bits says where the garbage starts. The encoder does. zlib's
LZ77 parse is a function of the input and the level, so compressing the
decoded output again reproduces the captured symbols exactly up to about where
the damage starts, and stops agreeing there. It was measured how close
"about" is and which encoders it holds on: zlib, GNU gzip, pigz,
busybox gzip and Info-ZIP reproduce; libdeflate and 7-Zip do not.

So the cut is where the replay diverges, moved back by a measured margin, and
everything from the cut to the stop is stored as RESUMED, unproven like the
bytes read past it. When no zlib level reproduces the stream, the cut falls
back to the start of the block before the one that stopped.

Damage in a block's HEADER is the case the replay cannot see. A wrong table
garbles the block from its first symbol, and recompressing the wrong output
reproduces the wrong symbols, since the parse is self-consistent: on
stack-rot.gz it agreed to output 699 with `^olume` and `BugTraz` before the cut.
Flipping 823 header bits that stopped a decode left 27 with
wrong bytes as facts, and in all 27 the replay diverged within 1,829 bytes of
the stopped block's start. So a divergence that early marks the whole block.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass

from .gap import replay
from .inflate import Decoder
from .tape import RESUMED, Tape

# How far back from the replay's divergence the cut goes, in output bytes.
# Measured over 560 bursts and nine encoders.
MARGIN = 512
# Symbols each level is ranked on before the best is decoded in full.
PROBE = 4000
# A block smaller than this, or one that is not dynamic, is what garbage after
# a garbage end-of-block looks like: two such blocks behind a burst sent the
# fallback one block short and left 29,052 wrong bytes as facts on 7-Zip.
PLAUSIBLE = 1024
# A divergence this close to the start of the block that stopped is where a
# damaged header would put it, and the whole block is marked. The worst
# header-damage leak measured diverged 1,829 bytes in.
HEADER_REACH = 4096
# A replay that reproduces at least this share of the symbols before the
# fallback cut has identified the encoder. libdeflate reached 2.9% and 7-Zip
# 0% in the sweep; every zlib-family encoder was above 90%.
IDENTIFIED = 0.5

CLEAN = ("end", "eob", "truncated")


@dataclass
class HeadCheck:
    cut: int                    # output position the unproven run starts at
    stop: int                   # output position the decode stopped at
    how: str                    # "replay" or "block"
    level: int | None = None    # the zlib level that reproduced it best
    agree: int = 0              # symbols that level reproduced
    diverge: int | None = None  # output position it first disagreed at

    @property
    def unproven(self) -> int:
        return self.stop - self.cut

    def summary(self) -> str:
        if self.how == "header":
            return (f"zlib level {self.level} stops reproducing it at output "
                    f"{self.diverge:,}, close enough to the block's start "
                    f"that the header may be what is damaged, so the "
                    f"{self.unproven:,} bytes from {self.cut:,} to the stop "
                    f"are unproven")
        if self.how == "replay":
            return (f"zlib level {self.level} reproduces the first "
                    f"{self.agree:,} symbols, to output {self.diverge:,}; the "
                    f"{self.unproven:,} bytes from {self.cut:,} to the stop "
                    f"are unproven")
        return (f"no zlib level reproduces this stream, so the "
                f"{self.unproven:,} bytes from block start {self.cut:,} to "
                f"the stop are unproven")


def _symbols(trace, limit=None):
    out = []
    for s in trace.symbols[:limit]:
        out.append((s.out_pos, s.kind,
                    s.literal if s.kind == "literal" else (s.length, s.distance)))
    return out


def _agreement(captured, out: bytes, level: int, limit=None):
    co = zlib.compressobj(level, zlib.DEFLATED, -15)
    blob = co.compress(out) + co.flush()
    got = _symbols(Decoder(blob).stream(0, max_symbols=limit), limit)
    for i, (a, b) in enumerate(zip(captured, got)):
        if a != b:
            return i, a[0]
    n = min(len(captured), len(got))
    return n, (captured[n][0] if n < len(captured) else len(out))


def _fallback(blocks) -> int:
    """The start of the block before the last one that looks real.

    The block that stopped may be garbage reached through a garbage
    end-of-block, and so may the blocks before it. Walk back over any that are
    small or not dynamic, then one more: the damage can be in the last real
    block's body, where its own garbage end-of-block came from.
    """
    i = len(blocks) - 1
    while i > 0 and (blocks[i].btype != 2 or
                     blocks[i].out_end - blocks[i].out_start < PLAUSIBLE):
        i -= 1
    return blocks[i - 1].out_start if i > 0 else 0


def check(trace) -> HeadCheck | None:
    """Where the unproven run before a damage stop starts, or None when the
    decode did not stop on damage."""
    if trace is None or trace.status in CLEAN or not trace.symbols:
        return None
    out = bytes(trace.tape.out[:len(trace.tape)])
    stop = len(out)
    captured = _symbols(trace)

    probe = captured[:PROBE]
    ranked = sorted(((_agreement(probe, out, L, PROBE)[0], L)
                     for L in range(1, 10)), reverse=True)
    best = None
    for _n, level in ranked[:2]:
        n, div = _agreement(captured, out, level)
        if best is None or n > best[0]:
            best = (n, div, level)
    agree, diverge, level = best

    blocks = trace.blocks
    fallback = _fallback(blocks)
    before = sum(1 for s in captured if s[0] < fallback)
    replay_cut = max(0, diverge - MARGIN)
    stopped = blocks[-1]
    if agree >= IDENTIFIED * max(before, 1) and agree >= 64:
        cut, how = replay_cut, "replay"
        if diverge - stopped.out_start < HEADER_REACH:
            cut, how = min(cut, stopped.out_start), "header"
    else:
        cut, how = fallback, "block"
    cut = min(cut, stop)
    # Back to the start of the symbol the cut falls in.
    cut = max((s.out_pos for s in trace.symbols if s.out_pos <= cut),
              default=0)
    return HeadCheck(cut, stop, how, level, agree, diverge)


def tape_with_cut(symbols, cut: int) -> Tape:
    """A tape for these symbols with everything from `cut` on unproven."""
    tape = Tape()
    for s in symbols:
        if s.out_pos >= cut and tape.taint is None:
            tape.taint = RESUMED
        replay([s], tape)
    tape.taint = None
    return tape
