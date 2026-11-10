"""Reading on past damage in the middle of a block.

A burst of overwritten bytes stops a decode part way through a block, and
everything after the stop is lost to it even though the compressed data past
the burst is intact. It is the lost-packet case with one difference that
matters: a capture says exactly where its hole is, and damage does not.

Two of the three unknowns still have answers.

Where to read from. DEFLATE self-synchronises, so a reading that starts where
the decode stopped settles onto the true symbol boundaries somewhere after the
burst. If it runs into an invalid code first, it starts again just past that.

Where its output goes. A container that recorded the decompressed size anchors
the tail at the END: the last byte the tail produces is the entry's last byte,
so every true symbol lands at its true output position whatever garbage came
before it.

Where the damage ends has no answer. The readings converge before the burst is
over (bit 61,336 on stack.zip's shredded.txt, where the burst runs to 61,816),
so their agreement says nothing about it, and a byte model scores the garbage
literals much like the text around them. Back-references carry the garbage on:
332 wrong bytes, the last of them 36 KB past the burst. So
every byte read here is stored as RESUMED, a provenance of its own, and stays
that way until something outside the data proves it: a suspected copy checked
against the recorded checksum.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .events import BlockInfo, Symbol, Trace
from .gap import replay
from .inflate import Decoder
from .tape import RESUMED, Tape

# Restarts allowed after a resumed reading runs into another invalid code
# before the reading that reaches the end is found.
MAX_RESTARTS = 256
# How far before the end of the data a clean reading may stop: a gzip trailer
# is 8 bytes, and the last byte of the stream is padding past its final bit.
END_SLACK_BITS = 9 * 8


@dataclass
class Resumed:
    trace: Trace | None
    resumed_at: int = 0             # bit the reading that reached the end starts at
    gap: int = 0                    # unknown output bytes between head and tail
    dropped: int = 0                # tail bytes dropped to make it fit the size
    tail: int = 0                   # bytes placed after the gap
    start: int = 0                  # output position of the first byte placed
    restarts: int = 0
    header: bool = False            # read from a later block header, not the stop's table
    reason: str = ""                # why it could not, when trace is None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.trace is None:
            return self.reason
        where = ("the block header at bit" if self.header
                 else "bit")
        out = (f"read on from {where} {self.resumed_at:,}: {self.tail:,} "
               f"bytes placed by the recorded size")
        if self.gap:
            out += f", {self.gap:,} unknown between"
        if self.dropped:
            out += (f"; {self.dropped:,} read before those were dropped, "
                    f"since they overrun the size or copy from before output "
                    f"0")
        return out + ". They stay unproven until a copy matches the checksum"


def applies(session) -> tuple[bool, str]:
    """Whether a stream decode stopped on damage with a size to place what
    follows. The reason is written for the operator either way."""
    tr = getattr(session, "trace", None)
    status = getattr(tr, "status", "")
    if getattr(session, "mode", "") != "stream":
        return False, "only a decode that starts at the stream's head stops " \
                      "on damage this way"
    if status in ("end", "eob"):
        return False, f"the decode already ends cleanly ({status!r})"
    if status == "truncated":
        return False, "the data runs out; there is nothing after the stop " \
                      "to read"
    size = getattr(getattr(session, "stream", None), "uncompressed_size", None)
    if size is None:
        return False, "nothing recorded the decompressed size, so what " \
                      "follows the damage cannot be placed"
    if len(tr.tape) >= size:
        return False, "the decode already produced the recorded size"
    if not any(b.litlen is not None for b in tr.blocks):
        return False, "no table was read before the stop to read on with"
    return True, (f"the decode stops at bit {tr.bit_end:,} ({status}) with "
                  f"{size - len(tr.tape):,} of the recorded size still to come")


def _read_to_end(data: bytes, bit: int, litlen, dist) -> list[Trace]:
    """A headerless reading from `bit`, then every block after it, until
    something ends. Clean when the last part ends on a final block."""
    parts = [Decoder(data, strict_window=False).fragment(bit, litlen, dist)]
    while parts[-1].status == "eob" and parts[-1].bit_end < len(data) * 8 - 8:
        nxt = Decoder(data, strict_window=False).stream(parts[-1].bit_end)
        parts.append(nxt)
        if nxt.status != "eob":
            break
    return parts


def _clean(parts: list[Trace], data: bytes) -> bool:
    """A final end-of-block at the end of the data.

    A final block on its own is not enough: garbage can decode as a short
    final block, and on one burst a reading ended that way 30 KB early, put
    105 wrong literals out of 107 in the tail, and called the rest a gap. The
    stream's real end is where the data ends.
    """
    last = parts[-1]
    if last.bit_end < len(data) * 8 - END_SLACK_BITS:
        return False
    # A tail that is all one block ends on that block's own end-of-block.
    return last.status == "end" or (len(parts) == 1 and last.status == "eob")


def _from_next_header(data: bytes, bit: int, progress=None):
    """The earliest block header after `bit` whose decode reaches the end.

    The fallback for when no reading with the damaged block's table gets
    there. A restart can land past that block's end, and from then on the old
    table reads the next block as noise; on 3 bursts in 70 nothing reached the
    end that way. A header carries its own table, so this does not care.
    """
    from .scan import scan
    points = scan(data, start_bit=bit, progress=progress)
    clean = [p for p in points if p.trace is not None
             and _clean([p.trace], data)]
    if not clean:
        return None
    return min(clean, key=lambda p: p.bit_offset)


def resume(session, progress=None) -> Resumed:
    """Read on past where the decode stopped and place it by the size."""
    ok, why = applies(session)
    if not ok:
        return Resumed(None, reason=why)
    head = session.trace
    data = session.data
    size = session.stream.uncompressed_size
    table = [b for b in head.blocks if b.litlen is not None][-1]

    bit, restarts, header = head.bit_end, 0, False
    while True:
        parts = _read_to_end(data, bit, table.litlen, table.dist)
        if _clean(parts, data):
            break
        restarts += 1
        if restarts > MAX_RESTARTS or parts[-1].bit_end >= len(data) * 8 - 8:
            found = _from_next_header(data, head.bit_end, progress)
            if found is None:
                return Resumed(None, reason=(
                    f"nothing from bit {head.bit_end:,} on decodes cleanly "
                    f"to the end of the stream, with the table before the "
                    f"stop or from a later block header; there may be more "
                    f"damage further in"))
            bit, parts, header = found.bit_offset, [found.trace], True
            break
        bit = max(parts[-1].bit_end, bit + 1)

    tail_syms: list[tuple[Symbol, int]] = []      # (symbol, tail-relative out)
    out = 0
    for part in parts:
        for s in part.symbols:
            tail_syms.append((s, out + s.out_pos))
        out += len(part.tape)
    tail_len = out

    # The tail is anchored at the end, so where each symbol lands does not
    # depend on what is dropped in front of it. Two things prove a symbol at
    # the front is garbage, and everything up to the last such one goes:
    # it lands on output the head already produced (garbage on both sides of
    # the stop made more output than the entry holds), or it is a match that
    # reaches before output 0, which no true symbol can.
    end = size - tail_len                          # where tail offset 0 would land
    first = 0
    for i, (s, rel) in enumerate(tail_syms):
        if end + rel < len(head.tape):
            first = i + 1
        elif s.kind == "match" and end + rel - s.distance < 0:
            first = i + 1
    kept = tail_syms[first:]
    dropped = (kept[0][1] if kept else tail_len)
    placed = tail_len - dropped
    gap = size - len(head.tape) - placed

    tape = Tape()
    hc = getattr(session, "headcheck", None)
    cut = hc.cut if hc is not None else len(head.tape)
    for s in head.symbols:
        if s.out_pos >= cut:
            tape.taint = RESUMED
        replay([s], tape)
    tape.taint = None
    if gap:
        tape.emit_unknown(gap)
    tape.taint = RESUMED
    replay([s for s, _ in kept], tape)
    tape.taint = None

    start = len(head.tape) + gap                   # where the tail's first byte goes
    delta = start - dropped
    symbols = list(head.symbols)
    for s, rel in kept:
        symbols.append(Symbol(
            kind=s.kind, out_pos=delta + rel, bit_start=s.bit_start,
            bit_end=s.bit_end, block=s.block, literal=s.literal,
            length=s.length, distance=s.distance, fields=s.fields,
            code_sym=s.code_sym, dist_sym=s.dist_sym))

    blocks = list(head.blocks)
    offset = delta
    for part in parts:
        for b in part.blocks:
            blocks.append(BlockInfo(
                index=len(blocks), bit_start=b.bit_start, bit_end=b.bit_end,
                bfinal=b.bfinal, btype=b.btype,
                btype_name=("resumed" if b.btype_name == "fragment"
                            else b.btype_name),
                out_start=max(start, b.out_start + offset),
                out_end=b.out_end + offset, fields=b.fields,
                clen_ops=b.clen_ops, litlen=b.litlen, dist=b.dist,
                codelen=b.codelen, hlit=b.hlit, hdist=b.hdist, hclen=b.hclen,
                stored_len=b.stored_len, n_literals=b.n_literals,
                n_matches=b.n_matches, status=b.status))
        offset += len(part.tape)

    trace = Trace(data, head.bit_start, parts[-1].bit_end, blocks, symbols,
                  tape, "end", None, headerless=False)
    got = Resumed(trace, resumed_at=bit, gap=gap, dropped=dropped,
                  tail=placed, start=start, restarts=restarts,
                  header=header)
    got.notes.append(got.summary())
    return got


def apply(session, got: Resumed) -> None:
    """Put the resumed decode in place of the stopped one."""
    session.trace = got.trace
    session.mode = "assembled"
    session._candidates = []
    session.resumed_from = got
    session._reapply_guesses()
    session.record_step("resume")
