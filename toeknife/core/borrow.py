"""Reading a block whose table is gone, with a table from another capture.

A carve that starts inside a dynamic block has the block's bits and not its
table. Resync recovers everything from the next block header on; what sits
before that header stays unreadable. A related capture -- another copy of the
same or similar content -- may carry the same table, and this tries every
table it has.

The candidates are the related capture's OWN blocks first. Recompressing its
content instead fixes the encoder, and the table is a function of the encoder
as much as the content: the Phrack article gzip'd by GNU gzip comes out in two
blocks, split at output 17,724, while zlib packs it in one, so a carve of the
first could never match a table lifted from the second.
Recompressions at zlib levels 1-9 follow, for a related capture that is plain
text.

Two tests, because a table that parses is no evidence at all: an unrelated
article "recovered" 20,468 bytes that way.

The block's end is known. Resync found the next header, so the right table
puts its end-of-block exactly there. On the Phrack carve, 1 of 13 candidate
tables landed there, from all 48 starting bits; the other 12 from none.

Landing proves the table's shape, not its meanings. A copy of the article with
"One" changed to "Two" gives a table that differs only in `@` and `^`, lands
exactly, and decodes 9,545 bytes where the truth is 9,963 -- everything after
the first difference misplaced. The next block tells them apart: it copies
from this one, so once the two are joined its holes read through the borrowed
bytes, and a wrong table makes those reads garbage. Scored by the model, the
true table's seam bytes averaged -4.53 and the near miss -6.10.

Nothing is adopted automatically, and what is adopted is BORROWED: unproven,
until a copy or a checksum says otherwise.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

from .events import BlockInfo, Symbol, Trace
from .gap import agreed_boundaries, replay
from .inflate import Decoder
from .tape import BORROWED, Tape

# Starting bits tried for the headerless part: a symbol cut by the carve can
# run 48 bits into it (see gap.MAX_SYMBOL_BITS).
STARTS = 48
# Seam bytes scored by the model, at most.
SCORE_LIMIT = 4000


@dataclass
class Landing:
    """One candidate table whose end-of-block lands on the next header."""
    label: str
    litlen: object
    dist: object
    start: int                       # agreed starting bit
    starts: int                      # of STARTS, how many landed
    out_len: int                     # bytes the borrowed part produces
    known: int                       # of them, known
    seam: int = 0                    # next-block bytes now read through it
    score: float | None = None       # model mean over those seam bytes

    def summary(self, baseline: float | None) -> str:
        head = (f"{self.label}: lands from {self.starts} of {STARTS} starts, "
                f"{self.out_len:,} bytes, {self.seam:,} read through by the "
                f"next block")
        if self.score is None:
            return head
        if baseline is None:
            return head + f", model {self.score:.2f}"
        return head + f", model {self.score:.2f} against {baseline:.2f}"


@dataclass
class Borrow:
    boundary: int | None             # bit the next block header starts at
    tried: int = 0
    landings: list[Landing] = field(default_factory=list)
    baseline: float | None = None    # model mean over the next block's own text
    reason: str = ""

    def summary(self) -> str:
        if self.boundary is None:
            return self.reason
        head = (f"{self.tried} table(s) tried against the block end at bit "
                f"{self.boundary:,}")
        if not self.landings:
            return (f"{head}: none lands there. None of them is this block's "
                    f"table.")
        n = len(self.landings)
        tail = ("" if self.baseline is not None or n == 1 else
                " With no model loaded they can't be ranked; f5 loads one.")
        return (f"{head}: {n} land{'s' if n == 1 else ''} there, with the "
                f"right shape. The model score of the bytes the next block "
                f"reads through it says whether the meanings are right too."
                + tail)


def _dedupe(tables):
    seen, out = set(), []
    for label, litlen, dist in tables:
        key = (tuple(litlen.lengths), tuple(dist.lengths))
        if key not in seen:
            seen.add(key)
            out.append((label, litlen, dist))
    return out


def candidate_tables(blob: bytes):
    """Every dynamic table in the related capture, then recompressions of its
    content. (label, litlen, dist), duplicates removed."""
    from .. import containers
    from .related import plain_bytes
    out = []
    report = None
    try:
        report = containers.parse(blob)
    except Exception:
        report = None
    streams = [blob]
    if report is not None and report.decodable:
        streams = []
        for st in report.decodable:
            try:
                streams.append(st.slice(blob))
            except Exception:
                continue
    for n, raw in enumerate(streams):
        try:
            tr = Decoder(raw, strict_window=False).stream(0)
        except Exception:
            continue
        for b in tr.blocks:
            if b.btype == 2 and b.litlen is not None:
                where = f"stream {n}, " if len(streams) > 1 else ""
                out.append((f"its own {where}block {b.index}", b.litlen,
                            b.dist))
    content = plain_bytes(blob)
    for level in range(1, 10):
        co = zlib.compressobj(level, zlib.DEFLATED, -15)
        comp = co.compress(content) + co.flush()
        for b in Decoder(comp).stream(0).blocks:
            if b.btype == 2 and b.litlen is not None:
                out.append((f"zlib level {level}, block {b.index}", b.litlen,
                            b.dist))
    return _dedupe(out)


def boundary(session) -> int | None:
    """The bit the next block header starts at: where a resynced decode
    begins, or the best boundary a scan finds."""
    if session.mode == "stream" and session.resumed:
        return session.bit_offset
    from .scan import scan
    for p in scan(session.data):
        if p.status == "end":
            return p.bit_offset
    return None


def _model_mean(tape, positions, model) -> float | None:
    if model is None or not positions:
        return None
    order = getattr(model, "order", 4)
    total = 0.0
    for p in positions[:SCORE_LIMIT]:
        ctx = bytearray()
        for q in range(p - order, p):
            v = tape.value_at(q)
            if v is None:
                ctx.clear()
            else:
                ctx.append(v)
        total += model.logprob(tape.value_at(p), bytes(ctx))
    return total / min(len(positions), SCORE_LIMIT)


def assemble(data: bytes, prefix: Trace, rest: Trace) -> Trace:
    """The borrowed part followed by everything from the next header, as one
    trace. The borrowed bytes are BORROWED; copies of them inherit it."""
    tape = Tape()
    tape.taint = BORROWED
    replay(prefix.symbols, tape)
    tape.taint = None
    base = tape.length
    replay(rest.symbols, tape)
    symbols = list(prefix.symbols)
    for s in rest.symbols:
        symbols.append(Symbol(
            kind=s.kind, out_pos=s.out_pos + base, bit_start=s.bit_start,
            bit_end=s.bit_end, block=s.block + 1, literal=s.literal,
            length=s.length, distance=s.distance, fields=s.fields,
            code_sym=s.code_sym, dist_sym=s.dist_sym))
    pb = prefix.blocks[0]
    blocks = [BlockInfo(index=0, bit_start=pb.bit_start, bit_end=pb.bit_end,
                        btype=-1, btype_name="borrowed", out_start=0,
                        out_end=base, litlen=pb.litlen, dist=pb.dist,
                        n_literals=pb.n_literals, n_matches=pb.n_matches,
                        status=pb.status)]
    for b in rest.blocks:
        blocks.append(BlockInfo(
            index=b.index + 1, bit_start=b.bit_start, bit_end=b.bit_end,
            bfinal=b.bfinal, btype=b.btype, btype_name=b.btype_name,
            out_start=b.out_start + base, out_end=b.out_end + base,
            fields=b.fields, clen_ops=b.clen_ops, litlen=b.litlen,
            dist=b.dist, codelen=b.codelen, hlit=b.hlit, hdist=b.hdist,
            hclen=b.hclen, stored_len=b.stored_len, n_literals=b.n_literals,
            n_matches=b.n_matches, status=b.status))
    return Trace(data, prefix.bit_start, rest.bit_end, blocks, symbols, tape,
                 rest.status, rest.error, headerless=True)


def borrow(session, blob: bytes, model=None, progress=None) -> Borrow:
    """Try every table in `blob` on the part of this capture before the next
    block header, and score the ones that land."""
    data = session.data
    end = boundary(session)
    if end is None:
        return Borrow(None, reason=(
            "no block header found to test against: a borrowed table is only "
            "checked by landing its end-of-block exactly on the next one, and "
            "this capture has none"))
    if end <= 0:
        return Borrow(None, reason="the decode already starts at the data's "
                                   "first bit; there is nothing before it")
    rest = Decoder(data, strict_window=False).stream(end)
    alone = Tape()
    replay(rest.symbols, alone)
    own = [p for p in range(alone.length) if alone.value_at(p) is not None]
    out = Borrow(end, baseline=_model_mean(alone, own, model))
    tables = candidate_tables(blob)
    out.tried = len(tables)
    head = data[:(end + 7) // 8 + 1]
    for n, (label, litlen, dist) in enumerate(tables):
        if progress is not None:
            progress(n, len(tables), f"table {n + 1} of {len(tables)}")
        landed = []
        for bit in range(min(STARTS, end)):
            try:
                t = Decoder(head, strict_window=False).fragment(bit, litlen,
                                                                dist)
            except Exception:
                continue
            if t.status == "eob" and t.bit_end == end:
                landed.append(t)
        if not landed:
            continue
        agreed = agreed_boundaries(landed)
        start = agreed[0] if agreed else landed[0].bit_start
        prefix = Decoder(head, strict_window=False).fragment(start, litlen,
                                                             dist)
        trace = assemble(data, prefix, rest)
        base = len(prefix.tape)
        seam = [base + p for p in range(alone.length)
                if alone.value_at(p) is None
                and trace.tape.value_at(base + p) is not None]
        st = prefix.tape.stats()
        out.landings.append(Landing(
            label, litlen, dist, start, len(landed), len(prefix.tape),
            st["known"], len(seam), _model_mean(trace.tape, seam, model)))
    if out.baseline is not None:
        out.landings.sort(key=lambda c: -(c.score if c.score is not None
                                          else -99))
    return out


def apply(session, landing: Landing, end: int) -> None:
    """Put the borrowed part and the rest in place of the current decode."""
    head = session.data[:(end + 7) // 8 + 1]
    prefix = Decoder(head, strict_window=False).fragment(
        landing.start, landing.litlen, landing.dist)
    rest = Decoder(session.data, strict_window=False).stream(end)
    session.trace = assemble(session.data, prefix, rest)
    session.mode = "assembled"
    session.assumed_note = ""          # its bytes are BORROWED on their own account
    session._candidates = []
    session.headcheck = None
    session._reapply_guesses()
    session.record_step("borrow", start=landing.start, end=end,
                        label=getattr(landing, "label", ""),
                        litlen=list(landing.litlen.lengths),
                        dist=list(landing.dist.lengths))
