"""Two fragments of one stream, with something missing between them.

The case this exists for is a capture that lost packets in the middle. You have
the head of a stream, so you have its Huffman tables, and you have a later piece,
and between them is a hole of unknown size.

There are two unknowns and they are different in kind. The *compressed* gap is
handled by finding where the later fragment's bitstream starts, which is the
ordinary alignment problem. The *output* gap is the harder one: the missing
compressed bytes produced some number of output bytes, and until that number is
right, every back-reference in the later fragment that reaches across the gap
lands on the wrong byte. Getting it wrong does not fail; it silently reads the
wrong text.

Where the container records the decompressed length, that number is arithmetic
rather than search: with one gap it is the recorded length minus what the two
fragments produced. gzip's ISIZE and a ZIP entry's uncompressed size both give
it outright. Without one, it has to be searched for, and the signal is that a
wrong offset makes the cross-gap references read gibberish.
"""
from __future__ import annotations

import os

import struct
from dataclasses import dataclass, field

from .events import BitField, BlockInfo, Symbol, Trace
from .inflate import Decoder
from .tape import Tape


@dataclass
class Piece:
    """One surviving fragment of a stream."""
    data: bytes
    bit_offset: int = 0
    label: str = ""
    has_header: bool = False        # starts at a block header, so carries tables


@dataclass
class Assembly:
    trace: Trace
    pieces: list[Piece]
    gaps: list[int]                          # output bytes missing before each piece
    starts: list[int] = field(default_factory=list)   # output position of each piece
    litlen: object | None = None
    dist: object | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def tape(self) -> Tape:
        return self.trace.tape

    def summary(self) -> str:
        st = self.tape.stats()
        gaps = ", ".join(str(g) for g in self.gaps) or "none"
        return (f"{len(self.pieces)} pieces, output gaps [{gaps}], "
                f"{st['out_len']} bytes out, {st['holes']} unknown classes "
                f"covering {st['hole_positions']} positions")


# Dragged bytes the model scores per candidate gap, at most.
SCORE_CAP = None
# How far above the sampled median a searched gap has to score before it is
# said to stand out. One measurement (Phrack pieces: 1.15 for the true gap,
# 0.20 for a wrong best), so it gates a warning and decides nothing.
SPREAD = 0.5
# The largest output gap assemble() will allocate. Every unknown byte is a
# Python object in the union-find, and a size read from the wrong 8 bytes once
# asked for 2.8 GB of them.
MAX_GAP = 1 << 26


def replay(symbols, tape: Tape) -> None:
    """Apply an already-decoded symbol stream to a different tape.

    Decoding a fragment and placing its output are separable: the symbols depend
    only on the bits and the tables, while where they land depends on what came
    before. That is what lets a fragment be decoded once and then positioned
    against several candidate gap lengths.
    """
    for sym in symbols:
        if sym.kind == "literal":
            tape.emit_literal(sym.literal)
        elif sym.kind == "match":
            tape.emit_copy(sym.distance, sym.length)


def _shift(sym: Symbol, out_delta: int, bit_delta: int) -> Symbol:
    return Symbol(
        kind=sym.kind, out_pos=sym.out_pos + out_delta,
        bit_start=sym.bit_start + bit_delta, bit_end=sym.bit_end + bit_delta,
        block=sym.block, literal=sym.literal, length=sym.length,
        distance=sym.distance,
        fields=tuple(BitField(f.label, f.start + bit_delta, f.end + bit_delta,
                              f.detail) for f in sym.fields),
        code_sym=sym.code_sym, dist_sym=sym.dist_sym)


def assemble(pieces: list[Piece], gaps: list[int],
             litlen=None, dist=None) -> Assembly:
    """Decode every piece and lay them out with the given output gaps between.

    The combined `data` is the pieces concatenated, which is not a valid stream
    and is not meant to be decoded again. It exists so that every symbol's bit
    span still points at the bytes it came from, which is what the inspector
    needs.
    """
    tape = Tape()
    symbols: list[Symbol] = []
    blocks: list[BlockInfo] = []
    combined = bytearray()
    starts: list[int] = []
    notes: list[str] = []
    status = "end"

    for index, piece in enumerate(pieces):
        if index:
            gap = gaps[index - 1]
            if gap > MAX_GAP:
                raise ValueError(f"an output gap of {gap:,} bytes is past the "
                                 f"{MAX_GAP:,} this will hold; the size it was "
                                 f"worked out from is probably wrong")
            if gap:
                tape.emit_unknown(gap)
            notes.append(f"{gap:,} unknown output bytes before {piece.label or index}")

        bit_delta = len(combined) * 8
        out_delta = tape.length
        starts.append(out_delta)

        if piece.has_header or (index == 0 and litlen is None):
            trace = Decoder(piece.data, strict_window=False).stream(piece.bit_offset)
            for block in trace.blocks:
                if block.litlen is not None:
                    litlen, dist = block.litlen, block.dist
        else:
            trace = Decoder(piece.data, strict_window=False).fragment(
                piece.bit_offset, litlen, dist)

        replay(trace.symbols, tape)
        symbols.extend(_shift(s, out_delta, bit_delta) for s in trace.symbols)
        for block in trace.blocks:
            shifted = BlockInfo(
                index=len(blocks), bit_start=block.bit_start + bit_delta,
                bit_end=block.bit_end + bit_delta, bfinal=block.bfinal,
                btype=block.btype, btype_name=block.btype_name,
                out_start=block.out_start + out_delta,
                out_end=block.out_end + out_delta,
                fields=[BitField(f.label, f.start + bit_delta, f.end + bit_delta,
                                 f.detail) for f in block.fields],
                clen_ops=block.clen_ops, litlen=block.litlen, dist=block.dist,
                codelen=block.codelen, hlit=block.hlit, hdist=block.hdist,
                hclen=block.hclen, stored_len=block.stored_len,
                n_literals=block.n_literals, n_matches=block.n_matches,
                status=block.status)
            blocks.append(shifted)
        combined += piece.data
        status = trace.status
        if trace.status not in ("end", "eob") and index < len(pieces) - 1:
            notes.append(f"{piece.label or index} ended {trace.status}")

    combined_trace = Trace(bytes(combined), 0, len(combined) * 8, blocks, symbols,
                           tape, status, None, headerless=False)
    return Assembly(combined_trace, pieces, list(gaps), starts, litlen, dist, notes)


def gap_from_size(pieces: list[Piece], uncompressed_size: int,
                  litlen=None, dist=None) -> int | None:
    """The output gap implied by a recorded decompressed length.

    Arithmetic, not search, and only valid for a single gap: with two or more
    the recorded total constrains their sum and nothing else.
    """
    if len(pieces) != 2:
        return None
    zero = assemble(pieces, [0], litlen, dist)
    produced = len(zero.tape)
    gap = uncompressed_size - produced
    return gap if gap >= 0 else None


@dataclass
class GapCandidate:
    gap: int
    score: float
    assembly: Assembly
    resolved: int = 0          # bytes the cross-gap references made known
    parts: dict = field(default_factory=dict)

    def summary(self) -> str:
        st = self.assembly.tape.stats()
        return (f"gap {self.gap:>7}  score {self.score:>8.3f}  "
                f"{st['out_len']:>8} out  {st['holes']:>6} holes  "
                f"{self.resolved:>6} bytes read across the gap")


def _cross_gap_bytes(assembly: Assembly) -> list[int]:
    """Output positions in the last piece that a cross-gap reference resolved.

    These are the bytes that only exist because a back-reference reached over
    the gap into the head fragment. They are the evidence: with the gap length
    wrong they are read from the wrong place and come out as noise.
    """
    if len(assembly.starts) < 2:
        return []
    boundary = assembly.starts[-1]
    tape = assembly.tape
    out = []
    for sym in assembly.trace.symbols:
        if sym.kind != "match" or sym.out_pos < boundary:
            continue
        source = sym.out_pos - sym.distance
        if source >= boundary:
            continue                    # reads within its own piece, not evidence
        for i in range(sym.length):
            p = sym.out_pos + i
            if p < len(tape) and tape.value_at(p) is not None:
                out.append(p)
    return out


def score_gap(assembly: Assembly, model=None) -> tuple[float, dict]:
    """How plausible the bytes read across the gap look."""
    tape = assembly.tape
    positions = _cross_gap_bytes(assembly)
    parts = {"resolved": float(len(positions))}
    if not positions:
        parts["model"] = 0.0
        parts["printable"] = 0.0
        return -1.0, parts

    values = [tape.value_at(p) for p in positions]
    printable = sum(1 for v in values
                    if v is not None and (32 <= v < 127 or v in (9, 10, 13)))
    parts["printable"] = printable / len(values)

    if model is None:
        parts["model"] = 0.0
        return parts["printable"] + min(len(positions), 400) / 4000.0, parts

    order = getattr(model, "order", 4)
    total = 0.0
    for p in positions:
        ctx = bytearray()
        for q in range(p - order, p):
            v = tape.value_at(q)
            if v is None:
                ctx.clear()
            else:
                ctx.append(v)
        total += model.logprob(tape.value_at(p), bytes(ctx))
    parts["model"] = total / len(positions)
    return parts["model"] + min(len(positions), 400) / 4000.0, parts


@dataclass
class _Prepared:
    """Everything about a two-piece assembly that does not depend on the gap.

    Decoding is by far the expensive part and the gap does not affect it: the
    same symbols come out whatever offset they end up at.  Only where the tail's
    references *land* changes, so a candidate can be scored by looking up a few
    hundred bytes rather than by decoding and replaying the whole thing.
    """
    head_tape: Tape
    head_len: int
    tail_tape: Tape                # the tail decoded on its own
    matches: list                  # (tail-relative out_pos, distance, length)
    litlen: object = None
    dist: object = None


def prepare(pieces: list[Piece], litlen=None, dist=None) -> _Prepared:
    head, tail = pieces[0], pieces[-1]
    if head.has_header or litlen is None:
        head_trace = Decoder(head.data, strict_window=False).stream(head.bit_offset)
        for block in head_trace.blocks:
            if block.litlen is not None:
                litlen, dist = block.litlen, block.dist
    else:
        head_trace = Decoder(head.data, strict_window=False).fragment(
            head.bit_offset, litlen, dist)

    if tail.has_header:
        tail_trace = Decoder(tail.data, strict_window=False).stream(tail.bit_offset)
    else:
        tail_trace = Decoder(tail.data, strict_window=False).fragment(
            tail.bit_offset, litlen, dist)

    matches = [(s.out_pos, s.distance, s.length)
               for s in tail_trace.symbols if s.kind == "match"]
    return _Prepared(head_trace.tape, len(head_trace.tape), tail_trace.tape,
                     matches, litlen, dist)


def fast_score(prep: _Prepared, gap: int, model=None) -> tuple[float, dict]:
    """Score a gap by whether the bytes it drags across read correctly in place.

    A match at tail position p with distance d reaches into the head exactly
    when d > gap + p, and then reads head position head_len - (d - gap - p).

    The byte has to be judged where it *lands*, against the tail's own text, not
    where it came from.  Scored in its original context inside the head, any
    real byte looks fine no matter which one the gap selected, and the score
    says nothing at all.  It is the seam in the tail that gives it away.
    """
    head, head_len, tail = prep.head_tape, prep.head_len, prep.tail_tape
    order = getattr(model, "order", 4) if model else 4

    dragged: dict[int, int] = {}
    for pos, distance, length in prep.matches:
        if distance <= gap + pos:
            continue                       # reads inside the tail, not evidence
        for i in range(length):
            back = distance - gap - pos - i
            if back <= 0:
                break                      # the rest of this match is tail-local
            src = head_len - back
            if src < 0:
                continue                   # before the head fragment: unknown
            v = head.value_at(src)
            if v is not None:
                dragged[pos + i] = v

    parts = {"resolved": float(len(dragged))}
    if not dragged:
        parts["printable"] = 0.0
        parts["model"] = 0.0
        return -99.0, parts

    def at(p):
        return dragged.get(p) if p in dragged else tail.value_at(p)

    values = list(dragged.values())
    printable = sum(1 for v in values if 32 <= v < 127 or v in (9, 10, 13))
    parts["printable"] = printable / len(values)
    if model is None:
        parts["model"] = 0.0
        return parts["printable"] + min(len(dragged), 400) / 4000.0, parts

    total, n = 0.0, 0
    for p in sorted(dragged)[:SCORE_CAP]:
        ctx = bytearray()
        for q in range(p - order, p):
            v = at(q)
            if v is None:
                ctx.clear()
            else:
                ctx.append(v)
        total += model.logprob(dragged[p], bytes(ctx))
        n += 1
        # and how well it predicts what the tail already knows follows it
        run = bytearray(ctx)
        run.append(dragged[p])
        for j in range(1, 3):
            nxt = at(p + j)
            if nxt is None:
                break
            total += model.logprob(nxt, bytes(run[-order:]))
            run.append(nxt)
            n += 1
    parts["model"] = total / max(n, 1)
    return parts["model"] + min(len(dragged), 400) / 4000.0, parts


# Neighbouring gaps drag mostly the same bytes into the same contexts: on the
# manual's two pieces 93% of the model's lookups repeat within 512 gaps. The
# cache is cleared that often, which holds it near 200,000 entries.
_MEMO_SPAN = 512


class _Memo:
    """A model's logprob, remembered. Same answers, fewer lookups."""

    def __init__(self, model):
        self.model = model
        self.order = getattr(model, "order", 4)
        self._seen: dict = {}

    def clear(self) -> None:
        self._seen.clear()

    def logprob(self, byte: int, ctx: bytes) -> float:
        key = (byte, ctx)
        got = self._seen.get(key)
        if got is None:
            got = self._seen[key] = self.model.logprob(byte, ctx)
        return got


def _score_chunk(prep, model, gaps) -> list:
    """Score a run of neighbouring gaps with one cache between them."""
    cached = _Memo(model) if model is not None else None
    return [(*fast_score(prep, gap, cached), gap) for gap in gaps]


def _workers(asked: int | None) -> int:
    """How many processes score gaps: the physical cores unless told. Past
    them it bought nothing: 5.2x at 8 workers and 5.2x at 15 on an 8-core,
    16-thread machine."""
    if asked is not None:
        return max(1, asked)
    try:
        logical = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        logical = os.cpu_count() or 1
    return max(1, logical // 2)


def _can_fork() -> bool:
    """Forking a process that runs other threads can leave a child holding
    a lock nobody will release, so a threaded caller scores in-process."""
    import multiprocessing
    import threading
    return (threading.active_count() == 1
            and "fork" in multiprocessing.get_all_start_methods())


_FORKED = None                  # (prep, model), inherited by forked workers


def _forked_chunk(gaps) -> list:
    prep, model = _FORKED
    return _score_chunk(prep, model, gaps)


def _score_forked(prep, model, chunks, n: int) -> list:
    """The gaps are independent, so the cores take chunks of them. Forked
    workers share the model's memory: 8 workers held 5.3 GB between them
    where one process holds 3.5 GB, and scored the same as one process."""
    import multiprocessing
    global _FORKED
    _FORKED = (prep, model)
    try:
        with multiprocessing.get_context("fork").Pool(n) as pool:
            return [x for part in pool.imap_unordered(_forked_chunk, chunks)
                    for x in part]
    finally:
        _FORKED = None


def search_gap(pieces: list[Piece], litlen=None, dist=None, model=None,
               max_gap: int = 65536, step: int = 1,
               known_size: int | None = None,
               keep: int = 12, workers: int | None = None) -> list[GapCandidate]:
    """Find the output gap length, best first.

    With a recorded decompressed length there is nothing to search: the answer
    is that number minus what the fragments produce. Otherwise every candidate
    is scored cheaply and only the best few are assembled in full.
    """
    if known_size is not None:
        exact = gap_from_size(pieces, known_size, litlen, dist)
        if exact is not None:
            asm = assemble(pieces, [exact], litlen, dist)
            score, parts = score_gap(asm, model)
            asm.notes.append(
                f"gap of {exact} bytes, from the recorded decompressed "
                f"size")
            return [GapCandidate(exact, score, asm, len(_cross_gap_bytes(asm)),
                                 parts)]

    prep = prepare(pieces, litlen, dist)
    gaps = range(0, max_gap + 1, step)
    chunks = [gaps[i:i + _MEMO_SPAN] for i in range(0, len(gaps), _MEMO_SPAN)]
    n = _workers(workers)
    if model is not None and n > 1 and len(chunks) > 1 and _can_fork():
        scored = _score_forked(prep, model, chunks, n)
    else:
        scored = [x for c in chunks for x in _score_chunk(prep, model, c)]
    scored.sort(key=lambda t: (-t[0], t[2]))

    out = []
    for score, parts, gap in scored[:keep]:
        asm = assemble(pieces, [gap], prep.litlen, prep.dist)
        out.append(GapCandidate(gap, score, asm, int(parts["resolved"]), parts))
    return out


# ---- a stream that arrived in packets ---------------------------------------

# The longest a single DEFLATE symbol can be: a 15-bit length code with 5 extra
# bits, then a 15-bit distance code with 13.
MAX_SYMBOL_BITS = 15 + 5 + 15 + 13
# Bytes a container may put after the last end-of-block: gzip's trailer.
TRAILER_MAX = 8
# How far into a resumed tail the readings are compared, in bytes.
CONVERGE_WITHIN = 4096


def agreed_boundaries(readings) -> list[int]:
    """Every bit at which all the live readings start a symbol, ascending.

    The same filter as `converged`: a reading that dies before half the
    furthest one's length is a wrong offset running into an invalid code and
    does not vote.
    """
    if not readings:
        return []
    furthest = max(r.bits_consumed for r in readings)
    live = [r for r in readings if r.symbols and r.bits_consumed * 2 >= furthest]
    if len(live) < 2:
        return []
    common = {s.bit_start for s in live[0].symbols}
    for r in live[1:]:
        common &= {s.bit_start for s in r.symbols}
    return sorted(common)


def converged(readings) -> int | None:
    """The first bit at which every reading of a resumed tail starts a symbol.

    A resumed tail can be read from any bit a symbol might start at, and DEFLATE
    self-synchronises: the readings disagree for a few symbols and then land on
    the same boundaries for good. Before that point, which bytes came out
    depends on a guess; after it, it does not. A reading that dies early is a
    wrong offset running into an invalid code and is not asked.
    """
    if not readings:
        return None
    furthest = max(r.bits_consumed for r in readings)
    live = [r for r in readings if r.symbols and r.bits_consumed * 2 >= furthest]
    if len(live) < 2:
        return None
    common = {s.bit_start for s in live[0].symbols}
    for r in live[1:]:
        common &= {s.bit_start for s in r.symbols}
    return min(common) if common else None


def from_holes(data: bytes, missing, header_skip: int = 0,
               trailer_skip: int = 0, total_size: int | None = None,
               model=None) -> Assembly:
    """Assemble a stream whose absent runs are already known.

    This is the capture case and it is the easy one, which is worth saying
    plainly: everywhere else a fragment that resumes after damage has to be
    found by scanning, and the width of what is gone has to be searched for. A
    pcap hands over both. TCP numbers every byte, so the resumption point is read
    rather than ranked and the hole's width is arithmetic on two sequence
    numbers.

    Two things it still does not know, and they are handled here:

    Where the surviving tail begins in BITS. A packet boundary is byte-aligned in
    the file and says nothing about which bit inside it resumes a Huffman symbol.
    DEFLATE turns out to rescue itself here: the code is self-synchronising, so a
    wrong offset costs a handful of symbols and then converges. All eight are
    tried and the one that decodes furthest is taken.

    How much OUTPUT the missing bytes carried. `total_size` -- a gzip ISIZE, a
    Content-Length -- makes that arithmetic. Without it the gap is unknown and
    the caller is told so rather than shown a number.
    """
    pieces: list[Piece] = []
    gaps: list[int] = []
    notes: list[str] = []
    body = data[header_skip:len(data) - trailer_skip if trailer_skip else None]
    # Positions are relative to the joined stream; shift them onto the body.
    holes = sorted((p - header_skip, n) for p, n in missing)
    holes = [(p, n) for p, n in holes if 0 <= p < len(body)]

    at = 0
    litlen = dist = None
    for index, (pos, width) in enumerate(holes):
        chunk = body[at:pos]
        if index == 0:
            pieces.append(Piece(chunk, 0, "the part before the loss", has_header=True))
            probe = Decoder(chunk, strict_window=False).stream(0)
            for block in probe.blocks:
                if block.litlen is not None:
                    litlen, dist = block.litlen, block.dist
        else:
            pieces.append(Piece(chunk, 0, f"between holes {index}"))
            gaps.append(0)
        # `pos` is where the two surviving sides MEET in the concatenation,
        # not an offset in a stream that still contains the hole.
        at = pos
        if width:
            notes.append(f"{width:,} bytes absent where the stream reaches "
                         f"{pos:,}, measured from the capture")

    rest = body[at:]
    if rest:
        # The tail resumes mid-symbol, and the symbols read before the eight
        # readings converge are not facts: on a phrack capture they were 5
        # wrong literals, copied on by back-references into 111 wrong "known"
        # bytes. So the tail starts at the first symbol
        # boundary every reading agrees on, and whatever came before it is
        # absorbed into the gap -- the recorded size still measures the gap
        # exactly, it just measures a slightly wider one.
        # Every start in the first 48 bits, not just the first byte's eight: a
        # symbol cut by the packet boundary can run that far into the tail
        # (15-bit length code, 5 extra, 15-bit distance code, 13 extra), so the
        # true reading may begin anywhere in it. Eight readings agreeing with
        # each other while all missing the true one put 334 wrong bytes on a
        # phrack capture.
        # Read only a prefix: the readings agree within a few hundred bits when
        # they agree at all, and decoding 48 whole tails cost seconds a capture.
        head = rest[:CONVERGE_WITHIN]
        readings = []
        for bit in range(MAX_SYMBOL_BITS):
            try:
                got = Decoder(head, strict_window=False).fragment(bit, litlen,
                                                                  dist)
            except Exception:
                continue
            readings.append(got)
        agreed = converged(readings)
        if agreed is not None:
            skip = agreed // 8
            rest = rest[skip:]
            best = agreed - skip * 8
            notes.append(f"the {len(readings)} ways of reading the first bits "
                         f"after the loss agree from bit {agreed}; what they "
                         f"disagree on before it is counted as missing")
        else:
            best = max(range(len(readings)),
                       key=lambda i: readings[i].bits_consumed,
                       default=0)
            best = readings[best].bit_start if readings else 0
            notes.append("the ways of reading the first bits after the loss "
                         "never agree, so the first few bytes after it may "
                         "be misread")
        frag = Decoder(rest, strict_window=False).fragment(best, litlen, dist)
        # Where it stopped, not how far it went: the two differ by the start
        # bit, which was nearly always 0 until the start became the agreed one.
        cut = frag.bit_end
        # An end-of-block with only a trailer's worth of bytes after it is
        # the stream's last: a gzip trailer read as a block header made a
        # third "piece" of garbage and hid the trailer itself.
        if frag.status == "eob" and len(rest) - (cut + 7) // 8 > TRAILER_MAX \
                and not _header_holds(rest, cut, total_size is not None):
            # An end-of-block, but what follows it is no block. Read with the
            # wrong table -- the real boundary went down with the packet -- the
            # tail produced an end-of-block symbol by chance, and the "header"
            # after it was more of the same garbage: on one capture 9,416 bytes
            # of it were kept as fact, the next block read `bad-table`
            # (wrap-up #5, the Holmes payload, segment 53). What is kept is
            # only what a real block header further on can vouch for.
            _resync_or_drop(body[at:], best, total_size is not None, pieces, gaps,
                            notes, "reaches an end-of-block with the table "
                            "from before it, but no block decodes after it")
        elif frag.status == "eob" and len(rest) - (cut + 7) // 8 > TRAILER_MAX:
            # It stopped on an end-of-block, so what follows carries its own
            # header and can be read properly rather than guessed at.
            cut_byte, cut_off = cut // 8, cut % 8
            pieces.append(Piece(rest[:cut_byte + 1], best, "the part after the loss"))
            pieces.append(Piece(rest[cut_byte:], cut_off, "the next block",
                                has_header=True))
            gaps.extend([0, 0])
            notes.append(f"the tail resumes mid-block at bit {best}; a block "
                         f"boundary {cut:,} bits in restores a header")
        elif total_size is not None and frag.status != "eob":
            # The trailer survived, so the stream is complete and its last
            # block ends on an end-of-block. A tail that runs out without one
            # was read with the wrong table: a block boundary went down with
            # the packet, and its header with it. Everything it produced would
            # be garbage drawn in the colour of fact -- 27,188 bytes of it on
            # one capture -- so none of it is kept. The
            # recorded size still places the unknown run exactly.
            _resync_or_drop(body[at:], best, True, pieces, gaps, notes,
                            "does not decode to a clean end with the table "
                            "from before it")
        else:
            pieces.append(Piece(rest, best, "the part after the loss"))
            gaps.append(0)

    if len(pieces) < 2:
        return assemble(pieces or [Piece(body, 0, "all of it", True)], [],
                        litlen, dist)

    # The first gap is the only unknown. Everything else is a real join.
    zero = assemble(pieces, [0] * (len(pieces) - 1), litlen, dist)
    if total_size is not None and total_size < len(zero.tape):
        # Smaller than what is already decoded, so it is wrong: a mistyped
        # --total-size, or a damaged trailer. Clamped to a 0-byte gap, it
        # joined the pieces with no gap and placed every byte after it at
        # the wrong position as known (wrap-up #2).
        notes.append(f"the recorded size, {total_size:,} bytes, is less than "
                     f"the {len(zero.tape):,} the pieces already decode to, "
                     f"so it is wrong and is not used")
        total_size = None
    if total_size is not None:
        first = max(0, total_size - len(zero.tape))
        notes.append(f"output gap of {first:,} bytes, from the recorded "
                     f"decompressed size")
    else:
        first = 0
        notes.append("no recorded decompressed size, so the output gap is "
                     "unknown and left at zero: what follows it is positioned "
                     "wrongly and should be read as a separate run")
    out_gaps = [first] + [0] * (len(pieces) - 2)
    asm = assemble(pieces, out_gaps, litlen, dist)
    asm.notes[:0] = notes
    return asm


def _resync_or_drop(rest, start, has_trailer, pieces, gaps, notes, why):
    """A tail read with the wrong table: keep only what a real header vouches for.

    A block boundary went down with the packet, so the tail's first block has
    no table. Its bytes are not kept. The first later boundary whose block
    decodes cleanly to the end (`_header_holds`) starts a piece with its own
    header; everything before it joins the unknown gap, which the recorded
    size still places exactly. With no such boundary nothing of the tail is
    kept -- the earlier answer to every case, which threw away blocks that
    carry their own tables (capture-tail.bin: a fresh block every 1 KB).
    """
    from .scan import KIND_DYNAMIC, KIND_STORED, KIND_SYNC, scan

    def first_holding(kinds, limit=None):
        # From bit 0, not from where the readings agreed: a tail cut at a
        # block boundary starts with a real header (capture-tail.bin), and
        # `_header_holds` keeps a false one out wherever it is.
        try:
            points = scan(rest, kinds=kinds, start_bit=0, max_bit=limit)
        except Exception:
            return None
        for sp in sorted(points, key=lambda p: p.bit_offset):
            if _header_holds(rest, sp.bit_offset, has_trailer):
                return sp.bit_offset
        return None

    # The earliest boundary of any kind. Flush markers and stored headers are
    # cheap to find; dynamic headers are searched only up to the first of
    # those, since a later one could not be the earliest.
    found = first_holding((KIND_SYNC, KIND_STORED))
    early = first_holding((KIND_DYNAMIC,), found)
    if early is not None:
        found = early
    if found is None:
        pieces.append(Piece(b"", 0, "the part after the loss"))
        gaps.append(0)
        notes.append(f"the tail after the loss {why}, so a block header was "
                     f"lost with the packet; the tail's table is gone, no "
                     f"later block header decodes, and its bytes stay "
                     f"unknown. f7 lists the routes to a lost table")
        return
    pieces.append(Piece(rest[found // 8:], found % 8, "the next block",
                        has_header=True))
    gaps.append(0)
    if found == 0:
        notes.append("the tail after the loss starts with a block header of "
                     "its own, which decodes cleanly to the end; it is read "
                     "from there")
    else:
        notes.append(f"the tail after the loss {why}, so a block header was "
                     f"lost with the packet; its first {found:,} bits stay "
                     f"unknown, and a block header {found:,} bits in decodes "
                     f"cleanly to the end and restores everything after it")


def _header_holds(rest: bytes, cut: int, has_trailer: bool) -> bool:
    """Does a real block start where the tail's end-of-block left off?

    It must decode from its own header with no bad table or symbol, and reach
    the stream's end with at most a trailer left over -- or, with no trailer
    to say the stream is whole, run out of data cleanly. A chance end-of-block
    in a tail read with the wrong table is followed by garbage, and garbage
    does not do that.
    """
    try:
        t = Decoder(rest[cut // 8:], strict_window=False).stream(cut % 8)
    except Exception:
        return False
    if t.status == "end":
        return len(rest) - cut // 8 - (t.bit_end + 7) // 8 <= TRAILER_MAX
    return not has_trailer and t.status == "truncated"


# ---- two pieces carved from one stream --------------------------------------

def tail_trailer(asm) -> tuple[int, int] | None:
    """(CRC-32, ISIZE) from the 8 bytes after the tail's final block, when the
    tail ends on one with exactly a gzip trailer's worth of bytes left."""
    tr = asm.trace
    if tr.status not in ("end", "eob"):
        return None
    if not tr.blocks:
        return None
    # The assembled trace ends where the joined data does; the last block's
    # own end is where the stream does.
    left = len(tr.data) - (tr.blocks[-1].bit_end + 7) // 8
    if left != 8:
        return None
    return struct.unpack("<II", tr.data[-8:])


def _mark_cross_gap(asm: Assembly) -> None:
    """Rebuild the tape so every copy reaching back across the gap is GAPPED.

    With a searched gap, which bytes those copies read rests on the search:
    a wrong width reads the wrong ones, in the colour of fact. Copies of them
    inherit it through the tape's unproven rule.
    """
    from .tape import GAPPED
    start = asm.starts[1]
    tape = Tape()
    gap = asm.gaps[0]
    syms = asm.trace.symbols
    i = 0
    while i < len(syms) and syms[i].out_pos < start - gap:
        i += 1
    replay(syms[:i], tape)
    if gap:
        tape.emit_unknown(gap)
    for s in syms[i:]:
        tape.taint = (GAPPED if s.kind == "match"
                      and s.out_pos - s.distance < start else None)
        replay([s], tape)
    tape.taint = None
    asm.trace.tape = tape


def two_pieces(head: bytes, tail: bytes, total_size: int | None = None,
               model=None, max_gap: int = 8192,
               workers: int | None = None) -> Assembly:
    """The start of a stream and a later piece of it, with the middle gone.

    The tail is read the way a packet capture's is after a lost segment
    (from_holes): it resumes mid-symbol, so every reading of its first 48
    bits is tried and it starts where they all agree. With the stream's
    decompressed size the gap is arithmetic. Without it the gap is searched
    for, scored by whether the tail's references across it read like text,
    and only a candidate.
    """
    joined = head + tail
    asm = from_holes(joined, [(len(head), 0)], total_size=total_size)
    if total_size is not None:
        return asm
    if len(asm.pieces) != 2:
        _mark_cross_gap(asm)        # the gap is unknown and left at zero
        return asm
    found = search_gap(asm.pieces, asm.litlen, asm.dist, model=model,
                       max_gap=max_gap, workers=workers)
    if not found:
        _mark_cross_gap(asm)
        return asm
    best = found[0]
    _mark_cross_gap(best.assembly)
    best.assembly.notes[:0] = [n for n in asm.notes
                               if "output gap" not in n]
    best.assembly.notes.append(
        f"gap of {best.gap:,} bytes, the best of {max_gap + 1:,} tried "
        f"({best.resolved} bytes read across it); a search result, which "
        f"nothing here proves")
    # How far the best stands above a typical gap. On the Phrack pieces the
    # true gap scored -1.69 against a median of -2.84; with the truth outside
    # the range searched, the best scored -2.64 -- barely off the median, and
    # wrong. Shown, so the second case can be told from the first.
    import random
    prep = prepare(asm.pieces, asm.litlen, asm.dist)
    rng = random.Random(0)
    sample = sorted(fast_score(prep, gap, model)[0]
                    for gap in rng.sample(range(max_gap + 1),
                                          min(200, max_gap + 1)))
    median = sample[len(sample) // 2]
    best.assembly.notes.append(
        f"it scores {best.score:.2f}, against a median of {median:.2f} over "
        f"{len(sample)} gaps sampled"
        + ("" if best.score - median >= SPREAD else
           f"; that barely stands out, so the gap may be past the "
           f"{max_gap:,} searched (--gap-max)"))
    return best.assembly
