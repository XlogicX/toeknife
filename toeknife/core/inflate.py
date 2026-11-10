"""DEFLATE decoder that keeps its work.

Unlike a normal inflate, this one is built to survive bad input and to record
where every decoded thing came from.  Two entry points:

  Decoder.stream()    -- a well-formed stream, headers and all ("super infgen")
  Decoder.fragment()  -- headerless bits starting mid-block at an arbitrary bit
                         offset, which is the forensic case

Both produce the same Trace, so the UI does not care which one ran.
"""
from __future__ import annotations

from .bitreader import BitEOF, BitReader
from .events import BitField, BlockInfo, CodeLengthOp, Symbol, Trace
from .huffman import HuffmanError, HuffmanTable, fixed_tables
from .tables import (
    BTYPE_DYNAMIC, BTYPE_FIXED, BTYPE_NAMES, BTYPE_RESERVED, BTYPE_STORED,
    CODE_LENGTH_ORDER, DISTANCE_BASE, DISTANCE_EXTRA, END_OF_BLOCK,
    LENGTH_BASE, LENGTH_EXTRA,
)
from .tape import Tape, WindowError


class Decoder:
    def __init__(self, data: bytes, strict_window: bool = True):
        self.data = data
        # In a complete stream a distance may not reach behind the start of the
        # output.  In a fragment it always does, and that is the whole point.
        self.strict_window = strict_window
        # Set only by fragment(), for a code set recovered without its meanings.
        # A stream with an intact header always knows what its literals are.
        self._identity: dict[int, tuple[int, int]] | None = None

    # ---- public entry points ----------------------------------------------

    def stream(self, bit_pos: int = 0, max_symbols: int | None = None) -> Trace:
        br = BitReader(self.data, bit_pos)
        tape = Tape()
        blocks: list[BlockInfo] = []
        symbols: list[Symbol] = []
        status, error = "end", None

        try:
            while True:
                block = BlockInfo(index=len(blocks), bit_start=br.pos,
                                  out_start=tape.length)
                blocks.append(block)
                p = br.pos
                bfinal = br.read(1)
                block.fields.append(BitField(
                    "BFINAL", p, br.pos,
                    "last block" if bfinal else "more blocks follow"))
                p = br.pos
                btype = br.read(2)
                block.bfinal, block.btype = bfinal, btype
                block.btype_name = BTYPE_NAMES[btype]
                block.fields.append(BitField("BTYPE", p, br.pos, block.btype_name))

                if btype == BTYPE_RESERVED:
                    block.status = status = "reserved-btype"
                    break
                if btype == BTYPE_STORED:
                    st = self._stored(br, block, tape, symbols)
                else:
                    if btype == BTYPE_FIXED:
                        litlen, dist = fixed_tables()
                    else:
                        try:
                            litlen, dist = self._dynamic_tables(br, block)
                        except (HuffmanError, ValueError) as e:
                            block.status = status = "bad-table"
                            error = str(e)
                            break
                    block.litlen, block.dist = litlen, dist
                    st = self._symbols(br, litlen, dist, tape, symbols, block,
                                       max_symbols, len(symbols))
                block.status = st
                block.bit_end = br.pos
                block.out_end = tape.length
                if st != "eob":
                    status = st
                    break
                if bfinal:
                    status = "end"
                    break
        except BitEOF:
            status = "truncated"
        except WindowError as e:
            status, error = "bad-distance", str(e)

        if blocks and not blocks[-1].bit_end:
            blocks[-1].bit_end = br.pos
            blocks[-1].out_end = tape.length
        return Trace(self.data, bit_pos, br.pos, blocks, symbols, tape,
                     status, error, headerless=False)

    def fragment(self, bit_pos: int, litlen: HuffmanTable | None = None,
                 dist: HuffmanTable | None = None,
                 max_symbols: int | None = None,
                 identity: dict[int, tuple[int, int]] | None = None) -> Trace:
        """Decode headerless symbols starting at an arbitrary bit offset.

        `identity` switches on the case where the code set is known and the
        meanings are not: pass the map `huffman.structural_tables` returns and
        every literal becomes an unknown byte tied to its code rather than a
        byte value, so the tape can carry the equality and the assignment can be
        read off later.
        """
        if litlen is None or dist is None:
            litlen, dist = fixed_tables()
        self._identity = identity
        br = BitReader(self.data, bit_pos)
        tape = Tape()
        block = BlockInfo(index=0, bit_start=bit_pos, btype=-1,
                          btype_name="fragment", litlen=litlen, dist=dist)
        symbols: list[Symbol] = []
        status, error = "eob", None
        try:
            status = self._symbols(br, litlen, dist, tape, symbols, block,
                                   max_symbols, 0)
        except BitEOF:
            status = "truncated"
        except WindowError as e:
            status, error = "bad-distance", str(e)
        block.bit_end = br.pos
        block.out_end = tape.length
        block.status = status
        return Trace(self.data, bit_pos, br.pos, [block], symbols, tape,
                     status, error, headerless=True, identity=identity)

    # ---- block bodies ------------------------------------------------------

    def _stored(self, br: BitReader, block: BlockInfo, tape: Tape,
                symbols: list[Symbol]) -> str:
        pad = br.align_to_byte()
        if pad:
            block.fields.append(BitField("pad", br.pos - pad, br.pos,
                                         f"{pad} bits to byte boundary"))
        p = br.pos
        ln = br.read(16)
        block.fields.append(BitField("LEN", p, br.pos, f"{ln} bytes"))
        p = br.pos
        nln = br.read(16)
        ok = ln == (~nln) & 0xFFFF
        block.fields.append(BitField("NLEN", p, br.pos,
                                     "ok" if ok else f"MISMATCH (~LEN={(~ln) & 0xFFFF})"))
        if not ok:
            return "bad-stored-len"
        block.stored_len = ln
        raw = br.read_bytes(ln)
        start_bit = br.pos - ln * 8
        for i, b in enumerate(raw):
            pos = tape.emit_literal(b)
            symbols.append(Symbol("literal", pos, start_bit + i * 8,
                                  start_bit + (i + 1) * 8, block.index, literal=b))
        block.n_literals += ln
        return "eob"

    def _dynamic_tables(self, br: BitReader,
                        block: BlockInfo) -> tuple[HuffmanTable, HuffmanTable]:
        f = block.fields
        p = br.pos
        hlit = br.read(5) + 257
        f.append(BitField("HLIT", p, br.pos, f"{hlit - 257} → {hlit} lit/len codes"))
        p = br.pos
        hdist = br.read(5) + 1
        f.append(BitField("HDIST", p, br.pos, f"{hdist - 1} → {hdist} distance codes"))
        p = br.pos
        hclen = br.read(4) + 4
        f.append(BitField("HCLEN", p, br.pos, f"{hclen - 4} → {hclen} code-length codes"))
        block.hlit, block.hdist, block.hclen = hlit, hdist, hclen

        # RFC 1951 3.2.7 allows 257..286 literal/length codes and 1..30 distance
        # codes, but the fields are five bits wide and so can encode 288 and 32.
        # zlib rejects the excess ("too many length or distance symbols") and so
        # must we: besides being wrong to accept, these are a meaningful share of
        # the false positives when scanning for block headers in damaged data.
        if hlit > 286:
            raise ValueError(f"HLIT {hlit} exceeds the 286 the format allows")
        if hdist > 30:
            raise ValueError(f"HDIST {hdist} exceeds the 30 the format allows")

        cl = [0] * 19
        for i in range(hclen):
            p = br.pos
            v = br.read(3)
            sym = CODE_LENGTH_ORDER[i]
            cl[sym] = v
            f.append(BitField(f"CL[{sym}]", p, br.pos, f"{v} bits"))
        codelen = HuffmanTable(tuple(cl))
        block.codelen = codelen

        # RFC 1951 3.2.7: the literal/length and distance code lengths are one
        # concatenated sequence.  A repeat code may straddle the boundary, so
        # they cannot be decoded as two independent runs.
        total = hlit + hdist
        lengths = [0] * total
        i = 0
        while i < total:
            p = br.pos
            sym, _, _ = codelen.decode(br)
            if sym < 16:
                lengths[i] = sym
                block.clen_ops.append(CodeLengthOp("lens", sym, 1, i, p, br.pos))
                i += 1
                continue
            if sym == 16:
                if i == 0:
                    raise ValueError("repeat code 16 with nothing to repeat")
                n = br.read(2) + 3
                val = lengths[i - 1]
                op = "repeat"
            elif sym == 17:
                n = br.read(3) + 3
                val = 0
                op = "zeros"
            else:
                n = br.read(7) + 11
                val = 0
                op = "zeros"
            if i + n > total:
                raise ValueError(f"code-length run overruns table ({i}+{n} > {total})")
            for _ in range(n):
                lengths[i] = val
                i += 1
            block.clen_ops.append(CodeLengthOp(op, val, n, i - n, p, br.pos))

        litlen = HuffmanTable(tuple(lengths[:hlit]))
        dist = HuffmanTable(tuple(lengths[hlit:]))
        return litlen, dist

    def _symbols(self, br: BitReader, litlen: HuffmanTable, dist: HuffmanTable,
                 tape: Tape, symbols: list[Symbol], block: BlockInfo,
                 max_symbols: int | None, already: int) -> str:
        strict = self.strict_window
        while True:
            if max_symbols is not None and len(symbols) - already >= max_symbols:
                return "limit"
            start = br.pos
            try:
                sym, nbits, code = litlen.decode(br)
            except HuffmanError:
                return "bad-code"

            if sym < 256:
                ident = self._identity
                if ident is None:
                    pos = tape.emit_literal(sym)
                    desc = f"sym {sym} = {_ch(sym)}  code {_fmt(code, nbits)}"
                    lit = sym
                else:
                    # The code set was recovered without its assignment, so the
                    # symbol id is a placeholder and reporting it as a byte
                    # would manufacture a meaning the bitstream never gave.
                    name = ident[sym]
                    pos = tape.emit_literal_unknown(name)
                    desc = (f"literal code {name[0]}/{name[1]} unassigned  "
                            f"code {_fmt(code, nbits)}")
                    lit = None
                symbols.append(Symbol(
                    "literal", pos, start, br.pos, block.index, literal=lit,
                    fields=(BitField("litlen code", start, br.pos, desc),),
                    code_sym=sym))
                block.n_literals += 1
                continue

            if sym == END_OF_BLOCK:
                symbols.append(Symbol(
                    "eob", tape.length, start, br.pos, block.index,
                    fields=(BitField("litlen code", start, br.pos,
                                     f"sym 256 end-of-block  code {_fmt(code, nbits)}"),),
                    code_sym=END_OF_BLOCK))
                return "eob"

            if sym > 285:
                return "bad-litlen"

            li = sym - 257
            fields = [BitField("litlen code", start, br.pos,
                               f"sym {sym} length base {LENGTH_BASE[li]}  "
                               f"code {_fmt(code, nbits)}")]
            nx = LENGTH_EXTRA[li]
            p = br.pos
            extra = br.read(nx)
            length = LENGTH_BASE[li] + extra
            if nx:
                fields.append(BitField("length extra", p, br.pos,
                                       f"+{extra} → length {length}"))

            p = br.pos
            try:
                dsym, dnbits, dcode = dist.decode(br)
            except HuffmanError:
                return "bad-code"
            if dsym > 29:
                return "bad-dist"
            fields.append(BitField("dist code", p, br.pos,
                                   f"sym {dsym} dist base {DISTANCE_BASE[dsym]}  "
                                   f"code {_fmt(dcode, dnbits)}"))
            nxd = DISTANCE_EXTRA[dsym]
            p = br.pos
            dextra = br.read(nxd)
            distance = DISTANCE_BASE[dsym] + dextra
            if nxd:
                fields.append(BitField("dist extra", p, br.pos,
                                       f"+{dextra} → distance {distance}"))

            if strict and distance > tape.length:
                return "bad-distance"

            pos = tape.emit_copy(distance, length)
            symbols.append(Symbol("match", pos, start, br.pos, block.index,
                                  length=length, distance=distance,
                                  fields=tuple(fields),
                                  code_sym=sym, dist_sym=dsym))
            block.n_matches += 1


def _ch(v: int) -> str:
    return repr(chr(v)) if 0x20 <= v < 0x7F else f"0x{v:02x}"


def _fmt(code: int, nbits: int) -> str:
    return format(code, f"0{nbits}b")


def inflate(data: bytes, bit_pos: int = 0, **kw) -> Trace:
    return Decoder(data).stream(bit_pos, **kw)


def carve(data: bytes, bit_pos: int, litlen=None, dist=None, **kw) -> Trace:
    return Decoder(data, strict_window=False).fragment(bit_pos, litlen, dist, **kw)
