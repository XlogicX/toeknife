"""What a block's tables declare, against what the block did with them."""
import zlib
from pathlib import Path

import pytest

from toeknife.core import tablestats as ts
from toeknife.core.events import BlockInfo, Trace
from toeknife.core.huffman import HuffmanTable
from toeknife.core.inflate import Decoder
from toeknife.core.session import Session
from toeknife.core.tables import BTYPE_STORED

CAPTURE = Path("samples/damaged/capture.gz")


def real_block():
    s = Session(CAPTURE.read_bytes())
    return s.trace, s.trace.blocks[0]


def bare(litlen=None, dist=None, codelen=None, symbols=(), clen_ops=()):
    """A block with the tables under test and nothing else, so a claim about the
    arithmetic is not also a claim about a sample file."""
    b = BlockInfo(index=0, bit_start=0, btype=2, btype_name="dynamic")
    b.litlen, b.dist, b.codelen = litlen, dist, codelen
    b.clen_ops = list(clen_ops)
    return Trace(b"", 0, 0, [b], list(symbols), None, "ok"), b


# ---- the declared side, which is arithmetic on the header --------------------

def test_a_declared_share_is_two_to_the_minus_code_length():
    """The whole reading rests on this: canonical Huffman spends a 3-bit code on
    something it expects one time in eight."""
    trace, block = bare(litlen=HuffmanTable((1, 2, 3, 3)))
    a = ts.alphabets(trace, block)[0]
    assert [(s.sym, s.bits, s.declared) for s in a.stats] == [
        (0, 1, 0.5), (1, 2, 0.25), (2, 3, 0.125), (3, 3, 0.125)]
    assert a.kraft == pytest.approx(1.0) and a.complete


def test_an_incomplete_table_is_reported_as_one():
    """Legal, and worth saying out loud: bit patterns that decode to nothing are
    how a truncated or mis-read table usually announces itself."""
    trace, block = bare(litlen=HuffmanTable((1, 0, 0)))
    a = ts.alphabets(trace, block)[0]
    assert a.kraft == pytest.approx(0.5)
    assert not a.complete
    # Shares are raw, not normalised to 100%: normalising would hide exactly the
    # thing the Kraft number is there to show.
    assert sum(s.declared for s in a.stats) == pytest.approx(0.5)


def test_the_code_space_summary_adds_up_to_the_kraft_sum():
    trace, block = real_block()
    for a in ts.alphabets(trace, block):
        assert sum(share for _, _, share in a.per_length) == \
            pytest.approx(a.kraft)
        assert sum(n for _, n, _ in a.per_length) == a.used


def test_heat_bands_are_absolute_so_two_blocks_can_be_compared():
    assert ts.heat_band(1) == ts.heat_band(4) == 0
    assert ts.heat_band(5) == ts.heat_band(6) == 1
    assert ts.heat_band(9) == 3
    assert ts.heat_band(13) == ts.heat_band(15) == 4
    assert ts.heat_band(0) == -1, "a symbol with no code is not cold, it is absent"


# ---- the measured side, which is counted off the wire -----------------------

def test_measured_counts_are_read_from_the_codes_not_inferred_from_values():
    trace, block = real_block()
    lit, dist, clen = ts.alphabets(trace, block)
    # Every symbol the decoder read is counted exactly once.
    assert sum(s.count for s in lit.stats) == lit.emitted
    assert lit.emitted == block.n_literals + block.n_matches + 1  # + end of block
    assert sum(s.count for s in dist.stats) == dist.emitted
    assert dist.emitted == block.n_matches
    assert clen.emitted == len(block.clen_ops)


def test_length_258_is_counted_under_the_code_that_carried_it():
    """258 is reachable as symbol 285 or as 284 with its extra bits maxed, so
    counting by length instead of by code would file it under whichever the
    reverse lookup happened to prefer."""
    data = b"x" * 4000                      # long runs, so 258-byte matches
    trace = Decoder(zlib.compress(data, 9)[2:-4]).stream()
    block = trace.blocks[0]
    lit = ts.alphabets(trace, block)[0]
    longest = [s for s in trace.symbols if s.length == 258]
    assert longest, "expected at least one maximum-length match"
    carried = {s.code_sym for s in longest}
    assert carried == {285}, f"this encoder used {carried}"
    by_sym = {s.sym: s.count for s in lit.stats}
    assert by_sym[285] == len(longest)
    assert by_sym.get(284, 0) == 0


def test_the_code_length_code_is_counted_from_the_header_ops():
    """Its symbols are emitted in the header, not the body, and the op record
    keeps the run length rather than the symbol -- 17 writes 3 to 10 zeros and 18
    writes 11 to 138, which do not overlap."""
    from toeknife.core.events import CodeLengthOp
    ops = [CodeLengthOp("lens", 5, 1, 0, 0, 0),
           CodeLengthOp("lens", 5, 1, 1, 0, 0),
           CodeLengthOp("repeat", 5, 4, 2, 0, 0),
           CodeLengthOp("zeros", 0, 7, 6, 0, 0),      # symbol 17
           CodeLengthOp("zeros", 0, 60, 13, 0, 0)]    # symbol 18
    lengths = [0] * 19
    for sym in (5, 16, 17, 18):
        lengths[sym] = 2
    trace, block = bare(codelen=HuffmanTable(tuple(lengths)), clen_ops=ops)
    a = ts.alphabets(trace, block)[0]
    counts = {s.sym: s.count for s in a.stats}
    assert counts[5] == 2 and counts[16] == 1
    assert counts[17] == 1 and counts[18] == 1
    assert a.emitted == 5


def test_a_truncated_block_says_its_counts_are_a_floor():
    s = Session(Path("samples/body-fragment.bin").read_bytes())
    block = s.trace.blocks[0]
    assert not any(sym.kind == "eob" for sym in s.trace.symbols)
    for a in ts.alphabets(s.trace, block):
        assert a.partial, "no end-of-block was reached, so nothing is final"


def test_nothing_decoded_leaves_the_measured_side_empty_rather_than_guessed():
    trace, block = bare(litlen=HuffmanTable((1, 1)))
    a = ts.alphabets(trace, block)[0]
    assert a.emitted == 0
    assert all(s.count == 0 and s.measured == 0.0 for s in a.stats)


# ---- what it refuses to say --------------------------------------------------

def test_a_stored_block_has_no_tables_to_show():
    data = bytes(range(256)) * 12
    trace = Decoder(zlib.compress(data, 0)[2:-4]).stream()
    stored = next(b for b in trace.blocks if b.btype == BTYPE_STORED)
    assert ts.alphabets(trace, stored) == []


def test_a_recovered_code_set_gets_no_byte_grid_and_no_byte_names():
    """The rule this tool runs on. A code set fixes which code words fall to
    literals without saying which byte any of them means, so drawing a byte map
    or printing "0x41 'A'" would manufacture a meaning the bitstream never
    carried."""
    from toeknife.core.huffman import fixed_litlen, structural_tables
    from toeknife.core.inflate import carve
    from toeknife.core.tables import FIXED_DIST_LENGTHS

    fx = fixed_litlen().lengths
    counts = [0] * 16
    for v in range(256):
        counts[fx[v]] += 1
    litlen, dist, ident = structural_tables(
        counts, fx[256], [fx[257 + i] for i in range(29)],
        list(FIXED_DIST_LENGTHS))
    trace = carve(Path("samples/fixed-fragment.bin").read_bytes(), 0,
                  litlen, dist, identity=ident)
    assert trace.identity is not None, "the trace has to carry this fact"

    a = ts.alphabets(trace, trace.blocks[0])[0]
    assert not a.byte_space, "there is no byte space to draw"
    assert "unrecovered" in a.name and "not known" in a.note
    literals = [s for s in a.stats if s.sym < 256]
    assert literals
    for s in literals:
        assert "unassigned" in s.name
        assert "0x" not in s.name, f"named a byte it does not know: {s.name}"
    # The codes themselves are entirely real and still measured.
    assert all(s.bits for s in a.stats)


def test_an_ordinary_trace_still_names_its_bytes():
    trace, block = real_block()
    a = ts.alphabets(trace, block)[0]
    assert a.byte_space
    named = {s.name for s in a.stats if s.sym < 256}
    assert "0x65 'e'" in named
    assert "0x20 ' '" in named, "a space must not print as blank"
    assert "0x0a '\\n'" in named


def test_symbols_above_the_length_codes_are_called_invalid_not_named():
    """The fixed table assigns codes to 286 and 287, which have no length base:
    they cannot legally appear, and seeing one means the decode went wrong."""
    assert "invalid" in ts.litlen_name(286)
    assert "invalid" in ts.dist_name(30)
