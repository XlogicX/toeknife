"""Two pieces of one gzip file, the middle gone."""
import zlib

import pytest

from toeknife.containers import parse
from toeknife.core import gap

from tests.test_resume import text


@pytest.fixture(scope="module")
def pieces():
    raw = text()
    co = zlib.compressobj(6, zlib.DEFLATED, 16 + 15)
    whole = co.compress(raw) + co.flush()
    cut1, cut2 = len(whole) * 2 // 5, len(whole) * 3 // 5
    return raw, whole[:cut1], whole[cut2:]


def test_a_cut_short_gzip_does_not_trust_its_last_8_bytes(pieces):
    raw, head, tail = pieces
    st = parse(head).streams[0]
    assert st.uncompressed_size is None and st.crc32 is None
    assert any("cut short" in n for n in st.notes)


def test_the_gap_refuses_a_size_it_cannot_hold():
    with pytest.raises(ValueError, match="probably wrong"):
        gap.assemble([gap.Piece(b"\x03\x00", 0, "a", True),
                      gap.Piece(b"\x03\x00", 0, "b", True)],
                     [gap.MAX_GAP + 1])


def test_the_tail_carries_the_trailer(pieces):
    raw, head, tail = pieces
    body = parse(head).streams[0].slice(head)
    zero = gap.from_holes(body + tail, [(len(body), 0)])
    # The trailer is never read as a block of its own.
    assert all(len(p.data) > 8 for p in zero.pieces[1:])
    crc, isize = gap.tail_trailer(zero)
    assert isize == len(raw) and crc == zlib.crc32(raw) & 0xFFFFFFFF


def test_with_the_size_the_gap_is_exact_and_nothing_known_is_wrong(pieces):
    raw, head, tail = pieces
    body = parse(head).streams[0].slice(head)
    asm = gap.two_pieces(body, tail, total_size=len(raw))
    tape = asm.tape
    assert len(tape) == len(raw)
    assert not [p for p in range(len(tape)) if tape.value_at(p) is not None
                and tape.value_at(p) != raw[p]]


def test_the_command_line_finds_the_trailer_and_a_copy_proves_it(
        pieces, tmp_path, capsys):
    from toeknife.cli import main
    raw, head, tail = pieces
    (tmp_path / "a.gz").write_bytes(head)
    (tmp_path / "b.bin").write_bytes(tail)
    (tmp_path / "copy.txt").write_bytes(raw)
    code = main([str(tmp_path / "a.gz"), "--gap", str(tmp_path / "b.bin"),
                 "--fill-from", str(tmp_path / "copy.txt"), "--verify"])
    out = capsys.readouterr()
    assert "a gzip trailer" in out.err
    assert code == 0 and "verified" in out.out


def test_without_a_size_cross_gap_copies_are_unproven(pieces):
    """Which bytes they copy rests on a gap nobody measured; drawn as facts
    they would be wrong wherever the gap is."""
    from toeknife.core.tape import COPY, GAPPED, LITERAL
    raw, head, tail = pieces
    body = parse(head).streams[0].slice(head)
    asm = gap.two_pieces(body, tail[:-8], None, max_gap=2000)
    tape = asm.tape
    start = asm.starts[1]
    assert any(tape.prov_at(p) == GAPPED for p in range(start, len(tape)))
    off = len(raw) - (len(tape) - start)
    assert not [p for p in range(start, len(tape))
                if tape.prov_at(p) in (LITERAL, COPY)
                and tape.value_at(p) != raw[off + p - start]]
