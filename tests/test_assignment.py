"""Decoding with a code set whose literal meanings are not known.

The state a recovered table leaves the tool in: the parse is known and the
meanings are not.  These cover the three claims that makes -- that the parse
really is recoverable from the counts alone, that every occurrence of one
literal code is one equality class, and that the assignment can then be read
back off whatever becomes known.
"""
import zlib

import pytest

from toeknife.core.events import Symbol
from toeknife.core.huffman import HuffmanError, structural_tables
from toeknife.core.inflate import Decoder, carve
from toeknife.core.tape import GUESS, GUESS_PROP, SOLVED, SOLVED_PROP, Tape

TEXT = (b"the quick brown fox jumps over the lazy dog. " * 40 +
        b"pack my box with five dozen liquor jugs. " * 40)


def _dynamic_block(data=TEXT):
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9)
    comp = co.compress(data) + co.flush()
    tr = Decoder(comp).stream()
    blk = next(b for b in tr.blocks if b.btype == 2)
    return comp, tr, blk


def _code_set(blk):
    """The block's table reduced to what a recovery actually recovers."""
    ll = blk.litlen.lengths
    a = [0] * 16
    for s in range(256):
        if ll[s]:
            a[ll[s]] += 1
    lens = [ll[257 + i] if 257 + i < len(ll) else 0 for i in range(29)]
    dist = (list(blk.dist.lengths) + [0] * 30)[:30]
    return a, ll[256], lens, dist


def _carve_structural(comp, blk):
    a, eob, lens, dist = _code_set(blk)
    litlen, dtab, ident = structural_tables(a, eob, lens, dist)
    return carve(comp, blk.clen_ops[-1].bit_end, litlen, dtab, identity=ident)


# ---- the parse survives losing the assignment ------------------------------

def test_structural_tables_reproduce_the_parse_exactly():
    """Counts alone fix which code words are literals, EOB and length codes."""
    comp, tr, blk = _dynamic_block()
    got = _carve_structural(comp, blk)
    assert got.status == "eob"
    truth = [(s.kind, s.length, s.distance) for s in tr.symbols
             if s.block == blk.index]
    mine = [(s.kind, s.length, s.distance) for s in got.symbols]
    assert mine == truth
    assert len(truth) > 50


def test_literals_carry_no_value_and_no_invented_one():
    comp, _tr, blk = _dynamic_block()
    got = _carve_structural(comp, blk)
    lits = [s for s in got.symbols if s.kind == "literal"]
    assert lits
    assert all(s.literal is None for s in lits)
    # The label must say so rather than formatting None as a byte.
    assert lits[0].label() == "literal (unassigned)"


def test_no_byte_is_known_before_anything_is_identified():
    comp, _tr, blk = _dynamic_block()
    tape = _carve_structural(comp, blk).tape
    st = tape.stats()
    assert st["known"] == 0
    assert st["codes"] > 0 and st["codes_known"] == 0
    assert all(h.code is not None for h in tape.holes())


# ---- reading the assignment back off ---------------------------------------

def test_identifying_one_occurrence_resolves_every_other():
    comp, _tr, blk = _dynamic_block()
    tape = _carve_structural(comp, blk).tape
    biggest = max(tape.holes(), key=lambda h: h.size)
    assert biggest.size > 1
    touched = tape.resolve(biggest.anchor, TEXT[biggest.anchor], GUESS)
    assert len(touched) == biggest.size
    assert tape.prov_at(biggest.anchor) == GUESS
    others = [p for p in touched if p != biggest.anchor]
    assert others and all(tape.prov_at(p) == GUESS_PROP for p in others)


def test_the_read_off_recovers_the_true_assignment_and_bytes():
    comp, _tr, blk = _dynamic_block()
    tape = _carve_structural(comp, blk).tape
    for hole in tape.holes():
        tape.resolve(hole.anchor, TEXT[hole.anchor], GUESS)
    assignment = tape.assignment()
    assert assignment
    for code, info in assignment.items():
        assert info["value"] == TEXT[info["first"]], code
        assert info["occurrences"] >= 1
    out, _mask = tape.recovered()
    assert bytes(out) == TEXT[:len(out)]
    assert tape.stats()["codes_known"] == tape.stats()["codes"]


def test_a_model_fill_stays_distinguishable_from_a_guess():
    """The provenance rule: four sources, four colours, through propagation."""
    comp, _tr, blk = _dynamic_block()
    tape = _carve_structural(comp, blk).tape
    hole = max(tape.holes(), key=lambda h: h.size)
    touched = tape.resolve(hole.anchor, TEXT[hole.anchor], SOLVED)
    assert tape.prov_at(hole.anchor) == SOLVED
    assert all(tape.prov_at(p) == SOLVED_PROP for p in touched
               if p != hole.anchor)


def test_undo_puts_the_whole_class_back():
    comp, _tr, blk = _dynamic_block()
    tape = _carve_structural(comp, blk).tape
    before = {h.anchor: h.size for h in tape.holes()}
    hole = max(tape.holes(), key=lambda h: h.size)
    touched = tape.resolve(hole.anchor, TEXT[hole.anchor], GUESS)
    tape.unresolve(touched)
    assert {h.anchor: h.size for h in tape.holes()} == before
    assert tape.stats()["known"] == 0


# ---- the tape mechanism on its own -----------------------------------------

def test_same_code_twice_is_one_class():
    tape = Tape()
    tape.emit_literal_unknown((3, 0))
    tape.emit_literal_unknown((4, 1))
    tape.emit_literal_unknown((3, 0))
    holes = tape.holes()
    assert len(holes) == 2
    by_code = {h.code: h for h in holes}
    assert by_code[(3, 0)].positions == (0, 2)
    assert by_code[(4, 1)].positions == (1,)
    assert tape.assignment()[(3, 0)]["occurrences"] == 2


def test_a_later_occurrence_of_a_settled_code_is_not_a_hole():
    """Once a code's meaning is known, the next one it emits is known too."""
    tape = Tape()
    tape.emit_literal_unknown((3, 0))
    tape.resolve(0, ord("x"), GUESS)
    p = tape.emit_literal_unknown((3, 0))
    assert tape.value_at(p) == ord("x")
    assert tape.prov_at(p) == GUESS_PROP        # still a guess, not a fact
    assert tape.hole_at(p) is None


def test_a_copy_onto_an_unassigned_literal_joins_its_class():
    """A back-reference cannot settle a literal code, only widen it.

    Worth pinning down, because it says where the meanings can come from at
    all.  `emit_copy` resolves immediately when its source is already known and
    otherwise unions, so a copy whose source is an unassigned literal joins that
    code's class rather than giving it a value.  Nothing in the bitstream ever
    injects a byte into such a class: the meanings arrive only from outside it,
    which is why every one of them carries a guess or model provenance.
    """
    tape = Tape()
    tape.emit_literal(ord("a"))
    tape.emit_literal_unknown((3, 0))     # position 1, no value
    tape.emit_copy(1, 1)                  # position 2 reads position 1
    assert tape.value_at(1) is None
    hole = tape.hole_at(1)
    assert hole.positions == (1, 2) and hole.code == (3, 0)
    tape.resolve(1, ord("z"), GUESS)
    assert tape.value_at(2) == ord("z")
    assert tape.prov_at(2) == GUESS_PROP


def test_two_codes_forced_to_one_byte_is_reported_as_a_contradiction():
    """Two literal codes are two byte values, so this cannot happen on a
    correct parse -- and it is the only free self-check the tape has."""
    tape = Tape()
    tape.emit_literal_unknown((3, 0))
    tape.emit_literal_unknown((3, 1))
    assert tape.assignment_conflicts() == []
    tape.emit_copy(2, 1)                  # ties position 2 to position 0 ...
    tape._union(2, 1)                     # ... and to position 1
    assert tape.assignment_conflicts() == [((3, 0), (3, 1))]


def test_two_codes_resolved_to_the_same_byte_is_also_a_contradiction():
    tape = Tape()
    tape.emit_literal_unknown((3, 0))
    tape.emit_literal_unknown((3, 1))
    tape.resolve(0, ord("q"), GUESS)
    tape.resolve(1, ord("q"), GUESS)
    assert tape.assignment_conflicts() == [((3, 0), (3, 1))]


def test_structural_tables_reject_an_impossible_code_set():
    with pytest.raises(HuffmanError):
        structural_tables([0] + [300] * 15, 6, [0] * 29, [0] * 30)


def test_the_codeset_file_contract(tmp_path, capsys):
    """A recovered code set arrives as JSON, and the file may say where to start.

    The contract with the research routes is four keys -- a, e, L, D -- plus an
    optional `bit`. That last one matters more than it looks: whoever recovered
    the table knew where the block body began, and a file that carries the table
    but makes the operator re-derive the offset is only most of a handover.
    """
    import json

    from toeknife.cli import build_parser, cmd_assignment, codeset_session

    comp, _tr, blk = _dynamic_block()
    a, eob, lens, dist = _code_set(blk)
    spec = tmp_path / "cs.json"
    spec.write_text(json.dumps(dict(a=a, e=eob, L=lens, D=dist,
                                    bit=blk.clen_ops[-1].bit_end,
                                    page="test", source="unit test")))
    args = build_parser().parse_args(["x", "--codeset", str(spec)])
    session = codeset_session(comp, args)
    assert session is not None
    assert cmd_assignment(session) == 0
    out = capsys.readouterr().out
    assert "parse:  eob" in out
    # Every literal code is listed and none has a meaning yet.
    assert "literal codes have a meaning" in out
    assert "still unassigned" in out


def test_a_wrong_offset_still_parses_cleanly_so_it_is_warned_about(tmp_path,
                                                                   capsys):
    """Why `bit` is worth carrying: its absence does not fail loudly.

    Decoding from the wrong offset with a valid code set reaches a clean
    end-of-block anyway -- here bit 0 parses 39 symbols and stops, against the
    true 89 -- so the output looks like a successful recovery. Nothing in the
    parse gives it away, which is exactly why the tool has to say the result is
    unanchored rather than let it pass.
    """
    import json

    from toeknife.cli import build_parser, cmd_assignment, codeset_session

    comp, _tr, blk = _dynamic_block()
    a, eob, lens, dist = _code_set(blk)
    spec = tmp_path / "cs.json"
    spec.write_text(json.dumps(dict(a=a, e=eob, L=lens, D=dist)))
    args = build_parser().parse_args(["x", "--codeset", str(spec)])
    session = codeset_session(comp, args)
    assert session is not None
    assert cmd_assignment(session) == 0
    cap = capsys.readouterr()
    assert "unanchored" in cap.err
    # And it really does look fine, which is the point of the warning.
    assert "parse:  eob" in cap.out

    with_bit = tmp_path / "cs2.json"
    with_bit.write_text(json.dumps(dict(a=a, e=eob, L=lens, D=dist,
                                        bit=blk.clen_ops[-1].bit_end)))
    args = build_parser().parse_args(["x", "--codeset", str(with_bit)])
    cmd_assignment(codeset_session(comp, args))
    anchored = capsys.readouterr()
    assert "unanchored" not in anchored.err
    assert anchored.out != cap.out          # a different, and correct, parse


def test_the_tape_pane_marks_positions_waiting_on_a_code(tmp_path):
    """The TUI has to distinguish the two kinds of unknown byte.

    Both are equally unknown, so neither gets a colour the other does not --
    colour is reserved for where a value came from, and neither has come from
    anywhere. What differs is what filling it buys, so the difference is carried
    by shape: a copy chain is usually a handful of positions, a common literal
    code can be hundreds.
    """
    import json

    from toeknife.cli import build_parser, codeset_session
    from toeknife.tui.render import UNASSIGNED_GLYPH, UNKNOWN_GLYPH
    from toeknife.tui.tape import TapeView

    assert UNASSIGNED_GLYPH != UNKNOWN_GLYPH

    comp, _tr, blk = _dynamic_block()
    a, eob, lens, dist = _code_set(blk)
    spec = tmp_path / "cs.json"
    spec.write_text(json.dumps(dict(a=a, e=eob, L=lens, D=dist,
                                    bit=blk.clen_ops[-1].bit_end)))
    args = build_parser().parse_args(["x", "--codeset", str(spec)])
    session = codeset_session(comp, args)
    view = TapeView(session)
    view.rebuild()
    assert view._code_positions
    assert set(view._code_positions) <= view._hole_positions
    # Every code the ledger knows about is represented on the tape.
    assert set(view._code_positions.values()) == set(session.tape.assignment())

    # And one identification really does move a whole class at once.
    hole = max(session.tape.holes(), key=lambda h: h.size)
    assert hole.code is not None and hole.size > 1
    before = session.tape.stats()["known"]
    session.set_guess(hole.anchor, TEXT[hole.anchor])
    after = session.tape.stats()
    assert after["known"] - before == hole.size
    assert after["codes_known"] == 1


def test_ordinary_decoding_is_untouched():
    """The new path is opt-in; without an identity map nothing changes."""
    comp, tr, _blk = _dynamic_block()
    assert tr.status == "end"
    out, _mask = tr.tape.recovered()
    assert bytes(out) == TEXT
    assert all(s.literal is not None
               for s in tr.symbols if s.kind == "literal")
