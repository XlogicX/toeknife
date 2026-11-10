"""The constraint tape, alignment ranking, and the session on top of them."""
import binascii

import pytest

from toeknife.core.align import rank
from toeknife.core.session import Session, tables_from_header
from toeknife.core.tape import GUESS, LITERAL, Tape, UNKNOWN

# The fixed-Huffman body fragment from the README.  Full text was
# "Some numbers are one, two, three, and four. four is the only of those that
#  has four letters in it"; this is the tail, carved mid-symbol.
FRAGMENT = binascii.unhexlify(
    "2f2dd203930a99c50a2519a90af97939950af9690a2519f9c5a90a251989250a"
    "1989c5102539a9252520e332f314324b00")
TRUE_OFFSET = 3


def test_union_find_groups_equal_positions():
    t = Tape()
    for b in b"abc":
        t.emit_literal(b)
    t.emit_copy(distance=5, length=4)      # reaches behind the fragment start
    # positions 3..6 pair with -2,-1,0,1; 0 and 1 are known literals
    assert t.value_at(5) == ord("a")
    assert t.value_at(6) == ord("b")
    assert t.value_at(3) is None and t.value_at(4) is None
    holes = t.holes()
    assert {h.anchor for h in holes} == {-2, -1}
    assert dict((h.anchor, h.positions) for h in holes)[-2] == (-2, 3)


def test_overlapping_copy_repeats():
    t = Tape()
    t.emit_literal(ord("a"))
    t.emit_literal(ord("b"))
    t.emit_copy(distance=2, length=5)      # distance < length: must repeat
    assert bytes(t.out) == b"ababab a".replace(b" ", b"")[:7]
    assert bytes(t.out) == b"abababa"
    assert not t.holes()


def test_guess_propagates_to_every_equal_position():
    t = Tape()
    for b in b"xy":
        t.emit_literal(b)
    t.emit_copy(distance=4, length=3)      # 2,3,4 <- -2,-1,0
    hole = t.hole_at(-2)
    assert hole is not None and hole.positions == (-2, 2)
    touched = t.resolve(-2, ord("Q"), GUESS)
    assert sorted(touched) == [-2, 2]
    assert t.value_at(-2) == ord("Q") and t.value_at(2) == ord("Q")
    assert t.prov_at(-2) == GUESS
    assert not any(h.anchor == -2 for h in t.holes())


def test_resolve_refuses_to_overwrite_a_known_byte():
    t = Tape()
    t.emit_literal(ord("a"))
    with pytest.raises(ValueError):
        t.resolve(0, ord("z"))


def test_readme_fragment_reduces_to_seven_unknown_bytes():
    cands = rank(FRAGMENT, profile="text")
    top = cands[0]
    assert top.bit_offset == TRUE_OFFSET
    assert top.trace.status == "eob"
    assert top.trace.bits_left == 0
    assert top.subset_of is None, "the winner should be a root, not a tail"
    assert top.trace.n_matches == 3
    st = top.trace.tape.stats()
    # 7 unknown bytes standing in for 17 output positions: that ratio is the
    # whole reason for modelling back-references as equalities.
    assert st["holes"] == 7
    assert st["hole_positions"] == 17


def test_tail_alignments_are_detected():
    cands = rank(FRAGMENT, profile="text")
    by = {c.bit_offset: c for c in cands}
    # bit 11 is bit 3's stream minus one leading literal
    assert by[11].subset_of is not None
    assert by[3].subset_of is None


def test_session_guessing_recovers_the_original_text():
    s = Session(FRAGMENT, mode="fragment", profile="text")
    assert s.bit_offset == TRUE_OFFSET
    for pos, ch in [(-3, " "), (-2, "f"), (-1, "o"),
                    (-31, "e"), (-30, "r"), (-29, "s"), (-28, " ")]:
        s.set_guess(pos, ord(ch))
    assert not s.tape.holes(), "every hole should be filled"
    buf, mask, start = s.full()
    assert start == -31
    assert all(mask[i] for i in range(len(mask)) if i >= 31)
    text = buf.decode("latin-1")
    assert text.endswith("four. four is the only of those that has four letters in it")


def test_undo_redo_round_trips():
    s = Session(FRAGMENT, mode="fragment", profile="text")
    before = s.tape.stats()["holes"]
    s.set_guess(-3, ord("X"))
    assert s.tape.stats()["holes"] == before - 1
    assert s.tape.value_at(3) == ord("X")       # propagated
    assert s.undo()
    assert s.tape.stats()["holes"] == before
    assert s.tape.value_at(3) is None
    assert s.redo()
    assert s.tape.value_at(3) == ord("X")


def test_guesses_survive_an_alignment_change():
    s = Session(FRAGMENT, mode="fragment", profile="text")
    s.set_guess(-3, ord("Z"))
    s.select_alignment(11)
    # A guess that no longer lands on a hole is dropped and reported, never
    # silently reinterpreted as a different byte.
    assert -3 not in s.guesses or s.tape.value_at(-3) == ord("Z")
    assert s.trace.status == "eob"


def test_header_fragment_supplies_tables_for_a_body_fragment():
    header = binascii.unhexlify(
        "8dcbd10980400c04d156b600b10b0b0961258133caada0767fda811fef676096dbb6")
    body = binascii.unhexlify("ac7bffb0940e2b6b8f5213ae480fa4de04fe7ae601")
    litlen, dist, block = tables_from_header(header)
    assert block.btype_name == "dynamic"
    assert (block.hlit, block.hdist, block.hclen) == (274, 12, 18)
    s = Session(body, mode="fragment", litlen=litlen, dist=dist, profile="text")
    # bit 2 is the true offset, verified against the intact stream
    s.select_alignment(2)
    buf, mask = s.recovered()
    text = "".join(chr(b) if m else "?" for b, m in zip(buf, mask))
    assert "forensic analysis, which is an" in text


def test_a_corpus_model_must_not_reward_decoding_less():
    """Regression: the model term was a mean over recovered bytes, so an
    alignment recovering 40 slightly-more-predictable bytes outranked one
    recovering 47.  A shorter decode is the worse answer, and the score has to
    say so."""
    from toeknife.predict.context import ContextModel
    corpus = (b"one two three four five six seven eight nine ten. "
              b"the word four has four letters in it. "
              b"of those that remain, only one is the answer. ") * 60
    model = ContextModel(order=4).train(corpus)
    ranked = rank(FRAGMENT, profile="text", model=model)
    top = ranked[0]
    assert top.bit_offset == TRUE_OFFSET, [c.bit_offset for c in ranked[:3]]
    assert top.trace.tape.stats()["holes"] == 7
    # and the model term should rise with how much is recovered
    by_offset = {c.bit_offset: c for c in ranked}
    assert by_offset[3].parts["model"] > by_offset[19].parts["model"]


def test_byte_prior_from_a_dynamic_table():
    header = binascii.unhexlify(
        "8dcbd10980400c04d156b600b10b0b0961258133caada0767fda811fef676096dbb6")
    litlen, _, _ = tables_from_header(header)
    prior = litlen.byte_prior()
    assert prior and abs(sum(prior.values()) - 1.0) < 1e-9
    # 'a' and 'e' are frequent in the source text, so they carry short codes and
    # therefore more probability mass than a byte that never appears.
    assert prior.get(ord("a"), 0) > 0
    assert 0xFE not in prior


def test_the_default_start_reads_only_what_every_reading_agrees_on(tmp_path):
    """A fragment read from a start that is not a symbol
    boundary decodes a few symbols wrong before it falls into step, and those
    went onto the tape as facts. The tool's own start now begins where every
    ranked reading agrees; a start the operator names is honoured exactly; and
    a saved session reloads under the same rule, or its guesses land wrong."""
    import zlib
    text = (b"The quick brown fox jumps over the lazy dog, then the dog "
            b"gets up and chases the fox around the yard for a while. ") * 60
    co = zlib.compressobj(9, zlib.DEFLATED, -15, 9, zlib.Z_FIXED)
    raw = co.compress(text) + co.flush()
    s = Session(raw[37:], mode="fragment", profile="text")
    agree = s.agreement_point()
    assert agree is not None and s.agreed_from >= agree
    buf, mask = s.recovered()
    runs, cur = [], bytearray()
    for b, m in zip(buf, mask):
        if m:
            cur.append(b)
        elif cur:
            runs.append(bytes(cur))
            cur = bytearray()
    runs.append(bytes(cur))
    assert any(runs), "the default start recovered nothing"
    assert all(r in text for r in runs), "every byte the default shows is real"

    s.select_alignment(s.bit_offset)              # the operator's own call
    assert s.agreed_from == s.bit_offset

    s.select_alignment(s.candidates()[0].bit_offset, agreed=True)
    hole = s.tape.holes()[0].anchor
    s.set_guess(hole, ord("x"))
    saved = tmp_path / "s.json"
    s.save(saved)
    back = Session(raw[37:], mode="fragment", profile="text")
    back.select_alignment(0, agreed=False)
    back.load(saved)
    assert back.agreed_from == s.agreed_from
    assert back.tape.value_at(hole) == ord("x")
