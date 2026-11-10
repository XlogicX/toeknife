"""Two fragments of one stream with something missing between them."""
import random
import zlib

import pytest

from toeknife.core.gap import (Piece, assemble, fast_score, gap_from_size,
                               prepare, replay, score_gap, search_gap)
from toeknife.core.session import Session
from toeknife.core.tape import Tape
from toeknife.predict.context import ContextModel

WORDS = (b"the quick brown fox jumps over lazy dogs while forensic analysis "
         b"proceeds steadily through the compressed evidence").split()


def prose(n, seed):
    rng = random.Random(seed)
    return b" ".join(rng.choice(WORDS) for _ in range(n))


def model():
    return ContextModel(order=4).train(prose(9000, 99))


def streamed(src, chunk=1024):
    """Compress with a flush every chunk, recording where each boundary lands."""
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9)
    comp, marks = b"", [(0, 0)]
    for i in range(0, len(src), chunk):
        comp += co.compress(src[i:i + chunk]) + co.flush(zlib.Z_SYNC_FLUSH)
        marks.append((len(comp), min(i + chunk, len(src))))
    return comp + co.flush(), marks


def split(src, keep=4, lose=2, chunk=1024):
    comp, marks = streamed(src, chunk)
    a_end, a_out = marks[keep]
    b_start, b_out = marks[keep + lose]
    pieces = [Piece(comp[:a_end], label="head", has_header=True),
              Piece(comp[b_start:], label="tail", has_header=True)]
    return pieces, b_out - a_out, len(src)


# ---- what an assembled session is -------------------------------------------

def test_an_assembled_session_carries_everything_an_ordinary_one_does():
    """`from_trace` builds a session field by field rather than through
    `__init__`, and the two lists drifted: gap mode had no ledger, no `related`
    and no `known_plaintext`, so any action that recorded itself raised
    AttributeError on a --gap capture while working on an ordinary one.

    Comparing the whole attribute set rather than those three names is the point.
    The bug was not the fields, it was that adding a field to one constructor
    could silently miss the other.
    """
    src = prose(3000, 7)
    pieces, gap, total = split(src)
    trace = assemble(pieces, [gap]).trace

    ordinary = Session(pieces[0].data)
    assembled = Session.from_trace(trace)
    missing = {k for k in vars(ordinary) if not k.startswith("_")} \
        - set(vars(assembled))
    assert not missing, f"an assembled session is missing {sorted(missing)}"
    private = {k for k in vars(ordinary) if k.startswith("_")} \
        - set(vars(assembled))
    assert not private, f"an assembled session is missing {sorted(private)}"

    # And they are usable, not merely present.
    assembled.ledger.add("test", "found", "writing to it works")
    assert len(assembled.ledger) == 1
    assert assembled.related is None and assembled.known_plaintext is None
    # An assembly has no alignment to reconsider, which is not the same as one
    # whose candidates have never been ranked.
    assert assembled.candidates() == []


# ---- the tape ---------------------------------------------------------------

def test_a_gap_becomes_one_unknown_class_per_byte():
    """Each missing byte is its own unknown, so a later reference into the gap
    joins that class rather than inventing a second unknown for the same byte."""
    t = Tape()
    for b in b"abc":
        t.emit_literal(b)
    t.emit_unknown(4)
    assert len(t) == 7
    assert len(t.holes()) == 4
    t.emit_copy(distance=5, length=2)      # positions 7 and 8
    # 7 reads position 2, a known literal, so it resolves outright; 8 reads
    # position 3, which is in the gap, so it joins that byte's class.
    assert t.value_at(7) == ord("c")
    holes = {h.anchor: h.positions for h in t.holes()}
    assert holes[3] == (3, 8), "the copy must join the gap byte's class"


def test_replay_reproduces_a_decode_on_another_tape():
    src = prose(400, 1)
    comp = zlib.compress(src, 6)[2:-4]
    from toeknife.core.inflate import Decoder
    trace = Decoder(comp, strict_window=False).stream(0)
    fresh = Tape()
    replay(trace.symbols, fresh)
    assert bytes(fresh.out) == bytes(trace.tape.out)


# ---- assembling -------------------------------------------------------------

def test_the_right_gap_recovers_only_correct_bytes():
    src = prose(9000, 1)
    pieces, true_gap, total = split(src)
    asm = assemble(pieces, [true_gap])
    buf, mask = asm.tape.recovered()
    known = [i for i, m in enumerate(mask) if m]
    assert len(known) > 2000
    wrong = sum(1 for i in known if i < len(src) and buf[i] != src[i])
    assert wrong == 0, f"{wrong} of {len(known)} known bytes disagree"


def test_a_wrong_gap_shifts_everything_and_is_wrong_everywhere():
    """Getting it wrong does not fail, it silently reads the wrong text.  That
    is the reason the gap has to be pinned rather than assumed."""
    src = prose(9000, 1)
    pieces, true_gap, _ = split(src)
    asm = assemble(pieces, [true_gap + 137])
    buf, mask = asm.tape.recovered()
    known = [i for i, m in enumerate(mask) if m]
    wrong = sum(1 for i in known if i < len(src) and buf[i] != src[i])
    assert wrong > 0, "a wrong gap should not reproduce the original"


def test_the_recorded_decompressed_size_gives_the_gap_arithmetically():
    src = prose(9000, 2)
    pieces, true_gap, total = split(src)
    assert gap_from_size(pieces, total) == true_gap


def test_gap_from_size_needs_exactly_two_pieces():
    src = prose(2000, 3)
    pieces, _, total = split(src)
    assert gap_from_size(pieces + [pieces[-1]], total) is None


# ---- searching --------------------------------------------------------------

def test_a_small_gap_is_found_by_searching():
    src = prose(9000, 4)
    pieces, true_gap, _ = split(src, keep=4, lose=2)
    prep = prepare(pieces)
    m = model()
    scored = sorted(((fast_score(prep, g, m)[0], g)
                     for g in range(0, true_gap + 600)), key=lambda t: (-t[0], t[1]))
    assert scored[0][1] == true_gap, f"best was {scored[0][1]}, true {true_gap}"


def test_the_dragged_byte_is_judged_where_it_lands_not_where_it_came_from():
    """Regression.  Scored in its original context inside the head, any real
    byte looks fine whichever one the gap selected, and the score is useless.
    Judged against the tail's text around it, a wrong gap gives itself away."""
    src = prose(9000, 5)
    pieces, true_gap, _ = split(src, keep=4, lose=2)
    prep = prepare(pieces)
    m = model()
    right, _ = fast_score(prep, true_gap, m)
    for delta in (-200, -37, 51, 311):
        wrong, _ = fast_score(prep, true_gap + delta, m)
        assert right > wrong, f"gap {true_gap} scored below {true_gap + delta}"


def test_searching_reports_how_much_evidence_it_had():
    src = prose(9000, 6)
    pieces, true_gap, _ = split(src)
    found = search_gap(pieces, model=model(), max_gap=true_gap + 400, keep=3)
    assert found[0].gap == true_gap
    assert found[0].resolved > 100, "cross-gap bytes are the evidence"
    assert "read across the gap" in found[0].summary()


def test_a_known_size_short_circuits_the_search():
    src = prose(9000, 7)
    pieces, true_gap, total = split(src)
    found = search_gap(pieces, known_size=total, max_gap=8)   # max_gap ignored
    assert len(found) == 1
    assert found[0].gap == true_gap
    assert any("from the recorded decompressed size" in n for n in found[0].assembly.notes)


def test_a_gap_too_large_to_reach_across_leaves_almost_no_evidence():
    """The window is 32K, so as the gap approaches it almost nothing reaches
    over and there is next to nothing to score.  Better that the shortage be
    visible than that a confident wrong answer come back.

    Measured across gap sizes on one stream: 1940 cross-gap bytes at a 1K gap,
    805 at 4K, 182 at 8K, 23 at 16K.  The search finds the answer outright up to
    about 4K and is unusable well before the window runs out.
    """
    src = prose(40000, 8)
    small, small_gap, _ = split(src, keep=2, lose=2, chunk=1024)
    big, big_gap, _ = split(src, keep=2, lose=24, chunk=1024)
    assert big_gap >= 24000

    m = model()
    _, near = fast_score(prepare(small), small_gap, m)
    _, far = fast_score(prepare(big), big_gap, m)
    assert near["resolved"] > 20 * max(far["resolved"], 1), (
        f"evidence should collapse with the gap: {near['resolved']} vs "
        f"{far['resolved']}")


# ---- the session it produces ------------------------------------------------

def test_an_assembly_becomes_a_session_the_rest_of_the_tool_can_use():
    src = prose(9000, 9)
    pieces, true_gap, _ = split(src)
    asm = assemble(pieces, [true_gap])
    session = Session.from_trace(asm.trace)
    assert session.mode == "assembled"
    assert len(session.tape) == len(asm.tape)
    assert session.tape.holes()

    hole = session.tape.holes()[0]
    session.set_guess(hole.anchor, ord("x"))
    assert session.tape.value_at(hole.anchor) == ord("x")
    assert session.undo()
    assert session.tape.value_at(hole.anchor) is None

    buf, mask, start = session.full()
    assert len(buf) == len(mask)


def test_an_assembled_session_does_not_pretend_to_have_an_alignment():
    """It was stitched from several fragments, so there is no single offset to
    re-decode from and no alignment to reconsider."""
    src = prose(4000, 10)
    pieces, true_gap, _ = split(src, keep=2, lose=1)
    session = Session.from_trace(assemble(pieces, [true_gap]).trace)
    before = len(session.tape)
    session.select_alignment(17)          # must not destroy the assembly
    assert len(session.tape) == before
    assert session.candidates() == []


def test_scoring_an_assembly_matches_the_fast_path():
    src = prose(9000, 11)
    pieces, true_gap, _ = split(src)
    m = model()
    asm = assemble(pieces, [true_gap])
    slow, slow_parts = score_gap(asm, m)
    fast, fast_parts = fast_score(prepare(pieces), true_gap, m)
    assert slow_parts["resolved"] > 0 and fast_parts["resolved"] > 0


def test_the_lookup_cache_changes_no_score():
    """The search remembers the model's answers between neighbouring
    gaps. It has to be the same search, and it has to find the same gap."""
    from toeknife.core import gap as gapmod
    src = prose(9000, 4)
    pieces, true_gap, _ = split(src, keep=4, lose=2)
    prep = prepare(pieces)
    m = model()
    plain = [fast_score(prep, g, m)[0] for g in range(0, true_gap + 600)]
    memo = gapmod._Memo(m)
    cached = [fast_score(prep, g, memo)[0] for g in range(0, true_gap + 600)]
    assert cached == plain
    best = gapmod.search_gap(pieces, model=m, max_gap=true_gap + 600)[0]
    assert best.gap == true_gap


def test_the_cores_score_the_same_as_one_process():
    """Forked workers take chunks of the range. Same gaps, same scores,
    same winner."""
    from toeknife.core import gap as gapmod
    src = prose(9000, 4)
    pieces, true_gap, _ = split(src, keep=4, lose=2)
    m = model()
    one = gapmod.search_gap(pieces, model=m, max_gap=2048, workers=1, keep=50)
    four = gapmod.search_gap(pieces, model=m, max_gap=2048, workers=4, keep=50)
    assert [(c.gap, c.score) for c in one] == [(c.gap, c.score) for c in four]
    assert one[0].gap == true_gap


def test_a_tail_that_starts_on_a_block_header_is_read_from_it():
    """capture-tail.bin is cut at a sync flush, so it starts with a header of
    its own. Read with the head's table it reached a chance end-of-block; the
    header after that was garbage and its block's bytes were kept as fact
    (wrap-up #5). Now the tail is read from the first boundary whose block
    decodes cleanly to the end -- here, bit 0 -- and the gap is the 2,048
    bytes the generator removed, with no known byte wrong."""
    from pathlib import Path
    from toeknife.core.session import Session
    from toeknife.core import gap as gapmod
    d = Path("samples/damaged")
    head = (d / "capture-head.bin").read_bytes()
    tail = (d / "capture-tail.bin").read_bytes()
    truth = (d / "payload.txt").read_bytes()[:60000]
    asm = gapmod.from_holes(head + tail, [(len(head), 0)], 0, 0, 60000)
    assert any("starts with a block header of its own" in n for n in asm.notes)
    assert any("output gap of 2,048 bytes" in n for n in asm.notes)
    buf, mask = Session.from_trace(asm.trace).tape.recovered()
    assert len(buf) == len(truth)
    assert not [i for i in range(len(buf)) if mask[i] and buf[i] != truth[i]]
