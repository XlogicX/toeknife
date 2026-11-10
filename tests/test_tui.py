"""Smoke tests for the interactive view: it starts, navigates, and edits."""
import binascii
import os
from pathlib import Path

import pytest

from textual.widgets import DataTable, Input, OptionList

from toeknife.core.session import Session
from toeknife.tui.app import ToeKnifeApp
from toeknife.tui.tape import TapeView

FRAGMENT = binascii.unhexlify(
    "2f2dd203930a99c50a2519a90af97939950af9690a2519f9c5a90a251989250a"
    "1989c5102539a9252520e332f314324b00")


def option_index(screen, option_id: str) -> int:
    """Where a row sits in the corpus picker, by the id it carries."""
    lst = screen.query_one(OptionList)
    for i in range(lst.option_count):
        if lst.get_option_at_index(i).id == option_id:
            return i
    raise AssertionError(f"no option {option_id!r} in the list")


def make_app():
    return ToeKnifeApp(Session(FRAGMENT, mode="fragment", profile="text"))


async def settle(app, pilot):
    """Wait out the background worker a slow action started.

    Training, scanning and scoring run on threads behind a progress screen now,
    so the assertion has to wait for the thread rather than for the next repaint.
    """
    await app.workers.wait_for_complete()
    await pilot.pause()


async def skip_save_offer(app, pilot):
    """Dismiss the "keep this model?" prompt that follows a successful train."""
    from toeknife.tui.modals import PromptModal
    if isinstance(app.screen, PromptModal):
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_app_starts_on_the_first_hole():
    app = make_app()
    async with app.run_test() as pilot:
        tape = app.query_one(TapeView)
        assert tape.cursor == app.session.tape.holes()[0].anchor
        await pilot.pause()


@pytest.mark.asyncio
async def test_typing_fills_a_hole_and_propagates():
    app = make_app()
    async with app.run_test() as pilot:
        tape = app.query_one(TapeView)
        tape.cursor = -3
        await pilot.press("space")
        assert app.session.tape.value_at(-3) == 0x20
        # -3 is equal to output positions 3 and 38, so one keystroke fills three
        assert app.session.tape.value_at(3) == 0x20
        assert app.session.tape.value_at(38) == 0x20


@pytest.mark.asyncio
async def test_typing_over_a_known_byte_is_refused():
    import zlib
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    whole = co.compress(b"read straight from the stream " * 40) + co.flush()
    app = ToeKnifeApp(Session(whole))          # a whole stream: byte 0 is a fact
    async with app.run_test() as pilot:
        tape = app.query_one(TapeView)
        tape.cursor = 0                      # a literal: a fact, not a guess
        before = app.session.tape.value_at(0)
        await pilot.press("z")
        assert app.session.tape.value_at(0) == before
        assert not app.session.guesses
        status = str(app.query_one("#status").content)
        assert "it is evidence" in status


@pytest.mark.asyncio
async def test_typing_over_a_model_fill_replaces_it():
    """The iterative loop only works both ways. The solver writes bytes at a
    confidence figure; the operator has the plaintext and a reason, and used to
    be locked out of correcting one because the tape had a value there."""
    from toeknife.core.tape import GUESS, SOLVED
    app = solving_app()
    async with app.run_test(size=(150, 36)) as pilot:
        await pilot.press("ctrl+f")
        await settle(app, pilot)
        filled = [p for p, rec in app.session._records.items()
                  if rec.provenance == SOLVED]
        assert filled, "the model should have filled something to override"
        pos = filled[0]
        was = app.session.tape.value_at(pos)

        app.tape_view.cursor = pos
        await pilot.press("Q")
        await pilot.pause()

        assert app.session.tape.value_at(pos) == ord("Q")
        # It is the operator's byte now, not the model's, and it says so in the
        # one place that matters: the provenance the colour comes from.
        assert app.session._records[pos].provenance == GUESS
        assert app.session.tape.prov_at(pos) == GUESS
        status = str(app.query_one("#status").content)
        assert "replaced the model's" in status and "'Q'" in status

        # And one undo puts the model's fill back, as the model's.
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert app.session.tape.value_at(pos) == was
        assert app.session.tape.prov_at(pos) == SOLVED, \
            "undo must not re-attribute the model's fill to the operator"


@pytest.mark.asyncio
async def test_typing_over_your_own_guess_replaces_it_too():
    app = make_app()
    async with app.run_test() as pilot:
        pos = app.session.tape.holes()[0].anchor
        app.tape_view.cursor = pos
        await pilot.press("a")
        await pilot.pause()
        await pilot.press("b")
        await pilot.pause()
        assert app.session.tape.value_at(pos) == ord("b")
        assert "replaced your 'a' with 'b'" in \
            str(app.query_one("#status").content)


@pytest.mark.asyncio
async def test_a_back_reference_says_where_the_byte_it_copies_lives():
    """A copy is not a guess and not an independent fact: correcting it means
    correcting its source, and a bare refusal does not say that."""
    import zlib
    from toeknife.core.tape import COPY
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    whole = co.compress(b"copy me, copy me, copy me again " * 30) + co.flush()
    app = ToeKnifeApp(Session(whole))     # a whole stream: copies are facts
    async with app.run_test(size=(150, 36)) as pilot:
        tape = app.session.tape
        copied = next(p for p in range(len(tape)) if tape.prov_at(p) == COPY)
        app.tape_view.cursor = copied
        await pilot.press("z")
        await pilot.pause()
        assert tape.value_at(copied) != ord("z")
        assert "copies from" in str(app.query_one("#status").content)


@pytest.mark.asyncio
async def test_correcting_a_byte_invalidates_the_cached_proposals():
    """The point of correcting one: the next round should see it."""
    app = make_app()
    async with app.run_test() as pilot:
        pos = app.session.tape.holes()[0].anchor
        app.tape_view.cursor = pos
        app.suggestion_for(pos)
        assert app._suggestions, "expected something cached to invalidate"
        await pilot.press("k")
        await pilot.pause()
        assert not app._suggestions


@pytest.mark.asyncio
async def test_tab_walks_holes_and_undo_restores():
    app = make_app()
    async with app.run_test() as pilot:
        tape = app.query_one(TapeView)
        anchors = sorted(h.anchor for h in app.session.tape.holes())
        tape.cursor = anchors[0]
        await pilot.press("tab")
        assert tape.cursor == anchors[1]
        await pilot.press("Q")
        assert app.session.tape.value_at(anchors[1]) == ord("Q")
        await pilot.press("ctrl+z")
        assert app.session.tape.value_at(anchors[1]) is None


@pytest.mark.asyncio
async def test_arrow_keys_move_the_byte_cursor_not_the_scrollbar():
    app = make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        tape = app.query_one(TapeView)
        tape.cursor = 10
        await pilot.press("right")
        assert tape.cursor == 11
        await pilot.press("left", "left")
        assert tape.cursor == 9
        await pilot.press("down")
        assert tape.cursor == 9 + tape.cols
        await pilot.press("up")
        assert tape.cursor == 9


@pytest.mark.asyncio
async def test_cursor_cannot_leave_the_tape():
    app = make_app()
    async with app.run_test() as pilot:
        tape = app.query_one(TapeView)
        tape.goto(tape.lo)
        await pilot.press("left", "left", "left")
        assert tape.cursor == tape.lo
        tape.goto(tape.hi - 1)
        await pilot.press("right", "right")
        assert tape.cursor == tape.hi - 1


@pytest.mark.asyncio
async def test_selecting_a_match_marks_its_source_bytes():
    app = make_app()
    async with app.run_test() as pilot:
        tape = app.query_one(TapeView)
        await pilot.press("ctrl+n")           # jump to the first back-reference
        sym = app.session.trace.symbol_at_out(tape.cursor)
        assert sym is not None and sym.kind == "match"
        assert tape._source_range == range(sym.out_pos - sym.distance,
                                           sym.out_pos - sym.distance + sym.length)


@pytest.mark.asyncio
async def test_every_row_renders_without_error():
    app = make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        tape = app.query_one(TapeView)
        for y in range(tape.virtual_size.height):
            strip = tape.render_line(y)
            assert strip is not None
        await pilot.pause()


@pytest.mark.asyncio
async def test_hex_fill_sets_a_non_printable_byte():
    app = make_app()
    async with app.run_test() as pilot:
        app.query_one(TapeView).cursor = -3
        await pilot.press("ctrl+x")
        await pilot.press("0", "a", "enter")
        await pilot.pause()
        assert app.session.tape.value_at(-3) == 0x0A
        assert app.session.tape.value_at(3) == 0x0A


@pytest.mark.asyncio
async def test_help_and_alignment_picker_open_and_close():
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.press("f1")
        await pilot.pause()
        assert app.screen_stack[-1].__class__.__name__ == "HelpScreen"
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("f2")
        await pilot.pause()
        assert app.screen_stack[-1].__class__.__name__ == "AlignmentPicker"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen_stack[-1].__class__.__name__ != "AlignmentPicker"


@pytest.mark.asyncio
async def test_the_help_scrolls_on_a_short_terminal():
    """It is longer than 24 rows, and in a plain container the tail was clipped
    away with nothing on screen to say there was more."""
    app = make_app()
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("f1")
        await pilot.pause()
        box = app.screen_stack[-1].query_one("#help-box")
        assert box.max_scroll_y > 0, "fixture must be a terminal it overflows"
        await pilot.press("pagedown")
        await pilot.pause()
        assert box.scroll_offset.y > 0, "the keyboard has to reach the bottom"
        await pilot.press("escape")           # and escape still closes it
        await pilot.pause()
        assert app.screen_stack[-1].__class__.__name__ != "HelpScreen"


@pytest.mark.asyncio
async def test_alignment_picker_changes_the_decode():
    app = make_app()
    async with app.run_test(size=(140, 34)) as pilot:
        before = app.session.bit_offset
        await pilot.press("f2")
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.session.bit_offset != before
        assert app.session.trace.status in ("eob", "end", "truncated")


@pytest.mark.asyncio
async def test_stream_mode_opens_with_block_structure():
    full = binascii.unhexlify(
        "8dcbd10980400c04d156b600b10b0b0961258133caada0767fda811fef676096"
        "dbb6a311629d564eac7bffb0940e2b6b8f5213ae480fa4de04fe7ae601")
    app = ToeKnifeApp(Session(full))
    async with app.run_test() as pilot:
        assert app.session.mode == "stream"
        assert not app.session.tape.holes()
        assert app.session.trace.blocks[0].btype_name == "dynamic"
        await pilot.pause()


# ---- the model in the interface --------------------------------------------

SOLVER_CORPUS = (b"one two three four five six seven eight nine ten. "
                 b"the word four has four letters in it, which is unusual. "
                 b"of those that remain, only one is the answer we want. ") * 60


def solving_app():
    from toeknife.predict.context import ContextModel
    session = Session(FRAGMENT, mode="fragment", profile="text")
    session.model = ContextModel(order=4).train(SOLVER_CORPUS)
    return ToeKnifeApp(session, solve_confidence=0.9)


@pytest.mark.asyncio
async def test_a_solved_tape_renders():
    """Regression: model fills had no colour, so every row containing one
    raised KeyError and took the whole view down."""
    from toeknife.core.tape import SOLVED
    from toeknife.tui.render import STYLES
    app = make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        hole = app.session.tape.holes()[0]
        app.session.set_guess(hole.anchor, ord("x"), SOLVED)
        assert app.session.tape.prov_at(hole.anchor) in STYLES
        tape = app.query_one(TapeView)
        tape.rebuild()
        for y in range(tape.virtual_size.height):
            assert tape.render_line(y) is not None
        await pilot.pause()


@pytest.mark.asyncio
async def test_model_fills_are_a_different_colour_from_typed_guesses():
    from toeknife.core.tape import GUESS, GUESS_PROP, SOLVED, SOLVED_PROP
    from toeknife.tui.render import STYLES
    assert STYLES[SOLVED] != STYLES[GUESS]
    assert STYLES[SOLVED_PROP] != STYLES[GUESS_PROP]
    assert STYLES[SOLVED].color.name == "magenta"
    assert STYLES[GUESS].color.name == "yellow"


@pytest.mark.asyncio
async def test_ctrl_f_fills_and_one_undo_reverts_the_whole_run():
    app = solving_app()
    async with app.run_test(size=(140, 34)) as pilot:
        before = app.session.tape.stats()["holes"]
        await pilot.press("ctrl+f")
        await settle(app, pilot)
        after = app.session.tape.stats()["holes"]
        assert after < before, "the model should have filled something"
        assert len(app.session.guesses) > 1
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert app.session.tape.stats()["holes"] == before


@pytest.mark.asyncio
async def test_ctrl_f_without_a_corpus_says_one_is_missing():
    """With no corpus the model learns from the stream's own bytes, and on a
    real hole that fills next to nothing. The status has to say which of
    "the model is weak" and "there is no model yet" it is."""
    bare = make_app()
    async with bare.run_test(size=(140, 34)) as pilot:
        await pilot.press("ctrl+f")
        await settle(bare, pilot)
        assert "no corpus loaded" in status_of(bare)
    trained = solving_app()
    async with trained.run_test(size=(140, 34)) as pilot:
        await pilot.press("ctrl+f")
        await settle(trained, pilot)
        assert "no corpus loaded" not in status_of(trained)


def test_ctrl_l_belongs_to_the_checksum_screen_only():
    """It was a hidden second key for f2 on the main screen and "model +
    checksum" one screen away: same key, unrelated verbs."""
    from toeknife.tui.modals import ChecksumScreen
    assert not any(b.key == "ctrl+l" for b in ToeKnifeApp.BINDINGS)
    assert any(b.key == "ctrl+l" for b in ChecksumScreen.BINDINGS)


@pytest.mark.asyncio
async def test_ctrl_e_takes_back_the_models_fills_and_keeps_yours():
    """A fill closes its class, so the model never reconsiders a byte it got
    wrong -- even after the operator reads one next to it. ctrl+e reopens only
    the model's classes, and one ctrl+z puts the whole thing back."""
    from toeknife.core.tape import GUESS, SOLVED
    app = solving_app()
    async with app.run_test(size=(140, 34)) as pilot:
        # The operator reads one byte, then lets the model take the rest.
        app.tape_view.cursor = app.session.tape.holes()[0].anchor
        await pilot.press("e")
        await pilot.pause()
        mine = {p for p, rec in app.session._records.items()
                if rec.provenance == GUESS}
        assert mine, "the typed byte did not land"

        await pilot.press("ctrl+f")
        await settle(app, pilot)
        assert any(rec.provenance == SOLVED
                   for rec in app.session._records.values()), \
            "nothing to retract means this test proves nothing"

        await pilot.press("ctrl+e")
        await settle(app, pilot)
        for pos in mine:
            assert app.session.tape.value_at(pos) is not None, \
                "retracting took the operator's byte as well as the model's"
            assert app.session.tape.prov_at(pos) == GUESS

        await pilot.press("ctrl+z")
        await pilot.pause()
        assert mine <= set(app.session._records), "undo lost the typed bytes"


@pytest.mark.asyncio
async def test_the_solver_panel_opens_and_accepts_one():
    app = solving_app()
    async with app.run_test(size=(150, 36)) as pilot:
        before = app.session.tape.stats()["holes"]
        await pilot.press("f4")
        await settle(app, pilot)
        assert app.screen_stack[-1].__class__.__name__ == "SolverScreen"
        assert app.screen_stack[-1].proposals, "panel should list proposals"
        await pilot.press("enter")
        await pilot.pause()
        assert app.session.tape.stats()["holes"] == before - 1


@pytest.mark.asyncio
async def test_the_threshold_moves_from_inside_the_panel():
    """The dial is the point of the panel: you cannot pick a threshold sensibly
    until you have seen the spread, which is only visible once it is open."""
    app = solving_app()                       # opens at 0.9
    async with app.run_test(size=(150, 36)) as pilot:
        await pilot.press("f4")
        await settle(app, pilot)
        panel = app.screen_stack[-1]
        marked = lambda: sum(1 for p in panel.proposals
                             if p.confidence >= panel.threshold)
        assert panel.threshold == 0.9
        at_default = marked()
        assert 0 < at_default < len(panel.proposals)

        await pilot.press("plus")
        await pilot.pause()
        assert panel.threshold == 0.95
        assert marked() < at_default, "stricter should mark fewer"

        for _ in range(4):                    # 0.95 -> .9 -> .8 -> .7 -> .6
            await pilot.press("minus")
        await pilot.pause()
        assert panel.threshold == 0.6
        assert marked() == len(panel.proposals), "loose enough to take them all"


@pytest.mark.asyncio
async def test_the_dial_stops_at_the_ends_of_the_ladder():
    app = solving_app()
    async with app.run_test(size=(150, 36)) as pilot:
        await pilot.press("f4")
        await settle(app, pilot)
        panel = app.screen_stack[-1]
        for _ in range(15):
            await pilot.press("minus")
        await pilot.pause()
        assert panel.threshold == 0.0
        for _ in range(15):
            await pilot.press("plus")
        await pilot.pause()
        assert panel.threshold == 0.999


@pytest.mark.asyncio
async def test_what_the_dial_was_left_on_is_what_ctrl_f_uses():
    """Otherwise the panel teaches you the right threshold and then throws it
    away, and the only way to act on it is to quit and pass a flag."""
    from toeknife.predict.context import ContextModel
    session = Session(FRAGMENT, mode="fragment", profile="text")
    session.model = ContextModel(order=4).train(SOLVER_CORPUS)
    app = ToeKnifeApp(session, solve_confidence=0.995)   # the shipped default
    async with app.run_test(size=(150, 36)) as pilot:
        await pilot.press("ctrl+f")           # what the strict bar reaches
        await settle(app, pilot)
        strict = len(app.session.guesses)
        assert app.session.tape.holes(), "0.995 should leave something over"
        await pilot.press("ctrl+z")
        await pilot.pause()

        await pilot.press("f4")
        await settle(app, pilot)
        for _ in range(5):                    # .995 -> .99 -> .95 -> .9 -> .8 -> .7
            await pilot.press("minus")
        await pilot.press("escape")
        await pilot.pause()
        assert app.solve_confidence == 0.7
        assert app.solver().min_confidence == 0.7

        await pilot.press("ctrl+f")
        await settle(app, pilot)
        assert len(app.session.guesses) > strict
        assert not app.session.tape.holes()


@pytest.mark.asyncio
async def test_the_panel_never_claims_the_bar_is_100_percent():
    """Regression: 0.995 formatted with .0% is "100%", so both the panel and the
    ctrl+f refusal told the operator the model had to be certain.  It reads as
    "this feature does not work" rather than "the bar is one notch too high"."""
    from textual.widgets import Label, Static
    from toeknife.predict.context import ContextModel
    session = Session(FRAGMENT, mode="fragment", profile="text")
    session.model = ContextModel(order=4).train(b"nothing like the fragment " * 40)
    app = ToeKnifeApp(session, solve_confidence=0.995)
    async with app.run_test(size=(150, 36)) as pilot:
        await pilot.press("ctrl+f")
        await settle(app, pilot)
        status = app.query_one("#status", Static).content.plain
        if "filled nothing" in status:
            assert "99.5%" in status and "100%" not in status
        await pilot.press("f4")
        await settle(app, pilot)
        content = app.screen_stack[-1].query_one("#solve-title", Label).content
        title = getattr(content, "plain", content)
        assert "99.5%" in title and "100%" not in title


@pytest.mark.asyncio
async def test_the_solver_panel_can_be_dismissed_without_applying_anything():
    app = solving_app()
    async with app.run_test(size=(150, 36)) as pilot:
        before = app.session.tape.stats()["holes"]
        await pilot.press("f4")
        await settle(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert app.session.tape.stats()["holes"] == before
        assert not app.session.guesses


@pytest.mark.asyncio
async def test_the_bits_pane_shows_a_suggestion_on_a_hole():
    app = solving_app()
    async with app.run_test(size=(150, 36)) as pilot:
        tape = app.query_one(TapeView)
        hole = app.session.tape.holes()[0]
        tape.cursor = hole.anchor
        await pilot.pause()
        suggestion = app.suggestion_for(hole.anchor)
        assert suggestion is not None
        assert 0 <= suggestion.confidence <= 1
        # and nothing on a byte that is already known
        tape.cursor = 0
        await pilot.pause()
        assert app.suggestion_for(0) is None


@pytest.mark.asyncio
async def test_the_model_declines_when_there_is_too_little_to_train_on():
    """47 recovered bytes is too few to self-train on, and a fixed-Huffman
    block's implied frequencies are flat, so there is no table prior either.
    Saying so beats inventing a predictor -- and f4 has to reach that verdict
    the same way --solve does, which it did not when the panel built its own."""
    app = make_app()                       # session.model is None
    async with app.run_test(size=(120, 30)) as pilot:
        app.solver()
        assert "empty" in app._model_source, app._model_source
        await pilot.pause()


@pytest.mark.asyncio
async def test_f5_trains_a_corpus_and_the_solver_is_rebuilt_against_it(tmp_path):
    """The corpus is the biggest lever on a fill and used to be reachable only
    from the command line, which meant deciding it before seeing the data."""
    corpus = tmp_path / "prose.txt"
    corpus.write_text("forensic analysis of recovered data. " * 400)
    app = make_app()                       # session.model is None
    async with app.run_test(size=(120, 30)) as pilot:
        stale = app.solver()
        assert "empty" in app._model_source, app._model_source

        await pilot.press("f5")            # the binding, the modal, the lot
        await pilot.pause()
        modal = app.screen_stack[-1]
        modal.query_one(OptionList).highlighted = option_index(modal, "ask:path")
        await pilot.press("enter")         # opens the freeform path field
        await pilot.pause()
        modal.query_one(Input).value = str(corpus)
        await pilot.press("enter")         # shows what it would read
        await pilot.pause()
        assert app.session.model is None, "the first enter must not train"
        await pilot.press("enter")         # and commits
        await pilot.pause()

        assert app.session.model is not None
        assert app.session.model.trained_bytes > 0

        # The solver is rebuilt rather than reused, so proposals come from the
        # new corpus instead of whatever the old model had already cached.
        assert app.solver() is not stale, "solver must be rebuilt, not reused"
        assert "corpus" in app._model_source, app._model_source
        assert app.solver().model.logprob(ord("s"), b"analysi") > -8.0
        await pilot.pause()


@pytest.mark.asyncio
async def test_f5_on_a_path_that_is_not_there_says_so_and_changes_nothing():
    app = solving_app()
    async with app.run_test(size=(120, 30)) as pilot:
        before = app.session.model
        msg = app.load_corpus("/nonexistent/corpus/path")
        assert "no such file" in msg, msg
        assert app.session.model is before
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_supplied_corpus_is_what_the_panel_predicts_from():
    app = solving_app()
    async with app.run_test(size=(120, 30)) as pilot:
        app.solver()
        assert "corpus" in app._model_source, app._model_source
        await pilot.pause()


# ---- the blocks pane and the tape track each other -------------------------

def multi_block_app():
    """Three fixed blocks with an empty stored block wedged between each pair.

    The empty blocks matter: they share an output position with the block that
    follows, so they are the case a naive position lookup gets wrong.
    """
    import zlib
    co = zlib.compressobj(9, zlib.DEFLATED, -15)
    raw = co.compress(b"the first block of text " * 4)
    raw += co.flush(zlib.Z_FULL_FLUSH)
    raw += co.compress(b"and a second run of words " * 4)
    raw += co.flush(zlib.Z_FULL_FLUSH)
    raw += co.compress(b"third and final section " * 4)
    raw += co.flush()
    return ToeKnifeApp(Session(raw, mode="stream"))


@pytest.mark.asyncio
async def test_moving_the_cursor_walks_the_tree_to_that_block():
    app = multi_block_app()
    async with app.run_test(size=(150, 40)) as pilot:
        from textual.widgets import Tree
        from toeknife.tui.panes import BlocksPane

        pane = app.query_one(BlocksPane)
        tape = app.query_one(TapeView)
        tree = pane.query_one(Tree)
        for pos, want in ((0, 0), (95, 0), (96, 2), (250, 4), (199, 2)):
            tape.cursor = pos
            await pilot.pause()
            assert tree.cursor_node is pane._block_nodes[want], \
                f"output {pos} should highlight block #{want}"
            assert tree.cursor_node.is_expanded


@pytest.mark.asyncio
async def test_only_the_pane_s_own_expansion_is_undone():
    """A block the user opened by hand stays open as the cursor moves on."""
    app = multi_block_app()
    async with app.run_test(size=(150, 40)) as pilot:
        from toeknife.tui.panes import BlocksPane

        pane = app.query_one(BlocksPane)
        tape = app.query_one(TapeView)
        tape.cursor = 0
        await pilot.pause()
        pane._block_nodes[4].expand()        # as if the user opened it
        tape.cursor = 96                     # auto-expands #2, collapses #0
        await pilot.pause()
        assert pane._block_nodes[4].is_expanded
        assert not pane._block_nodes[0].is_expanded


@pytest.mark.asyncio
async def test_choosing_a_block_sends_the_cursor_to_its_first_byte():
    app = multi_block_app()
    async with app.run_test(size=(150, 40)) as pilot:
        from textual.widgets import Tree
        from toeknife.tui.panes import BlocksPane

        pane = app.query_one(BlocksPane)
        tape = app.query_one(TapeView)
        tree = pane.query_one(Tree)
        tree.select_node(pane._block_nodes[4])
        await pilot.pause()
        assert tape.cursor == app.session.trace.blocks[4].out_start == 200
        assert app.focused is tape       # typing must go on filling holes


@pytest.mark.asyncio
async def test_the_status_line_names_the_block_under_the_cursor():
    app = multi_block_app()
    async with app.run_test(size=(150, 40)) as pilot:
        from textual.widgets import Static

        tape = app.query_one(TapeView)
        tape.cursor = 250
        await pilot.pause()
        assert "block #4" in app.query_one("#status", Static).content.plain


@pytest.mark.asyncio
async def test_the_pane_survives_a_fragment_with_no_block_under_the_cursor():
    app = make_app()                       # headerless, and the window is < 0
    async with app.run_test(size=(150, 40)) as pilot:
        from textual.widgets import Static
        from toeknife.tui.panes import BlocksPane

        tape = app.query_one(TapeView)
        tape.cursor = -3                   # the pre-fragment window
        await pilot.pause()
        assert app.query_one(BlocksPane).highlight(-3) is None
        assert "pre-fragment" in app.query_one("#status", Static).content.plain


@pytest.mark.asyncio
async def test_clicking_a_byte_moves_the_cursor_and_walks_the_tree():
    """A real mouse click, not a programmatic cursor move: the coordinate math
    has to survive the widget's own padding and the address gutter."""
    app = multi_block_app()
    async with app.run_test(size=(150, 40)) as pilot:
        from textual.widgets import Tree
        from toeknife.tui.panes import BlocksPane
        from toeknife.tui.tape import GUTTER, LINES_PER_ROW

        pane = app.query_one(BlocksPane)
        tape = app.query_one(TapeView)
        tree = pane.query_one(Tree)
        pad_x, pad_y = tape.gutter.top_left  # border+padding around the content
        for pos, want in ((10, 0), (250, 4)):
            row, col = divmod(pos - tape.lo, tape.cols)
            await pilot.click(TapeView, offset=(
                pad_x + GUTTER + col, pad_y + row * LINES_PER_ROW + 1))
            await pilot.pause()
            assert tape.cursor == pos, \
                f"click on the glyph for {pos} put the cursor at {tape.cursor}"
            assert tree.cursor_node is pane._block_nodes[want], \
                f"output {pos} should walk the tree to block #{want}"
        assert app.focused is tape           # clicking the tape lets you type


@pytest.mark.asyncio
async def test_clicking_a_block_in_the_tree_jumps_the_tape():
    app = multi_block_app()
    async with app.run_test(size=(150, 40)) as pilot:
        from textual.widgets import Tree
        from toeknife.tui.panes import BlocksPane

        pane = app.query_one(BlocksPane)
        tape = app.query_one(TapeView)
        tree = pane.query_one(Tree)
        node = pane._block_nodes[4]
        await pilot.click(Tree, offset=(6, node.line - tree.scroll_offset.y))
        await pilot.pause()
        assert tape.cursor == app.session.trace.blocks[4].out_start == 200
        assert app.focused is tape           # typing must go on filling holes


def test_resync_moves_the_cursor_onto_the_recovered_data():
    """A successful resync must not leave the cursor in the window it dropped.

    Opening a mid-stream capture as a fragment puts the cursor deep in the
    negative pre-fragment window.  Resyncing re-decodes as a stream, where that
    window does not exist -- and a cursor left at -13,414 renders as unknown
    space forever, so a recovery of 321,399 bytes with no holes looked to the
    operator like it had recovered nothing.
    """
    import asyncio
    import zlib

    from toeknife.core.scan import scan
    from toeknife.core.session import Session
    from toeknife.tui.app import ToeKnifeApp

    raw = b"the quick brown fox jumps over the lazy dog. " * 400
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9, zlib.Z_FIXED)
    out = bytearray()
    for i in range(0, len(raw), 4096):
        out += co.compress(raw[i:i + 4096])
        out += co.flush(zlib.Z_FULL_FLUSH)
    out += co.flush()
    frag = bytes(out)[len(bytes(out)) // 3:]

    async def drive():
        app = ToeKnifeApp(Session(frag), "test")
        async with app.run_test() as pilot:
            assert app.tape_view.cursor < 0, "expected a pre-fragment cursor"
            points = scan(frag)
            assert points

            def done(_):
                pass
            app.session.resync_to(points[0].bit_offset)
            app.refresh_all()
            app.tape_view.goto(0)
            await pilot.pause()
            st = app.session.tape.stats()
            assert st["holes"] == 0
            # In range, and pointing at a byte that is actually known.
            assert 0 <= app.tape_view.cursor < st["out_len"]
            assert app.session.tape.value_at(app.tape_view.cursor) is not None

    asyncio.run(drive())


def test_rebuild_reclamps_a_cursor_left_outside_the_new_range():
    """The general form of the same bug: any re-decode can shrink the range."""
    import zlib

    from toeknife.core.scan import scan
    from toeknife.core.session import Session

    raw = b"pack my box with five dozen liquor jugs. " * 400
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9, zlib.Z_FIXED)
    out = bytearray()
    for i in range(0, len(raw), 4096):
        out += co.compress(raw[i:i + 4096])
        out += co.flush(zlib.Z_FULL_FLUSH)
    out += co.flush()
    frag = bytes(out)[len(bytes(out)) // 3:]

    import asyncio

    from toeknife.tui.app import ToeKnifeApp

    async def drive():
        session = Session(frag)
        app = ToeKnifeApp(session, "test")
        async with app.run_test() as pilot:
            await pilot.pause()                  # let the panes mount
            view = app.tape_view
            view.goto(session.tape.min_pos)      # far into the negative window
            assert view.cursor < 0
            session.resync_to(scan(frag)[0].bit_offset)
            app.refresh_all()
            assert view.cursor >= view.lo, "cursor left below the new range"
            assert view.cursor < view.hi, "cursor left above the new range"

    asyncio.run(drive())


# ---- the checksum, in the view ----------------------------------------------

CRC_BODY = (b"the forensic analyst reads compressed data repeatedly. "
            b"forensic work repeats, and so do the words inside it. ") * 40


def whole_app(recorded_crc_wrong=False):
    """A complete stream whose container recorded a size and a checksum."""
    import zlib
    from toeknife.containers import parse
    blob = zlib.compress(CRC_BODY, 6)
    sm = parse(blob).streams[0]
    crc = zlib.crc32(CRC_BODY) & 0xFFFFFFFF
    sm.crc32 = (crc ^ 0xFFFF) if recorded_crc_wrong else crc
    sm.uncompressed_size = len(CRC_BODY)
    return ToeKnifeApp(Session(sm.slice(blob), mode="stream", stream=sm))


def dict_app(tail=b"sic"):
    """A stream with real unknown bytes and a container that knows the answer:
    compressed against a preset dictionary the file does not carry."""
    import zlib
    from toeknife.containers import parse
    zdict = b"\xa5" * 80 + tail
    co = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY,
                          zdict=zdict)
    blob = co.compress(CRC_BODY) + co.flush()
    sm = parse(blob).streams[0]
    sm.crc32 = zlib.crc32(CRC_BODY) & 0xFFFFFFFF
    sm.uncompressed_size = len(CRC_BODY)
    return ToeKnifeApp(Session(sm.slice(blob), mode="stream", stream=sm)), zdict


def status_of(app):
    from textual.widgets import Static
    return app.query_one("#status", Static).content.plain


@pytest.mark.asyncio
async def test_the_status_bar_says_the_checksum_agrees():
    """The one thing on screen that can tell a right decode from a plausible
    one. Everything else looks identical either way."""
    app = whole_app()
    async with app.run_test(size=(140, 30)) as pilot:
        assert "crc ok" in status_of(app)
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_status_bar_says_the_checksum_disagrees():
    app = whole_app(recorded_crc_wrong=True)
    async with app.run_test(size=(140, 30)) as pilot:
        assert "crc MISMATCH" in status_of(app)
        await pilot.pause()


@pytest.mark.asyncio
async def test_an_unfinished_recovery_says_the_checksum_is_undecided():
    """A checksum over output that still has holes is not evidence, and the
    view must not imply that it is."""
    app, _ = dict_app()
    async with app.run_test(size=(140, 30)) as pilot:
        assert app.session.tape.holes()
        assert "crc ?" in status_of(app)
        await pilot.pause()


@pytest.mark.asyncio
async def test_f8_opens_the_checksum_panel_with_the_verdict():
    from toeknife.tui.modals import ChecksumScreen
    app = whole_app(recorded_crc_wrong=True)
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f8")
        await pilot.pause()
        assert isinstance(app.screen, ChecksumScreen)
        body = str(app.screen.query_one("#help-body").content)
        assert "MISMATCH" in body and "FAILED" in body
        # and it names the thing that knows, rather than only complaining
        assert "crc32" in body
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_checksum_panel_refuses_when_there_is_nothing_to_check():
    app = make_app()                        # a bare fragment, no container
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f8")
        await pilot.pause()
        body = str(app.screen.query_one("#help-body").content)
        assert "Nothing to check against" in body
        assert "crc" not in status_of(app)
        await pilot.pause()


@pytest.mark.asyncio
async def test_f8_then_enter_pins_the_last_unknown_bytes_with_the_checksum():
    """Unknown bytes enter CRC32 linearly, so a small enough system is solved
    outright rather than searched."""
    app, zdict = dict_app()
    async with app.run_test(size=(140, 40)) as pilot:
        before = len(app.session.tape.holes())
        assert 0 < before <= 4
        await pilot.press("f8")
        await pilot.pause()
        await pilot.press("enter")          # the panel returns "pin"
        await pilot.pause()

        assert not app.session.tape.holes(), status_of(app)
        window, _, _ = app.session.window()
        assert bytes(window) == zdict[-len(window):]
        assert "crc ok" in status_of(app)
        assert any(e.route == "checksum" for e in app.session.ledger.entries)
        await pilot.pause()


@pytest.mark.asyncio
async def test_pinned_bytes_undo_as_one_step():
    app, _ = dict_app()
    async with app.run_test(size=(140, 40)) as pilot:
        before = len(app.session.tape.holes())
        await pilot.press("f8")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert len(app.session.tape.holes()) == before
        await pilot.pause()


# ---- table statistics --------------------------------------------------------

def stats_app():
    """A real dynamic block from capture.gz's first member, 40 KB."""
    from toeknife.core.session import Session
    from toeknife.tui.app import ToeKnifeApp
    data = Path("samples/damaged/capture.gz").read_bytes()
    return ToeKnifeApp(Session(data))


def first_block_facts(app):
    """What the screen should say, worked out from the decode rather than
    pasted from the screen: the figures move whenever the sample's payload
    does, and a constant would only test that it did not."""
    blk = app.session.trace.blocks[0]
    syms = [y for y in app.session.trace.symbols if y.block == 0]
    used = lambda lengths: sum(1 for x in lengths if x)
    e_len = blk.litlen.lengths[0x65]
    return dict(
        lit=f"{used(blk.litlen.lengths)} of {len(blk.litlen.lengths)} symbols "
            f"carry a code",
        dist=f"{used(blk.dist.lengths)} of {len(blk.dist.lengths)} symbols "
             f"carry a code",
        uses=f"{len(syms):,}",
        e_share=f"{100 * 2 ** -e_len:.2f}%",
        e_seen=f"({sum(1 for y in syms if y.code_sym == 0x65)})")


def stats_text(app) -> str:
    body = app.screen.query_one("#stats-body").content
    return body.plain if hasattr(body, "plain") else str(body)


@pytest.mark.asyncio
async def test_the_statistics_row_under_a_block_opens_the_screen():
    """The launch point asked for: from the expanded block in the left pane."""
    from textual.widgets import Tree
    from toeknife.tui.panes import BlocksPane, StatsLink
    app = stats_app()
    async with app.run_test(size=(150, 46)) as pilot:
        pane = app.query_one(BlocksPane)
        tree = pane.query_one(Tree)
        node = next(n for n in tree.root.children[1].children
                    if isinstance(n.data, StatsLink))
        tree.select_node(node)
        await pilot.pause()
        await pilot.pause()
        assert app.screen.__class__.__name__ == "TableStatsScreen"
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_ctrl_t_opens_the_statistics_for_the_block_under_the_cursor():
    app = stats_app()
    async with app.run_test(size=(150, 46)) as pilot:
        app.tape_view.cursor = 100
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert app.screen.__class__.__name__ == "TableStatsScreen"
        assert app.screen.block.index == 0
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_screen_shows_declared_beside_measured_and_the_heat_grid():
    app = stats_app()
    async with app.run_test(size=(150, 46)) as pilot:
        app.tape_view.cursor = 100
        await pilot.press("ctrl+t")
        await pilot.pause()
        body = stats_text(app)

        facts = first_block_facts(app)
        assert facts["lit"] in body
        assert "Kraft 1.0000" in body and "complete" in body
        assert facts["uses"] in body, "how many times the block used the table"
        # the grid, and a byte that is present beside one that is not
        assert "code length per byte value" in body
        assert " 60  " in body and "·" in body
        # both columns, on a row whose numbers are known
        assert "0x65 'e'" in body and facts["e_share"] in body
        assert facts["e_seen"] in body, "how often the block used 'e'"
        assert "where the code space went" in body
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_tab_cycles_the_three_alphabets():
    app = stats_app()
    async with app.run_test(size=(150, 46)) as pilot:
        app.tape_view.cursor = 100
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert "code length per byte value" in stats_text(app)

        await pilot.press("tab")
        await pilot.pause()
        body = stats_text(app)
        assert first_block_facts(app)["dist"] in body
        assert "distance 8193-12288" in body
        assert "code length per byte value" not in body, \
            "distance codes are not a byte space"

        await pilot.press("tab")
        await pilot.pause()
        body = stats_text(app)
        assert "15 of 19 symbols carry a code" in body
        assert "used in the header, not the body" in body

        await pilot.press("tab")
        await pilot.pause()
        assert first_block_facts(app)["lit"] in stats_text(app), \
            "wraps back to the first"
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_block_with_no_table_says_so_instead_of_opening_empty():
    import zlib
    from toeknife.core.session import Session
    from toeknife.tui.app import ToeKnifeApp
    raw = zlib.compress(bytes(range(256)) * 12, 0)[2:-4]      # stored
    app = ToeKnifeApp(Session(raw))
    async with app.run_test(size=(150, 46)) as pilot:
        app.tape_view.cursor = 10
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert app.screen.__class__.__name__ != "TableStatsScreen"
        assert "no Huffman table" in str(app.query_one("#status").content)
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_statistics_row_is_absent_from_a_stored_block():
    import zlib
    from textual.widgets import Tree
    from toeknife.core.session import Session
    from toeknife.tui.app import ToeKnifeApp
    from toeknife.tui.panes import BlocksPane, StatsLink
    raw = zlib.compress(bytes(range(256)) * 12, 0)[2:-4]
    app = ToeKnifeApp(Session(raw))
    async with app.run_test(size=(150, 46)) as pilot:
        tree = app.query_one(BlocksPane).query_one(Tree)
        links = [n for b in tree.root.children for n in b.children
                 if isinstance(n.data, StatsLink)]
        assert links == [], "nothing to summarise, so nothing to click"
        await pilot.pause()


# ---- the corpus picker -------------------------------------------------------

@pytest.mark.asyncio
async def test_the_picker_lists_this_machine_the_catalogue_and_saved_models():
    from toeknife.predict import corpora
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f5")
        await pilot.pause()
        lst = app.screen.query_one(OptionList)
        ids = [lst.get_option_at_index(i).id for i in range(lst.option_count)]
        assert "type:text-en" in ids and "type:text-en all" in ids
        assert any(i and i.startswith("dl:") for i in ids)
        assert "ask:path" in ids and "ask:url" in ids
        # every catalogue entry is offered, not just named
        for d in corpora.DOWNLOADS:
            assert f"dl:{d.key}" in ids
        await pilot.press("escape")
        await pilot.pause()


def _tiny_model(order: int = 2):
    """A real model, small enough to train in a test. Provenance included,
    because the whole point of saving is that the lineage travels."""
    from toeknife.predict.context import Bidirectional
    m = Bidirectional.train_on(b"the quick brown fox " * 40, order=order)
    m.provenance = {"type": "text-en", "files": 1, "order": order}
    return m


@pytest.mark.asyncio
async def test_saving_writes_the_trained_model_with_its_lineage(tmp_path):
    """Training was the expensive step and used to die with the process. The
    file has to be loadable and has to still know what it read."""
    from toeknife.predict.context import ContextModel
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        app.session.model = _tiny_model()
        target = tmp_path / "elsewhere" / "english-o2.tkm"
        app.action_save_model()     # f5's save row and ^k both land here
        await pilot.pause()
        app.screen.query_one(Input).value = str(target)
        await pilot.press("enter")
        await pilot.pause()

        assert target.is_file(), "the parent directory must be created"
        back = ContextModel.load(target)
        assert back.provenance == {"type": "text-en", "files": 1, "order": 2}
        assert back.forward.trained_bytes == 800
        status = app.query_one("#status").content
        assert "wrote" in str(status) and str(target) in str(status)
        # A forensic tool writing a file records that it did
        assert any(e.route == "model" and e.detail == str(target)
                   for e in app.session.ledger.entries)
        await pilot.pause()


@pytest.mark.asyncio
async def test_training_from_a_path_records_what_it_read(tmp_path):
    """The commonest route -- a path on this machine -- recorded nothing, so a
    model saved from it was a hypothesis generator of unknown parentage and got
    offered the name "corpus.tkm"."""
    from toeknife.predict import corpora
    src = tmp_path / "moby-dick.txt"
    src.write_bytes(b"call me ishmael. " * 200)
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        app.load_corpus(str(src))
        prov = app.session.model.provenance
        assert prov["path"] == str(src) and prov["files"] == 1
        assert prov["bytes"] == 3400 and prov["order"]
        assert corpora.suggest_model_name(prov).startswith("moby-dick-o")
        await pilot.pause()


@pytest.mark.asyncio
async def test_saving_with_nothing_trained_says_so_rather_than_prompting():
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        screen = app.screen
        app.action_save_model()
        await pilot.pause()
        assert app.screen is screen, "no prompt for a model that does not exist"
        assert "no model to save" in str(app.query_one("#status").content)
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_save_that_cannot_be_written_is_reported_not_raised(tmp_path):
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        app.session.model = _tiny_model()
        blocked = tmp_path / "afile"
        blocked.write_bytes(b"x")
        msg = app._save_model(str(blocked / "under" / "m.tkm"))
        assert "could not write" in msg
        assert any(e.route == "model" and e.outcome == "failed"
                   for e in app.session.ledger.entries)
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_picker_offers_to_save_what_is_loaded_and_says_where(tmp_path,
                                                                      monkeypatch):
    """The section he went looking in is where the offer belongs: nothing was
    listed there because nothing in the TUI ever wrote one."""
    from toeknife.predict import corpora
    monkeypatch.setattr(corpora, "MODEL_DIRS", (str(tmp_path / "models"),))
    monkeypatch.delenv("TOEKNIFE_MODELS", raising=False)
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f5")
        await pilot.pause()
        lst = app.screen.query_one(OptionList)
        ids = [lst.get_option_at_index(i).id for i in range(lst.option_count)]
        assert "save:" not in ids, "nothing trained yet, so nothing to write"
        rows = "\n".join(str(lst.get_option_at_index(i).prompt)
                         for i in range(lst.option_count))
        # The empty section used to name only the directories that exist, which
        # told the operator where it looked and never where to put one.
        assert str(corpora.model_dir()) in rows
        await pilot.press("escape")
        await pilot.pause()

        app.session.model = _tiny_model()
        await pilot.press("f5")
        await pilot.pause()
        screen = app.screen
        lst = screen.query_one(OptionList)
        lst.highlighted = option_index(screen, "save:")
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is screen, "the first enter must not write"
        detail = str(screen.query_one("#corpus-detail").content)
        assert "text-en-o2.tkm" in detail and "800 training bytes" in detail
        assert not (tmp_path / "models").exists()

        await pilot.press("enter")
        await pilot.pause()
        # Committing hands over to the same editable prompt f9 uses
        app.screen.query_one(Input).value = str(tmp_path / "models" / "m.tkm")
        await pilot.press("enter")
        await pilot.pause()
        assert (tmp_path / "models" / "m.tkm").is_file()
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_saved_model_is_then_listed_by_the_picker(tmp_path, monkeypatch):
    """The round trip that was broken: save, and find it where you saved it."""
    from toeknife.predict import corpora
    monkeypatch.setattr(corpora, "MODEL_DIRS", (str(tmp_path),))
    monkeypatch.delenv("TOEKNIFE_MODELS", raising=False)
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        app.session.model = _tiny_model()
        app._save_model(str(corpora.suggest_model_path(
            app.session.model.provenance)))
        await pilot.press("f5")
        await pilot.pause()
        lst = app.screen.query_one(OptionList)
        ids = [lst.get_option_at_index(i).id for i in range(lst.option_count)]
        saved = [i for i in ids if i and i.startswith("saved:")]
        assert len(saved) == 1 and saved[0].endswith("text-en-o2.tkm")
        rows = "\n".join(str(lst.get_option_at_index(i).prompt)
                         for i in range(lst.option_count))
        assert "type=text-en" in rows, "the lineage travels with it"
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_training_offers_to_save_before_it_hands_back_the_keyboard(
        tmp_path, monkeypatch):
    """The whole point of the change: f9 exists, but a model is only worth
    saving if you remember to before the process ends. The moment the cost has
    just been paid is the only moment the question answers itself."""
    from toeknife.predict import corpora
    from toeknife.tui.modals import PromptModal
    monkeypatch.setattr(corpora, "MODEL_DIRS", (str(tmp_path / "models"),))
    monkeypatch.delenv("TOEKNIFE_MODELS", raising=False)
    src = tmp_path / "prose.txt"
    src.write_bytes(b"forensic analysis of compressed evidence. " * 400)

    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        app.load_corpus_slow(str(src))
        await settle(app, pilot)

        assert isinstance(app.screen, PromptModal), \
            "the offer must arrive on its own, not wait to be remembered"
        assert app.session.model is not None, "trained before it asks"
        prompt = app.screen
        assert "keep this model" in str(prompt.prompt)
        target = Path(prompt.query_one(Input).value)
        # Named from what it read and the order it earned; which order that is
        # belongs to the tier tests, not this one.
        assert target.name.startswith("prose-o") and target.suffix == ".tkm"
        assert target.parent == tmp_path / "models"

        await pilot.press("enter")
        await pilot.pause()
        assert target.is_file()
        assert "wrote" in str(app.query_one("#status").content)
        await pilot.pause()


@pytest.mark.asyncio
async def test_skipping_the_offer_keeps_the_model_and_says_how_to_save_it_later(
        tmp_path, monkeypatch):
    from toeknife.predict import corpora
    monkeypatch.setattr(corpora, "MODEL_DIRS", (str(tmp_path / "models"),))
    monkeypatch.delenv("TOEKNIFE_MODELS", raising=False)
    src = tmp_path / "prose.txt"
    src.write_bytes(b"forensic analysis of compressed evidence. " * 400)

    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        app.load_corpus_slow(str(src))
        await settle(app, pilot)
        await pilot.press("escape")
        await pilot.pause()

        assert app.session.model is not None, "skipping saving is not discarding"
        assert not (tmp_path / "models").exists()
        assert "f5" in str(app.query_one("#status").content)
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_slow_run_shows_progress_and_keeps_answering_the_keyboard():
    """The failure this replaces: the work ran in the handler, so the status it
    set never painted and the app ignored keys until it was done."""
    import threading
    from toeknife.tui.progress import ProgressScreen
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        release = threading.Event()
        reports = []

        def work(report):
            report(1, 4, "first quarter")
            reports.append("reported")
            release.wait(10)
            report(4, 4, "done")
            return "finished"

        out = []
        app.run_slow("a slow thing", work, lambda r, e: out.append((r, e)))
        await pilot.pause()
        await pilot.pause()

        assert isinstance(app.screen, ProgressScreen)
        assert reports == ["reported"], "it is running, not waiting to start"
        # The event loop is still turning: the clock ticks and keys arrive.
        bar = app.screen.query_one("#progress-bar")
        assert bar.total == 4 and bar.progress == 1
        assert "first quarter" in str(app.screen.query_one("#progress-note").content)

        release.set()
        await settle(app, pilot)
        assert out == [("finished", None)]
        assert not isinstance(app.screen, ProgressScreen), "the screen comes down"


@pytest.mark.asyncio
async def test_escape_cancels_a_slow_run_and_nothing_is_committed():
    import threading
    from toeknife.tui.progress import Cancelled, ProgressScreen
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        out = []
        gate = threading.Event()
        # The worker sits at the gate while the keypress is delivered, so this
        # tests the cancel rather than which of the two won a race.
        def work(report):
            report(0, 2, "before the gate")
            gate.wait(10)
            report(1, 2, "after the gate")
            return "should not get here"

        app.run_slow("cancel me", work, lambda r, e: out.append((r, e)))
        await pilot.pause()
        await pilot.pause()
        assert isinstance(app.screen, ProgressScreen)
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen.cancelled, "escape must reach the progress screen"
        assert isinstance(app.screen, ProgressScreen), \
            "the screen stays up until the work actually stops"
        gate.set()
        await settle(app, pilot)

        assert len(out) == 1
        result, error = out[0]
        assert result is None and isinstance(error, Cancelled)


@pytest.mark.asyncio
async def test_cancelling_a_train_leaves_the_previous_model_alone(tmp_path):
    """Nothing is assigned until a run finishes, so an abandoned train is not a
    half-trained model."""
    from toeknife.predict.context import ContextModel
    src = tmp_path / "prose.txt"
    src.write_bytes(b"forensic analysis of compressed evidence. " * 4000)
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        before = ContextModel(order=2).train(b"an earlier model " * 40)
        app.session.model = before
        app.load_corpus_slow(str(src))
        # Cancelled before the worker has run at all, which is the deterministic
        # version of pressing escape early: the first report raises.
        app.screen_stack[-1].cancelled = True
        await settle(app, pilot)

        assert app.session.model is before
        assert "cancelled" in str(app.query_one("#status").content)
        assert any(e.outcome == "skipped" for e in app.session.ledger.entries)
        await pilot.pause()


@pytest.mark.asyncio
async def test_f3_scans_on_a_thread_and_then_offers_the_boundaries():
    """The scan is 4.6 seconds on a 95 KB capture, and it used to hold the event
    loop for all of it."""
    import zlib
    from toeknife.core.session import Session
    from toeknife.tui.app import ToeKnifeApp

    raw = b"the quick brown fox jumps over the lazy dog. " * 200
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9, zlib.Z_FIXED)
    out = bytearray()
    for i in range(0, len(raw), 2048):
        out += co.compress(raw[i:i + 2048])
        out += co.flush(zlib.Z_FULL_FLUSH)
    out += co.flush()
    frag = bytes(out)[len(bytes(out)) // 3:]

    app = ToeKnifeApp(Session(frag))
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f3")
        await settle(app, pilot)
        assert app.screen.__class__.__name__ == "ResyncPicker"
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_picker_shows_what_a_row_would_do_before_it_does_it():
    """Nothing is read, fetched or loaded on the first enter: the whole point
    of the screen is that a fetch is something the operator caused."""
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f5")
        await pilot.pause()
        screen = app.screen
        screen.query_one(OptionList).highlighted = option_index(
            screen, "dl:gutenberg-de-verwandlung")
        await pilot.press("enter")
        await pilot.pause()

        assert app.screen is screen, "the first enter must not commit"
        detail = str(screen.query_one("#corpus-detail").content)
        assert "gutenberg.org" in detail and "KB" in detail
        assert "Nothing has been sent yet" in detail or "cache" in detail
        assert app.session.model is None
        await pilot.pause()


@pytest.mark.asyncio
async def test_arming_a_scan_row_scans_off_the_event_loop(tmp_path, monkeypatch):
    """The sweep row reads directory listings until it runs out of filesystem.
    It used to run in the key handler, where the "scanning…" it wrote could not
    paint -- and it ran the scan twice, once for the plan and once for the size
    hint."""
    from toeknife.predict import corpora
    from toeknife.tui.progress import ProgressScreen
    src = tmp_path / "prose.txt"
    src.write_bytes(b"prose to train on. " * 200)
    calls = []

    def fake_plan(key, roots, whole):
        calls.append((key, whole))
        return corpora.Plan(corpora.BY_KEY[key], [tmp_path], [src], whole)

    monkeypatch.setattr(corpora, "plan", fake_plan)

    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f5")
        await pilot.pause()
        screen = app.screen
        screen.query_one(OptionList).highlighted = option_index(
            screen, "type:text-en all")
        await pilot.press("enter")
        await pilot.pause()
        # The progress screen goes up over the picker, and comes back down to it.
        assert isinstance(app.screen, (ProgressScreen, type(screen)))
        await settle(app, pilot)

        assert app.screen is screen, "back on the picker, not committed"
        assert calls == [("text-en", True)], f"scanned {len(calls)} times"
        detail = str(screen.query_one("#corpus-detail").content)
        assert "found 1 file(s)" in detail and "context order" in detail
        assert screen.armed == "type:text-en all"

        await pilot.press("enter")
        await settle(app, pilot)
        assert app.session.model is not None
        assert app.session.model.provenance["type"] == "text-en"
        await skip_save_offer(app, pilot)


@pytest.mark.asyncio
async def test_escape_backs_out_of_an_armed_row_rather_than_closing():
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f5")
        await pilot.pause()
        screen = app.screen
        screen.query_one(OptionList).highlighted = option_index(
            screen, "dl:zlib-inflate-c")
        await pilot.press("enter")
        await pilot.pause()
        assert screen.armed == "dl:zlib-inflate-c"
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is screen and screen.armed is None
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_second_enter_fetches_and_trains_on_what_came_back(monkeypatch,
                                                                  tmp_path):
    """The commit path, with the network stubbed: what matters here is that a
    fetched blob becomes the model and the fetch is written into the ledger."""
    from toeknife.predict import corpora
    prose = b"forensic analysis of compressed evidence, repeated often. " * 60
    local = tmp_path / "fetched.txt"
    local.write_bytes(prose)
    monkeypatch.setattr(corpora, "fetch",
                        lambda url, *a, **k: (local, len(prose), False))

    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f5")
        await pilot.pause()
        screen = app.screen
        screen.query_one(OptionList).highlighted = option_index(
            screen, "dl:gutenberg-en-pride")
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("enter")
        await settle(app, pilot)

        assert app.session.model is not None
        assert app.session.model.trained_bytes > 0
        prov = app.session.model.provenance
        assert prov["download"] == "gutenberg-en-pride"
        assert prov["url"].startswith("https://")
        routes = [e.route for e in app.session.ledger.entries]
        assert "fetch" in routes and "corpus" in routes
        await skip_save_offer(app, pilot)
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_url_typed_by_hand_is_still_confirmed_before_it_is_fetched(
        monkeypatch, tmp_path):
    from toeknife.predict import corpora
    prose = b"some prose fetched from somewhere else entirely. " * 60
    local = tmp_path / "url.txt"
    local.write_bytes(prose)
    calls = []

    def fake(url, *a, **k):
        calls.append(url)
        return local, len(prose), False

    monkeypatch.setattr(corpora, "fetch", fake)
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f5")
        await pilot.pause()
        screen = app.screen
        screen.query_one(OptionList).highlighted = option_index(screen, "ask:url")
        await pilot.press("enter")
        await pilot.pause()
        field = screen.query_one(Input)
        assert field.display
        field.value = "https://example.invalid/corpus.txt"
        await pilot.press("enter")
        await pilot.pause()
        assert not calls, "typing a URL must not fetch it"

        await pilot.press("enter")          # the confirm
        await pilot.pause()
        assert calls == ["https://example.invalid/corpus.txt"]
        assert app.session.model is not None
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_saved_model_is_discovered_and_loaded_from_the_picker(
        tmp_path, monkeypatch):
    """Models land wherever --save-model was pointed. Making the operator
    remember that path was the thing wrong with the old screen."""
    from toeknife.predict import corpora
    from toeknife.predict.context import ContextModel
    model = ContextModel(order=3).train(b"forensic prose about bytes. " * 80)
    path = tmp_path / "english.tkm"
    model.save(path, {"corpus": "docs"})
    monkeypatch.setenv("TOEKNIFE_MODELS", str(tmp_path))
    # f9 writes real models into the canonical directory, so counting rows means
    # excluding whatever this machine has actually saved.
    monkeypatch.setattr(corpora, "MODEL_DIRS", ())

    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f5")
        await pilot.pause()
        screen = app.screen
        lst = screen.query_one(OptionList)
        rows = [lst.get_option_at_index(i) for i in range(lst.option_count)]
        saved = [r for r in rows if r.id and r.id.startswith("saved:")]
        assert len(saved) == 1 and "english.tkm" in str(saved[0].prompt)
        assert "order 3" in str(saved[0].prompt)

        lst.highlighted = option_index(screen, f"saved:{path}")
        await pilot.press("enter")
        await pilot.pause()
        assert "corpus=docs" in str(screen.query_one("#corpus-detail").content)
        await pilot.press("enter")
        await pilot.pause()

        assert app.session.model is not None
        assert app.session.model.order == 3
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_picker_offers_prose_that_is_already_to_hand(tmp_path,
                                                              monkeypatch):
    """The corpus that fits a capture is usually in the project it came from,
    and that was the one thing the old screen made you type from memory."""
    docs = tmp_path / "docs"
    docs.mkdir()
    for i in range(4):
        (docs / f"note{i}.md").write_text("forensic prose about bytes. " * 60)
    monkeypatch.chdir(tmp_path)

    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f5")
        await pilot.pause()
        screen = app.screen
        row = option_index(screen, f"path:{docs}")
        lst = screen.query_one(OptionList)
        assert "here" in str(lst.get_option_at_index(row).prompt)
        assert "4 file(s)" in str(lst.get_option_at_index(row).prompt)

        lst.highlighted = row
        await pilot.press("enter")            # look
        await pilot.pause()
        assert app.session.model is None
        assert "4 file(s)" in str(screen.query_one("#corpus-detail").content)
        await pilot.press("enter")            # commit
        await pilot.pause()
        assert app.session.model is not None
        assert app.session.model.trained_bytes > 0
        await pilot.pause()


@pytest.mark.asyncio
async def test_each_corpus_gets_the_order_its_size_earns(tmp_path, monkeypatch):
    """Order is a per-corpus property, not a session-wide setting: swapping the
    corpus in f5 has to re-decide it."""
    small = tmp_path / "small"
    small.mkdir()
    (small / "a.md").write_text("forensic prose. " * 200)          # ~3 KB
    big = tmp_path / "big"
    big.mkdir()
    for i in range(40):
        (big / f"{i}.md").write_text("forensic prose about bytes. " * 400)  # ~450 KB

    app = make_app()
    async with app.run_test(size=(140, 30)) as pilot:
        assert "order-3" in app.load_corpus(str(small))
        assert "order-5" in app.load_corpus(str(big))
        # and an operator who names an order keeps it, whatever the size
        app.order = 6
        assert "order-6" in app.load_corpus(str(small))
        await pilot.pause()


# ---- the route board, as a launcher ------------------------------------------

def rotted_app():
    """A capture whose header has one bit wrong: the case f7's repair route is
    for, and the only state in which it offers itself."""
    import zlib
    from toeknife.core import bitrot
    from toeknife.core.session import Session
    from toeknife.tui.app import ToeKnifeApp
    body = b"forensic analysis of compressed evidence, at some length. " * 90
    co = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    comp = co.compress(body) + co.flush()
    lo, _hi = bitrot.header_span(Session(comp, mode="stream"))
    damaged = bitrot.flip(comp, [lo + 61])
    return ToeKnifeApp(Session(damaged, mode="stream")), comp, body


def route_row(screen, key):
    lst = screen.query_one(OptionList)
    for i in range(lst.option_count):
        if lst.get_option_at_index(i).id == f"route:{key}":
            return i, lst.get_option_at_index(i)
    raise AssertionError(f"no row for route {key!r}")


@pytest.mark.asyncio
async def test_the_board_opens_with_a_runnable_row_under_the_cursor():
    """It used to open with nothing highlighted, so enter did nothing at all --
    which reads as the screen being broken rather than as an invitation to press
    an arrow key first."""
    app, _comp, _body = rotted_app()
    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.press("f7")
        await pilot.pause()
        lst = app.screen.query_one(OptionList)
        assert lst.highlighted is not None
        assert not lst.get_option_at_index(lst.highlighted).disabled
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_route_that_needs_evidence_is_offered_rather_than_refused():
    """"needs a second capture" is not a dead end, it is a prompt waiting to be
    asked. Both of these were reachable only from the command line before."""
    app, _comp, _body = rotted_app()
    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.press("f7")
        await pilot.pause()
        for key in ("related", "kp-judge"):
            i, opt = route_row(app.screen, key)
            assert not opt.disabled, f"{key} should be selectable"
            assert "ask" in str(opt.prompt)
            app.screen.query_one(OptionList).highlighted = i
            await pilot.pause()
            detail = str(app.screen.query_one("#routes-detail").content)
            assert "enter asks for what it needs" in detail
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_an_unwired_research_route_is_listed_but_not_offered():
    """Visible, so it cannot be confused with a route that does not apply; not
    selectable, so the board never offers something it cannot start."""
    app, _comp, _body = rotted_app()
    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.press("f7")
        await pilot.pause()
        i, opt = route_row(app.screen, "dict-head")
        assert opt.disabled
        assert "unshipped" in str(opt.prompt)
        # The reason sits on the rows beneath it, not in the detail pane: a
        # disabled row cannot take the highlight, so a reason kept in the pane
        # would be unreachable.
        lst = app.screen.query_one(OptionList)
        after = " ".join(str(lst.get_option_at_index(j).prompt)
                         for j in range(i + 1, min(i + 4, lst.option_count)))
        assert "not in this release" in after
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_selecting_the_repair_route_runs_it_and_applying_restores_the_bytes():
    """The whole point of the change: a route listed as available can be started
    from the board, and what it produces can be acted on."""
    from toeknife.tui.modals import BitRotPicker
    app, comp, body = rotted_app()
    async with app.run_test(size=(150, 46)) as pilot:
        assert app.session.trace.status == "bad-table"
        await pilot.press("f7")
        await pilot.pause()
        i, _opt = route_row(app.screen, "bitrot")
        app.screen.query_one(OptionList).highlighted = i
        await pilot.press("enter")
        await settle(app, pilot)

        assert isinstance(app.screen, BitRotPicker)
        summary = str(app.screen.query_one("#routes-detail").content)
        assert "1-bit flips" in summary
        assert app.session.data != comp, "nothing applied yet"

        await pilot.press("enter")
        await pilot.pause()
        assert app.session.data == comp, "the capture should be repaired"
        assert app.session.trace.status == "end"
        assert bytes(app.session.tape.out[:len(body)]) == body
        status = str(app.query_one("#status").content)
        assert "f8 says whether it is the original" in status
        assert any(e.route == "bitrot" for e in app.session.ledger.entries)


@pytest.mark.asyncio
async def test_leaving_the_repair_alone_changes_nothing():
    app, _comp, _body = rotted_app()
    async with app.run_test(size=(150, 46)) as pilot:
        before = app.session.data
        await pilot.press("f7")
        await pilot.pause()
        i, _opt = route_row(app.screen, "bitrot")
        app.screen.query_one(OptionList).highlighted = i
        await pilot.press("enter")
        await settle(app, pilot)
        await pilot.press("escape")
        await pilot.pause()
        assert app.session.data is before
        assert "nothing applied" in str(app.query_one("#status").content)


@pytest.mark.asyncio
async def test_the_repair_route_is_not_offered_on_a_capture_that_decodes():
    app = solving_app()
    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.press("f7")
        await pilot.pause()
        _i, opt = route_row(app.screen, "bitrot")
        assert opt.disabled
        await pilot.press("escape")
        await pilot.pause()


# ---- table cells are data, not markup ----------------------------------------

@pytest.mark.asyncio
async def test_a_proposed_slash_does_not_crash_the_solver_panel():
    """Reported from the field on a pcap: f4 died with
    `MarkupError: closing tag '[/]' has nothing to close`.

    The context column wraps the proposed byte in brackets, and Textual runs a
    string cell through Rich's markup parser, where `[/]` is a closing tag.
    """
    from toeknife.tui.modals import SolverScreen
    app = solving_app()
    async with app.run_test(size=(150, 36)) as pilot:
        pos = app.session.tape.holes()[0].anchor
        screen = SolverScreen(app.session, app.solver(), 0.9)
        app.push_screen(screen)
        await pilot.pause()
        # Force the exact shape that crashed: a proposal whose byte is '/'.
        from toeknife.predict.solver import Proposal
        screen.proposals = [Proposal(pos, [pos], ord("/"), -1.0, 0.5,
                                     ((ord("b"), -2.0),))]
        screen._draw()
        await pilot.pause()
        await pilot.pause()          # the crash happened on idle, not on draw
        table = screen.query_one(DataTable)
        assert table.row_count == 1
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_proposed_letter_is_not_eaten_as_a_style_tag():
    """The quieter half of the same bug. Rich reads `[b]` as bold, so the panel
    silently dropped the proposed byte for every single letter -- showing an
    unbroken run of block characters where the suggestion should have been."""
    from toeknife.predict.solver import Proposal
    from toeknife.tui.modals import SolverScreen
    app = solving_app()
    async with app.run_test(size=(150, 36)) as pilot:
        pos = app.session.tape.holes()[0].anchor
        screen = SolverScreen(app.session, app.solver(), 0.9)
        app.push_screen(screen)
        await pilot.pause()
        for ch in "biudsrx/":
            screen.proposals = [Proposal(pos, [pos], ord(ch), -1.0, 0.5, ())]
            screen._draw()
            await pilot.pause()
            table = screen.query_one(DataTable)
            context = table.get_row_at(0)[-1]
            shown = context.plain if hasattr(context, "plain") else str(context)
            assert f"[{ch}]" in shown, f"{ch!r} vanished from the preview: {shown!r}"
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_recovered_preview_containing_brackets_survives():
    """The same trap in the alignment and resync pickers, whose preview column is
    decoded text and can contain anything at all."""
    from toeknife.tui.modals import _cells
    rows = _cells("plain", "[/]", "[b]bold?", "░░[x]░░", "100%")
    assert [r.plain for r in rows] == ["plain", "[/]", "[b]bold?", "░░[x]░░",
                                       "100%"]


@pytest.mark.asyncio
async def test_the_solver_panel_previews_a_readable_occurrence():
    """Reported from a captured stream: every row of f4 read as an unbroken run
    of block characters. The panel was previewing each class's FIRST occurrence,
    which on a recovery sits inside the lost region where every neighbour is also
    unknown. The back-reference copies are the judgeable ones."""
    from toeknife.tui.modals import SolverScreen
    app = solving_app()
    async with app.run_test(size=(160, 36)) as pilot:
        screen = SolverScreen(app.session, app.solver(), 0.9)
        app.push_screen(screen)
        await pilot.pause()
        proposals = [p for p in screen.proposals if p.size > 1]
        if not proposals:
            pytest.skip("this fragment has no multi-position class")
        from toeknife.predict.solver import most_readable, readable_neighbours
        tape = app.session.tape
        for p in proposals:
            shown = most_readable(tape, p.positions)
            best = max(readable_neighbours(tape, q) for q in p.positions)
            assert readable_neighbours(tape, shown) == best, \
                "the preview must use the most readable occurrence"
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_panel_says_which_occurrence_the_context_is_from():
    """Otherwise the context column and the "at" column disagree silently: one
    is the class anchor, the other a different position entirely."""
    from textual.widgets import Label
    from toeknife.tui.modals import SolverScreen
    app = solving_app()
    async with app.run_test(size=(160, 36)) as pilot:
        screen = SolverScreen(app.session, app.solver(), 0.9)
        app.push_screen(screen)
        await pilot.pause()
        title = screen.query_one("#solve-title", Label).content
        text = getattr(title, "plain", str(title))
        assert "most readable occurrence" in text
        await pilot.press("escape")
        await pilot.pause()


# ---- training screen and progress ------------------------------------------

def _fake_run(tmp_path, monkeypatch, log, **spec):
    """A run directory as a later session would find it: files, no memory."""
    import json
    import time
    from toeknife.predict import training
    monkeypatch.setenv("TOEKNIFE_TRAINING", str(tmp_path / "training"))
    d = training.runs_dir() / "r"
    d.mkdir(parents=True, exist_ok=True)
    body = dict(name="r", command=["/bin/true"], started=time.time(),
                marker="r", pid=os.getpid())
    body.update(spec)
    (d / training.SPEC).write_text(json.dumps(body))
    (d / training.LOG).write_text(log)
    return d


REPORT_ACTIVE = ("  600:00    50,000,000       1.600     1.900     50.0%   "
                 "lr 1.0e-04  25.0%  active   300:00")


def test_the_status_line_shows_a_run_that_is_going(tmp_path, monkeypatch):
    """A four-day run gets a place on the status line rather than only existing
    behind a keystroke."""
    _fake_run(tmp_path, monkeypatch, REPORT_ACTIVE, marker="python")
    app = ToeKnifeApp(Session(FRAGMENT, mode="fragment"))
    token = app.training_token()
    assert token is not None
    assert "25%" in token[0] and "left" in token[0]


def test_the_eta_ignores_time_the_machine_spent_asleep(tmp_path, monkeypatch):
    """600 minutes of wall clock, 300 of running, a quarter done. The answer is
    15 hours of running left, not 30 -- the machine was suspended for half of
    it and will not be suspended for the remainder on demand."""
    d = _fake_run(tmp_path, monkeypatch, REPORT_ACTIVE, marker="python")
    from toeknife.predict import training
    r = training.load(d)
    p = r.progress()
    assert p.elapsed_s == 600 * 60 and p.active_s == 300 * 60
    assert r.eta_seconds() == pytest.approx(300 * 60 * 3)


def test_an_older_run_without_the_active_clock_falls_back_to_wall_time(
        tmp_path, monkeypatch):
    old = "  600:00    50,000,000       1.600     1.900     50.0%   lr 1.0e-04  25.0%"
    d = _fake_run(tmp_path, monkeypatch, old, marker="python")
    from toeknife.predict import training
    p = training.load(d).progress()
    assert p.active_s is None and p.working_s == 600 * 60


def test_no_run_means_no_badge(tmp_path, monkeypatch):
    monkeypatch.setenv("TOEKNIFE_TRAINING", str(tmp_path / "nothing"))
    app = ToeKnifeApp(Session(FRAGMENT, mode="fragment"))
    assert app.training_token() is None


def test_training_refuses_to_start_without_a_corpus(tmp_path, monkeypatch):
    """Training on the file being recovered makes every number afterwards
    fiction, and nothing here can know the capture's source, so it refuses
    rather than guessing one."""
    monkeypatch.setenv("TOEKNIFE_TRAINING", str(tmp_path / "training"))
    app = ToeKnifeApp(Session(FRAGMENT, mode="fragment"))
    msg = app._start_training(dict(d_model=256, layers=6, heads=4,
                                   window=129, steps=10))
    assert "no corpus chosen" in msg


def test_the_training_screen_does_not_shadow_a_textual_property():
    """`size` and `window` are both properties on a Textual screen, and
    shadowing one fails at render time rather than at construction."""
    from toeknife.tui.modals import TrainingScreen
    s = TrainingScreen()
    assert s.pick_size == 2 and s.pick_window == 0
    assert TrainingScreen.SIZES[s.pick_size][3] == "19.2M"


@pytest.mark.asyncio
async def test_no_binding_collides_with_a_terminal_alias():
    """ctrl+m is carriage return, and three others are just as invisible.

    The fill mode was bound to ctrl+m for a release. A terminal sends CR for
    it and textual resolves that to `enter`, so the enter binding -- follow
    match -- ran instead, the status line answered "cursor is not on a
    back-reference", and the mode silently never changed. Nothing in the app
    could report this, because from its side the wrong action simply ran.
    """
    from textual.keys import KEY_ALIASES

    aliased = {alias for aliases in KEY_ALIASES.values() for alias in aliases}
    # Anchored to textual's own table rather than to a list copied out of it:
    # if a release stops folding ctrl+m into enter, this stops asserting a
    # rule that no longer holds.
    assert "ctrl+m" in aliased and "ctrl+i" in aliased
    # ctrl+j and ctrl+h are the same trap one layer down -- the terminal sends
    # newline and backspace for them -- and textual lists neither as an alias,
    # so they are ours to remember.
    bound = {b.key for b in ToeKnifeApp.BINDINGS}
    clashes = bound & (aliased | set(ToeKnifeApp.TERMINAL_ALIASES))
    assert not clashes, (f"{clashes} cannot be told apart from the key it "
                         f"aliases; bind something else")
    # And the same shadowing one layer up: the command palette took ctrl+p,
    # so "previous back-reference" opened a search screen for a release. Its
    # key may be bound here only to open the palette, so the footer can show it.
    if ToeKnifeApp.ENABLE_COMMAND_PALETTE:
        assert {b.action for b in ToeKnifeApp.BINDINGS
                if b.key == ToeKnifeApp.COMMAND_PALETTE_BINDING} <= \
            {"command_palette"}


@pytest.mark.asyncio
async def test_ctrl_p_reaches_the_tape_and_not_the_command_palette():
    from textual.command import CommandPalette

    app = make_app()
    async with app.run_test(size=(200, 40)) as pilot:
        await pilot.pause()
        await pilot.press("ctrl+p")
        await pilot.pause()
        assert not isinstance(app.screen, CommandPalette)
        assert "palette" not in footer_keys(app)


@pytest.mark.asyncio
async def test_fill_mode_cycles_and_says_which():
    app = make_app()
    async with app.run_test() as pilot:
        app.action_fill_mode()
        await pilot.pause()
        # No byte model: it says so rather than cycling to a mode that would
        # not be used.
        assert "no byte model loaded" in str(app.query_one("#status").content)
        assert not hasattr(app, "_fill_mode")

        app._filler = FakeFiller()
        app.action_fill_mode()
        await pilot.pause()
        assert app._fill_mode == "slm"
        status = str(app.query_one("#status").content)
        assert "byte model alone" in status and "fill: slm" in status
        app.action_fill_mode()
        await pilot.pause()
        assert app._fill_mode == "ngram"


@pytest.mark.asyncio
async def test_f5_holds_the_fill_mode_and_byte_model_training():
    """f10 and ctrl+o went when the footer ran out of room. Their rows on the
    model screen take one enter, since neither reads or fetches anything."""
    from toeknife.tui.modals import CorpusScreen
    app = make_app()
    async with app.run_test(size=(160, 50)) as pilot:
        app._filler = FakeFiller()
        await pilot.press("f5")
        await pilot.pause()
        assert isinstance(app.screen, CorpusScreen)
        lst = app.screen.query_one(OptionList)
        assert option_index(app.screen, "train:") >= 0
        lst.highlighted = option_index(app.screen, "mode:ngram")
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, CorpusScreen)
        assert app._fill_mode == "ngram"


class FakeFiller:
    """Enough of a ClassFiller for the keys that only ask it questions.

    The real one is 77 MB of tensors and a torch import; nothing here needs it
    to predict anything, only to exist and to be able to price a fill.
    """

    def describe(self):
        return "byte model, test stub"

    def rate(self):
        return 72.0


def footer_keys(app):
    """What the footer is actually drawing, by key."""
    from toeknife.tui.app import WorkflowFooter
    return {w.key for w in app.query_one(WorkflowFooter).query("FooterKey")}


@pytest.mark.asyncio
async def test_the_footer_follows_the_stage_of_the_recovery():
    """At 140 columns the full row ran off the edge before f8. The footer now
    draws the keys for where the recovery is, plus help, menu, what next and
    quit -- and the stage with no model is the one that has to show f5."""
    app = make_app()
    async with app.run_test(size=(200, 40)) as pilot:
        await pilot.pause()
        assert app.stage() == "no model"
        keys = footer_keys(app)
        assert {"f1", "ctrl+k", "f7", "ctrl+q", "f5", "ctrl+f", "f8"} <= keys
        assert not {"ctrl+z", "ctrl+r", "f4", "ctrl+e", "ctrl+w"} & keys
        # A fragment asks where it starts.
        assert "f2" in keys

        pos = app.session.tape.holes()[0].anchor
        app.tape_view.goto(pos)
        await pilot.press("Q")
        await pilot.pause()
        assert "ctrl+z" in footer_keys(app)
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert "ctrl+r" in footer_keys(app)

        app._filler = FakeFiller()
        app.update_status()
        await pilot.pause()
        assert app.stage() == "filling"
        assert {"ctrl+f", "f4", "ctrl+e", "f8"} <= footer_keys(app)
        assert "f5" not in footer_keys(app)


@pytest.mark.asyncio
async def test_ctrl_k_lists_every_command_with_its_key():
    """The menu is where everything the footer does not draw lives, so every
    main-screen action a key reaches must be in it."""
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        listed = {a for _g, _t, a, _k in app.COMMANDS_MENU}
        bound = {b.action for b in app.BINDINGS} - {"command_palette",
                                                    "focus_tape", "follow"}
        assert bound <= listed, f"not in the menu: {bound - listed}"
        # The removed keys' verbs are still reachable.
        assert {"train", "save_model", "fill_mode"} <= listed
        titles = [c.title for c in app.get_system_commands(app.screen)]
        assert any("(f5)" in t for t in titles)
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_threshold_is_visible_and_settable_without_f4():
    """It was settable only on f4, behind two undrawn keys, after scoring every
    class -- minutes with a byte model loaded."""
    from toeknife.tui.modals import ThresholdPicker
    app = solving_app()
    async with app.run_test(size=(160, 40)) as pilot:
        assert "floor 90%" in status_of(app)
        app.action_set_threshold()
        await pilot.pause()
        assert isinstance(app.screen, ThresholdPicker)
        lst = app.screen.query_one(OptionList)
        lst.highlighted = [o.id for o in
                           (lst.get_option_at_index(i)
                            for i in range(lst.option_count))].index("0.7")
        await pilot.press("enter")
        await pilot.pause()
        assert app.solve_confidence == 0.7
        assert app.solver().min_confidence == 0.7, "^f must use it"
        assert "floor 70%" in status_of(app)
        titles = [c.title for c in app.get_system_commands(app.screen)]
        assert any("confidence threshold" in t for t in titles)


@pytest.mark.asyncio
async def test_a_model_fill_says_how_it_got_there():
    """Magenta only says a model suggested it. The pane under the cursor says
    which model, at what confidence, beside what that band measured -- the
    number alone is not a probability."""
    from toeknife.core.tape import SOLVED
    from toeknife.tui.panes import BitsPane
    app = solving_app()
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.press("ctrl+f")
        await settle(app, pilot)
        pos = next(p for p, r in app.session._records.items()
                   if r.provenance == SOLVED)
        app.tape_view.goto(pos)
        app.update_inspector(pos)
        await pilot.pause()
        body = app.query_one(BitsPane)._body().content
        text = getattr(body, "plain", str(body))
        assert "filled by" in text and "the n-gram" in text
        assert "confidence" in text and "measured phrack carve" in text


@pytest.mark.asyncio
async def test_a_long_run_stop_offers_to_continue():
    from toeknife.predict.solver import SolveResult
    from toeknife.tui.modals import LongRunStop
    app = make_app()
    async with app.run_test(size=(160, 40)) as pilot:
        held = SolveResult(held=3, stopped="held")
        app._offer_long_runs(held, "", "")
        await pilot.pause()
        assert isinstance(app.screen, LongRunStop)
        asked = []
        app._solve_pass = lambda retract, long_runs=False: asked.append(long_runs)
        await pilot.press("enter")
        await pilot.pause()
        assert asked == [True], "continue anyway re-runs ^f with long runs"


@pytest.mark.asyncio
async def test_what_next_puts_the_model_first_when_there_is_none():
    from toeknife.tui.modals import RouteBoard
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press("f7")
        await pilot.pause()
        assert isinstance(app.screen, RouteBoard)
        first = app.screen.next_steps()
        assert [a for a, _l, _w in first if a][:2] == ["align", "corpus"]
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_hidden_footer_row_still_answers_its_key():
    """Hiding the row must not silence the key.

    check_action was the obvious way to trim the footer and it does both: f9
    with nothing trained stopped saying "no model to save -- f5 trains one
    first" and did nothing at all. Every key in this app explains itself when
    it cannot act; the footer only decides what to draw.
    """
    app = make_app()
    async with app.run_test(size=(200, 40)) as pilot:
        await pilot.pause()
        assert "f4" not in footer_keys(app)
        await pilot.press("f4")
        await pilot.pause()
        from toeknife.tui.modals import SolverScreen
        await settle(app, pilot)
        assert isinstance(app.screen, SolverScreen)
        await pilot.press("escape")
        await pilot.pause()
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert "nothing to undo" in str(app.query_one("#status").content)


@pytest.mark.asyncio
async def test_rounds_stack_under_the_overall_bar_and_old_ones_collapse():
    """^f cannot know its round count in advance, so each round gets its own
    bar under the overall one, a finished round stays full, and past
    ROUNDS_SHOWN the oldest collapse into one line."""
    from toeknife.tui.progress import ProgressScreen
    app = make_app()
    async with app.run_test(size=(140, 50)) as pilot:
        scr = ProgressScreen("filling")
        app.push_screen(scr)
        await pilot.pause()
        for n in range(1, 12):
            scr.set_progress(n * 10, 200, f"round {n}", stage=(n, 50, 50))
        scr.set_progress(110, 200, "round 12", stage=(12, 5, 40))
        await pilot.pause()
        body = scr.query_one("#progress-rounds").content
        text = getattr(body, "plain", str(body))
        assert text.startswith("overall")
        assert "rounds 1–4 done" in text
        assert "round 11" in text and "50/50  ✓" in text
        assert "5/40" in text and "5/40  ✓" not in text


@pytest.mark.asyncio
async def test_a_suspected_copy_is_laid_over_the_capture_and_ruled_on(tmp_path):
    """The lossy pcap with its true payload as the candidate: filled as the
    operator's own guesses, and the CRC proves it."""
    import struct
    from pathlib import Path
    from toeknife import containers
    from toeknife.core import gap as gapmod
    from toeknife.core.tape import GUESS
    from toeknife.tui.modals import CopyVerdict, PromptModal
    lossy = Path("samples/damaged/http-gzip-loss.pcap")
    intact = Path("samples/damaged/http-gzip.pcap")
    if not lossy.is_file():
        pytest.skip("run samples/build/makepcap.py to build the captures")
    raw = lossy.read_bytes()
    st = containers.parse(raw).streams[0]
    asm = gapmod.from_holes(st.slice(raw, allow_gaps=True), st.missing, 10, 8,
                            st.uncompressed_size)
    import gzip
    good = intact.read_bytes()
    gst = containers.parse(good).streams[0]
    truth = gzip.decompress(b"".join(good[o:o + n] for o, n in gst.segments))
    copy = tmp_path / "copy.md"
    copy.write_bytes(truth)
    app = ToeKnifeApp(Session.from_trace(asm.trace, stream=st))
    async with app.run_test(size=(160, 50)) as pilot:
        app.action_fill_from_copy()
        await pilot.pause()
        assert isinstance(app.screen, PromptModal)
        app.screen.query_one(Input).value = str(copy)
        await pilot.press("enter")
        await pilot.pause()
        assert isinstance(app.screen, CopyVerdict)
        assert app.screen.result.proven
        assert not app.session.tape.holes()
        assert all(r.provenance == GUESS for r in app.session._records.values())
        await pilot.press("escape")
        await pilot.press("ctrl+z")
        await pilot.pause()
        assert app.session.tape.holes(), "one ctrl+z takes the copy back out"


@pytest.mark.asyncio
async def test_f5_moves_a_model_to_the_trash_only_after_its_name_is_typed(
        tmp_path, monkeypatch):
    """Never a delete: a model can be days of compute. The models directory
    is pointed at a temporary one, so nothing real is in reach."""
    from toeknife.predict import corpora
    from toeknife.tui.modals import CorpusScreen, TrashModel
    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setattr(corpora, "MODEL_DIRS", (str(models),))
    monkeypatch.delenv("TOEKNIFE_MODELS", raising=False)
    monkeypatch.chdir(tmp_path)
    throwaway = models / "throwaway-o2.tkm"
    _tiny_model().save(throwaway, {"type": "test"})
    app = make_app()
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.press("f5")
        await pilot.pause()
        assert isinstance(app.screen, CorpusScreen)
        lst = app.screen.query_one(OptionList)
        lst.highlighted = option_index(app.screen, f"saved:{throwaway}")
        await pilot.pause()
        foot = app.screen.query_one("#corpus-foot").content
        assert "^d move to trash" in str(foot), "the key is drawn where it applies"
        await pilot.press("ctrl+d")
        await pilot.pause()
        assert isinstance(app.screen, TrashModel)
        app.screen.query_one(Input).value = "throwaway"     # not its name
        await pilot.press("enter")
        await pilot.pause()
        assert throwaway.is_file(), "a wrong name must keep it"
        app.screen.query_one(Input).value = "throwaway-o2.tkm"
        await pilot.press("enter")
        await pilot.pause()
        assert not throwaway.exists()
        gone = list((tmp_path / "trash").glob("*throwaway-o2.tkm"))
        assert len(gone) == 1, "it moved to the trash, it was not deleted"


@pytest.mark.asyncio
async def test_the_pane_says_what_its_suggestion_number_is():
    """'model suggests n 99.4%' with no corpus loaded was
    the stream's own bytes talking, shown as a bare percentage."""
    from toeknife.tui.panes import BitsPane
    app = make_app()
    async with app.run_test(size=(160, 40)) as pilot:
        pos = app.session.tape.holes()[0].anchor
        app.tape_view.goto(pos)
        app.update_inspector(pos)
        await pilot.pause()
        body = app.query_one(BitsPane)._body().content
        text = getattr(body, "plain", str(body))
        if "model suggests" in text:
            assert "no corpus loaded" in text


def _evidence():
    from pathlib import Path
    from toeknife import containers
    p = Path("samples/damaged/evidence.zip")
    if not p.is_file():
        pytest.skip("run samples/build/makedamaged.py to build the samples")
    raw = p.read_bytes()
    return raw, containers.parse(raw)


def test_a_damaged_zip_entry_decodes_as_a_stream():
    """A strict decode stopped by damage fell back to
    fragment ranking and recovered 125 bytes of an entry whose clean prefix
    is tens of KB. An entry starts where its header says."""
    raw, report = _evidence()
    st = next(s for s in report.streams if s.name == "shredded.md")
    s = Session(st.slice(raw), stream=st)
    assert s.mode == "stream"
    assert s.tape.stats()["out_len"] > 10_000


@pytest.mark.asyncio
async def test_a_multi_entry_archive_offers_its_entries():
    """It printed a table and exited."""
    from toeknife.tui.entries import EntryPicker
    _raw, report = _evidence()
    picker = EntryPicker(report, "evidence.zip")
    async with picker.run_test(size=(140, 40)) as pilot:
        lst = picker.query_one(OptionList)
        opts = [lst.get_option_at_index(i) for i in range(lst.option_count)]
        assert [o.id for o in opts if not o.disabled] == ["0", "1", "2"], \
            "only the decodable entries can be chosen"
        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
    assert picker.return_value == 1


@pytest.mark.asyncio
async def test_a_decode_that_stopped_early_does_not_say_complete():
    """'complete' on an entry missing most of its bytes."""
    raw, report = _evidence()
    st = next(s for s in report.streams if s.name == "shredded.md")
    app = ToeKnifeApp(Session(st.slice(raw), stream=st))
    async with app.run_test(size=(160, 40)) as pilot:
        await pilot.pause()
        assert " complete " not in status_of(app)
        assert "no holes" in status_of(app) or "stopped at" in status_of(app)


# ---- reading on past mid-block damage ----------------------------------

def shredded_app():
    from tests import test_resume as tr
    from toeknife.containers.base import Stream
    import zlib
    raw = tr.text()
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    blob = tr.burst(co.compress(raw) + co.flush())
    stream = Stream(name="t", container="zip", offset=0, length=len(blob),
                    crc32=zlib.crc32(raw) & 0xFFFFFFFF,
                    uncompressed_size=len(raw))
    return ToeKnifeApp(Session(blob, stream=stream)), raw


@pytest.mark.asyncio
async def test_the_board_reads_on_past_the_damage_and_marks_it_unproven():
    from toeknife.core.tape import RESUMED
    app, raw = shredded_app()
    async with app.run_test(size=(150, 46)) as pilot:
        stopped = len(app.session.tape)
        assert stopped < len(raw)
        await pilot.press("f7")
        await pilot.pause()
        i, opt = route_row(app.screen, "resume")
        assert not opt.disabled
        app.screen.query_one(OptionList).highlighted = i
        await pilot.press("enter")
        await settle(app, pilot)
        assert len(app.session.tape) == len(raw)
        assert app.session.tape.prov_at(len(raw) - 1) == RESUMED
        assert "read on from" in str(app.query_one("#status").content)
        assert any(e.route == "resume" for e in app.session.ledger.entries)
        # and an unproven byte cannot be typed over as if it were a hole
        app.tape_view.goto(len(raw) - 1)
        await pilot.press("x")
        await pilot.pause()
        assert "unproven" in str(app.query_one("#status").content)


@pytest.mark.asyncio
async def test_the_resume_route_is_not_offered_on_a_capture_that_decodes():
    app = solving_app()
    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.press("f7")
        await pilot.pause()
        _i, opt = route_row(app.screen, "resume")
        assert opt.disabled
        await pilot.press("escape")
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_failed_header_parse_puts_the_repair_under_next():
    app, _comp, _body = rotted_app()
    async with app.run_test(size=(150, 46)) as pilot:
        await pilot.press("f7")
        await pilot.pause()
        lst = app.screen.query_one(OptionList)
        ids = [lst.get_option_at_index(i).id for i in range(lst.option_count)]
        assert "act:bitrot" in ids
        assert ids.index("act:bitrot") < ids.index("route:decode")
        await pilot.press("escape")
        await pilot.pause()


def _tiny_byte_checkpoint(path, held_out=None, calibrated=False):
    torch = pytest.importorskip("torch")
    from toeknife.predict import calibration
    from toeknife.predict.bytemodel import ByteFiller
    torch.manual_seed(0)
    m = ByteFiller(d_model=32, layers=1, heads=2, window=33)
    blob = dict(model=m.state_dict(), cfg=m.cfg, val_acc=0.5)
    if held_out is not None:
        blob["held_out"] = held_out
    if calibrated:
        blob["calibration"] = calibration.fit_filler(
            [(c / 200, 0, False, c % 3 != 0) for c in range(200)]).to_dict()
    torch.save(blob, path)
    return path


@pytest.mark.asyncio
async def test_a_finished_run_is_saved_from_the_training_screen(tmp_path,
                                                                 monkeypatch):
    """A run's checkpoint sat in its run directory and f5 never listed it:
    nothing copied it to the models directory."""
    import json
    import time as _t
    from toeknife.predict import corpora, training
    monkeypatch.setenv("TOEKNIFE_TRAINING", str(tmp_path / "training"))
    run = training.runs_dir() / "slm-1"
    run.mkdir(parents=True)
    (run / training.SPEC).write_text(json.dumps(dict(
        name="slm-1", started=_t.time(), marker="slm-1",
        corpus=str(tmp_path / "phrack"), config=dict(steps=30))))
    (run / training.LOG).write_text(
        "    0:05        17,811       5.842     5.813      0.5%   lr 1.58e-05"
        "  100.0%  active     0:05\n"
        "done (budget reached): 30 of 30 steps (17,811 predictions)\n")
    _tiny_byte_checkpoint(run / "bytemodel-best.pt", held_out=["x"],
                          calibrated=True)
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        app.action_train()
        await pilot.pause()
        screen = app.screen
        lst = screen.query_one(OptionList)
        lst.highlighted = option_index(screen, f"run:{run}")
        await pilot.pause()
        detail = str(screen.query_one("#training-detail").content)
        assert "save it to the models directory" in detail
        await pilot.press("enter")
        await pilot.pause()
        box = app.screen.query_one(Input)
        assert box.value.endswith("phrack-slm-0.0M.tks")
        await pilot.press("enter")
        await settle(app, pilot)
        [m] = corpora.find_models()
        assert m["kind"] == corpora.SLM and m["provenance"]["run"] == "slm-1"
        status = str(app.query_one("#status").content)
        assert "saved" in status and "calibrated" in status
        await pilot.pause()


@pytest.mark.asyncio
async def test_f5_calibrates_the_loaded_byte_model_and_loads_the_copy(
        tmp_path):
    """An uncalibrated byte model gets a row in f5 that measures it on the
    files it held back and swaps in the calibrated copy."""
    from test_calibration import _prose
    doc = tmp_path / "doc.txt"
    doc.write_bytes(_prose(12_000))
    src = _tiny_byte_checkpoint(tmp_path / "m.tks", held_out=[str(doc)])
    app = make_app()
    async with app.run_test(size=(140, 40)) as pilot:
        app.adopt_byte_model(src)
        assert app._filler.calibration is None
        await pilot.press("f5")
        await pilot.pause()
        screen = app.screen
        lst = screen.query_one(OptionList)
        lst.highlighted = option_index(screen, "cal:")
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen is screen, "the first enter only says what it costs"
        detail = str(screen.query_one("#corpus-detail").content)
        assert "held back 1 file(s)" in detail and "m-cal.tks" in detail
        await pilot.press("enter")
        await settle(app, pilot)
        assert (tmp_path / "m-cal.tks").is_file()
        assert app._filler.calibration is not None
        assert app._filler.path == str(tmp_path / "m-cal.tks")
        await pilot.press("f5")
        await pilot.pause()
        lst = app.screen.query_one(OptionList)
        ids = [lst.get_option_at_index(i).id for i in range(lst.option_count)]
        assert "cal:" not in ids, "a calibrated model has nothing to do here"
        await pilot.pause()


@pytest.mark.asyncio
async def test_the_floor_picker_says_what_each_floor_rests_on():
    """Near a gap a calibrated 90% floor rested on 9 held-out classes; the
    picker says so beside the floor rather than in a doc."""
    from toeknife.predict.calibration import Calibration
    from toeknife.tui.modals import ThresholdPicker
    cal = Calibration(short=[(0.5, 0.6, 400), (0.9, 0.92, 50)],
                      near=[(0.5, 0.55, 300), (0.99, 0.91, 9)],
                      classes=759)
    app = make_app()
    async with app.run_test(size=(160, 40)) as pilot:
        app.push_screen(ThresholdPicker(0.9, cal))
        await pilot.pause()
        lst = app.screen.query_one(OptionList)
        rows = {lst.get_option_at_index(i).id:
                str(lst.get_option_at_index(i).prompt)
                for i in range(lst.option_count)}
        assert "on 50 classes, 9 near a gap" in rows["0.9"]
        assert "on 450 classes, 309 near a gap" in rows["0.5"]
        assert "fills nothing" in rows["0.95"]
        await pilot.pause()


@pytest.mark.asyncio
async def test_a_short_fixed_table_reading_is_unproven_and_says_so():
    """A fragment read with the assumed fixed table is not confirmed by
    50 bytes; its bytes are unproven, and the status line names f3."""
    from toeknife.core.tape import BORROWED
    app = make_app()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.session.tape.prov_at(0) == BORROWED
        assert "fixed table assumed, unproven" in app.sub_title
        assert "f3" in app.sub_title

