"""The ToeKnife terminal application.

Key scheme note: every printable character is reserved for filling a hole,
because typing the byte you think belongs somewhere is the primary action of
this tool.  Commands therefore live on ctrl- and function keys.  Nothing you can
type will ever trigger a command by accident while you are solving.
"""
from __future__ import annotations

import time
from pathlib import Path

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Header, Static, Tree

from ..core import ledger
from ..core.tape import UNKNOWN
from ..predict.solver import measured_text, pct
from .modals import (AlignmentPicker, BitRotPicker, TablePicker, ChecksumScreen,
                     CorpusScreen, HelpScreen, PromptModal, ResyncPicker,
                     RouteBoard, SolverScreen, TableStatsScreen)
from .panes import BitsPane, BlocksPane
from .progress import Cancelled, ProgressScreen
from .tape import TapeView


def _show_byte(value: int | None) -> str:
    if value is None:
        return "nothing"
    return repr(chr(value)) if 0x20 <= value < 0x7F else f"0x{value:02x}"


def _size_of(paths) -> int:
    total = 0
    for p in paths:
        try:
            total += Path(p).stat().st_size
        except OSError:
            pass
    return total


class WorkflowFooter(Footer):
    """The footer, minus the keys that cannot do anything yet.

    It had grown to nineteen entries, which is longer than an operator reads
    and long enough that ctrl+f and ctrl+e -- the two keys that actually fill
    holes -- were hidden inside it while "next match" was not. Four of the
    rest only mean something in a state the session may not be in: there is
    nothing to undo until something is filled, and no model to save until one
    is trained.

    Filtering happens here, on the drawing, and not in `check_action`, which
    also stops the key. The app answers `footer_shows`; the key still works
    and still says why it cannot.
    """

    def compose(self):
        for widget in super().compose():
            action = getattr(widget, "action", None)
            if action and not self.app.footer_shows(action):
                continue
            yield widget


class ToeKnifeApp(App):
    CSS = """
    Screen { layers: base overlay; }
    #main { height: 1fr; }
    #blocks {
        width: 34; min-width: 22;
        border-right: solid $panel-lighten-2;
        padding: 0 1;
    }
    #tape { width: 1fr; padding: 0 0 0 1; }
    #bits {
        width: 32; min-width: 22;
        border-left: solid $panel-lighten-2;
        padding: 0 1;
    }
    #status { height: 1; background: $panel; color: $text; padding: 0 1; }
    #align-box {
        width: 92%; height: 80%;
        background: $surface; border: thick $primary;
        padding: 1 2;
    }
    #align-title { margin-bottom: 1; }
    #prompt-box {
        width: 56; height: auto;
        background: $surface; border: thick $primary;
        padding: 1 2; margin: 4 8;
    }
    #help-box {
        /* wide enough for the widest line plus the scrollbar the box needs on a
           short terminal; the match-label examples wrap into nonsense if the
           text is allowed to reflow */
        width: 80; max-width: 96%; height: auto; max-height: 90%;
        background: $surface; border: thick $primary;
        padding: 1 2;
    }
    #help-foot { color: $text-muted; margin-top: 1; }
    #solve-box {
        width: 92%; height: 80%;
        background: $surface; border: thick $primary;
        padding: 1 2;
    }
    #solve-title { margin-bottom: 1; }
    /* Without a height the table grows past the box and pushes the footer --
       the only place "- / + move the threshold" is written -- out of sight. */
    #solve-table { height: 1fr; }
    #threshold-box {
        width: 64; height: auto;
        background: $surface; border: thick $primary;
        padding: 1 2;
    }
    #threshold-list { height: auto; max-height: 14; margin: 1 0; }
    #threshold-note, #threshold-foot { color: $text-muted; }
    #solve-foot { color: $text-muted; margin-top: 1; }
    #corpus-box {
        width: 92%; height: 84%;
        background: $surface; border: thick $primary;
        padding: 1 2;
    }
    #corpus-title { margin-bottom: 1; }
    #corpus-list { height: 1fr; min-height: 8; }
    /* The detail pane is the whole point of the two-step, so it gets room
       whether or not anything is in it: a panel that appears and shoves the
       list around reads as a glitch rather than an answer. */
    #corpus-detail {
        height: 9; margin-top: 1; padding: 0 1;
        border-top: solid $panel-lighten-2;
        overflow-y: auto;
    }
    #corpus-input { margin-top: 1; }
    #corpus-foot { color: $text-muted; margin-top: 1; }
    #progress-box {
        /* Wide enough for a ^f run's overall line -- label, bar, count and
           percentage -- without wrapping, since a wrapped bar is not one. */
        width: 72; height: auto;
        background: $surface; border: thick $primary;
        padding: 1 2; margin: 4 8;
    }
    #progress-title { margin-bottom: 1; }
    #progress-bar { width: 100%; }
    #progress-note { color: $text-muted; margin-top: 1; }
    #progress-rounds { margin-top: 1; }
    #progress-foot { color: $text-muted; margin-top: 1; }
    #stats-box {
        /* Wide enough for the ranked row -- symbol, bits, declared, bar and
           measured -- without reflowing it, since a wrapped bar chart is not
           one. The 16x16 grid needs only 37 columns. */
        width: 88; max-width: 96%; height: auto; max-height: 90%;
        background: $surface; border: thick $primary;
        padding: 1 2;
    }
    #stats-foot { color: $text-muted; margin-top: 1; }
    #routes-box {
        width: 88; max-width: 96%; height: 84%;
        background: $surface; border: thick $primary;
        padding: 1 2;
    }
    #routes-title { margin-bottom: 1; }
    #routes-list { height: 1fr; min-height: 8; }
    #routes-detail {
        height: 7; margin-top: 1; padding: 0 1;
        border-top: solid $panel-lighten-2;
        overflow-y: auto;
    }
    #routes-foot { color: $text-muted; margin-top: 1; }
    AlignmentPicker, PromptModal, HelpScreen, SolverScreen, ChecksumScreen,
    ResyncPicker, RouteBoard, CorpusScreen, ProgressScreen, TableStatsScreen,
    BitRotPicker, TablePicker, ThresholdPicker, LongRunStop, CopyVerdict, TrashModel,
    WriteChoice, DictVerdict, DictionaryStop, ChunkRepairScreen,
    ConfirmScreen {
        align: center middle;
    }
    """

    # The command palette is the one place every command lives, so nothing has
    # to fit in the footer to be findable. It was off once, because it bound
    # ctrl+p -- previous back-reference -- and offered only textual's own theme
    # and quit entries. On ctrl+k and carrying this app's commands (see
    # get_system_commands) it is the menu the footer could never be.
    ENABLE_COMMAND_PALETTE = True
    COMMAND_PALETTE_BINDING = "ctrl+k"

    # Every key the main screen answers. Which of them the footer draws is
    # decided per workflow stage by footer_shows, not here: at 140 columns the
    # full row ran off the edge before f8, the key that finishes a recovery.
    # The order is the order a recovery is walked in, because the footer keeps
    # it.
    BINDINGS = [
        Binding("f1", "help", "help"),
        # The palette's own binding is never drawn, and a menu nobody can see
        # is the footer problem again. This one is.
        Binding("ctrl+k", "command_palette", "menu"),
        Binding("f7", "routes", "what next?"),
        Binding("f2", "align", "align"),
        Binding("f3", "resync", "resync"),
        Binding("f5", "corpus", "model"),
        Binding("ctrl+f", "solve_confident", "fill confident"),
        Binding("f4", "solver_panel", "proposals"),
        Binding("ctrl+e", "refill", "re-fill fresh"),
        Binding("f8", "checksum", "checksum"),
        Binding("ctrl+z", "undo", "undo"),
        Binding("ctrl+r", "redo", "redo"),
        Binding("ctrl+w", "export", "write"),
        Binding("ctrl+s", "save_session", "save"),
        Binding("ctrl+q", "quit", "quit"),
        # Never drawn: navigation detail, and views that answer a question the
        # operator already has when they go looking. f1 and ctrl+k list them.
        Binding("ctrl+g", "goto", "goto", show=False),
        Binding("ctrl+n", "next_match", "next match", show=False),
        Binding("ctrl+p", "prev_match", "prev match", show=False),
        Binding("ctrl+x", "hex_fill", "hex fill", show=False),
        Binding("ctrl+t", "table_stats", "table stats", show=False),
        Binding("enter", "follow", "follow match", show=False),
        Binding("backspace", "clear_guess", "clear", show=False),
        Binding("f6", "toggle_blocks", "blocks", show=False),
        Binding("ctrl+b", "focus_blocks", "block list", show=False),
        Binding("escape", "focus_tape", "back to tape", show=False),
    ]
    # f9 save model, f10 train a byte model and ctrl+o fill mode had keys of
    # their own until the footer ran out of room. All three are model verbs, so
    # they live on the f5 model screen now, and in ctrl+k.

    # Keys the terminal cannot tell apart from another key, whatever textual
    # calls them. Binding one of these means binding the other, silently.
    TERMINAL_ALIASES = ("ctrl+m", "ctrl+i", "ctrl+j", "ctrl+h")

    # The three fill modes, named the same way everywhere they appear: on the
    # status line, in the fill-mode report, and in the model description the solver
    # records. Three and not four -- linear mixing is measured closed.
    MODE_NAMES = {"both": "byte model, where the n-gram agrees and is sure",
                  "slm": "byte model alone",
                  "ngram": "n-gram alone"}
    # The same three, short enough to sit on the status line beside everything
    # else it has to carry.
    MODE_TAGS = {"both": "slm+ngram", "slm": "slm", "ngram": "n-gram"}

    # The keys each stage of a recovery draws, besides help, what next, the
    # menu and quit, which every stage draws. A stage is read off the session,
    # never set: see stage().
    STAGE_KEYS = {
        "no model": {"corpus", "solve_confident", "checksum"},
        "filling": {"solve_confident", "solver_panel", "refill", "checksum"},
        "complete": {"checksum", "export", "save_session"},
    }
    ALWAYS_SHOWN = {"help", "command_palette", "routes", "quit"}

    def stage(self) -> str:
        """Where this recovery is: no model yet, filling, or nothing left open.

        "no model" is the stage the manual found people stuck in: ^f with no
        corpus learns from the stream alone and fills next to nothing, so f5 is
        the key this stage has to put in front of them.
        """
        if not self.session.tape.holes():
            return "complete"
        if (getattr(self.session, "model", None) is None
                and getattr(self, "_filler", None) is None):
            return "no model"
        return "filling"

    def _start_bit(self) -> int:
        """The bit the decode really starts at: the agreed boundary when the
        tool chose the start, the operator's bit when they did."""
        agreed = getattr(self.session, "agreed_from", None)
        return self.session.bit_offset if agreed is None else agreed

    def footer_shows(self, action: str) -> bool:
        """Whether this key has earned a row in the footer right now.

        Not `check_action`, which was tried first and is the wrong tool: it
        hides the row AND stops the key, so a hidden key went from saying why it
        could not act to doing nothing at all. Every key in this app explains
        itself when it cannot act, and that is worth more than the row. This
        only draws.
        """
        if action in self.ALWAYS_SHOWN:
            return True
        if action == "undo":
            return bool(self.session._undo)
        if action == "redo":
            return bool(self.session._redo)
        if action in ("align", "resync"):
            # A fragment's first question is where it starts; a stream with
            # its headers intact never asks it.
            return self.session.mode == "fragment"
        return action in self.STAGE_KEYS[self.stage()]

    # The menu, in groups. Each entry is (group, title, action, key); the key
    # is printed so the menu teaches the shortcut rather than replacing it.
    COMMANDS_MENU = [
        ("Next", "what can I do with this capture?", "routes", "f7"),
        ("Navigate", "go to an output position", "goto", "ctrl+g"),
        ("Navigate", "next back-reference", "next_match", "ctrl+n"),
        ("Navigate", "previous back-reference", "prev_match", "ctrl+p"),
        ("Navigate", "show / hide the blocks pane", "toggle_blocks", "f6"),
        ("Navigate", "move into the blocks pane", "focus_blocks", "ctrl+b"),
        ("Model", "choose, load, train or save a model", "corpus", "f5"),
        ("Model", "review the model's proposals", "solver_panel", "f4"),
        ("Model", "train a byte model (slow, detaches)", "train", ""),
        ("Model", "save the loaded model", "save_model", ""),
        ("Model", "switch fill mode: n-gram / byte model / both",
         "fill_mode", ""),
        ("Fill", "fill confident, surest first", "solve_confident", "ctrl+f"),
        ("Fill", "set the confidence threshold ^f fills down to",
         "set_threshold", ""),
        ("Fill", "take back the model's fills and fill again", "refill",
         "ctrl+e"),
        ("Fill", "type a byte as hex", "hex_fill", "ctrl+x"),
        ("Fill", "fill from a suspected copy of the content…",
         "fill_from_copy", ""),
        ("Fill", "find the preset dictionary in a file…",
         "fill_dictionary", ""),
        ("Recover", "repair flipped bits with the chunk CRC",
         "chunk_repair", ""),
        ("Fill", "clear the guess under the cursor", "clear_guess",
         "backspace"),
        ("Fill", "undo", "undo", "ctrl+z"),
        ("Fill", "redo", "redo", "ctrl+r"),
        ("Recover", "align: pick the starting bit", "align", "f2"),
        ("Recover", "resync: block boundaries found by scanning", "resync",
         "f3"),
        ("Recover", "table statistics for this block", "table_stats",
         "ctrl+t"),
        ("Verify", "checksum: verdict, and pin bytes with it", "checksum",
         "f8"),
        ("File", "write the recovered output", "export", "ctrl+w"),
        ("File", "save this session", "save_session", "ctrl+s"),
        ("File", "help: every key", "help", "f1"),
        ("File", "quit", "quit", "ctrl+q"),
    ]

    def get_system_commands(self, screen):
        """What ctrl+k lists: this app's commands, grouped, with their keys.

        Only on the main screen. Every command here acts on the tape, and
        running one from inside a modal would stack a second screen on a first
        that is still waiting for an answer.
        """
        from textual.app import SystemCommand
        if screen is not self.screen_stack[0]:
            return
        for group, title, action, key in self.COMMANDS_MENU:
            shown = f"  ({key.replace('ctrl+', '^')})" if key else ""
            yield SystemCommand(f"{group}: {title}{shown}", "",
                                getattr(self, f"action_{action}"))

    def __init__(self, session, title: str = "ToeKnife",
                 solve_confidence: float = 0.995, self_train: bool = False,
                 order="auto", shortlist: int = 16,
                 crc_limit: int = 1 << 21):
        super().__init__()
        self.session = session
        self._title = title
        self.solve_confidence = solve_confidence
        self.self_train = self_train
        self.order = order
        # The same knobs --shortlist and --crc-limit set, so the checksum does
        # the same thing here as it does headless.
        self.shortlist = shortlist
        self.crc_limit = crc_limit
        self._solver = None
        self._suggestions: dict[int, object] = {}
        self._verdict = None
        self._verdict_key = None

    # ---- the model ---------------------------------------------------------

    def solver(self):
        """Build the solver on first use.

        The same assembly the headless path uses, so f4 and --solve cannot
        disagree about what the model is: a corpus model if one was supplied at
        launch, the stream's own recovered bytes when there are enough of them
        to be evidence rather than a rumour, and the block's own Huffman table
        wherever it says anything. With none of those there is nothing to
        predict from, and describe() says so rather than inventing a predictor.
        """
        if self._solver is None:
            from ..predict import corpora
            from ..predict.profiles import build
            from ..predict.solver import HoleSolver
            corpus = getattr(self.session, "model", None)
            # The self-trained half learns from the stream's own recovered
            # bytes, which is a corpus of its own size and wants its own order.
            known = self.session.tape.stats().get("known", 0)
            mixture, _profile = build(self.session, corpus_model=corpus,
                                      self_train=self.self_train or corpus is None,
                                      order=corpora.resolve_order(self.order, known))
            # A byte model, if one has been adopted. It is passed separately
            # because it is not a Predictor and cannot go in the mixture: it
            # reads the tape, where UNKNOWN is a token it was trained on.
            filler = getattr(self, "_filler", None)
            mode = getattr(self, "_fill_mode", "both") if filler else "ngram"
            self._solver = HoleSolver(self.session, mixture,
                                      min_confidence=self.solve_confidence,
                                      filler=filler, mode=mode,
                                      calibration=getattr(
                                          corpus, "calibration", None))
            self._model_source = mixture.describe()
            if filler:
                self._model_source += f" + {filler.describe()} [{mode}]"
        return self._solver

    def mode_name(self, mode: str) -> str:
        """MODE_NAMES, except "both" with a calibrated byte model: the byte
        model's own curve gates it then, so it fills what "slm" fills and the
        n-gram's agreement is only recorded (predict/calibration.py
        fit_filler)."""
        if mode == "both" and getattr(getattr(self, "_filler", None),
                                      "calibration", None) is not None:
            return ("byte model on its measured curve, n-gram agreement "
                    "noted (fills as byte model alone)")
        return self.MODE_NAMES[mode]

    def floor_calibration(self):
        """What the floor is measured against, for the fill mode in use.

        Read without building a solver, because the status line asks on every
        redraw and a solver costs a mixture.
        """
        from ..predict.solver import floor_calibration
        filler = getattr(self, "_filler", None)
        mode = getattr(self, "_fill_mode", "both") if filler else "ngram"
        return floor_calibration(
            mode, getattr(getattr(self.session, "model", None),
                          "calibration", None), filler)

    def suggestion_for(self, pos: int):
        """The model's proposal for the hole at pos, cached per class."""
        hole = self.session.tape.hole_at(pos)
        if hole is None:
            return None
        if hole.anchor not in self._suggestions:
            try:
                solver = self.solver()
                p = solver.propose_one(hole)
                if p is not None:
                    # The same annotation ^f reads, so the pane and the fill
                    # quote one number.
                    from ..predict.solver import evidence_runs, interior_gaps
                    tape = self.session.tape
                    solver.annotate(p, evidence_runs(tape), interior_gaps(tape))
                self._suggestions[hole.anchor] = p
            except Exception:
                self._suggestions[hole.anchor] = None
        return self._suggestions[hole.anchor]

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            yield BlocksPane(self.session, id="blocks")
            yield TapeView(self.session, id="tape")
            yield BitsPane(self.session, id="bits")
        yield Static("", id="status")
        # The menu key is drawn in the row with the rest, not alone at the
        # right-hand end where textual would put a second copy of it.
        yield WorkflowFooter(show_command_palette=False)

    def _subtitle(self) -> str:
        """Mode and start bit, and when the bytes rest on the assumed fixed
        table, that too: it lives in the header because a status message
        lasts only until the cursor next moves."""
        sub = f"{self.session.mode}  ·  bit {self._start_bit()}"
        if getattr(self.session, "assumed_note", ""):
            sub += "  ·  fixed table assumed, unproven (f3 finds a real header)"
        return sub

    def on_mount(self) -> None:
        self.title = self._title
        self.sub_title = self._subtitle()
        tape = self.query_one(TapeView)
        tape.focus()
        holes = self.session.tape.holes()
        if holes:
            tape.goto(holes[0].anchor)
        self.refresh_all()
        # The first goto runs before layout, when the view has no height and
        # cannot scroll; on a capture whose first hole is 31 KB in, the screen
        # opened on the top of the file with the cursor out of sight.
        self.call_after_refresh(tape.scroll_to_cursor)

    # ---- shared refresh ----------------------------------------------------

    @property
    def tape_view(self) -> TapeView:
        return self.query_one(TapeView)

    def refresh_all(self, rebuild_blocks: bool = True) -> None:
        tape = self.tape_view
        tape.rebuild()
        # A re-decode moves the range the cursor lives in: resyncing a fragment
        # into a stream drops the entire negative pre-fragment window, and a
        # cursor left outside the new range renders as blank space forever.
        # That is how a successful resync looked like a failed one -- 321,399
        # bytes recovered with no holes, cursor still parked at -13,414, nothing
        # but unknowns on screen.  goto() clamps, and lives here rather than in
        # rebuild() because touching the reactive needs a running app.
        tape.goto(tape.cursor)
        blocks = self.query_one(BlocksPane)
        if rebuild_blocks:
            blocks.rebuild()
        blocks.highlight(tape.cursor)
        self.update_inspector(tape.cursor)
        self.update_status()
        self.sub_title = self._subtitle()

    # A checksum over the whole output is cheap (zlib does it in C) but the
    # buffer it runs on has to be rebuilt, so it is recomputed when the tape
    # has actually changed rather than on every cursor move.
    VERIFY_LIMIT = 8 << 20

    def verdict(self, st: dict | None = None):
        """The container's verdict on the decode as it stands, or None."""
        s = self.session
        if s.stream is None:
            return None
        st = st or s.tape.stats()
        if st["out_len"] > self.VERIFY_LIMIT:
            return None                     # f8 still checks it on demand
        # The trace's identity too: a repair that flips a bit decodes again to
        # the same length and count, and the old verdict would stand.
        key = (id(s.trace), st["out_len"], st["known"], len(s.guesses))
        if key != self._verdict_key:
            try:
                self._verdict = s.verify()
            except Exception:
                self._verdict = None
            self._verdict_key = key
        return self._verdict

    def checksum_token(self, st: dict) -> tuple[str, str] | None:
        """One word in the status bar for the thing that knows whether this
        decode is real. A silent corruption looks exactly like a good decode
        everywhere else on screen."""
        v = self.verdict(st)
        if v is None:
            return None
        name = ("crc" if v.crc_expected is not None else
                "adler" if v.adler_expected is not None else "size")
        if v.ok is True:
            return f" {name} ok ", "bold green"
        if v.ok is False:
            if v.size_expected is not None and v.size_got != v.size_expected:
                return " size MISMATCH ", "bold red"
            return f" {name} MISMATCH ", "bold red"
        if name == "size":
            return None                     # nothing was recorded to check
        return f" {name} ? ", "grey42"

    def update_inspector(self, pos: int) -> None:
        bits = self.query_one(BitsPane)
        suggestion = (self.suggestion_for(pos)
                      if self.session.tape.value_at(pos) is None else None)
        bits.show_symbol(self.session.trace.symbol_at_out(pos), pos, suggestion)

    def update_status(self, message: str = "") -> None:
        # Every action that can change whether undo, redo, save-model or the
        # fill mode is available ends up here, so this is where the footer is
        # told to re-ask check_action.
        self.refresh_bindings()
        s = self.session
        st = s.tape.stats()
        cur = self.tape_view.cursor
        t = Text()
        t.append(f" out {cur} ", style="bold white")
        block = s.trace.block_at_out(cur)
        if block is not None:
            t.append(f" block #{block.index} {block.btype_name} ", style="cyan")
        elif cur < 0:
            t.append(" dictionary " if s.has_dictionary else " pre-fragment ",
                     style="grey42")
        t.append(f" {st['out_len']} bytes ", style="grey62")
        if st["holes"]:
            t.append(f" {st['holes']} holes / {st['hole_positions']} slots ",
                     style="red")
        elif (s.trace.status not in ("end", "eob")
              and not getattr(s.verify(), "ok", False)):
            # No holes, but the decode ended early: "complete" read as
            # "recovered" on a zip entry missing 36 KB.
            t.append(f" stopped at {st['out_len']:,} ", style="yellow")
        else:
            # What it means, exactly: nothing is open. "complete" read as
            # "recovered" next to a size MISMATCH.
            t.append(" no holes ", style="green")
        # Only shown when the code set was recovered without its assignment,
        # because that is the only time literal meanings are a separate unknown
        # worth counting.  It is the number that says how much of the block one
        # more identified byte could move.
        if st.get("codes"):
            t.append(f" {st['codes_known']}/{st['codes']} codes ",
                     style="green" if st["codes_known"] == st["codes"]
                     else "yellow")
        from ..core.tape import SOLVED
        typed = sum(1 for pos in s.guesses
                    if s._records.get(pos) and s._records[pos].provenance != SOLVED)
        filled = len(s.guesses) - typed
        if typed:
            t.append(f" {typed} guessed ", style="yellow")
        if filled:
            t.append(f" {filled} model-filled ", style="magenta")
        # Which predictor a fill would use. Only once there is a choice --
        # with no byte model loaded there is one answer and printing it every
        # frame says nothing. It is on the status line and not only behind
        # the f5 screen because the mode decides what every subsequent fill means,
        # and a mode you cannot see is one you assume.
        if getattr(self, "_filler", None):
            mode = getattr(self, "_fill_mode", "both")
            t.append(f" fill: {self.MODE_TAGS[mode]} ", style="magenta")
        # The floor ^f stops at. It was settable only from f4, and a dial
        # nobody can see is one nobody knows they moved.
        cal = self.floor_calibration()
        t.append(f" floor {pct(self.solve_confidence)}"
                 f"{' measured' if cal is not None else ''} ", style="grey62")
        # A training run is minutes-to-days of the operator's machine and it
        # outlives this app, so it gets a place on the status line rather than
        # only existing behind a keystroke. Cheap: reads one log file's tail.
        train = self.training_token()
        if train:
            t.append(train[0], style=train[1])
        token = self.checksum_token(st)
        if token:
            t.append(token[0], style=token[1])
        t.append(f" {s.trace.status} ", style="grey62")
        if message:
            t.append(f" · {message}", style="bold cyan")
        elif s.note:
            t.append(f" · {s.note}", style="bold red")
        self.query_one("#status", Static).update(t)

    def on_tape_view_cursor_moved(self, event: TapeView.CursorMoved) -> None:
        self.update_inspector(event.pos)
        self.query_one(BlocksPane).highlight(event.pos)
        self.update_status()

    def on_blocks_pane_block_selected(
            self, event: BlocksPane.BlockSelected) -> None:
        """Picking a block on the left puts the cursor on its first byte."""
        tape = self.tape_view
        tape.goto(event.block.out_start)
        # The move posts CursorMoved, which updates the status line itself, so
        # there is nothing to say here that would survive being said.
        tape.focus()

    def on_blocks_pane_stats_requested(
            self, event: BlocksPane.StatsRequested) -> None:
        self.push_screen(TableStatsScreen(self.session, event.block))

    def action_table_stats(self) -> None:
        """The same screen for the block under the cursor, so it does not have
        to be reached by walking the tree."""
        block = self.session.trace.block_at_out(self.tape_view.cursor)
        if block is None:
            self.update_status("the cursor is not inside a block; move it "
                               "onto decoded output first")
            return
        if block.litlen is None:
            self.update_status(f"block #{block.index} is {block.btype_name} "
                               f"and carries no Huffman table")
            return
        self.push_screen(TableStatsScreen(self.session, block))

    # ---- editing -----------------------------------------------------------

    def on_key(self, event) -> None:
        ch = event.character
        if ch is None or not (0x20 <= ord(ch) < 0x7F):
            return
        if self.focused is not self.tape_view:
            return
        event.stop()
        event.prevent_default()
        self._fill(ord(ch))

    def _writable(self, pos: int):
        """(target position, what is being displaced) for a fill at `pos`.

        Three cases, and only one of them is a refusal. An empty hole takes the
        byte. A byte that is there because the model filled it, or because the
        operator did, is a HYPOTHESIS and can be overruled -- the operator has
        the plaintext, the surrounding text and a reason, and the model has a
        confidence figure. A byte read from the bitstream or copied by a
        back-reference is a FACT, and typing over it would be editing evidence.

        The old check asked only whether the tape had a value, which put a model
        fill in the same class as a literal and made an iterative session
        one-way: the solver could write bytes the operator could not correct.
        """
        from ..core.tape import GUESS, GUESS_PROP, SOLVED, SOLVED_PROP
        owner = self.session.guess_owner(pos)
        if owner is not None:
            rec = self.session._records.get(owner)
            prov = rec.provenance if rec else GUESS
            was = self.session.tape.value_at(pos)
            kind = ("the model's" if prov in (SOLVED, SOLVED_PROP) else "your")
            return owner, (kind, was)
        if self.session.tape.value_at(pos) is None:
            return pos, None
        return None, None

    def _refusal(self, pos: int) -> str:
        from ..core.tape import BORROWED, COPY, GAPPED, PROV_NAMES, RESUMED
        prov = self.session.tape.prov_at(pos)
        if prov == GAPPED:
            return ("that byte was copied across a searched gap and is "
                    "unproven; ctrl+k checks it against a suspected copy")
        if prov == BORROWED:
            return ("that byte was decoded with a table from elsewhere and is "
                    "unproven; ctrl+k checks it against a suspected copy")
        if prov == RESUMED:
            return ("that byte was decoded near the damage and is unproven; "
                    "ctrl+k checks it against a suspected copy")
        if prov == COPY:
            return ("that byte is a copy of an earlier one; correct the byte "
                    "it copies from and this one follows")
        return (f"that byte was read from the bitstream "
                f"({PROV_NAMES.get(prov, prov)}); it is evidence and stays "
                f"as decoded")

    def _fill(self, value: int) -> None:
        tape_view = self.tape_view
        pos = tape_view.cursor
        target, displaced = self._writable(pos)
        if target is None:
            self.update_status(self._refusal(pos))
            return
        try:
            if displaced is None:
                touched = self.session.set_guess(target, value)
            else:
                # Clear and set as one undo step, so ctrl+z puts back what was
                # there -- with its own provenance -- instead of leaving a hole
                # where the model's fill used to be. set_guess would have
                # replaced it silently and unrecoverably.
                self.session.begin_group()
                self.session.clear_guess(target)
                touched = self.session.set_guess(target, value)
                self.session.end_group()
        except (ValueError, KeyError) as e:
            self.update_status(str(e))
            return
        self._after_manual_fill()
        tape_view.rebuild()
        self.update_inspector(pos)
        others = len(touched) - 1
        extra = f", and {others} position{'s' if others != 1 else ''} equal to it" \
            if others else ""
        shown = _show_byte(value)
        if displaced is not None:
            kind, was = displaced
            self.update_status(f"replaced {kind} {_show_byte(was)} with {shown} "
                               f"at {target}{extra}")
        else:
            self.update_status(f"set {shown} at {target}{extra}")

    def _after_manual_fill(self) -> None:
        """A byte the operator changed is context the model has not seen.

        The proposals are cached per class, so they go. The solver itself is
        rebuilt only when it has a self-trained part, which is the only piece
        that reads the recovered bytes once at build time rather than live: the
        corpus model and the table prior do not change when a byte does, and
        retraining on every keystroke would make the loop the operator is in
        slower for nothing.
        """
        self._suggestions.clear()
        if self._solver is not None and "self-trained" in \
                (getattr(self, "_model_source", "") or ""):
            self._solver = None

    def action_clear_guess(self) -> None:
        pos = self.tape_view.cursor
        owner = self.session.guess_owner(pos)
        if owner is None:
            self.update_status("no guess here")
            return
        self.session.clear_guess(owner)
        self._after_manual_fill()
        self.tape_view.rebuild()
        self.update_inspector(pos)
        self.update_status(f"cleared guess at {owner}")

    def action_undo(self) -> None:
        if self.session.undo():
            self._after_manual_fill()
            self.tape_view.rebuild()
            self.update_status("undo")
        else:
            self.update_status("nothing to undo")

    def action_redo(self) -> None:
        if self.session.redo():
            self._after_manual_fill()
            self.tape_view.rebuild()
            self.update_status("redo")
        else:
            self.update_status("nothing to redo")

    # ---- navigation --------------------------------------------------------

    def action_next_hole(self) -> None:
        if not self.tape_view.next_hole():
            self.update_status("no holes left")

    def action_prev_hole(self) -> None:
        if not self.tape_view.next_hole(backwards=True):
            self.update_status("no holes left")

    def action_next_match(self) -> None:
        if not self.tape_view.next_match():
            self.update_status("no back-references in this decode")

    def action_prev_match(self) -> None:
        if not self.tape_view.next_match(backwards=True):
            self.update_status("no back-references in this decode")

    def action_follow(self) -> None:
        if not self.tape_view.jump_to_source():
            self.update_status("cursor is not on a back-reference")

    def action_toggle_blocks(self) -> None:
        pane = self.query_one(BlocksPane)
        pane.display = not pane.display

    def action_focus_blocks(self) -> None:
        """Move to the block list, already sitting on the current block."""
        pane = self.query_one(BlocksPane)
        pane.display = True
        pane.highlight(self.tape_view.cursor)
        pane.query_one(Tree).focus()

    def action_focus_tape(self) -> None:
        self.tape_view.focus()

    # ---- modals ------------------------------------------------------------

    def action_solver_panel(self) -> None:
        """Rank every class, then show the list.

        The ranking is the slow part -- 1,742 classes against an order-5 model is
        2.25 seconds, more with a bigger one -- and it is read-only, so it runs on
        a thread and the screen only opens once there is something in it.
        """
        if not self.session.tape.holes():
            self.update_status("nothing unknown left to fill")
            return

        def done(choice) -> None:
            if choice is None:
                return
            kind, payload = choice
            if kind == "one":
                self._apply_proposals([payload], "accepted")
                return
            from ..predict.solver import pct

            # Re-scoring for the bulk accept is the same slow pass again.
            def work(report):
                return [p for p in self.solver().propose(progress=report)
                        if p.gate >= payload]

            def then(confident, error):
                if error is not None:
                    self.update_status(self._slow_failed("scoring", error))
                    return
                self._apply_proposals(confident, f"accepted all at or above "
                                                 f"{pct(payload)}")

            self.run_slow(f"scoring every class at or above {pct(payload)}",
                          work, then)

        def work(report):
            solver = self.solver()
            return solver, solver.propose(progress=report)

        def then(result, error):
            if error is not None:
                self.update_status(self._slow_failed("scoring", error))
                return
            solver, proposals = result
            self.push_screen(SolverScreen(self.session, solver,
                                          self.solve_confidence,
                                          proposals=proposals), done)

        self.run_slow("scoring every unfilled class", work, then)

    def action_chunk_repair(self) -> None:
        """One or two flipped bits in a PNG chunk, found by its CRC."""
        from ..core import chunkcrc
        from .modals import ChunkRepairScreen
        s = self.session
        total = len(s.stream.segments or [])

        def work(report):
            report(0, 0, "solving each failing chunk's CRC for the flipped bits")
            return chunkcrc.repair(s, s.file_data)

        def then(result, error):
            if error is not None:
                self.update_status(self._slow_failed("chunk repair", error))
                return
            proven = [f for f in result.fixes if f.proven]
            self.session.ledger.add(
                "chunk-crc", ledger.FOUND if proven else ledger.NOTHING,
                result.summary(total),
                counts={"candidates": result.candidates})

            def chosen(go: bool) -> None:
                if not go or len(proven) != 1:
                    self.update_status("chunk repair: " + result.summary(total))
                    return
                chunkcrc.apply(s, proven[0])
                self._solver = None
                self._suggestions.clear()
                self.refresh_all()
                ok = s.verify()
                self.update_status(
                    f"applied {proven[0].label(total)}; the stream now "
                    + ("matches its Adler-32" if ok is not None and ok.ok
                       else "decodes; f8 checks it"))

            self.push_screen(ChunkRepairScreen(result, total), chosen)

        self.run_slow("repairing with the chunk CRC", work, then)

    def action_fill_dictionary(self) -> None:
        """Find a zlib preset dictionary inside a file, prove it, fill it in."""
        from ..core import dictfill
        from .modals import DictVerdict

        def done(path: str | None) -> None:
            if not path:
                return
            p = Path(path).expanduser()
            try:
                blob = p.read_bytes()
            except OSError as e:
                self.update_status(f"cannot read {p}: {e}")
                return

            def work(report):
                return dictfill.find(self.session, blob, progress=report)

            def then(result, error) -> None:
                if error is not None:
                    self.update_status(f"dictionary search stopped: {error}")
                    return
                if result.start is not None:
                    dictfill.apply(self.session, result, blob, p.name)
                outcome = (ledger.NOTHING if result.start is None
                           else ledger.FOUND)
                self.session.ledger.add("dictionary", outcome,
                                        result.summary(),
                                        counts={"filled": result.filled},
                                        params={"path": str(p)})
                self._suggestions.clear()
                self.refresh_all(rebuild_blocks=True)
                self.update_status(result.summary())
                self.push_screen(DictVerdict(result, str(p)))

            self.run_slow("searching the file for the dictionary", work, then)

        self.push_screen(PromptModal(
            "a file that may hold the preset dictionary:",
            "the program that wrote the stream, its data files, a copy"),
            done)

    def action_fill_from_copy(self) -> None:
        """Lay a candidate copy over the capture and let the checksum rule."""
        from ..core import copyfill
        from .modals import CopyVerdict

        def done(path: str | None) -> None:
            if not path:
                return
            p = Path(path).expanduser()
            try:
                candidate = p.read_bytes()
            except OSError as e:
                self.update_status(f"cannot read {p}: {e}")
                return
            result = copyfill.fill(self.session, candidate, p.name)
            outcome = (ledger.FOUND if result.proven or result.copy_matches
                       else
                       ledger.CONTRADICTED if result.refuted
                       else ledger.NOTHING)
            self.session.ledger.add("copy", outcome, result.summary(),
                                    counts={"filled": result.filled,
                                            "contradicted":
                                                len(result.contradictions)},
                                    params={"path": str(p)})
            self._suggestions.clear()
            self.refresh_all(rebuild_blocks=True)
            self.update_status(result.summary())
            self.push_screen(CopyVerdict(result, str(p)))

        self.push_screen(PromptModal(
            "a file you suspect holds this content:",
            "e.g. a later fetch, a cached or public copy"), done)

    def action_set_threshold(self) -> None:
        """Choose the confidence floor without scoring every class first."""
        from .modals import ThresholdPicker

        def done(value) -> None:
            if value is None:
                return
            self.solve_confidence = value
            if self._solver is not None:
                self._solver.min_confidence = value
            self.update_status(f"confidence floor set to {pct(value)}; ^f "
                               f"fills down to it")

        cal = self.floor_calibration()
        self.push_screen(ThresholdPicker(self.solve_confidence, cal), done)

    def action_solve_confident(self) -> None:
        """Fill from the surest class downwards, stopping at the dial.

        The f4 threshold is a floor, not a cutoff: one press descends from the
        most confident class in the tape to the dial and stops there, which is
        what walking the ladder by hand used to do -- and measurably better than
        filling everything above the dial at once, because each fill becomes
        context before the doubtful ones are scored.

        The undo group is opened and closed here rather than in the worker, so a
        run cancelled half way is still one ctrl+z -- what it managed to fill
        stays, and one keystroke takes it back out.
        """
        self._solve_pass(retract=False)

    def action_refill(self) -> None:
        """Take back the model's own fills and score every one of them again.

        A fill resolves its class, so the solver never looks at it twice: a byte
        it got wrong early cannot be dislodged by context that arrives later,
        which is exactly what happens when the operator reads a word next to it.
        This reopens the model's classes and leaves the operator's alone.

        Measured on a phrack carve, after 200 classes were typed: filling again on
        top of its own mistakes left 754 wrong bytes, and retracting first left
        76.  It costs coverage -- reopened classes that no longer clear the floor
        stay unknown -- which is the trade this tool is built to prefer.
        """
        self._solve_pass(retract=True)

    def _solve_pass(self, retract: bool, long_runs: bool = False,
                    dictionary: bool = False) -> None:
        # Not _fill: that name already belongs to writing one typed byte,
        # and defining it twice in this class silently replaced it.

        if not self.session.tape.holes() and not retract:
            self.update_status("nothing unknown left to fill")
            return
        self.session.begin_group()
        took_back = []

        def work(report):
            # Inside the worker and inside the group, so the retraction and the
            # refill are one action: a cancel half way cannot leave the tape with
            # the model's fills gone and nothing put back.
            if retract:
                report(0, 0, "taking back the model's own fills")
                took_back.extend(self.session.retract_model_fills())
                self._suggestions.clear()
            return self.solver().solve(on_round=self._solve_round,
                                       progress=report, long_runs=long_runs,
                                       dictionary=dictionary)

        def then(result, error):
            self.session.end_group()
            self._suggestions.clear()
            self.refresh_all(rebuild_blocks=False)
            back = (f"took back {len(took_back)} of its own classes, then "
                    if took_back else "")
            # With no corpus and no byte model the solver predicts from the
            # stream's own bytes alone, which fills next to nothing on a real
            # hole. Saying so is the difference between "the model is weak" and
            # "you have not given it a model yet".
            bare = (getattr(self.session, "model", None) is None
                    and getattr(self, "_filler", None) is None)
            hint = ("; no corpus loaded, f5 gives the model similar data"
                    if bare else "")
            if error is not None:
                self.update_status(
                    self._slow_failed("solving", error)
                    + "; ctrl+z takes back whatever it had filled")
                return
            if result.held_dictionary:
                self._offer_dictionary(result, back, hint)
                return
            if result.held:
                # Stopped on the regime, not the floor. A dialog rather than a
                # status line, because what it says changes what the next
                # keystroke should be, and it offers the override in place.
                self._offer_long_runs(result, back, hint)
                return
            if result.filled:
                # Name the floor it descended to.  The operator set that number
                # and it is the whole reason the run stopped where it did, so
                # reporting the count without it hides the one control they hold.
                self.update_status(
                    f"model {back}filled {len(result.filled)} classes "
                    f"({result.positions_filled} positions), surest first, "
                    f"down to {pct(self.solve_confidence)} in {result.rounds} "
                    f"rounds; ctrl+z undoes the lot{hint}")
            elif took_back:
                # Retracting and then filling nothing is a real outcome, not a
                # failure: with the operator's bytes in place the model no longer
                # believes what it had said, and unknown beats wrong.
                self.update_status(
                    f"took back {len(took_back)} classes the model had filled "
                    f"and it no longer clears {pct(self.solve_confidence)} on "
                    f"any of them: {result.stopped}. ctrl+z puts them back")
            else:
                # Say where the dial is, not just that nothing cleared it: a
                # bare refusal reads as "the model has nothing", which is a
                # different and much more discouraging claim than "the bar is
                # set high".
                self.update_status(
                    f"model filled nothing: {result.stopped}. "
                    f"f4 lists what it does have, and moves the bar{hint}")

        self.run_slow("taking back the model's fills and scoring again"
                      if retract else
                      "filling every class the model is confident about",
                      work, then)

    def _offer_dictionary(self, result, back: str, hint: str) -> None:
        from ..predict.solver import DICTIONARY_PRECISION
        from .modals import DictionaryStop
        done_so_far = (f"The model {back}filled {len(result.filled)} classes "
                       f"({result.positions_filled} positions) first; ctrl+z "
                       f"undoes them." if result.filled else
                       "Nothing has been filled.")

        def chosen(go: str | None) -> None:
            if go == "fill":
                self._solve_pass(retract=False, dictionary=True)
            elif go == "find":
                self.action_fill_dictionary()
            else:
                self.update_status(
                    f"stopped before filling dictionary bytes; "
                    f"{len(result.filled)} classes filled{hint}")

        self.push_screen(DictionaryStop(result.held_dictionary,
                                        DICTIONARY_PRECISION, done_so_far,
                                        self.solve_confidence), chosen)

    def _offer_long_runs(self, result, back: str, hint: str) -> None:
        from ..predict.solver import LONG_RUN, LONG_RUN_PRECISION
        from .modals import LongRunStop
        done_so_far = (f"The model {back}filled {len(result.filled)} classes "
                       f"({result.positions_filled} positions) first; ctrl+z "
                       f"undoes them." if result.filled else
                       "Nothing has been filled.")

        def chosen(go: bool) -> None:
            if go:
                self._solve_pass(retract=False, long_runs=True)
            else:
                self.update_status(
                    f"stopped before filling inside long unknown runs; "
                    f"{len(result.filled)} classes filled{hint}")

        self.push_screen(LongRunStop(result.held, LONG_RUN,
                                     LONG_RUN_PRECISION, done_so_far,
                                     self.solve_confidence), chosen)

    def _solve_round(self, result, confident: int) -> None:
        """Called on the worker thread after each round. Records what happened
        rather than drawing it: the ledger is a list append and is safe here,
        while the widgets are not."""
        self.session.ledger.add("solve", ledger.FOUND,
                                f"round {result.rounds}",
                                counts={"classes": confident})

    def _apply_proposals(self, proposals, what: str) -> None:
        from ..core.tape import SOLVED
        if not proposals:
            self.update_status("nothing to accept")
            return
        self.session.begin_group()
        applied = 0
        for p in proposals:
            if self.session.tape.value_at(p.anchor) is not None:
                continue
            try:
                self.session.set_guess(p.anchor, p.value, SOLVED)
                applied += 1
            except (ValueError, KeyError):
                continue
            from ..core.session import FillNote
            self.session.note_fill(p.anchor, FillNote(
                how="f4", confidence=p.confidence, agreement=p.agreement,
                runners=tuple(v for v, _ in p.runners), grounded=p.grounded,
                run=p.run, floor=self.solve_confidence,
                precision=p.precision, near=p.near))
        self.session.end_group()
        self._suggestions.clear()
        self.refresh_all(rebuild_blocks=False)
        self.update_status(f"{what}: {applied} class(es) filled, ctrl+z undoes "
                           f"the lot")

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    # Routes the board can start, and which of those can ask for the evidence
    # they are missing rather than being a dead end. Keys are routes.ROUTES keys;
    # anything absent is listed but not offered, which is the honest state for a
    # route that is not in this release.
    ROUTE_RUNNABLE = ("decode", "resume", "resync", "related", "kp-judge",
                      "bitrot")
    ROUTE_ASKABLE = ("related", "kp-judge")

    def action_routes(self) -> None:
        """What next: the steps worth taking now, then what could be tried on a
        lost table and why the rest cannot be. enter runs one."""
        def done(key: str | None) -> None:
            if key and key.startswith("act:"):
                getattr(self, f"action_{key[4:]}")()
            elif key:
                self._run_route(key)

        self.push_screen(RouteBoard(self.session, self.ROUTE_RUNNABLE,
                                    self.ROUTE_ASKABLE), done)

    def _run_route(self, key: str) -> None:
        {"decode": self._route_decode,
         "resume": self.action_resume,
         "resync": self.action_resync,
         "related": self._route_related,
         "kp-judge": self._route_plaintext,
         "bitrot": self._route_bitrot}[key]()

    def action_bitrot(self) -> None:
        self._route_bitrot()

    def action_related(self) -> None:
        self._route_related()

    def action_resume(self) -> None:
        """Read on past mid-block damage, placed by the recorded size.

        The bytes it adds are drawn as unproven: where the damage ends cannot
        be read off the data, and garbage decoded from it is carried forward
        by back-references. A copy that matches the
        checksum is what proves them.
        """
        from ..core import resume

        def work(report):
            return resume.resume(self.session, progress=report)

        def then(got, error):
            if error is not None:
                self.update_status(self._slow_failed("resume", error))
                return
            if got.trace is None:
                self.session.ledger.add("resume", ledger.NOTHING, got.reason)
                self.update_status("resume: " + got.reason)
                return
            resume.apply(self.session, got)
            self.session.ledger.add("resume", ledger.FOUND, got.summary(),
                                    counts={"placed": got.tail,
                                            "gap": got.gap,
                                            "dropped": got.dropped})
            self._solver = None
            self._suggestions.clear()
            # Onto the seam: the first unproven byte.
            self.tape_view.cursor = got.start
            self.refresh_all()
            # After the move's own CursorMoved, which repaints the status
            # line without a message.
            self.call_after_refresh(self.update_status,
                                    "resume: " + got.summary())

        self.run_slow("reading on past the damage", work, then)

    def _route_decode(self) -> None:
        """Already ran: this is the engine. Say what it produced rather than
        re-running it, since a re-decode of the same bytes at the same offset
        cannot say anything new."""
        st = self.session.tape.stats()
        self.update_status(
            f"decode: {self.session.trace.status!r}, {st['out_len']:,} bytes, "
            f"{st['known']:,} known, {st['holes']:,} hole classes. f8 checks it "
            f"against the container")

    # ---- routes that need evidence the operator has ------------------------

    def _route_related(self) -> None:
        """Transplant the table from a second capture of similar content.

        Measured at 39.5% on verified sibling pairs and 10.1% on the same URL
        fetched later, and the number that decides which you are
        looking at is token overlap, which is computed before anything is tried.
        """
        from ..core import related
        from ..core.tape import UNKNOWN

        def done(path: str | None) -> None:
            if not path:
                self.update_status("no related capture given")
                return
            p = Path(path).expanduser()
            if not p.is_file():
                self.update_status(f"no such file: {p}")
                return

            def work(report):
                report(0, 0, f"reading {p.name}")
                blob = p.read_bytes()
                tape = self.session.tape
                # Only bytes actually determined here. Feeding placeholders into
                # the overlap would corrupt the one number this route asks the
                # operator to decide on.
                have = bytes(tape.out[i] for i in range(tape.length)
                             if tape.prov[i] != UNKNOWN)
                report(0, 0, "measuring token overlap")
                rep = related.report(have, blob)
                from ..core import borrow
                return rep, borrow.borrow(self.session, blob,
                                          model=self.session.model,
                                          progress=report)

            self.run_slow(f"related capture: {p.name}", work,
                          lambda rep, err: self._related_done(p, rep, err))

        self.push_screen(
            PromptModal("path to a related capture (same or similar content):",
                        "a second .gz, .zip or raw stream"), done)

    def _related_done(self, path, got, error) -> None:
        if error is not None:
            self.update_status(self._slow_failed("related", error))
            return
        rep, result = got
        self.session.related = str(path)
        over = ("not measurable" if rep["overlap"] is None
                else f"{rep['overlap']:.3f}")
        if not result.landings:
            self.session.ledger.add("related", ledger.NOTHING,
                                    result.summary(),
                                    params={"overlap": rep["overlap"]})
            self.update_status(f"related: overlap {over}; {result.summary()}")
            return
        self.session.ledger.add("related", ledger.FOUND, result.summary(),
                                counts={"landed": len(result.landings)},
                                params={"overlap": rep["overlap"]})

        def take(landing) -> None:
            from ..core import borrow
            if landing is None:
                self.update_status("related: no table adopted; the capture "
                                   "is as it was")
                return
            before = self.session.tape.stats()["known"]
            borrow.apply(self.session, landing, result.boundary)
            self._solver = None
            self._suggestions.clear()
            self.tape_view.cursor = 0          # the first borrowed byte
            self.refresh_all()
            st = self.session.tape.stats()
            self.session.ledger.add("related", ledger.FOUND,
                                    f"adopted {landing.label}",
                                    counts={"known": st["known"]})
            self.update_status(
                f"related: adopted {landing.label}; {st['known']:,} bytes "
                f"known, was {before:,}. The borrowed bytes are unproven "
                f"(blue) until a copy matches them")

        self.push_screen(TablePicker(result), take)

    def _route_plaintext(self) -> None:
        """Judge this decode against bytes the operator already knows."""
        from ..core import plaintext

        def got_path(path: str | None) -> None:
            if not path:
                self.update_status("no plaintext given")
                return
            p = Path(path.split()[0]).expanduser()
            at = 0
            parts = path.split()
            if len(parts) > 1:
                try:
                    at = int(parts[1], 0)
                except ValueError:
                    self.update_status(f"not an output position: {parts[1]}")
                    return
            if not p.is_file():
                self.update_status(f"no such file: {p}")
                return
            blob = p.read_bytes()
            self.session.known_plaintext = {at + i: b
                                            for i, b in enumerate(blob)}
            v = plaintext.judge(self.session, self.session.known_plaintext)
            if not v.ruled:
                self.session.ledger.abstained("plaintext", v.testable,
                                              plaintext.FLOOR)
                self.update_status(
                    f"plaintext: REFUSED to rule: {v.testable} assertion(s) "
                    f"reachable, {plaintext.FLOOR} required")
                return
            self.session.ledger.add(
                "plaintext", ledger.NOTHING if v.bad else ledger.FOUND,
                v.summary().split(":")[0],
                counts={"tested": v.testable, "failed": v.bad})
            self.update_status(f"plaintext: {v.summary()}")

        self.push_screen(PromptModal(
            "path to known plaintext, and the output position it starts at:",
            "e.g. /tmp/head.txt 0"), got_path)

    # ---- repairing a header whose bits are wrong ---------------------------

    def _route_bitrot(self, k: int = 1) -> None:
        from ..core import bitrot
        span = (bitrot.spans(self.session) or [None])[0]
        if span is None:
            self.update_status("no block header could be bounded to search")
            return

        def work(report):
            return bitrot.repair(self.session, k=k, progress=report)

        def then(result, error):
            if error is not None:
                self.update_status(self._slow_failed("bitrot", error))
                return
            self.session.ledger.add(
                "bitrot", ledger.FOUND if result.matching else ledger.NOTHING,
                result.summary(), counts={"survivors": len(result.candidates),
                                          "tried": result.tried},
                params={"bits": k})
            stream = self.session.stream
            checksum = stream is not None and (
                stream.crc32 is not None
                or getattr(stream, "adler32", None) is not None)
            if k == 1 and not result.matching and checksum:
                # One bit is not it. Two bits over a few hundred is minutes,
                # and the checksum still picks the right pair.
                self._offer_two_bits(result)
                return
            if not result.candidates:
                self.update_status("bitrot: " + result.summary())
                return
            self.push_screen(BitRotPicker(result), self._bitrot_chosen)

        lo, hi = span
        what = "one-bit flips" if k == 1 else f"{k}-bit flips"
        self.run_slow(f"{what} over {hi - lo:,} header bits", work, then)

    def _offer_two_bits(self, result) -> None:
        from ..predict.corpora import _duration
        from .modals import ConfirmScreen
        flips, secs = result.next_cost()
        lo, hi = result.span
        body = (f"{result.summary()}\n\n"
                f"Two flipped bits are {flips:,} pairs over these {hi - lo:,} "
                f"bits: about {_duration(secs)} at this search's rate. The "
                f"recorded checksum still picks the one that is the original.")

        def chosen(go: bool) -> None:
            if go:
                self._route_bitrot(k=2)
            elif result.candidates:
                self.push_screen(BitRotPicker(result), self._bitrot_chosen)
            else:
                self.update_status("bitrot: " + result.summary())

        self.push_screen(ConfirmScreen(
            "No single flipped bit repairs this header", body,
            "enter try every pair   esc stop here"), chosen)

    def _bitrot_chosen(self, candidate) -> None:
        from ..core import bitrot
        if candidate is None:
            self.update_status("bitrot: nothing applied; the capture is as it "
                               "was")
            return
        self.session.flip_bits(candidate.bits)
        self._solver = None
        self._suggestions.clear()
        self.refresh_all()
        st = self.session.tape.stats()
        self.session.ledger.add("bitrot", ledger.FOUND,
                                f"applied {candidate.bits}",
                                counts={"known": st["known"]})
        self.update_status(
            f"bitrot: flipped {', '.join(f'bit {b:,}' for b in candidate.bits)}"
            f"; now {self.session.trace.status!r}, {st['out_len']:,} bytes, "
            f"{st['known']:,} known. " + {
                True: "It matches the recorded size and checksum",
                False: "It parses and does not match the recorded checksum",
                None: "It parses; f8 says whether it is the original",
            }[candidate.matches])

    # ---- the checksum ------------------------------------------------------

    def action_checksum(self) -> None:
        """What the container promised, measured against what came out.

        This is the only check in the tool that can tell a right recovery from
        a plausible one, and until now it lived on the command line -- outside
        the view where the recovery actually happens.
        """
        def done(choice) -> None:
            if choice == "pin":
                self._checksum_pin()
            elif choice == "loop":
                self._checksum_loop()

        has_model = getattr(self.session, "model", None) is not None
        self.push_screen(ChecksumScreen(self.session, has_model), done)

    def _checksum_pin(self) -> None:
        """Let the checksum determine what it can, and say so when it cannot.

        CRC32 is affine over GF(2), so unknown bytes enter it linearly and the
        remaining classes are a system of equations rather than a search. Below
        five classes it is exact; above that it removes 32 bits of freedom and
        the model picks from what survives.
        """
        from ..core.tape import SOLVED
        from ..predict.crcsolve import solve_by_shortlist, solve_from_crc

        before = len(self.session.tape.holes())
        if not before:
            self.update_status("nothing unknown left for the checksum to pin")
            return
        self.update_status("solving against the checksum…")
        linear = solve_from_crc(self.session)
        assignment, note = None, linear.summary()
        if linear.ok and not linear.free_bits:
            assignment = linear.assignment
        elif getattr(self.session, "model", None) is None:
            # Below five classes the checksum decides alone; past that it needs
            # something to choose between, and that something is the model.
            note = ("the checksum cannot finish this on its own (" +
                    linear.summary() + "), and there is no model to shortlist "
                    "with. Press f5 for a corpus, then try again")
        elif linear.ok or "no CRC32" in linear.reason:
            shortlisted = solve_by_shortlist(self.session, self.session.model,
                                             per_class=self.shortlist,
                                             max_combinations=self.crc_limit)
            note = shortlisted.summary()
            if shortlisted.ok:
                assignment = shortlisted.assignment

        if not assignment:
            outcome = (ledger.CONTRADICTED if "contradicts" in note
                       else ledger.NOTHING if linear.ok else ledger.ABSTAINED)
            self.session.ledger.add("checksum", outcome, note,
                                    counts={"classes": before})
            self.update_status(f"checksum: {note}")
            return

        self.session.begin_group()
        pinned = 0
        from ..core.session import FillNote
        for anchor, value in sorted(assignment.items()):
            if self.session.tape.value_at(anchor) is None:
                self.session.set_guess(anchor, value, SOLVED)
                self.session.note_fill(anchor, FillNote(how="checksum"))
                pinned += 1
        self.session.end_group()
        self._suggestions.clear()
        self.refresh_all(rebuild_blocks=False)
        self.session.ledger.add("checksum", ledger.FOUND, note,
                                counts={"classes": pinned})
        self.update_status(f"checksum pinned {pinned} class(es); {note}. "
                           f"ctrl+z undoes the lot")

    def _checksum_loop(self) -> None:
        """Alternate the model and the checksum until neither moves.

        Worth having because a checksum is not a second opinion, it is a
        different kind of evidence: each class the model fills is one fewer
        unknown in the linear system, and each class the checksum pins is new
        context for the model.
        """
        from ..predict.pipeline import solve_loop

        if not self.session.tape.holes():
            self.update_status("nothing unknown left to fill")
            return
        self.update_status("model and checksum, in rounds…")
        solver = self.solver()
        result = solve_loop(self.session, solver.model,
                            min_confidence=self.solve_confidence,
                            shortlist=self.shortlist,
                            max_combinations=self.crc_limit,
                            calibration=solver.calibration,
                            filler=solver.filler, mode=solver.mode)
        self._suggestions.clear()
        self.refresh_all(rebuild_blocks=False)
        outcome = (ledger.CONTRADICTED if result.contradicted
                   else ledger.FOUND if result.filled else ledger.NOTHING)
        self.session.ledger.add("checksum-loop", outcome, result.summary(),
                                counts={"filled": result.filled,
                                        "rounds": len(result.rounds)})
        self.update_status(result.summary() + "; ctrl+z undoes each round")

    def action_align(self) -> None:
        if self.session.mode != "fragment":
            self.update_status("alignment picker applies to fragments only")
            return

        def done(offset: int | None) -> None:
            if offset is None:
                return
            self.session.choose_start(offset, agreed=False)
            self._solver = None
            self._suggestions.clear()
            self.refresh_all()
            holes = self.session.tape.holes()
            if holes:
                self.tape_view.goto(holes[0].anchor)
            agree = self.session.agreement_point()
            if agree is not None and offset < agree:
                # A start the operator chose is honoured, but before the point
                # where every reading agrees, which bytes come out depends on
                # that choice -- a wrong one puts misread bytes on the tape in
                # the colour of fact.
                self.update_status(
                    f"decoding from bit {offset}, as chosen. The readings "
                    f"only agree from bit {agree}: bytes before it are right "
                    f"only if bit {offset} is the true start")
            else:
                self.update_status(f"aligned at bit {offset}")

        self.push_screen(AlignmentPicker(self.session), done)

    def action_resync(self) -> None:
        """Find block boundaries and re-open the stream at one of them.

        The decision this exposes was previously available only as a launch
        flag, which meant an operator who opened a mid-stream capture without
        `--resync` saw the fragment path and no hint that a far better one
        existed -- on `samples/fixed-fragment.bin`, 2,211 known bytes and 2,449
        holes against 163,840 known and none.
        """
        from ..core.scan import scan

        total_bits = len(self.session.data) * 8
        # The earliest boundary recovers the most, so a quarter of the data
        # finds the best candidate while costing a third of the time: measured
        # 4.6s against 14.6s on a 95 KB capture, same winner. Widen only if that
        # turns up nothing, which is the case where the head is badly damaged.
        limit = max(1, total_bits // 4)

        def work(report):
            points = scan(self.session.data, max_bit=limit, progress=report)
            # Something found is not enough: a garbage "header" that builds a
            # table and dies a symbol later fires about once every 4,000 bits.
            # On a carve whose real header sat at 25.6% of the data, the
            # quarter held only such a one, and the picker offered nothing
            # else. Widen unless a point decodes clean.
            if any(p.status == "end" for p in points):
                return points, limit
            # The widened pass is four times the work and says so, rather than
            # looking like the first one stalled at the end.
            report(0, 0, "nothing in the first quarter decodes to the end; "
                         "scanning all of it")
            return scan(self.session.data, progress=report), total_bits

        def then(result, error):
            if error is not None:
                self.update_status(self._slow_failed("resync scan", error))
                return
            points, scanned = result
            if not points:
                self.update_status("no block boundary found to resync on")
                return
            self.push_screen(ResyncPicker(points, scanned, total_bits,
                                          self.session.bit_offset), done)

        def done(bit: int | None) -> None:
            if bit is None:
                self.update_status("resync cancelled")
                return
            self.session.resync_to(bit)
            self._solver = None
            self._suggestions.clear()
            self.refresh_all()
            st = self.session.tape.stats()
            holes = self.session.tape.holes()
            # Land on the first thing worth looking at: an unknown if there is
            # one, otherwise the start of the recovered output.  A resync that
            # leaves nothing to fill is the best outcome, not a reason to leave
            # the cursor wherever it happened to be.
            self.tape_view.goto(holes[0].anchor if holes else 0)
            self.update_status(
                f"resynced at bit {bit}: {st['out_len']:,} bytes, "
                f"{st['known']:,} known, {st['holes']:,} holes")

        self.run_slow(f"scanning {limit:,} bit positions for block boundaries",
                      work, then)

    # ---- slow work, on a thread, with something to look at ------------------

    def run_slow(self, title: str, work, then, cancellable: bool = True) -> None:
        """Run `work(report)` off the event loop behind a progress screen.

        `work` is called on a worker thread and handed a `report(done, total,
        note)` it should call as it goes; `then(result, error)` runs back on the
        event loop once the screen is down. `error` is a `Cancelled`, an
        exception, or None.

        Everything slow here used to run inside the handler that started it,
        which meant the status line it set never painted and the app stopped
        answering keys until it finished. A minute of that is indistinguishable
        from a crash.
        """
        screen = ProgressScreen(title, cancellable)
        self.push_screen(screen)
        last = [0.0]
        started = time.monotonic()

        def report(done: int = 0, total: int = 0, note: str = "",
                   stage=None) -> None:
            if screen.cancelled:
                raise Cancelled(note or title)
            # Throttled: training reports every 64 KB, which on a large corpus
            # is thousands of hops onto the event loop for a bar that redraws
            # at best 60 times a second. A stage reaching its end is never
            # dropped, or a finished round could be left drawn part-full.
            now = time.monotonic()
            finished = stage is not None and stage[1] >= stage[2]
            if now - last[0] < 0.1 and not finished:
                return
            last[0] = now
            self.call_from_thread(screen.set_progress, done, total, note,
                                  stage)

        def finish(result, error) -> None:
            # What it cost, for whoever wants to report it: the save offer uses
            # it to say what skipping would throw away.
            self._last_slow_seconds = time.monotonic() - started
            if self.screen is screen:
                self.pop_screen()
            then(result, error)

        def runner() -> None:
            try:
                result, error = work(report), None
            except Cancelled as e:
                result, error = None, e
            except Exception as e:                  # noqa: BLE001 -- reported
                result, error = None, e
            self.call_from_thread(finish, result, error)

        self.run_worker(runner, thread=True, exclusive=False,
                        name=title, group="slow")

    def _corpus_paths(self, spec: str):
        """The files `spec` names, or a message saying why there are none."""
        paths = []
        for word in spec.split():
            p = Path(word).expanduser()
            if p.is_dir():
                paths += sorted(str(f) for f in p.rglob("*") if f.is_file())
            elif p.is_file():
                paths.append(str(p))
            else:
                return None, f"no such file or directory: {word}"
        if not paths:
            return None, f"nothing readable in {spec}"
        # Remembered so a training run can be pointed at the same corpus the
        # n-gram was trained on. The spec, not the expanded file list: a byte
        # model carves the corpus itself and wants the root, and a list of 967
        # paths on a command line is not a thing to hand to a process.
        self._corpus_root = spec.split()[0] if spec.split() else None
        return paths, None

    def _path_verdict(self, model, spec: str, paths, order: int) -> str:
        """Adopt a model trained off this machine and say how it fits.

        Split from the training so the training can run on a thread: everything
        here touches the session or the widgets and belongs on the event loop.
        """
        from ..predict.profiles import sanity_check
        if not model.trained_bytes:
            return f"read 0 bytes from {len(paths)} file(s)"
        # This route recorded nothing about what it read, so a model saved from
        # it read "provenance unknown" -- which is exactly the parentage problem
        # the save path exists to avoid. The root is what names it: one file
        # gives its own name, several give the directory they came from.
        root = Path(spec.split()[0]).expanduser()
        prov = {"path": str(root), "files": len(paths),
                "bytes": model.trained_bytes, "order": order}
        model.provenance = prov
        self.session.model = model
        self._solver = None                 # rebuild against the new model
        self._suggestions.clear()           # cached per class, now stale
        # Recorded like every other corpus route. This one alone was not, which
        # left "what was this model trained on" out of the ledger for the
        # commonest route into it.
        self.session.ledger.add("corpus", ledger.FOUND, str(root),
                                counts={"bytes": model.trained_bytes},
                                params=prov)
        self.refresh_all()

        score, warning = sanity_check(model, self.session.tape)
        head = (f"order-{order} model on {model.trained_bytes} bytes "
                f"from {len(paths)} file(s); {self._calibration_note(model)}")
        if warning:
            return f"{head}. WARNING: {warning}"
        return f"{head}, scoring {score:.1f} bits/byte on recovered data"

    def load_corpus(self, spec: str) -> str:
        """Train on the paths in `spec` and predict from that from now on.

        Synchronous, and returns the message to show. This is the path the
        launch flag and the headless callers take; the interactive one is
        `load_corpus_slow`, which is the same work on a thread. The fit is
        reported the same way either way, warning included: a corpus of the
        wrong KIND does not degrade to ignorance, it degrades past it, and the
        number that says so is the one thing worth seeing before trusting a fill.
        """
        from ..predict.context import Bidirectional

        paths, problem = self._corpus_paths(spec)
        if problem:
            return problem
        from ..predict import calibration, corpora
        order = corpora.resolve_order(self.order, _size_of(paths),
                                      corpora.sample_of(paths))
        model, _cal, train = calibration.train_files(paths, order)
        return self._path_verdict(model, spec, train, order)

    def load_corpus_slow(self, spec: str) -> None:
        """`load_corpus` with the training on a thread and a bar over it.

        A directory of prose is minutes of work. Doing it in the handler meant
        the app stopped answering keys for those minutes with the previous
        status still on screen.
        """
        from ..predict.context import Bidirectional
        from ..predict import corpora

        paths, problem = self._corpus_paths(spec)
        if problem:
            self.update_status(problem)
            return
        size = _size_of(paths)
        order, note = corpora.order_note(size, corpora.sample_of(paths),
                                         self.order)

        def work(report):
            # Holds ~5% of the corpus back and measures the model on it, so
            # the confidence floor means precision for this model.
            from ..predict import calibration
            return calibration.train_files(paths, order, progress=report)

        def then(result, error):
            if error is not None:
                self.update_status(self._slow_failed("training", error))
                return
            model, _cal, train = result
            self._trained(self._path_verdict(model, spec, train, order))

        # The whole corpus is read, so the cost goes in the title where it sits
        # in front of the operator for as long as the training runs.
        self.run_slow(f"{len(paths)} file(s), {size:,} bytes · {note}",
                      work, then)

    def _slow_failed(self, what: str, error) -> str:
        """One place for what a cancelled or broken background run says."""
        if isinstance(error, Cancelled):
            self.session.ledger.add(what, ledger.SKIPPED, "cancelled")
            return f"{what} cancelled; nothing changed"
        self.session.ledger.add(what, ledger.FAILED,
                                f"{type(error).__name__}: {error}")
        return f"{what} failed: {type(error).__name__}: {error}"

    def _trained(self, status: str) -> None:
        """Every route that trains ends here: say how it went, then ask whether
        to keep it, while the operator is still thinking about this model.

        Asking later does not work. Saving from f5 exists and is documented, but a model is
        only worth saving if you remember to before the process ends, and the
        moment the cost has just been paid is the only moment the question is
        obviously worth answering.
        """
        self.update_status(status)
        self._offer_to_save()

    def _offer_to_save(self) -> None:
        model = self.session.model
        if model is None:
            return
        from ..predict import corpora
        prov = getattr(model, "provenance", {}) or {}
        cost = getattr(self, "_last_slow_seconds", 0.0)
        from .progress import human_time
        took = f" it took {human_time(cost)} to build." if cost >= 1 else ""

        def done(name: str | None) -> None:
            if not name:
                self.update_status("not saved. f5 can save it any time before you "
                                   "quit; after that it is gone")
                return
            self.update_status(self._save_model(name))

        self.push_screen(
            PromptModal(f"keep this model?{took}  enter saves, esc skips:",
                        "path", str(corpora.suggest_model_path(prov))), done)

    def adopt_byte_model(self, path) -> str:
        """Load a byte model and say whether it is any use on THIS capture.

        The n-gram's rule, and the reason it exists, applies here and harder. A
        corpus model on the wrong kind of data scores -17.2 bits per byte
        against the -8.0 of knowing nothing; a byte model on the wrong kind
        reaches -28, measured. It does not degrade to ignorance, it degrades
        past it, and confidence is what the solver's threshold reads -- so the
        verdict is shown before the model is trusted rather than after a fill
        has gone wrong.
        """
        from ..predict.bytemodel import ClassFiller, fill_cost, sanity_check
        try:
            filler = ClassFiller.load(path)
        except Exception as e:
            return f"could not load that model: {type(e).__name__}: {e}"
        bits, top1, warning = sanity_check(filler, self.session.tape)
        self._filler = filler
        self._solver = None                 # rebuilt with the byte model in it
        self._suggestions.clear()
        head = (f"{filler.describe()}: {bits:.1f} bits/byte, "
                f"{top1:.0%} of known bytes right")
        if warning:
            return f"{head}. WARNING: {warning}"
        return f"{head}. {self.fill_estimate()} f5 switches the fill mode."

    def fill_estimate(self) -> str:
        """What filling everything open would cost, before it is started.

        The rate is this model's own, measured on this machine from the work it
        has already done -- sanity_check pushes a couple of hundred windows
        through it at load, so there is a figure before the operator can act.
        """
        from ..predict.bytemodel import fill_cost
        filler = getattr(self, "_filler", None)
        if filler is None:
            return ""
        windows, seconds = fill_cost(filler, self.session.tape)
        if not windows:
            return "Nothing left to fill."
        if not seconds:
            return f"Filling all of it is {windows:,} passes."
        how = (f"{seconds:.0f} s" if seconds < 90
               else f"{seconds / 60:.0f} min" if seconds < 5400
               else f"{seconds / 3600:.1f} h")
        return (f"Filling all {windows:,} passes takes about {how} "
                f"at {filler.rate():.0f}/s.")

    def action_fill_mode(self) -> None:
        """Cycle n-gram / byte model / both.

        Three modes and not four: linear mixing is measured closed, and so is
        routing on confidence. `both` takes the byte
        model's answer -- where they disagree it is right far more often -- and
        uses the n-gram's agreement as a confidence signal.
        """
        order = ["both", "slm", "ngram"]
        now = getattr(self, "_fill_mode", "both")
        self._set_fill_mode(order[(order.index(now) + 1) % len(order)])

    def _set_fill_mode(self, mode: str) -> None:
        if getattr(self, "_filler", None) is None:
            self.update_status("no byte model loaded; f5 lists them")
            return
        self._fill_mode = mode
        self._solver = None
        self._suggestions.clear()
        how = self.mode_name(self._fill_mode)
        cost = self.fill_estimate() if self._fill_mode != "ngram" else ""
        self.update_status(f"filling with {how}. {cost}".rstrip())

    def training_token(self):
        """A compact "there is a model training" badge, or None.

        Percentage and time REMAINING rather than time spent, because the
        question an operator has while a four-day run is going is when they get
        their machine back. The remaining figure is running-time, which excludes
        anything the box spent suspended.
        """
        try:
            from ..predict import training
            live = [r for r in training.runs()
                    if r.state() in ("running", "quiet", "finishing")]
        except Exception:
            return None
        if not live:
            return None
        r = live[0]
        p = r.progress()
        if p is None:
            return (f" training {r.name}: starting ", "magenta")
        eta = r.eta_seconds()
        left = f", ~{eta / 3600:.1f} h left" if eta else ""
        style = "yellow" if r.state() == "quiet" else "magenta"
        note = " (silent)" if r.state() == "quiet" else ""
        return (f" training {p.fraction:.0%}{left}{note} ", style)

    def action_train(self) -> None:
        """Start or watch a byte model. Never blocks: the run is detached."""
        from .modals import TrainingScreen

        def done(result):
            if not result:
                return
            what, payload = result
            if what == "open":
                self._show_training_log(payload)
            elif what == "save":
                self._save_run(payload)
            elif what == "start":
                self.update_status(self._start_training(payload))

        self.push_screen(TrainingScreen(self.session), done)

    def _calibrate_byte_model(self) -> None:
        """Calibrate the loaded byte model into a -cal copy and load that.

        On the held-out files it recorded when they are all on this machine;
        otherwise on text the operator names.
        """
        from ..predict import calibration
        filler = getattr(self, "_filler", None)
        if filler is None or not getattr(filler, "path", None):
            self.update_status("no byte model loaded; f5 lists them")
            return
        try:
            names = calibration.recorded_held_out(filler.path)
        except Exception:
            names = []
        root = getattr(self, "_corpus_root", None)
        specs, missing = calibration.find_held_out(names,
                                                   [root] if root else ())
        if names and not missing:
            self._run_calibration(filler.path, specs)
            return

        def done(text: str | None) -> None:
            if text and text.split():
                self._run_calibration(filler.path, text.split())
            else:
                self.update_status("not calibrated")

        self.push_screen(PromptModal(
            "text it never trained on, FILE or FILE@OFFSET, space-separated:",
            "~/corpus/a.txt ~/corpus/b.txt@48000"), done)

    def _run_calibration(self, src, specs) -> None:
        from ..predict import calibration
        specs = [str(Path(s).expanduser()) for s in specs]
        ngram = self.session.model

        def work(report):
            dst, cal = calibration.calibrate_checkpoint(
                src, specs, ngram,
                lambda d, n, note="": report(d, n, note))
            if cal is None:
                return ("not calibrated: too few classes in that text to fit "
                        "a curve; nothing written")
            loaded = self.adopt_byte_model(dst)
            return (f"{cal.summary()}; wrote {dst.name} and loaded it. "
                    f"{loaded}")

        self.run_slow("calibrating the byte model", work,
                      lambda msg, err: self.update_status(
                          msg if err is None
                          else self._slow_failed("calibration", err)))

    def _save_run(self, path) -> None:
        """Copy a finished run's model into the models directory, named."""
        from ..predict import corpora, training
        r = training.load(path)
        try:
            suggested = str(training.suggest_name(r))
        except Exception as e:
            self.update_status(f"cannot read {r.name}'s checkpoint: "
                               f"{type(e).__name__}: {e}")
            return

        def done(dest: str | None) -> None:
            if not dest:
                self.update_status(f"{r.name} not saved; its checkpoints stay "
                                   f"in {corpora.tilde(r.path)}")
                return
            self.run_slow("saving the byte model",
                          lambda _r: training.save_run(r, dest),
                          lambda out, err: self.update_status(
                              self._saved_run_note(out) if err is None
                              else self._slow_failed("saving", err)))

        self.push_screen(PromptModal(f"save {r.name} as:", "path", suggested),
                         done)

    def _saved_run_note(self, dest) -> str:
        from ..predict import corpora
        meta = corpora._model_meta(Path(dest))
        cal = meta.get("calibrated")
        how = (f"calibrated (best {measured_text(cal['best'])})" if cal
               else "uncalibrated; f5 can calibrate it")
        return f"saved {corpora.tilde(dest)}, {how}. f5 lists it"

    def _show_training_log(self, path) -> None:
        from ..predict import training
        r = training.load(path)
        p = r.progress()
        if p is None:
            self.update_status(f"{r.name}: {r.state()}, no reports yet")
            return
        self.update_status(
            f"{r.name}: {r.state()} {p.fraction:.1%}, "
            f"val top-1 {p.val_top1:.1f}%, log at {r.log}")

    def _start_training(self, cfg: dict) -> str:
        """Launch a run, detached, and hand back a line for the status bar.

        The corpus is whichever one the operator last pointed the model at, and
        the exclusions matter more than they look: training on the file being
        recovered makes every number afterwards fiction. There is no way to know
        the capture's source from here, so this refuses rather than guessing.
        """
        import sys
        import time
        from ..predict import training

        corpus = getattr(self, "_corpus_root", None)
        if not corpus:
            return ("no corpus chosen: pick one in f5 first, so the run knows "
                    "what to carve")
        from ..predict import carves
        paths, problem = self._corpus_paths(str(corpus))
        if problem:
            return problem
        usable, tried, _pos, verdict = carves.feasible(paths)
        if not usable:
            return f"will not train: {verdict}"

        # Training on the file being recovered makes every number afterwards
        # fiction, and the operator has no way to know their capture came out of
        # the corpus they just pointed at. The tape is the evidence, so it is
        # searched for -- and the offenders are EXCLUDED rather than the run
        # being refused, because a corpus that contains the capture is usually
        # the right corpus with one file too many in it.
        excluded = list(cfg.get("exclude") or [])
        if carves.enough_evidence(self.session.tape):
            hits = carves.contamination(self.session.tape, paths)
            for _frac, path in hits:
                name = Path(path).name
                if name not in excluded:
                    excluded.append(name)
            contaminated = (
                f". EXCLUDED {len(hits)} file(s) that contain this capture: "
                f"{', '.join(Path(p).name for _f, p in hits[:3])}"
                f"{'...' if len(hits) > 3 else ''}" if hits else "")
        else:
            contaminated = (". Too few long recovered runs to check the "
                            "corpus for this capture; exclude its source by "
                            "hand if it is in there")
        name = time.strftime("slm-%Y%m%d-%H%M%S")
        cmd = [sys.executable, "-m", "toeknife.predict.slmtrain",
               "--corpus", str(corpus),
               "--out", str(training.runs_dir() / name),
               "--d-model", str(cfg["d_model"]), "--layers", str(cfg["layers"]),
               "--heads", str(cfg["heads"]), "--window", str(cfg["window"]),
               "--steps", str(cfg["steps"])]
        if excluded:
            cmd += ["--exclude"] + excluded
        try:
            run = training.launch(name, cmd,
                                  spec={"marker": "slmtrain", "config": cfg,
                                        "corpus": str(corpus)})
        except Exception as e:
            return f"could not start training: {type(e).__name__}: {e}"
        note = (f". NOTE: {verdict}" if verdict else "") + contaminated
        return (f"training {run.name} started (pid {run.pid}). It carves the "
                f"corpus first, ~23 min before the first step, and "
                f"calibrates on held-back files at the end; quitting "
                f"ToeKnife leaves it running{note}")

    def action_corpus(self) -> None:
        """Choose a corpus, rather than being asked for a path and nothing else.

        The screen returns a spec; acting on it lives here so the screen only
        ever chooses. Whatever is picked, the model is measured against this
        stream afterwards -- picking the wrong KIND is worse than picking none.
        """
        def done(spec: str | None) -> None:
            if not spec:
                return
            if spec == "train:":
                self.action_train()
                return
            if spec == "cal:":
                self._calibrate_byte_model()
                return
            if spec.startswith("mode:"):
                self._set_fill_mode(spec[5:])
                return
            if spec == "save:":
                # The one row on that screen that writes rather than reads. It
                # goes through the same prompt as the save command so the path stays editable.
                self.action_save_model()
                return
            # Each of these is minutes of work on a corpus worth having, so they
            # go to a thread behind a progress screen. Loading a saved model is
            # the one that is genuinely fast (1.1s for a 71 MB model) and stays
            # on the event loop.
            if spec.startswith("type:"):
                self._corpus_by_type_slow(spec[5:].strip())
            elif spec.startswith("slm:"):
                # Loading is seconds (torch, then 77 MB of tensors) and scoring
                # it against the capture is seconds more, so it goes on the
                # slow path like every other model adoption.
                self.run_slow("loading the byte model",
                              lambda _r, s=spec[4:].strip():
                                  self.adopt_byte_model(s),
                              lambda msg, err: self.update_status(
                                  msg if err is None
                                  else self._slow_failed("byte model", err)))
            elif spec.startswith("saved:"):
                self._corpus_saved_slow(spec[6:].strip())
            elif spec.startswith("dl:"):
                self._corpus_download_slow(spec[3:].strip())
            elif spec.startswith("url:"):
                self._corpus_url_slow(spec[4:].strip())
            elif spec.startswith("path:"):
                self.load_corpus_slow(spec[5:].strip())
            else:
                self.load_corpus_slow(spec)

        self.push_screen(CorpusScreen(self.session), done)

    # ---- corpora from outside ----------------------------------------------

    def _adopt(self, model, provenance: dict, what: str) -> str:
        """Make this the session's model. Event loop only: it touches widgets.

        The lineage is attached here, on the one path every trained model takes,
        rather than by each caller -- which is where it went missing once already.
        """
        model.provenance = provenance
        self.session.model = model
        self._solver = None
        self._suggestions.clear()
        self.session.ledger.add("corpus", ledger.FOUND, what,
                                counts={"bytes": model.trained_bytes},
                                params={k: v for k, v in provenance.items()
                                        if k != "url"})
        return self._corpus_verdict(model, what)

    def _fetch_and_train(self, url: str, provenance: dict, name: str,
                         title: str) -> None:
        """Fetch, read, train, adopt -- one progress screen over all of it.

        The stages are separate work with separate failure modes but they are one
        wait as far as the operator is concerned, so they share one bar and the
        note says which stage is running.
        """
        from ..predict import corpora
        from ..predict.context import Bidirectional

        def work(report):
            report(0, 0, f"fetching {url}")
            path, size, cached = corpora.fetch(url, progress=report)
            report(0, 0, "reading what came back")
            blob, how = corpora.corpus_bytes(path)
            if not blob:
                return None, size, cached, how, 0
            order = corpora.resolve_order(self.order, len(blob),
                                          corpora.sample_of(blob))
            from ..predict import calibration
            model, _cal = calibration.train_blob(blob, order, progress=report)
            return model, size, cached, how, order

        def then(result, error):
            if error is not None:
                if isinstance(error, corpora.FetchError):
                    self.session.ledger.add("fetch", ledger.FAILED,
                                            f"{url}: {error}")
                    self.update_status(f"fetch failed: {error}")
                    return
                self.update_status(self._slow_failed("corpus", error))
                return
            model, size, cached, how, order = result
            self.session.ledger.add("fetch", ledger.FOUND, url,
                                    counts={"bytes": size},
                                    params={"cached": cached})
            where = "from the cache" if cached else "fetched"
            what = f"{name} ({where}, {size:,} bytes, {how})"
            if model is None:
                self.update_status(f"{what}: no training bytes came out of it")
                return
            self._trained(self._adopt(model, {**provenance, "read": how,
                                              "order": order}, what))

        self.run_slow(title, work, then)

    def _corpus_download_slow(self, key: str) -> None:
        """Fetch one of the catalogue entries. The screen has already shown the
        URL and the size and taken a second confirmation; this is the doing."""
        from ..predict import corpora
        entry = next((d for d in corpora.DOWNLOADS if d.key == key), None)
        if entry is None:
            self.update_status(f"no such download {key!r}")
            return
        self._fetch_and_train(
            entry.url, {"download": entry.key, "url": entry.url,
                        "licence": entry.licence}, entry.name,
            f"fetching and training on {entry.name}")

    def _corpus_url_slow(self, url: str) -> None:
        self._fetch_and_train(url, {"url": url}, url,
                              f"fetching and training on {url}")

    def _corpus_by_type_slow(self, spec: str) -> None:
        """Scan for files of one kind, read them, train.

        All three stages are slow and the first is the one with no natural
        percentage: a whole-filesystem sweep reads directory listings until it
        runs out of filesystem. It reports what it is doing rather than how far
        through it is, which is the honest version.
        """
        from ..predict import corpora
        from ..predict.context import Bidirectional
        parts = spec.split()
        key = parts[0] if parts else ""
        whole = "all" in parts[1:]
        if key not in corpora.BY_KEY:
            self.update_status(f"unknown corpus type {key!r}; the screen "
                               f"lists them")
            return

        def work(report):
            report(0, 0, "the whole filesystem" if whole
                   else "scanning the suggested roots")
            plan = corpora.plan(key, None, whole)
            if not plan.files:
                return None, plan, 0
            chunks = []
            for n, p in enumerate(plan.files, 1):
                report(0, 0, f"reading file {n} of {len(plan.files)}"
                             f"{' (inflating)' if plan.type.needs_inflating else ''}")
                chunks.append(corpora.training_bytes(plan.type, p))
            blob = b"".join(chunks)
            if not blob:
                return None, plan, 0
            order = corpora.resolve_order(self.order, len(blob),
                                          corpora.sample_of(blob))
            from ..predict import calibration
            model, _cal = calibration.train_blob(blob, order, progress=report)
            return model, plan, order

        def then(result, error):
            if error is not None:
                self.update_status(self._slow_failed("corpus", error))
                return
            model, plan, order = result
            if model is None:
                if not plan.files:
                    self.update_status(
                        f"no {plan.type.name} files found "
                        f"{'anywhere' if whole else 'in the suggested roots'}")
                else:
                    self.update_status("those files yielded no training bytes")
                return
            prov = {"type": key, "files": len(plan.files),
                    "bytes": model.trained_bytes, "order": order}
            self._trained(self._adopt(
                model, prov,
                f"{len(plan.files)} file(s), {model.trained_bytes:,} bytes"))

        self.run_slow(f"corpus of {corpora.BY_KEY[key].name}", work, then)

    def _adopt_saved(self, model, path: str = "") -> str:
        """Load a saved model and say so in one line that fits.

        It used to print the whole provenance as k=v pairs under "trained
        on", which ran off the status line before the calibration note and
        read as if a training had just happened. The
        lineage is one enter away in f5 and in the ledger.
        """
        from ..predict.profiles import sanity_check
        self.session.model = model
        self._solver = None
        self._suggestions.clear()
        prov = getattr(model, "provenance", {}) or {}
        self.session.ledger.add("corpus", ledger.FOUND,
                                f"loaded {path or 'a saved model'}",
                                counts={"bytes": getattr(model, "trained_bytes", 0)},
                                params={k: v for k, v in prov.items()
                                        if k != "url"})
        self.refresh_all()
        fwd = getattr(model, "forward", model)
        cal = getattr(model, "calibration", None)
        name = Path(path).name if path else "saved model"
        # Calibration first: it decides what the floor means, and a long line
        # loses its end off the status bar.
        head = (f"loaded {name}, "
                + (f"calibrated (best {measured_text(cal.best())})" if cal is not None
                   else "not calibrated")
                + f": order {getattr(fwd, 'order', '?')}, "
                  f"{getattr(fwd, 'trained_bytes', 0):,} training bytes")
        score, warning = sanity_check(model, self.session.tape)
        if warning:
            return f"{head}. WARNING: {warning}"
        return f"{head}; fits this capture at {score:.1f} bits/byte"

    def _corpus_saved_slow(self, spec: str) -> None:
        """Reading a model back is not free either: 71 MB of pickle is 1.1
        seconds, which is long enough to look like a dropped keystroke."""
        from ..predict import corpora
        from ..predict.context import ContextModel
        if Path(spec).expanduser().is_file():
            path = str(Path(spec).expanduser())
        else:
            saved = corpora.find_models()
            try:
                path = saved[int(spec) - 1]["path"]
            except (ValueError, IndexError):
                self.update_status(f"no saved model {spec!r}")
                return

        def work(report):
            report(0, 0, f"reading {corpora.tilde(path)}")
            return ContextModel.load(path)

        def then(model, error):
            if error is not None:
                self.update_status(f"cannot load that model: {error}")
                return
            self.update_status(self._adopt_saved(model, path))

        self.run_slow(f"loading {Path(path).name}", work, then,
                      cancellable=False)

    def _calibration_note(self, model) -> str:
        cal = getattr(model, "calibration", None)
        if cal is None:
            return ("not calibrated (too little held-out text), so the floor "
                    "is raw confidence")
        return (f"{cal.summary()}, so the floor is measured precision")

    def _corpus_verdict(self, model, what: str) -> str:
        """Say what was trained, then say whether it is any use HERE.

        It does not mention saving: every route that reaches this after training goes
        on to offer the save itself, and saying it twice in two lines reads as
        two different things to do.
        """
        from ..predict.profiles import sanity_check
        score, warning = sanity_check(model, self.session.tape)
        self.refresh_all()
        if warning:
            return f"trained on {what}. WARNING: {warning}"
        return (f"trained on {what}; scores {score:.1f} bits per byte on the "
                f"recovered data (-8.0 is knowing nothing); "
                f"{self._calibration_note(model)}")

    def action_hex_fill(self) -> None:
        pos = self.tape_view.cursor
        target, displaced = self._writable(pos)
        if target is None:
            self.update_status(self._refusal(pos))
            return
        # Say what is about to be overruled in the prompt itself: replacing a
        # model fill and filling an empty hole are different acts and the
        # keystroke is the same one.
        over = "" if displaced is None else \
            f" (over {displaced[0]} {_show_byte(displaced[1])})"

        def done(value: str | None) -> None:
            if not value:
                return
            try:
                byte = int(value.strip().removeprefix("0x"), 16)
            except ValueError:
                self.update_status(f"not a hex byte: {value}")
                return
            if not 0 <= byte <= 0xFF:
                self.update_status(f"0x{byte:x} is more than one byte; "
                                   f"enter 00 to ff")
                return
            self._fill(byte)

        self.push_screen(
            PromptModal(f"hex byte for position {pos}{over}:", "e.g. 0a or ff"),
            done)

    def action_goto(self) -> None:
        def done(value: str | None) -> None:
            if not value:
                return
            try:
                self.tape_view.goto(int(value, 0))
            except ValueError:
                self.update_status(f"not a number: {value}")

        self.push_screen(PromptModal("go to output position:", "e.g. -32 or 1024"), done)

    def action_export(self) -> None:
        def done(name: str | None) -> None:
            if not name:
                return
            buf, _mask, start = self.session.full()
            Path(name).write_bytes(buf)
            self.update_status(
                f"wrote {len(buf)} bytes to {name} (starts at output {start})")

        from .modals import WriteChoice
        s = self.session
        if getattr(s, "whole", None) is not None:
            _data, report = s.whole
            total = sum(x.uncompressed_size or 0 for x in report.streams)
            body = (f"This file is {len(report.streams)} gzip members, and "
                    f"{s.stream.name} is one of them. gunzip writes every "
                    f"member's output end to end.")
            options = [
                ("m", "member", f"{s.stream.name}: {len(s.tape):,} bytes"),
                ("a", "all", f"the whole file: {total:,} bytes by the "
                             f"trailers, this member as it stands and the "
                             f"others as they decode; a member that falls "
                             f"short is padded with '?'")]
        elif getattr(s, "file_fixes", None):
            body = (f"{len(s.file_fixes)} bit(s) of the input were repaired. "
                    f"The decoded bytes are what the stream holds; the "
                    f"repaired file is the input with those bits flipped "
                    f"back, readable by anything that reads the original.")
            options = [
                ("d", "decoded", f"the decoded bytes: {len(s.tape):,}"),
                ("r", "repaired", f"the repaired input file: "
                                  f"{len(s.file_data):,} bytes")]
        else:
            options = None
        if options is not None:
            def chosen(which: str | None) -> None:
                if which in ("member", "decoded"):
                    self.push_screen(PromptModal(
                        "write the decoded bytes to:", "path",
                        "recovered.bin"), done)
                elif which == "all":
                    self.push_screen(PromptModal(
                        "write the whole file to:", "path", "recovered.bin"),
                        self._export_all)
                elif which == "repaired":
                    self.push_screen(PromptModal(
                        "write the repaired file to:", "path",
                        f"repaired.{s.stream.container}"),
                        self._export_repaired)

            self.push_screen(WriteChoice(body, options), chosen)
            return
        self.push_screen(
            PromptModal("write recovered bytes to:", "path", "recovered.bin"), done)

    def _export_repaired(self, name: str | None) -> None:
        if not name:
            return
        from ..core import chunkcrc
        fixed = chunkcrc.repaired_file(self.session)
        Path(name).write_bytes(fixed)
        self.update_status(
            f"wrote {len(fixed):,} bytes to {name}: the input with "
            f"{len(self.session.file_fixes)} bit(s) flipped back")

    def _export_all(self, name: str | None) -> None:
        """Every member of the gzip, this one as it stands on the tape."""
        if not name:
            return
        from ..core import members
        data, report = self.session.whole
        here = next(i for i, st in enumerate(report.streams)
                    if st.offset == self.session.stream.offset)
        sessions = members.open_all(data, report)
        sessions[here] = self.session
        buf, mask, parts = members.join(report, sessions)
        Path(name).write_bytes(buf)
        ok = sum(1 for p in parts if p.verdict is not None and p.verdict.ok)
        self.update_status(
            f"wrote {len(buf):,} bytes to {name}: {len(parts)} members, "
            f"{sum(mask):,} known; {ok} of {len(parts)} match their trailers")

    def action_save_session(self) -> None:
        def done(name: str | None) -> None:
            if not name:
                return
            self.session.save(name)
            self.update_status(f"saved session to {name}")

        self.push_screen(
            PromptModal("save session to:", "path", "toeknife-session.json"), done)

    def action_save_model(self) -> None:
        """Keep the trained model, which until now died with the process.

        Training is the expensive step -- minutes and gigabytes for the corpus
        sizes that pay -- and f5 could only ever spend it again. ctrl+s saves the
        session's guesses; this saves the thing that produced them.
        """
        if self.session.model is None:
            self.update_status("no model to save; f5 trains or loads one")
            return
        from ..predict import corpora
        prov = getattr(self.session.model, "provenance", {}) or {}

        def done(name: str | None) -> None:
            if not name:
                return
            self.update_status("writing the model…")
            self.update_status(self._save_model(name))

        self.push_screen(
            PromptModal("save the trained model to:", "path",
                        corpora.tilde(corpora.suggest_model_path(prov))), done)

    def _save_model(self, name: str) -> str:
        """Write it, with its lineage, and say what it cost.

        A model with no provenance is a hypothesis generator of unknown
        parentage, so what it was trained on goes into the file rather than into
        the operator's memory.
        """
        from ..predict import corpora
        model = self.session.model
        if model is None:
            return "no model to save; f5 trains or loads one"
        prov = getattr(model, "provenance", {}) or {}
        path = Path(name).expanduser()
        try:
            if path.parent != Path(""):
                path.parent.mkdir(parents=True, exist_ok=True)
            model.save(path, prov)
        except (OSError, TypeError, ValueError) as e:
            self.session.ledger.add("model", ledger.FAILED,
                                    f"{path}: {type(e).__name__}: {e}")
            return f"could not write {path}: {e}"
        size = path.stat().st_size
        self.session.ledger.add("model", ledger.FOUND, str(path),
                                counts={"bytes": size},
                                params={k: v for k, v in prov.items()
                                        if k != "url"})
        lineage = "" if prov else " (no lineage recorded)"
        return (f"wrote {size:,} bytes to {corpora.tilde(path)}{lineage}; f5 lists it under "
                f"models you built earlier")


def run(session, title: str = "ToeKnife", solve_confidence: float = 0.995,
        self_train: bool = False, order: int = 4, shortlist: int = 16,
        crc_limit: int = 1 << 21) -> None:
    ToeKnifeApp(session, title, solve_confidence, self_train, order,
                shortlist, crc_limit).run()
