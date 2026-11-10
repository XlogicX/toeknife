"""Overlay screens: alignment picker, goto, hex entry, export, help."""
from __future__ import annotations

from pathlib import Path

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from ..predict.solver import (DICTIONARY_FLOOR, confidence_text,
                              measured_text, pct)
from .render import glyph

# Rungs for the confidence dial.  Even steps are wrong here: everything that
# separates a useful threshold from a useless one is crushed up against 1.0, and
# a linear 0.05 step from the 0.995 default steps straight over 0.99 and 0.95.
# These are the values the measurements in predict/solver.py are quoted at.
THRESHOLDS = (0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999)


def _cells(*values):
    """Table cells as Text, never as markup.

    Textual runs a string cell through `Text.from_markup`, and every column here
    carries decoded bytes: a preview of recovered text, or a proposed byte shown
    in brackets. Rich reads `[b]` as bold and `[/]` as a closing tag, so a
    proposal for 'b' silently vanished from the panel and a proposal for '/'
    crashed it outright. Wrapping in Text turns the markup parser off, which is
    the only correct setting for content that is data rather than formatting.
    """
    return [Text(v) if isinstance(v, str) else v for v in values]


def _tv(value):
    """A lineage value for the screen: paths under the operator's home as
    ~/..., everything else as it is."""
    from ..predict.corpora import tilde
    return tilde(value) if isinstance(value, str) else value


class AlignmentPicker(ModalScreen[int]):
    """Ranked starting offsets, each with the text it actually recovers.

    The ranking is advisory.  On a self-synchronising code several offsets are
    all individually valid, and when the disputed evidence is a handful of bytes
    no scorer separates them reliably -- but a person reading the previews picks
    the right one instantly.  So the previews are the point, and the score is
    just the sort order.
    """

    BINDINGS = [
        Binding("escape", "cancel", "cancel"),
        Binding("enter", "choose", "select"),
    ]

    def __init__(self, session, preview_width: int = 62):
        super().__init__()
        self.session = session
        self.preview_width = preview_width
        self.offsets: list[int] = []

    def compose(self) -> ComposeResult:
        with Vertical(id="align-box"):
            yield Label("select starting bit offset   (enter = use, esc = cancel)",
                        id="align-title")
            yield Static(self._explain(), id="align-note")
            yield DataTable(id="align-table", cursor_type="row", zebra_stripes=True)

    def _explain(self) -> Text:
        """What the rows mean, on the screen where they are read. The CLI's
        --list explained ⊂N and this did not, and `holes` read like a quality
        score."""
        s = self.session
        agree = s.agreement_point()
        t = Text()
        if agree is not None:
            t.append(f"Every reading agrees from bit {agree}; the rows differ "
                     f"only before it. The default start reads from there, so "
                     f"it shows only what all of them agree on"
                     f"{f' (now: bit {s.agreed_from})' if s.agreed_from else ''}"
                     f". Choosing a row starts exactly there, and its first "
                     f"bytes are right only if that row is the true start.\n",
                     style="dim")
        t.append("Fewer holes can mean a wrong start: a start off a symbol "
                 "boundary misreads a few bytes before it falls into step. "
                 "⊂N: a tail of offset N's decode, which recovers less.",
                 style="dim")
        return t

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("bit", "score", "lit", "mch", "out", "holes", "status", "recovered")
        cands = self.session.candidates()
        current = self.session.bit_offset
        cursor_row = 0
        for i, c in enumerate(cands):
            t = c.trace
            st = t.tape.stats()
            prev = "".join(glyph(t.tape.value_at(p))
                           for p in range(min(t.tape.length, self.preview_width)))
            status = t.status + (f" ⊂{c.subset_of}" if c.subset_of is not None else "")
            self.offsets.append(c.bit_offset)
            mark = "▶" if c.bit_offset == current else " "
            if c.bit_offset == current:
                cursor_row = i
            table.add_row(*_cells(f"{mark}{c.bit_offset}", f"{c.score:.2f}",
                                  str(t.n_literals), str(t.n_matches),
                                  str(st["out_len"]), str(st["holes"]), status,
                                  prev))
        table.move_cursor(row=cursor_row)
        table.focus()

    def action_choose(self) -> None:
        table = self.query_one(DataTable)
        if 0 <= table.cursor_row < len(self.offsets):
            self.dismiss(self.offsets[table.cursor_row])
        else:
            self.dismiss(None)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if 0 <= event.cursor_row < len(self.offsets):
            self.dismiss(self.offsets[event.cursor_row])

    def action_cancel(self) -> None:
        self.dismiss(None)


class ResyncPicker(ModalScreen[int]):
    """Block boundaries found by scanning, each with what it actually recovers.

    A fragment that starts inside a dynamic block cannot be read: the table
    defining its symbols is behind the damage.  The way out is not to rebuild
    that table but to skip to the next one, because streams emit block headers
    periodically and each carries a fresh table.  This is that choice, made
    visible: every row is decode-validated, so the output and hole counts are
    measured rather than predicted, and the operator picks by reading them.

    Ranked by how much each recovers, which usually means the earliest boundary
    wins -- everything before it is unaccounted for either way.
    """

    BINDINGS = [
        Binding("escape", "cancel", "cancel"),
        Binding("enter", "choose", "select"),
    ]

    def __init__(self, points, scanned_bits: int, total_bits: int,
                 current: int, preview_width: int = 52):
        super().__init__()
        self.points = points
        self.scanned_bits = scanned_bits
        self.total_bits = total_bits
        self.current = current
        self.preview_width = preview_width

    def compose(self) -> ComposeResult:
        with Vertical(id="align-box"):
            pct = 100 * self.scanned_bits / max(self.total_bits, 1)
            yield Label(
                f"block boundaries: {len(self.points)} found in the first "
                f"{pct:.0f}% of the data   (enter = resync here, esc = cancel)",
                id="align-title")
            yield DataTable(id="align-table", cursor_type="row",
                            zebra_stripes=True)

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns("bit", "byte", "kind", "blocks", "out", "holes",
                          "recovered")
        cursor_row = 0
        for i, p in enumerate(self.points):
            prev = ""
            if p.trace is not None:
                prev = "".join(glyph(p.trace.tape.value_at(q))
                               for q in range(min(p.trace.tape.length,
                                                 self.preview_width)))
            mark = "▶" if p.bit_offset == self.current else " "
            if p.bit_offset == self.current:
                cursor_row = i
            table.add_row(*_cells(f"{mark}{p.bit_offset}", f"{p.byte_offset:,}",
                                  p.kind, str(p.blocks), f"{p.out_bytes:,}",
                                  str(p.holes), prev))
        table.move_cursor(row=cursor_row)
        table.focus()

    def action_choose(self) -> None:
        table = self.query_one(DataTable)
        if 0 <= table.cursor_row < len(self.points):
            self.dismiss(self.points[table.cursor_row].bit_offset)
        else:
            self.dismiss(None)

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if 0 <= event.cursor_row < len(self.points):
            self.dismiss(self.points[event.cursor_row].bit_offset)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SolverScreen(ModalScreen[object]):
    """What the model would fill, and how sure it is about each one.

    Nothing is applied on opening. A suggestion is a hypothesis until the
    operator says otherwise, and the confidence figure ranks well without being
    a probability, so the decision belongs here rather than in the solver.

    The threshold moves from in here, because the right value is not knowable
    before you see the spread.  A stream self-trained on 47 recovered bytes tops
    out around 75%, one with a real corpus behind it clears 99%, and the default
    0.995 is the correct bar for exactly one of those.  Nudging it re-marks the
    list in place so you can watch the cut move over real proposals rather than
    guess a number, quit, and relaunch.
    """

    BINDINGS = [
        Binding("escape", "cancel", "close"),
        Binding("enter", "accept_one", "accept"),
        Binding("ctrl+a", "accept_all", "accept all shown"),
        Binding("ctrl+d", "drop", "skip this one"),
        Binding("plus", "raise_threshold", "stricter", show=False),
        Binding("equals_sign", "raise_threshold", "stricter", show=False),
        Binding("minus", "lower_threshold", "looser", show=False),
        Binding("underscore", "lower_threshold", "looser", show=False),
    ]

    def __init__(self, session, solver, threshold: float = 0.995,
                 proposals=None):
        super().__init__()
        self.session = session
        self.solver = solver
        self.threshold = threshold
        # Scoring every class is seconds of work, and the caller has usually just
        # done it behind a progress screen. Handing the result in means the panel
        # opens on it rather than repeating it with the app frozen.
        self.proposals = list(proposals) if proposals is not None else []
        self._scored = proposals is not None

    def compose(self) -> ComposeResult:
        with Vertical(id="solve-box"):
            yield Label("", id="solve-title")
            yield DataTable(id="solve-table", cursor_type="row",
                            zebra_stripes=True)
            yield Label("enter accept   ^a accept every ✓   - / + move the "
                        "threshold   ^d skip   esc close", id="solve-foot")

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_columns(" ", "at", "seen", "byte", "confidence", "runners up",
                          "context")
        self.refresh_rows()
        table.focus()

    def refresh_rows(self) -> None:
        """Re-ask the model.  Costs a full propose(); moving the dial does not."""
        if self._scored:
            # Already scored by whoever opened this, once.
            self._scored = False
        else:
            self.proposals = self.solver.propose()
        self._draw()

    def _draw(self) -> None:
        table = self.query_one(DataTable)
        row = table.cursor_row
        table.clear()
        for p in self.proposals:
            alts = " ".join(_show(v) for v, _ in p.runners[:4])
            # The mark follows the same number ^f reads: measured precision
            # when the model is calibrated, raw confidence when it is not.
            shown = confidence_text(p.confidence)
            if p.precision is not None:
                shown += f" → {measured_text(p.precision)}"
            table.add_row(*_cells(
                "✓" if p.gate >= self.threshold else " ",
                str(p.anchor), f"x{p.size}", _show(p.value),
                shown, alts,
                self._context_preview(p)))
        if table.row_count:
            table.move_cursor(row=max(0, min(row, table.row_count - 1)))
        above = sum(1 for p in self.proposals if p.gate >= self.threshold)
        self.query_one("#solve-title", Label).update(
            f"{len(self.proposals)} open classes, {above} at or above the "
            f"{pct(self.threshold)} floor (✓). Context is from each class's "
            f"most readable occurrence.")

    # ---- the dial ----------------------------------------------------------

    def _set_threshold(self, value: float) -> None:
        self.threshold = value
        # The value the operator settled on here is the one they meant, so ctrl+f
        # and any later panel inherit it rather than snapping back to the flag.
        self.solver.min_confidence = value
        self.app.solve_confidence = value
        self._draw()

    def action_raise_threshold(self) -> None:
        higher = [t for t in THRESHOLDS if t > self.threshold]
        if higher:
            self._set_threshold(higher[0])

    def action_lower_threshold(self) -> None:
        lower = [t for t in THRESHOLDS if t < self.threshold]
        if lower:
            self._set_threshold(lower[-1])

    def _context_preview(self, proposal) -> str:
        """The text around the class's most readable occurrence, so the
        suggestion can be read in place rather than judged as a bare byte.

        Not the first occurrence, which is what this used to show. On a real
        recovery a class's early positions sit inside the lost region where every
        neighbour is also unknown, so the column rendered as an unbroken run of
        block characters and told the operator nothing -- on a captured stream
        with a dropped packet, every row of it. The back-reference copies are the
        ones in text that decoded, and they are what makes a byte judgeable.
        """
        from ..predict.solver import most_readable
        tape = self.session.tape
        pos = most_readable(tape, proposal.positions)
        out = []
        for p in range(pos - 14, pos + 15):
            if p == pos:
                out.append(f"[{_show(proposal.value).strip(chr(39))}]")
                continue
            v = tape.value_at(p)
            out.append(chr(v) if v is not None and 0x20 <= v < 0x7F
                       else ("·" if v is not None else "░"))
        return "".join(out)

    def _selected(self):
        table = self.query_one(DataTable)
        if 0 <= table.cursor_row < len(self.proposals):
            return self.proposals[table.cursor_row]
        return None

    def action_accept_one(self) -> None:
        p = self._selected()
        if p is not None:
            self.dismiss(("one", p))

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        # DataTable claims enter for its own row selection before a screen
        # binding sees it, so accept from here too.
        if 0 <= event.cursor_row < len(self.proposals):
            self.dismiss(("one", self.proposals[event.cursor_row]))

    def action_accept_all(self) -> None:
        self.dismiss(("all", self.threshold))

    def action_drop(self) -> None:
        row = self.query_one(DataTable).cursor_row
        if 0 <= row < len(self.proposals):
            self.proposals.pop(row)
            self._draw()          # redraw, so the counts in the title agree

    def action_cancel(self) -> None:
        self.dismiss(None)


def _show(value: int) -> str:
    return repr(chr(value)) if 0x20 <= value < 0x7F else f"0x{value:02x}"


class PromptModal(ModalScreen[str]):
    """One-line text prompt."""

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    def __init__(self, prompt: str, placeholder: str = "", value: str = ""):
        super().__init__()
        self.prompt = prompt
        self.placeholder = placeholder
        self.value = value

    def compose(self) -> ComposeResult:
        with Vertical(id="prompt-box"):
            yield Label(self.prompt)
            yield Input(placeholder=self.placeholder, value=self.value, id="prompt-input")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(None)


HELP_TEXT = """\
Type a character and it fills the hole under the cursor, along with
every position the stream proves is the same byte.  Commands are on
ctrl and function keys, so typing never runs one.

  ctrl+k            menu: every command, searchable, with its key
  f7                what next: steps for this file, then the routes
                    to a lost table and which of them apply
  f1                this help
  ctrl+q            quit

The footer shows the keys for where you are: no model yet, filling,
or done.  The rest are in ctrl+k.

[b]moving[/b]
  ← → ↑ ↓           move the byte cursor
  home / end        start / end of row
  pgup / pgdn       page through the tape
  tab / shift+tab   next / previous hole
  ctrl+n / ctrl+p   next / previous back-reference
  enter             follow the selected match to its source bytes
  ctrl+g            go to an output position

[b]filling by hand[/b]
  any character     fill the hole under the cursor.  Over a model
                    fill or an earlier guess, it replaces it.  Over a
                    byte read from the bitstream it refuses: that
                    byte is evidence
  ctrl+x            the same, as a hex byte value
  backspace         clear the guess under the cursor
  ctrl+z / ctrl+r   undo / redo

Every byte you type is context the model did not have.  Fill what
you are sure of, then ctrl+f again.

[b]filling with a model[/b]
  f5                model: pick a corpus to train on (this machine,
                    a download, or a path), load a saved model, save
                    the loaded one, train or calibrate a byte model,
                    set the fill mode.  enter looks at a row, a second
                    enter acts.
                    ^d on a saved model moves it to the trash
  ctrl+f            fill the surest classes first, 32 at a time,
                    down to the floor
  ctrl+e            take back the model's fills, keep yours, and
                    fill again
  f4                proposals: every open class, ranked.  enter takes
                    one, ctrl+a takes every ✓, - and + move the floor
  ctrl+k            "set the confidence threshold" sets the floor
                    without scoring anything first

The floor is on the status line.  With a calibrated model it is a
precision: "floor 90% measured" fills only what this model got right
at least 90% of the time on text held back from its training.  f5
marks each saved model calibrated or not; every n-gram trained here
is.  A byte model brings its own curve when it has one, and the
floor follows the fill mode: "slm" with an uncalibrated byte model
is raw confidence.  The bits pane shows the figure for the byte
under the cursor.

ctrl+f stops and asks before filling inside a run of 256 or more
unread bytes: no model's confidence ranks the fills there.

ctrl+e is for after you correct something near an early wrong fill.
A filled class is never scored again, so the model keeps its mistake
until ctrl+e reopens it.

Pick a corpus like the data you are carving.  f5 scores a new model
on the bytes already recovered and warns below -8.0 bits per byte,
the score of knowing nothing: a model of the wrong kind of data does
worse than no model.

[b]the byte model[/b]
A byte model (.tks, needs torch) reads the tape itself, holes and
all, and answers a whole class at once.  Load one from f5's list of
saved models; f5 then offers three fill modes: byte model, n-gram,
or both.  With a calibrated byte model, [b]both[/b] fills as the byte
model alone and notes where the n-gram agrees; with an uncalibrated
one it takes the byte model's answer only where the n-gram names the
same byte and is sure.  A byte model is slower per hole than the
n-gram; f5 prints what a fill will cost before it starts.

[b]proving a fill[/b]
  f8                checksum: what the container recorded, what the
                    decode produced, and the verdict.  enter lets
                    CRC32 solve the last open classes (under 512);
                    ctrl+l alternates model and checksum
  ctrl+k            "fill from a suspected copy" lays a candidate file
                    over the capture.  Every decoded byte tests it;
                    the checksum proves or refutes the rest.  This is
                    how a lost packet gets recovered

The status line carries the verdict as one word: crc ok, crc
MISMATCH, or crc ? while holes remain.

[b]views[/b]
  f2                start bit for a fragment, with previews
  f3                resync: block boundaries found by scanning
  f6                show / hide the blocks pane
  ctrl+b            move into the blocks pane
  ctrl+t            table statistics for the block under the cursor
  escape            back to the tape

f2 picks where a headerless fragment starts, decoding with assumed
tables.  f3 finds a block header further on, after which every
block brings its own table.  On a capture that starts mid-stream, f3
usually recovers more.

Table statistics compares what the block's Huffman code lengths claim
(a 3-bit code means about one byte in eight) with what the block
emitted.  A table that does not fit its own block is a foreign or
wrongly recovered one.  tab switches alphabets.

The blocks pane follows the cursor; selecting a block jumps to its
first byte.

[b]keeping the result[/b]
  ctrl+w            write the recovered bytes to a file (? for unknown)
  ctrl+s            save the session: start, guesses, model fills

Slow work (training, scanning, scoring, ctrl+f) runs behind a progress
screen with a clock.  esc cancels it; nothing changes until it
finishes, and a cancelled ctrl+f keeps what it had filled, one ctrl+z
away.

[b]colours[/b]
  [green b]green[/]    read straight from the bitstream
  [cyan]cyan[/]     copied from a byte that is already known
  [yellow b]yellow[/]   typed by you, or implied by something you typed
  [magenta b]magenta[/]  filled by a model, or implied by a model fill
  [red d]red ░[/]    unknown and fillable
  [grey42]grey ░[/]   window space no back-reference reaches

Green and cyan are facts.  Yellow and magenta are guesses: yours, and
a model's.  Red is open.

[b]back-reference labels[/b]
A match is drawn as wide as the bytes it produced, so the span width
is the length.  The label shrinks to fit:

  ◀───6,35───▶   length 6, distance 35
  ─6,35─         same, arrows dropped
  ←4521          distance only; the width shows the length
  ───            no room; select it to read the bits

A comma means length,distance.  A number alone is a distance.
"""


def _si(n: int) -> str:
    """Bytes, in the units a person reads."""
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.0f} KB"
    return f"{n} B"


class CorpusScreen(ModalScreen[object]):
    """Where a corpus comes from: pick it off a list, look at it, then commit.

    The screen it replaces printed every option as text and asked the operator
    to retype one of them, which meant the two questions that decide whether a
    corpus is any use -- what KIND of data, and where it actually is -- were
    left as homework. Everything selectable here was discovered: the roots that
    exist on this machine, the models already on disk, the files a scan would
    read.

    Nothing acts on the first keystroke. Selecting a row shows what it would
    do -- which directories, how many files, how many bytes, which URL and how
    large -- and only a second enter commits. That two-step is the module's own
    rule ("nothing is read beyond directory listings until the operator accepts
    it") made visible, and it is the difference between a fetch you asked for
    and a fetch that just happened.

    Returns a spec the app acts on, so this screen only ever chooses.
    """

    BINDINGS = [
        Binding("escape", "close", "back / close"),
        Binding("ctrl+d", "trash", "move model to trash"),
    ]

    def __init__(self, session, model_dir="~/.toeknife/models"):
        super().__init__()
        self.session = session
        self.model_dir = model_dir
        self.armed: str | None = None       # what a second enter would commit
        self._ids: list[str] = []

    # ---- building the list -------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="corpus-box"):
            yield Static("Where should the model learn from?  A corpus of the "
                         "wrong KIND is worse than none: -17.2 bits per byte "
                         "against -8.0 for knowing nothing.", id="corpus-title")
            yield OptionList(id="corpus-list")
            yield Static("", id="corpus-detail")
            yield Input(placeholder="", id="corpus-input")
            yield Label("↑↓ move   enter look / commit   esc back", id="corpus-foot")

    def on_mount(self) -> None:
        self.query_one("#corpus-input", Input).display = False
        self._build()
        lst = self.query_one(OptionList)
        lst.focus()

    def _build(self) -> None:
        from ..predict import corpora
        lst = self.query_one(OptionList)
        lst.clear_options()

        lst.add_option(Option("1 · ON THIS MACHINE", id=None, disabled=True))
        # What is already to hand comes first: it is usually the best answer and
        # it was the one thing the old screen made you type from memory.
        for path, files, total in corpora.nearby_corpora(self._extra_dirs()):
            shown = self._short(path)
            lst.add_option(Option(
                f"  {'here':<9} {shown:<20} {files} file(s), {_si(total)}",
                id=f"path:{path}"))
        for ty in corpora.TYPES:
            roots = corpora.suggested_roots(ty)
            where = (", ".join(corpora.tilde(p) for p in roots[:3]) if roots
                     else "no suggested root exists here")
            mark = "  ⚠ inflated first" if ty.needs_inflating else ""
            lst.add_option(Option(
                f"  {ty.key:<9} {ty.name:<20} {where}{mark}",
                id=f"type:{ty.key}", disabled=not roots))
            lst.add_option(Option(
                f"      ↳ sweep the whole filesystem for {ty.key} "
                f"(slow, and reads everywhere)", id=f"type:{ty.key} all"))
        lst.add_option(Option("  ⌨  type a path instead…", id="ask:path"))

        lst.add_option(Option("", id=None, disabled=True))
        lst.add_option(Option("2 · FROM OUTSIDE: nothing is fetched until you "
                              "confirm", id=None, disabled=True))
        for d in corpora.DOWNLOADS:
            cached = corpora.cache_path(d.url)
            here = "  ✓ already cached" if cached.is_file() else ""
            lst.add_option(Option(
                f"  {d.type_key:<9} {d.name:<44} ~{_si(d.approx)}{here}",
                id=f"dl:{d.key}"))
        lst.add_option(Option("  ⌨  fetch a URL you name…", id="ask:url"))

        lst.add_option(Option("", id=None, disabled=True))
        # The directory, named where the models are listed: removing one by
        # hand was the only way, and nothing on screen said where to look.
        where = (corpora.tilde(corpora.model_dir()) if corpora.MODEL_DIRS else
                 ", ".join(corpora.tilde(p) for p in
                           corpora.model_search_paths(self._extra_dirs())))
        lst.add_option(Option(
            f"3 · MODELS YOU BUILT EARLIER: {where}   "
            f"(^d moves the highlighted one to the trash)", id=None,
            disabled=True))
        # Both kinds live in one directory and they are not interchangeable, so
        # the rows say which is which rather than leaving the operator to infer
        # it from a filename.
        self.models = corpora.find_models(self._extra_dirs())
        for m in self.models:
            if m.get("error"):
                lst.add_option(Option(
                    f"  {m['name']:<28} unreadable: {m['error']}",
                    id=None, disabled=True))
                continue
            if m.get("kind") == corpora.SLM:
                # A byte model answers a whole equality class off the tape and
                # loads as a byte filler, not as a Predictor the way an n-gram
                # does, so its row says what it is: size, window, calibration.
                prov = m.get("provenance") or {}
                name = (m["name"] if len(m["name"]) <= 28
                        else m["name"][:27] + "…")
                cal = m.get("calibrated")
                mark = (f"calibrated (best {measured_text(cal['best'])})  " if cal
                        else "uncalibrated  ")
                lst.add_option(Option(
                    f"  {name:<28} SLM {prov.get('params', '?'):>6}  "
                    f"window {m.get('window', '?')}  {mark}"
                    f"p38-5 {prov.get('p38_5', 'unscored')}",
                    id=f"slm:{m['path']}"))
                continue
            prov = m.get("provenance") or {}
            tag = ", ".join(f"{k}={_tv(v)}" for k, v in sorted(prov.items())) \
                or "provenance unknown"
            # Names are generated from provenance now and run long, so the
            # column is held rather than pushed: order and size are what the eye
            # is scanning for, and the full path is one enter away.
            name = m["name"] if len(m["name"]) <= 28 else m["name"][:27] + "…"
            cal = m.get("calibrated")
            mark = (f"calibrated (best {measured_text(cal['best'])})  " if cal
                    else "uncalibrated  ")
            lst.add_option(Option(
                f"  {name:<28} order {m.get('order', '?')}  "
                f"{m.get('trained_bytes', 0):,} B  {mark}{tag}",
                id=f"saved:{m['path']}"))
        if not self.models:
            # Naming only the directories that exist told the operator where it
            # looked and never where to put one, which reads as "this section is
            # empty for some other reason". The canonical directory is named
            # whether or not it exists yet.
            searched = ", ".join(str(p) for p in
                                 corpora.model_search_paths(
                                     self._extra_dirs()))
            lst.add_option(Option(
                f"  none saved yet. Looked in {searched}; models written from "
                f"here go to {corpora.tilde(corpora.model_dir())}", id=None,
                disabled=True))
        loaded = getattr(self.session, "model", None)
        if loaded is not None:
            prov = getattr(loaded, "provenance", {}) or {}
            # The path, not just the name: saving twice off one corpus must not
            # offer the row's name and then write a different file.
            lst.add_option(Option(
                f"  💾 save the model now loaded  → "
                f"{corpora.suggest_model_path(prov).name}", id="save:"))
        else:
            lst.add_option(Option(
                "  💾 save a model: train or load one first", id=None,
                disabled=True))

        # Byte-model training and the fill mode had keys of their own (f10 and
        # ctrl+o) until the footer ran out of room. They are model verbs, and
        # this is the model screen.
        lst.add_option(Option("", id=None, disabled=True))
        lst.add_option(Option("4 · BYTE MODEL", id=None, disabled=True))
        lst.add_option(Option(
            "  ⏳ train a byte model on the corpus chosen above "
            "(slow, detaches)", id="train:"))
        filler = getattr(self.app, "_filler", None)
        if filler is None:
            lst.add_option(Option(
                "  fill mode: load a byte model (an SLM row in 3) first",
                id=None, disabled=True))
        else:
            now = getattr(self.app, "_fill_mode", "both")
            for mode in ("both", "slm", "ngram"):
                mark = "✓" if mode == now else " "
                lst.add_option(Option(
                    f"  {mark} fill with {self.app.mode_name(mode)}",
                    id=f"mode:{mode}"))
            if (getattr(filler, "calibration", None) is None
                    and getattr(filler, "path", None)):
                lst.add_option(Option(
                    f"  ⚖ calibrate the loaded byte model  → "
                    f"{corpora.calibrated_name(filler.path)}", id="cal:"))

    def action_trash(self) -> None:
        """Move the highlighted saved model to the trash, after the operator
        types its name. Never a delete: see corpora.trash_model."""
        from ..predict import corpora
        lst = self.query_one(OptionList)
        if lst.highlighted is None:
            return
        oid = lst.get_option_at_index(lst.highlighted).id or ""
        if not (oid.startswith("saved:") or oid.startswith("slm:")):
            self._detail(Text("highlight a saved model first: ^d only "
                              "moves models", style="yellow"))
            return
        path = oid.split(":", 1)[1]
        meta = next((m for m in self.models if m["path"] == path), {})

        def done(gone) -> None:
            if not gone:
                self._detail(Text("kept", style="dim"))
                return
            self._build()
            highlight_first(self.query_one(OptionList))
            t = Text()
            t.append(f"moved to {gone}\n", style="bold")
            t.append(f"to restore it: mv {corpora.tilde(gone)} "
                     f"{corpora.tilde(path)}\n", style="dim")
            self._detail(t)

        loaded = getattr(self.session, "model", None)
        in_use = (loaded is not None and str(getattr(loaded, "path", "")) == path)
        self.app.push_screen(TrashModel(meta or {"path": path,
                                                 "name": Path(path).name},
                                        in_use), done)

    def _extra_dirs(self):
        """The capture's own directory: a model or a corpus built for this job
        is usually sitting next to the capture."""
        import os
        path = getattr(self.session, "path", None)
        return [os.path.dirname(os.path.abspath(str(path)))] if path else []

    def _short(self, path) -> str:
        """A path the operator can place at a glance, relative where it helps."""
        import os
        from ..predict import corpora
        try:
            rel = os.path.relpath(str(path), os.getcwd())
        except ValueError:
            return corpora.tilde(path)
        return f"./{rel}" if not rel.startswith("..") else corpora.tilde(path)

    # ---- looking before leaping --------------------------------------------

    def _detail(self, text: Text) -> None:
        self.query_one("#corpus-detail", Static).update(text)

    def _foot(self, msg: str) -> None:
        self.query_one("#corpus-foot", Label).update(msg)

    def on_option_list_option_highlighted(self, event) -> None:
        """Cheap description only. Nothing is read from disk or the network
        until a row is actually chosen."""
        from ..predict import corpora
        oid = event.option.id
        self.armed = None
        t = Text()
        # ^d exists only on a saved model, so it is drawn only there: a key
        # named in a section header nobody scrolls to was a key nobody knew.
        is_model = bool(oid) and (oid.startswith("saved:")
                                  or oid.startswith("slm:"))
        self._foot("↑↓ move   enter look / commit   "
                   + ("^d move to trash   " if is_model else "") + "esc back")
        if not oid:
            self._detail(t)
            return
        if oid.startswith("type:"):
            key = oid[5:].split()[0]
            ty = corpora.BY_KEY[key]
            whole = oid.endswith(" all")
            t.append(f"{ty.name}\n", style="bold")
            t.append(f"the stream would hold: {ty.contains}\n")
            t.append("where: " + ("the whole filesystem" if whole else
                                  ", ".join(str(p) for p in
                                            corpora.suggested_roots(ty))
                                  or "nowhere on this machine") + "\n",
                     style="dim")
            if ty.needs_inflating:
                t.append("these files are already compressed and will be "
                         "INFLATED before training\n", style="yellow")
            if ty.note:
                t.append(ty.note + "\n", style="italic dim")
            t.append("enter to scan and see what it found", style="cyan")
        elif oid.startswith("dl:"):
            d = next(x for x in corpora.DOWNLOADS if x.key == oid[3:])
            t.append(f"{d.name}\n", style="bold")
            t.append(f"{d.url}\n", style="dim")
            t.append(f"about {_si(d.approx)} · {d.licence} · trains as "
                     f"{d.type_key}\n")
            if d.note:
                t.append(d.note + "\n", style="italic dim")
            t.append("enter to see exactly what would be fetched", style="cyan")
        elif oid.startswith("slm:"):
            t.append("a byte model\n", style="bold")
            t.append(corpora.tilde(oid[4:]) + "\n", style="dim")
            t.append("Reads the tape, holes included, and answers a whole "
                     "class at once. Slower per hole than the "
                     "n-gram.\n", style="italic dim")
            t.append("enter to load it and score it against this capture",
                     style="cyan")
            t.append("    ^d moves it to the trash", style="dim")
        elif oid.startswith("saved:"):
            t.append("a model you built earlier\n", style="bold")
            t.append(corpora.tilde(oid[6:]) + "\n", style="dim")
            t.append("enter to read its lineage", style="cyan")
            t.append("    ^d moves it to the trash", style="dim")
        elif oid == "save:":
            t.append("keep the model that is loaded\n", style="bold")
            t.append("Training it took the minutes and the memory; without "
                     "this it goes away with the process, and every session "
                     "after this one spends them again.\n", style="italic dim")
            t.append("enter to see where it would go", style="cyan")
        elif oid.startswith("path:"):
            t.append("prose you already have\n", style="bold")
            t.append(oid[5:] + "\n", style="dim")
            t.append("Found next to the capture or in the project you are "
                     "working in, which is usually the closest-fitting corpus "
                     "available and the hardest one to remember the path of.\n",
                     style="italic dim")
            t.append("enter to see what it would read", style="cyan")
        elif oid == "train:":
            t.append("train a byte model\n", style="bold")
            t.append("Opens the training screen, which says what the run "
                     "costs at this machine's measured speed before anything "
                     "starts. The run detaches: quitting ToeKnife does not "
                     "stop it.\n", style="italic dim")
            t.append("enter to open it", style="cyan")
        elif oid.startswith("mode:"):
            t.append(f"fill with {self.app.mode_name(oid[5:])}\n",
                     style="bold")
            t.append("Decides what every fill after it means, which is why the "
                     "status line always shows the mode in force.\n",
                     style="italic dim")
            t.append("enter to switch", style="cyan")
        elif oid == "ask:path":
            t.append("Train on files you name: a directory is read "
                     "recursively, and several paths can be given at once.\n")
        elif oid == "ask:url":
            t.append("Fetch something not on the list. Only http and https, "
                     "and the size is checked against a 32 MB ceiling.\n")
        self._detail(t)

    def on_option_list_option_selected(self, event) -> None:
        oid = event.option.id
        if not oid:
            return
        if oid == "ask:path":
            self._ask("path", "a directory or file, e.g. docs  /usr/share/doc")
            return
        if oid == "ask:url":
            self._ask("url", "https://…")
            return
        # Neither reads, fetches or writes anything: one opens a screen with its
        # own confirmation and the other is undone by choosing again. The two
        # steps exist for what costs something, so these take one.
        if oid == "train:" or oid.startswith("mode:"):
            self.dismiss(oid)
            return
        if self.armed == oid:
            self.dismiss(self._spec(oid))
            return
        self._preview(oid)

    def _ask(self, kind: str, placeholder: str) -> None:
        inp = self.query_one("#corpus-input", Input)
        inp.display = True
        inp.placeholder = placeholder
        inp.value = ""
        inp.focus()
        self._kind = kind
        self._foot("type it, enter to look at it, esc to go back")

    def _spec(self, oid: str) -> str:
        return oid

    def _calibrate_preview(self) -> Text:
        """What calibrating the loaded byte model would measure it on."""
        from ..predict import calibration, corpora
        filler = self.app._filler
        t = Text()
        t.append("calibrate the loaded byte model\n", style="bold")
        t.append(f"  {corpora.tilde(filler.path)} → "
                 f"{corpora.calibrated_name(filler.path)} beside it\n",
                 style="dim")
        try:
            names = calibration.recorded_held_out(filler.path)
        except Exception:
            names = []
        root = getattr(self.app, "_corpus_root", None)
        found, missing = calibration.find_held_out(
            names, [root] if root else ())
        if names and not missing:
            t.append(f"  It held back {len(names)} file(s) when it trained, "
                     "and all of them are here. Each is damaged three ways "
                     "and every hole scored against the original.\n")
        elif names:
            t.append(f"  It held back {len(names)} file(s); {len(missing)} "
                     "are not on this machine. Pick their corpus first, or "
                     "name other text at the prompt.\n", style="yellow")
        else:
            t.append("  It does not record what it held back. The prompt "
                     "asks for files it never trained on; FILE@48000 starts "
                     "past the 48,000 bytes the trainer reads of each "
                     "file.\n", style="yellow")
        pool = ("the loaded n-gram" if getattr(self.session, "model", None)
                is not None else "an n-gram trained on each damaged document")
        t.append(f"  Candidates for each hole come from {pool}, as they "
                 "would in a fill.\n", style="dim")
        t.append("  The 19M model took 2.5 to 10 minutes per document.\n",
                 style="bold yellow")
        return t

    def _preview(self, oid: str) -> None:
        """Do the read-only work that says what committing would cost."""
        from ..predict import corpora
        t = Text()
        if oid == "cal:":
            self._detail(self._calibrate_preview())
            self.armed = oid
            self._foot("enter again = calibrate   esc = back")
            return
        if oid.startswith("path:"):
            self._detail(self._freeform_preview("path", oid[5:]))
            self.armed = oid
            self._foot("enter again = train on it   esc = back")
            return
        if oid.startswith("type:"):
            rest = oid[5:]
            key, whole = rest.split()[0], rest.endswith(" all")
            # The sweep row reads directory listings until it runs out of
            # filesystem, and "scanning…" written here never painted, because
            # the scan held the event loop that would have drawn it.
            def scan(report):
                report(0, 0, "the whole filesystem" if whole
                       else "the suggested roots")
                return corpora.plan(key, None, whole)

            def scanned(plan, error):
                if error is not None:
                    self._detail(Text(f"scan stopped: {error}", style="yellow"))
                    self._foot("esc = back")
                    return
                got = Text()
                got.append(plan.render() + "\n")
                if not plan.files:
                    self._detail(got)
                    self._foot("nothing to train on there; pick another row")
                    return
                self._order_line(got, plan.total_bytes)
                got.append("\nenter again to commit, esc to go back",
                           style="bold cyan")
                self._detail(got)
                self.armed = oid
                self._foot("enter again = do it   esc = back")

            self._detail(Text("scanning…", style="dim"))
            self.app.run_slow(f"scanning for {corpora.BY_KEY[key].name}",
                              scan, scanned)
            return
        elif oid.startswith("dl:"):
            d = next(x for x in corpora.DOWNLOADS if x.key == oid[3:])
            cached = corpora.cache_path(d.url)
            t.append(f"{d.name}\n", style="bold")
            t.append(f"  fetch  {d.url}\n")
            t.append(f"  about  {_si(d.approx)}, {d.licence}\n")
            t.append(f"  into   {cached}\n", style="dim")
            if cached.is_file():
                t.append("  already in the cache: committing reads it from "
                         "there\n", style="green")
            else:
                t.append("  this reaches the network. Nothing has been sent "
                         "yet.\n", style="yellow")
        elif oid == "save:":
            loaded = getattr(self.session, "model", None)
            if loaded is None:
                self._detail(Text("nothing is loaded to save"))
                return
            prov = getattr(loaded, "provenance", {}) or {}
            fwd = getattr(loaded, "forward", loaded)
            t.append("save the loaded model\n", style="bold")
            t.append(f"  into   {corpora.suggest_model_path(prov)}\n",
                     style="dim")
            t.append(f"  order  {getattr(fwd, 'order', '?')}, "
                     f"{'forward+reverse' if hasattr(loaded, 'forward') else 'forward only'}, "
                     f"{getattr(fwd, 'trained_bytes', 0):,} training bytes\n")
            t.append("  lineage: " + (", ".join(f"{k}={_tv(v)}" for k, v in
                                                sorted(prov.items())) if prov
                                      else "none recorded; the file will say so") +
                     "\n", style="dim" if prov else "yellow")
            # Measured, so the operator can decide before spending the disk:
            # 17.9 MB of Gutenberg prose at order 6 is 130 s to train and
            # 1.36 GB of RSS, against a 71.6 MB file that loads back in 1.1 s.
            t.append("  on prose the file runs 4-6x the corpus and writes in "
                     "seconds; loading it back is ~100x faster than training "
                     "it again. The path is editable at the prompt\n",
                     style="italic dim")
            self._detail(t)
            self.armed = oid
            self._foot("enter again = write it   esc = back")
            return
        elif oid.startswith("slm:"):
            path = oid[4:]
            m = next((x for x in self.models if x["path"] == path), None)
            prov = (m or {}).get("provenance") or {}
            t.append(f"{(m or {}).get('name', path)}\n", style="bold")
            t.append(f"  {corpora.tilde(path)}\n", style="dim")
            t.append(f"  {prov.get('params', '?')} parameters, window "
                     f"{(m or {}).get('window', '?')}, "
                     f"{prov.get('steps') or prov.get('predictions', '?')} "
                     f"steps/predictions trained\n")
            t.append("  trained on: " + (", ".join(f"{k}={_tv(v)}" for k, v in
                                                   sorted(prov.items()))
                                         if prov else "unknown") + "\n",
                     style="dim" if prov else "yellow")
            # Two carves and never an average: the same checkpoint reads 64.6%
            # of bytes on one article and 90.7% on another nine bytes apart in
            # size, so one figure would be a lie whichever it was.
            t.append("  Those scores are on the corpus it was trained for. "
                     "Loading it scores it against THIS capture.\n",
                     style="italic dim")
        elif oid.startswith("saved:"):
            path = oid[6:]
            m = next((x for x in self.models if x["path"] == path), None)
            if m is None:
                t.append("that model has gone away")
                self._detail(t)
                return
            prov = m.get("provenance") or {}
            t.append(f"{m['name']}\n", style="bold")
            t.append(f"  {corpora.tilde(m['path'])}\n", style="dim")
            t.append(f"  order {m.get('order', '?')}, "
                     f"{m.get('direction', 'unknown direction')}, "
                     f"{m.get('trained_bytes', 0):,} training bytes\n")
            t.append("  trained on: " + (", ".join(f"{k}={_tv(v)}" for k, v in
                                                   sorted(prov.items()))
                                         if prov else
                                         "unknown: saved before lineage was "
                                         "recorded") + "\n",
                     style="dim" if prov else "yellow")
        self._order_line(t, self._size_hint(oid))
        t.append("\nenter again to commit, esc to go back", style="bold cyan")
        self._detail(t)
        self.armed = oid
        self._foot("enter again = do it   esc = back")

    def _size_hint(self, oid: str) -> int | None:
        """How much this row would train on, where that is known in advance."""
        from ..predict import corpora
        if oid.startswith("dl:"):
            entry = next((d for d in corpora.DOWNLOADS if d.key == oid[3:]), None)
            return entry.approx if entry else None
        if oid.startswith("type:"):
            key = oid[5:].split()[0]
            plan = corpora.plan(key, None, oid.endswith(" all"))
            return plan.total_bytes
        return None

    def _order_line(self, t: Text, size: int | None) -> None:
        """The context order this corpus earns and what it will cost.

        Both, before the second enter, because the second enter is the only
        chance to say no. Nothing here caps anything: an estimate over the budget
        is a warning with the override beside it, and committing anyway is a
        decision the operator is allowed to make.
        """
        from ..predict import corpora
        if size is None:
            return
        chosen = getattr(self.app, "order", "auto")
        # The kind is unknown until something is read, so the estimate assumes
        # the expensive kind. Reading a sample here would break the screen's own
        # rule that nothing is read until the row is committed.
        order, note = corpora.order_note(size, None, chosen)
        over = "over the" in note
        t.append(f"  {note}\n", style="yellow" if over else "dim")
        # Sized for the expensive kind, because the cheap-vs-expensive question
        # is answered by reading the data and this screen has not read any yet.
        # Committing samples it for real and may land on a longer context or a
        # smaller figure -- about a third of this, if it turns out to be prose.
        t.append("  costed as if it were binary; nothing has been read yet\n",
                 style="dim")
        if over:
            t.append("  committing anyway is allowed; esc backs out\n",
                     style="yellow")

    def on_input_submitted(self, event) -> None:
        """Typed entries get the same two steps as picked ones.

        Especially the URL: an address typed into a box and a request going out
        should not be the same keystroke.
        """
        value = event.value.strip()
        if not value:
            return
        spec = f"{self._kind}:{value}"
        if self.armed == spec:
            self.dismiss(spec)
            return
        self._detail(self._freeform_preview(self._kind, value))
        self.armed = spec
        self._foot("enter again = do it   esc = back")

    def _freeform_preview(self, kind: str, value: str) -> Text:
        from pathlib import Path
        t = Text()
        if kind == "url":
            t.append("fetch\n", style="bold")
            t.append(f"  {value}\n")
            t.append("  http and https only; the size is unknown until the "
                     "server answers, and anything past 32 MB is refused\n",
                     style="dim")
            t.append("  this reaches the network. Nothing has been sent yet.\n",
                     style="yellow")
            t.append("\nenter again to fetch, esc to go back", style="bold cyan")
            return t
        files, total, missing = [], 0, []
        for word in value.split():
            p = Path(word).expanduser()
            if p.is_dir():
                found = [f for f in sorted(p.rglob("*")) if f.is_file()]
                files += found
            elif p.is_file():
                files.append(p)
            else:
                missing.append(word)
        for f in files:
            try:
                total += f.stat().st_size
            except OSError:
                pass
        t.append("train on what you named\n", style="bold")
        for word in missing:
            t.append(f"  no such file or directory: {word}\n", style="red")
        t.append(f"  {len(files)} file(s), {total:,} bytes\n")
        if files:
            t.append("  " + ", ".join(f.name for f in files[:6])
                     + (" …" if len(files) > 6 else "") + "\n", style="dim")
        if not files:
            t.append("  nothing readable there\n", style="yellow")
            return t
        self._order_line(t, total)
        t.append("\nenter again to train on it, esc to go back",
                 style="bold cyan")
        return t

    def action_close(self) -> None:
        inp = self.query_one("#corpus-input", Input)
        if inp.display:
            inp.display = False
            self.query_one(OptionList).focus()
            self._foot("↑↓ move   enter look / commit   esc back")
            return
        if self.armed:
            self.armed = None
            self._detail(Text(""))
            self._foot("↑↓ move   enter look / commit   esc back")
            return
        self.dismiss(None)


class ChecksumScreen(ModalScreen[object]):
    """What the wrapper recorded, what the decode produced, and what the
    checksum can still do about the difference.

    This existed only on the command line, which put the one check that can
    tell a good recovery from a plausible one outside the tool you recover in.
    A decode that is quietly wrong looks exactly like a decode that is right --
    every byte green or cyan, no holes, status `complete` -- and the only thing
    that knows better is thirty-two bits at the end of the entry.

    Read-only until asked. It returns a choice and the app performs it, because
    pinning bytes is an edit and edits belong in one place with the undo stack.
    """

    BINDINGS = [
        Binding("escape", "close", "close"),
        Binding("q", "close", "close"),
        Binding("enter", "pin", "pin with the checksum"),
        Binding("ctrl+l", "loop", "model + checksum, until neither moves"),
    ]

    def __init__(self, session, has_model: bool = False):
        super().__init__()
        self.session = session
        self.has_model = has_model

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="help-box"):
            yield Static(self._body(), id="help-body")
            yield Label(self._foot(), id="help-foot")

    def _foot(self) -> str:
        if self.session.stream is None or not self.session.tape.holes():
            return "esc to close"
        return ("enter = let the checksum pin what it can    "
                "^l = alternate model and checksum    esc = close")

    def _body(self) -> Text:
        s = self.session
        t = Text()
        t.append("Checksum\n\n", style="bold")
        if s.stream is None:
            t.append("Nothing to check against.\n", style="yellow")
            t.append("These bytes came from no container, so nothing recorded "
                     "a size or a checksum. Fills here can be judged by eye "
                     "and by the model's figures; none can be proven.\n",
                     style="dim")
            return t

        st = s.tape.stats()
        stream = s.stream
        t.append("what the container recorded\n", style="bold")
        # stream.container is the wrapper this came out of; session.container is
        # whatever the sliced bytes look like on their own, which is usually
        # "raw" and would be a lie here.
        t.append(f"  {stream.container}: {stream.name}\n")
        if stream.source and stream.source != "central":
            t.append(f"  metadata from: {stream.source}\n", style="dim")
        t.append(f"  size   {'unrecorded' if stream.uncompressed_size is None else f'{stream.uncompressed_size:,} bytes'}\n")
        for label, value in (("crc32", stream.crc32),
                             ("adler32", getattr(stream, "adler32", None))):
            if value is not None:
                t.append(f"  {label}  {value:08x}\n")
        if getattr(stream, "dictid", None) is not None:
            t.append(f"  dictionary  Adler-32 {stream.dictid:08x}, absent "
                     f"from this file\n")
        if stream.crc32 is None and getattr(stream, "adler32", None) is None:
            t.append("  no checksum: this container records none for this "
                     "entry\n", style="dim")

        t.append("\nwhat this decode produced\n", style="bold")
        t.append(f"  {st['out_len']:,} bytes, {st['known']:,} known, "
                 f"{st['holes']:,} unknown class(es) over "
                 f"{st['hole_positions']:,} position(s)\n")

        verdict = s.verify()
        t.append("\nverdict\n", style="bold")
        if verdict is None:
            t.append("  nothing to check against\n", style="dim")
            return t
        t.append("  " + verdict.summary() + "\n")
        if verdict.ok is True:
            t.append("  verified: the recovered bytes are the original bytes.\n",
                     style="bold green")
        elif verdict.ok is False:
            t.append("  FAILED: the decode does not match what the container "
                     "recorded.\n", style="bold red")
            if not st["holes"] and s.trace.status not in ("end", "eob"):
                hc = getattr(s, "headcheck", None)
                t.append(f"  The decode stopped at output "
                         f"{st['out_len']:,} ({s.trace.status}): everything "
                         f"after it is missing.", style="italic")
                if hc is not None:
                    t.append(f" The damage starts before the stop: "
                             f"{hc.summary()}.\n", style="italic")
                elif not st["out_len"]:
                    t.append("\n", style="italic")
                else:
                    t.append(" The bytes just before it may already be "
                             "damaged.\n", style="italic")
                from ..core import resume
                if resume.applies(s)[0]:
                    t.append("  f7 \"read on past the damage\" decodes the "
                             "rest and places it by the recorded size.\n",
                             style="italic")
            elif not st["holes"]:
                t.append("  Every byte decoded, so some bytes read from the "
                         "bitstream are wrong. ctrl+k \"fill from a suspected "
                         "copy\" shows where they differ from a copy.\n",
                         style="italic")
        elif getattr(s, "resumed_from", None) is not None:
            t.append("  inconclusive: the bytes decoded near the damage are "
                     "unproven, and the bytes on either side of the stop "
                     "hold some of its garbage, so the checksum cannot pass "
                     "on this tape. ctrl+k checks a suspected copy against "
                     "it instead.\n", style="yellow")
        else:
            t.append(f"  inconclusive: {st['holes']} unknown class(es) remain. "
                     f"Fill them and check again.\n", style="yellow")

        if st["holes"]:
            t.append("\nwhat the checksum can still do\n", style="bold")
            t.append(self._system())
        return t

    def _system(self) -> Text:
        """The linear view: unknown bytes enter CRC32 linearly, so this is a
        system of equations rather than a search. Adler-32 is a different
        animal and gets its own answer."""
        from ..predict.crcsolve import solve_from_crc
        stream = self.session.stream
        t = Text()
        if stream.crc32 is None:
            if getattr(stream, "adler32", None) is None:
                t.append("  nothing: this container recorded no checksum at "
                         "all, so there is no constraint to apply.\n",
                         style="dim")
                return t
            t.append("  Adler-32 only: two sums modulo 65521. It cannot be "
                     "solved like CRC32, but it still rules candidates out.\n")
            t.append("  enter shortlists each unknown byte with the model and "
                     "keeps only the combinations that reproduce the checksum. "
                     "Survivors are counted and shown: five unknown bytes under "
                     "Adler-32 alone once produced a wrong answer that "
                     "checksummed correctly.\n", style="dim")
            if not self.has_model:
                t.append("  There is no model loaded to shortlist with. f5 "
                         "gives it a corpus.\n", style="yellow")
            return t
        try:
            r = solve_from_crc(self.session)
        except Exception as e:              # a refusal is data; a crash is not
            t.append(f"  could not build the system: {e}\n", style="red")
            return t
        if not r.ok:
            t.append(f"  {r.reason}\n", style="dim")
            # The reason says "fill some with the model first", and with no
            # corpus the model is only this stream's own bytes -- one class out
            # of 4,041 on the lossy pcap. Without this line the advice sends the
            # operator to ^f, ^f sends them back here, and neither names f5.
            if not self.has_model:
                t.append("  No corpus is loaded, so the model has only this "
                         "stream's own bytes to learn from. f5 gives it "
                         "similar data first.\n", style="yellow")
        elif not r.free_bits:
            t.append(f"  {r.summary()}\n", style="green")
            t.append("  enter applies it.\n")
        else:
            chooses = ("the model chooses among what survives" if self.has_model
                       else "a model has to choose among what survives; f5 "
                            "loads one")
            t.append(f"  {r.summary()}\n")
            t.append(f"  Four or fewer classes are determined outright; past "
                     f"that the checksum removes 32 bits of freedom and "
                     f"{chooses}.\n", style="dim")
        return t

    def action_pin(self) -> None:
        self.dismiss("pin" if self.session.tape.holes() else None)

    def action_loop(self) -> None:
        self.dismiss("loop" if self.session.tape.holes() else None)

    def action_close(self) -> None:
        self.dismiss(None)


# Heat, by code length. Deliberately outside the tape's palette: green, cyan,
# yellow, magenta and red all mean provenance there, and a statistics pane that
# borrowed one would be claiming something about where a byte came from. This
# ramp is brightness and one cold hue, and the digit in each cell carries the
# actual number, so the colour never has to be decoded to read the grid.
HEAT_STYLES = ("bold bright_white", "bright_white", "bright_blue", "blue",
               "dim blue")
ABSENT_STYLE = "dim"


class TableStatsScreen(ModalScreen[None]):
    """What a block's Huffman tables say, beside what the block did with them.

    The tree already lists every code in the table, which answers "what is the
    code for 'e'" and nothing else. The question this answers is the one the
    table is actually evidence for: what did the encoder think this data looked
    like, and does the body agree.

    Two columns, and they are two different kinds of fact. The declared share is
    2**-bits, a property of the header alone -- a canonical Huffman code spends a
    3-bit code on something it expects one time in eight. The measured share is
    counted from the symbols the decoder read out of this block's body. Neither
    is an estimate of the other, and the gap between them is the interesting
    part: a table that does not fit its own block is what a foreign or a
    wrongly-recovered table looks like from here.
    """

    BINDINGS = [
        Binding("escape", "close", "close"),
        Binding("q", "close", "close"),
        Binding("tab", "next_alphabet", "next table"),
        Binding("shift+tab", "prev_alphabet", "previous table", show=False),
    ]

    def __init__(self, session, block):
        super().__init__()
        self.session = session
        self.block = block
        from ..core import tablestats
        self.alphabets = tablestats.alphabets(session.trace, block)
        self.which = 0

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="stats-box"):
            yield Static("", id="stats-body")
            yield Label("tab next table   esc close", id="stats-foot")

    def on_mount(self) -> None:
        self._draw()

    def action_close(self) -> None:
        self.dismiss(None)

    def action_next_alphabet(self) -> None:
        if self.alphabets:
            self.which = (self.which + 1) % len(self.alphabets)
            self._draw()

    def action_prev_alphabet(self) -> None:
        if self.alphabets:
            self.which = (self.which - 1) % len(self.alphabets)
            self._draw()

    def _draw(self) -> None:
        self.query_one("#stats-body", Static).update(self._body())

    # ---- the sections ------------------------------------------------------

    def _body(self) -> Text:
        from ..core import tablestats
        b = self.block
        t = Text()
        size = b.out_end - b.out_start
        t.append(f"Huffman tables: block #{b.index}  {b.btype_name}"
                 f"{'  [final]' if b.bfinal else ''}\n", style="bold")
        t.append(f"output {b.out_start:,}..{b.out_end - 1:,} ({size:,} B), "
                 f"header and body {b.nbits:,} bits\n", style="dim")
        if not self.alphabets:
            t.append("\nThis block carries no Huffman table. A stored block is "
                     "literal bytes with a length, which is the one case where "
                     "there is nothing here to look at.\n")
            return t

        for n, a in enumerate(self.alphabets):
            style = "bold reverse" if n == self.which else "dim"
            t.append(f" {a.name} ", style=style)
            if n < len(self.alphabets) - 1:
                t.append(" ")
        t.append("\n\n")

        a = self.alphabets[self.which]
        self._summary(t, a)
        if a.byte_space:
            self._grid(t, a)
        self._ranked(t, a)
        self._spend(t, a, tablestats)
        return t

    def _summary(self, t: Text, a) -> None:
        t.append(f"{a.used} of {a.size} symbols carry a code", style="bold")
        if a.used:
            t.append(f", {a.min_bits}-{a.max_bits} bits")
        t.append(f".  Kraft {a.kraft:.4f} ")
        if a.complete:
            t.append("(complete: the code space is exactly used up)\n",
                     style="dim")
        else:
            # Legal, and worth saying out loud: an incomplete code has bit
            # patterns that decode to nothing, which is how a truncated or
            # mis-read table usually announces itself.
            t.append("(INCOMPLETE: some bit patterns decode to nothing)\n",
                     style="yellow")
        if a.byte_space and a.absent_bytes:
            t.append(f"{a.absent_bytes} of the 256 byte values never appear as "
                     f"a literal here, which is itself evidence\n", style="dim")
        if a.emitted:
            t.append(f"the block used it {a.emitted:,} times")
            if a.partial:
                t.append("  (the block never reached its end-of-block, so "
                         "every measured count is a floor)", style="yellow")
            t.append("\n")
        else:
            t.append("nothing was decoded from it, so there is nothing "
                     "measured to compare against\n", style="yellow")
        if a.note:
            t.append(a.note + "\n", style="italic dim")
        t.append("\n")

    def _grid(self, t: Text, a) -> None:
        from ..core.tablestats import band_label, heat_band
        t.append("code length per byte value, in bits\n", style="bold")
        t.append("     " + " ".join(f"{c:x}" for c in range(16)) + "\n",
                 style="dim")
        for row in range(16):
            t.append(f" {row:x}0  ", style="dim")
            for col in range(16):
                v = row * 16 + col
                bits = a.lengths[v] if v < len(a.lengths) else 0
                if bits:
                    t.append(f"{bits:x}", style=HEAT_STYLES[heat_band(bits)])
                else:
                    t.append("·", style=ABSENT_STYLE)
                t.append(" " if col < 15 else "\n")
        # Two lines: the whole ladder on one would wrap inside an 88-column box,
        # and a wrapped legend reads as a paragraph rather than a key.
        for band in range(len(HEAT_STYLES)):
            t.append(f"  {band_label(band)}", style=HEAT_STYLES[band])
            if band == 2:
                t.append("\n")
        t.append("   · absent\n\n", style=ABSENT_STYLE)

    def _ranked(self, t: Text, a) -> None:
        from ..core.tablestats import heat_band
        # A short alphabet is shown whole: 30 distance codes or 19 code-length
        # codes fit, and truncating them hides symbols the grid does not cover.
        limit = len(a.stats) if len(a.stats) <= 32 else 20
        t.append("cheapest codes first\n", style="bold")
        t.append(f"  {'symbol':<20}{'bits':>5}{'declared':>10}"
                 f"  {'':<20}{'measured':>18}\n", style="dim")
        for st in a.stats[:limit]:
            t.append(f"  {st.name:<20}", style=HEAT_STYLES[heat_band(st.bits)])
            t.append(f"{st.bits:>4}b")
            t.append(f"{st.declared * 100:>9.2f}%  ")
            # 20 columns is 25% of the code space. A 1-bit code is legal in a
            # two-symbol alphabet and would otherwise run the column off the
            # edge and take the measured column with it.
            t.append(f"{'█' * min(20, round(st.declared * 80)):<20}",
                     style=HEAT_STYLES[heat_band(st.bits)])
            if a.emitted:
                t.append(f"{st.measured * 100:>8.2f}% ({st.count:,})\n")
            else:
                t.append(f"{'—':>10}\n", style="dim")
        rest = len(a.stats) - limit
        if rest > 0:
            t.append(f"  … {rest} more, down to "
                     f"{a.stats[-1].declared * 100:.3f}% at "
                     f"{a.stats[-1].bits} bits\n", style="dim")
        t.append("\n")

    def _spend(self, t: Text, a, tablestats) -> None:
        t.append("where the code space went\n", style="bold")
        for bits, n, share in a.per_length:
            t.append(f"  {bits:>2}b  {n:>3} symbol{'s' if n != 1 else ' '}  ",
                     style="dim")
            t.append(f"{'█' * min(40, round(share * 40)):<40}",
                     style=HEAT_STYLES[tablestats.heat_band(bits)])
            t.append(f"{share * 100:>6.2f}%\n")
        t.append("  each extra bit halves what a code claims, so a few short "
                 "codes can outweigh many long ones\n", style="italic dim")


def _wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width) or [""]


def highlight_first(lst: OptionList) -> None:
    """Put the cursor on the first row that can actually be chosen.

    Textual leaves `highlighted` at None until something moves, so a list whose
    first rows are disabled headings opens with nothing selected and enter does
    nothing at all -- which reads as the screen being broken rather than as an
    invitation to press an arrow key first.
    """
    for i in range(lst.option_count):
        if not lst.get_option_at_index(i).disabled:
            lst.highlighted = i
            return


class BitRotPicker(ModalScreen[object]):
    """Every flip that leaves the block parsing, and none of them chosen.

    This is a list rather than an answer, for a measured reason. Undoing the
    damage is guaranteed to be among the one-bit candidates -- inverting the
    flipped bit is one of the flips enumerated -- and measured over 250 flips of
    a real header it was alone only 129 times. Ordering is by how much the
    candidate recovers, which put the true repair first 78% of the time; the
    other 22% is why nothing here is applied without being picked.
    """

    BINDINGS = [
        Binding("escape", "cancel", "leave it alone"),
        Binding("q", "cancel", "leave it alone", show=False),
    ]

    def __init__(self, repair):
        super().__init__()
        self.repair = repair

    def compose(self) -> ComposeResult:
        with Vertical(id="routes-box"):
            yield Label("Header repairs that parse", id="routes-title")
            yield OptionList(id="routes-list")
            yield Static("", id="routes-detail")
            yield Label("↑↓ move   enter apply   esc leave it alone",
                        id="routes-foot")

    def on_mount(self) -> None:
        lst = self.query_one(OptionList)
        for n, c in enumerate(self.repair.candidates):
            lst.add_option(Option(f"  {n + 1:>3}. {c.label}", id=str(n)))
        self.query_one("#routes-detail", Static).update(
            Text(self.repair.summary(), style="bold"))
        lst.focus()
        highlight_first(lst)

    def on_option_list_option_highlighted(self, event) -> None:
        if event.option.id is None:
            return
        c = self.repair.candidates[int(event.option.id)]
        t = Text()
        t.append(self.repair.summary() + "\n", style="dim")
        t.append(f"flipping {', '.join(f'bit {b:,}' for b in c.bits)} "
                 f"leaves the parse ending {c.status!r}\n")
        t.append(f"  {c.out_len:,} bytes out, {c.known:,} known, "
                 f"{c.holes:,} hole class(es)\n", style="dim")
        if c.matches is True:
            t.append("  matches the recorded size and checksum\n",
                     style="bold green")
        elif c.matches is False:
            t.append("  does not match the recorded size and checksum\n",
                     style="red")
        t.append("enter applies it to the capture; esc leaves it alone",
                 style="bold cyan")
        self.query_one("#routes-detail", Static).update(t)

    def on_option_list_option_selected(self, event) -> None:
        if event.option.id is not None:
            self.dismiss(self.repair.candidates[int(event.option.id)])

    def action_cancel(self) -> None:
        self.dismiss(None)


class TablePicker(ModalScreen[object]):
    """Tables from a related capture that land on the next block header.

    Landing proves the shape; the score of the bytes the next block reads
    through the borrowed part is the evidence on the meanings. Nothing is
    adopted without being picked, and what is adopted stays unproven.
    """

    BINDINGS = [
        Binding("escape", "cancel", "leave it alone"),
        Binding("q", "cancel", "leave it alone", show=False),
    ]

    def __init__(self, result):
        super().__init__()
        self.result = result

    def compose(self) -> ComposeResult:
        with Vertical(id="routes-box"):
            yield Label("Borrowed tables that land", id="routes-title")
            yield OptionList(id="routes-list")
            yield Static("", id="routes-detail")
            yield Label("↑↓ move   enter adopt   esc leave it alone",
                        id="routes-foot")

    def on_mount(self) -> None:
        lst = self.query_one(OptionList)
        base = self.result.baseline
        for n, c in enumerate(self.result.landings):
            score = ("" if c.score is None else
                     f"   model {c.score:.2f}" + ("" if base is None else
                                                  f" / text {base:.2f}"))
            lst.add_option(Option(f"  {n + 1:>2}. {c.label}: "
                                  f"{c.out_len:,} B{score}", id=str(n)))
        self.query_one("#routes-detail", Static).update(
            Text(self.result.summary(), style="bold"))
        lst.focus()
        highlight_first(lst)

    def on_option_list_option_highlighted(self, event) -> None:
        if event.option.id is None:
            return
        c = self.result.landings[int(event.option.id)]
        base = self.result.baseline
        t = Text()
        t.append(self.result.summary() + "\n", style="dim")
        t.append(f"{c.label} lands its end-of-block on bit "
                 f"{self.result.boundary:,} from {c.starts} of 48 starts\n")
        t.append(f"  {c.out_len:,} bytes, {c.known:,} known; the next block "
                 f"reads {c.seam:,} of its bytes through them\n", style="dim")
        if c.score is not None and base is not None:
            good = c.score >= base
            t.append(f"  those score {c.score:.2f} against {base:.2f} for the "
                     f"next block's own text: "
                     + ("they read like it\n" if good else
                        "they read worse than it\n"),
                     style="green" if good else "yellow")
        t.append("enter adopts it; the borrowed bytes stay unproven (blue)",
                 style="bold cyan")
        self.query_one("#routes-detail", Static).update(t)

    def on_option_list_option_selected(self, event) -> None:
        if event.option.id is not None:
            self.dismiss(self.result.landings[int(event.option.id)])

    def action_cancel(self) -> None:
        self.dismiss(None)


class RouteBoard(ModalScreen[object]):
    """What next: the few steps worth taking now, then every route to a lost
    table, why the unavailable ones are not, and enter to run the ones that are.

    The NEXT section was added when the manual found people stuck between ^f
    and f8 with no corpus loaded. It reads the same stage the footer does.

    It used to be read-only, which turned out to be the wrong half of the idea.
    Making an inapplicable route say why is the point and stays: a silently
    skipped check and a thrown-away right answer look
    identical from outside. But a board that listed a route as available and gave
    no way to start it left the operator to work out which function key it hid
    behind, and for two routes there was no key at all: the related-capture
    transplant and the known-plaintext judge were reachable only from the command
    line.

    Rows that need evidence are selectable too. A route that says "needs a second
    capture" is not a dead end; it is a prompt waiting to be asked.

    Nothing runs on highlight, and nothing here picks an answer. Returns a route
    key for the app to dispatch, so the screen still only ever chooses.
    """

    BINDINGS = [
        Binding("escape", "close", "close"),
        Binding("q", "close", "close"),
    ]

    def __init__(self, session, runnable=(), askable=()):
        super().__init__()
        self.session = session
        # What the app can actually dispatch, and which of those can ask for the
        # evidence they lack. The screen does not guess: a route it cannot start
        # is shown as listed-only rather than offered and then refused.
        self.runnable = set(runnable)
        self.askable = set(askable)
        self.verdicts = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="routes-box"):
            yield Label("What next for this capture", id="routes-title")
            yield OptionList(id="routes-list")
            yield Static("", id="routes-detail")
            yield Label("↑↓ move   enter run   esc close", id="routes-foot")

    def on_mount(self) -> None:
        self._build()
        lst = self.query_one(OptionList)
        lst.focus()
        highlight_first(lst)

    def next_steps(self):
        """The few things worth doing now, as (action, label, why) rows.

        Read off the session the way the footer's stage is, so the board and
        the footer can never disagree about where a recovery is. An action of
        None is a fact the operator needs and cannot act on from here -- the
        checksum being out of reach is the one that sent people round in
        circles.
        """
        from ..predict.crcsolve import MAX_CLASSES as limit
        s = self.session
        app = self.app
        stage = app.stage() if hasattr(app, "stage") else "filling"
        holes = len(s.tape.holes())
        stream = s.stream
        crc = stream is not None and stream.crc32 is not None
        rows = []
        from ..core import resume
        from ..core.routes import BY_KEY
        if BY_KEY["bitrot"].verdict(s).ok:
            rows.append(("bitrot", "repair the header: try every one-bit flip",
                         "the parse fails with its bits present; each flip "
                         "that parses is checked against the recorded "
                         "checksum"))
        if s.mode == "stream" and s.resumed and s.bit_offset >= 8:
            rows.append(("related", "read the part before this header with a "
                         "table from a related capture",
                         f"{s.bit_offset // 8:,} bytes before bit "
                         f"{s.bit_offset:,} belong to a block whose table is "
                         f"gone; another copy of the content may carry it"))
        if resume.applies(s)[0]:
            rows.append(("resume", "read on past the damage",
                         "the decode stopped part way and the recorded size "
                         "says where the rest goes; what it reads stays "
                         "unproven until a copy matches the checksum"))
        from ..core import chunkcrc
        bad = chunkcrc.bad_segments(s)
        if bad:
            rows.append(("chunk_repair", "repair flipped bits with the chunk "
                         "CRC",
                         f"chunk {bad[0] + 1} of {len(stream.segments)} fails "
                         f"its CRC32, which covers the compressed bytes; one "
                         f"or two flipped bits can be solved for"))
        if (stream is not None and getattr(stream, "dictid", None) is not None
                and s.tape.min_pos < 0 and holes):
            rows.append(("fill_dictionary", "find the preset dictionary in a "
                         "file",
                         f"the stream copies from {-s.tape.min_pos:,} bytes "
                         f"of a dictionary with Adler-32 "
                         f"{stream.dictid:08x}; the program that wrote it "
                         f"usually holds it, and the Adler-32 finds it"))
        if (crc and stream.uncompressed_size and not len(s.tape)
                and s.trace.status not in ("end", "eob")):
            # Nothing decoded, so nothing on the tape can test a copy; the
            # recorded size and CRC still can, on a slice of it alone.
            rows.append(("fill_from_copy", "check a suspected copy against "
                         "the recorded checksum",
                         f"nothing decoded, but the container recorded "
                         f"{stream.uncompressed_size:,} bytes and their "
                         f"CRC32; a slice of a copy that matches both is the "
                         f"original"))
        if s.mode == "fragment" and s.trace.status not in ("end", "eob"):
            rows.append(("resync", "scan for the next block header (f3)",
                         f"the decode stops after {len(s.tape):,} bytes: a "
                         f"carve that starts inside a block has no table, "
                         f"and the next header carries its own"))
        if s.mode == "fragment" and holes:
            rows.append(("align", "check the starting bit (f2)",
                         "a fragment's decode is only as good as where it "
                         "starts; the picker ranks the candidates"))
        if stage == "no model":
            rows.append(("corpus", "choose a model (f5)",
                         "no corpus is loaded, so a fill learns from this "
                         "stream's own bytes and finds next to nothing"))
            rows.append(("solve_confident", "fill confident anyway (^f)",
                         "cheap to try, and one ^z takes it back"))
        elif stage == "filling":
            rows.append(("solve_confident", "fill confident (^f)",
                         f"{holes:,} classes are open; it fills surest first "
                         f"and stops at the threshold"))
            rows.append(("solver_panel", "review the proposals (f4)",
                         "what the model would put where, and how sure it "
                         "is; lower the threshold there to fill more"))
        from ..predict.crcsolve import unproven
        if unproven(s) and (crc or (stream is not None and stream.adler32)):
            # Read past damage: the checksum cannot pass while garbage sits on
            # the tape, and solving it would make it lie. A copy that
            # matches it on its own is the proof left.
            rows.append(("fill_from_copy", "check against a suspected copy "
                         "of the content",
                         "the bytes decoded near the damage are unproven; a copy "
                         "that matches the recorded checksum proves the "
                         "entry and shows where the capture is damaged"))
        elif holes and crc:
            if holes <= limit:
                rows.append(("checksum", "pin the rest with the checksum (f8)",
                             f"{holes} classes is inside the {limit} the "
                             f"checksum can work on"))
            else:
                rows.append((None, f"the checksum can help below {limit} "
                             f"open classes; {holes:,} are open now",
                             "fill with a model first"))
            # The route that ends in a proof when a model cannot: a lost
            # packet is beyond any corpus tried, but a candidate copy of the
            # content can be checked against the CRC byte for byte.
            rows.append(("fill_from_copy", "fill from a suspected copy of "
                         "the content",
                         "a later fetch, a cached or public copy: every "
                         "decoded byte tests it, and the checksum proves or "
                         "refutes the rest"))
        verdict = s.verify() if not holes else None
        if (not holes and not unproven(s)
                and s.trace.status in ("end", "eob")
                and verdict is not None and verdict.ok is False):
            # Everything decoded and the checksum still disagrees: some bytes
            # read from the bitstream are wrong, and nothing on the tape says
            # which. A copy of the original is the one
            # thing that can point at them.
            rows.append((None, "the checksum says this decode is wrong "
                         "somewhere", "every byte decoded, so the damage is "
                         "in bytes that look like facts"))
            rows.append(("fill_from_copy", "compare against a suspected "
                         "copy of the content", "where the copy and the "
                         "decode differ is where to look"))
        if not holes:
            if crc or (stream is not None and stream.uncompressed_size):
                rows.append(("checksum", "check against what the container "
                             "recorded (f8)",
                             "the only check that can tell a right recovery "
                             "from a plausible one"))
            if len(s.tape):
                rows.append(("export", "write the recovered output (^w)", ""))
                rows.append(("save_session", "save this session (^s)", ""))
        if stream is None:
            rows.append((None, "no container: nothing recorded a size or a "
                         "checksum", "so no fill here can be confirmed, only "
                         "judged"))
        return rows

    def _build(self) -> None:
        from ..core import routes
        lst = self.query_one(OptionList)
        lst.clear_options()
        lst.add_option(Option("NEXT", id=None, disabled=True))
        self.steps = {}
        for action, label, why in self.next_steps():
            if action is None:
                lst.add_option(Option(f"  ·      {label}", id=None,
                                      disabled=True))
                if why:
                    lst.add_option(Option(f"          {why}", id=None,
                                          disabled=True))
                continue
            self.steps[action] = (label, why)
            lst.add_option(Option(f"  ▸ do   {label}", id=f"act:{action}"))
        lst.add_option(Option("", id=None, disabled=True))
        lst.add_option(Option("ROUTES TO A LOST TABLE", id=None,
                              disabled=True))
        fam = None
        for v in routes.board(self.session):
            r = v.route
            self.verdicts[r.key] = v
            if r.family != fam:
                fam = r.family
                lst.add_option(Option("", id=None, disabled=True))
                lst.add_option(Option(f"{fam.upper()}", id=None, disabled=True))
            can = r.key in self.runnable and (v.ok or r.key in self.askable)
            if v.ok:
                mark = "▸ run  " if can else "  ok   "
            elif r.key in self.askable and r.key in self.runnable:
                mark = "▸ ask  "
            elif v.blocked_by_status:
                mark = "  unshipped"
            else:
                mark = "  no   "
            lst.add_option(Option(f"  {mark} {r.name}",
                                  id=f"route:{r.key}", disabled=not can))
            if not can:
                # Cost, provenance and reason go on the rows rather than in the
                # detail pane. A disabled row cannot take the highlight, so
                # anything kept in the pane would be unreachable for exactly the
                # routes whose explanation matters most -- and "a route that does
                # not apply says why" is the whole point of the board. kp-climb is
                # the case that proves it: what an operator needs to know is not
                # "needs plaintext" but that this was measured and does not pay,
                # and that sentence lives in `source`.
                lst.add_option(Option(f"          {r.cost}"
                                      + (f" · {r.source}" if r.source else ""),
                                      id=None, disabled=True))
                for line in _wrap(v.reason, 72):
                    lst.add_option(Option(f"          {line}", id=None,
                                          disabled=True))
        if len(self.session.ledger):
            lst.add_option(Option("", id=None, disabled=True))
            lst.add_option(Option("WHAT HAS RUN ON THIS CAPTURE", id=None,
                                  disabled=True))
            for line in self.session.ledger.render().splitlines():
                lst.add_option(Option("  " + line, id=None, disabled=True))

    def _detail(self, text: Text) -> None:
        self.query_one("#routes-detail", Static).update(text)

    def on_option_list_option_highlighted(self, event) -> None:
        oid = event.option.id
        if oid and oid.startswith("act:"):
            label, why = self.steps[oid[4:]]
            t = Text()
            t.append(f"{label}\n", style="bold")
            if why:
                t.append(f"  {why}\n", style="dim")
            t.append("enter does it", style="bold cyan")
            self._detail(t)
            return
        if not oid or not oid.startswith("route:"):
            self._detail(Text())
            return
        v = self.verdicts[oid[6:]]
        r = v.route
        t = Text()
        t.append(f"{r.name}\n", style="bold")
        t.append(f"  needs   {r.needs}\n", style="dim")
        t.append(f"  costs   {r.cost}\n", style="dim")
        if r.source:
            t.append(f"  from    {r.source}\n", style="dim")
        if v.ok:
            t.append(f"  {v.reason}\n", style="green")
            t.append("enter runs it", style="bold cyan")
        elif r.key in self.askable and r.key in self.runnable:
            t.append(f"  {v.reason}\n", style="yellow")
            t.append("enter asks for what it needs", style="bold cyan")
        elif v.blocked_by_status:
            t.append(f"  {v.reason}\n", style="yellow")
            t.append("not in this release", style="dim")
        else:
            t.append(f"  {v.reason}\n", style="italic")
            t.append("does not apply to this capture", style="dim")
        self._detail(t)

    def on_option_list_option_selected(self, event) -> None:
        oid = event.option.id
        if oid and (oid.startswith("route:") or oid.startswith("act:")):
            self.dismiss(oid[6:] if oid.startswith("route:") else oid)

    def action_close(self) -> None:
        self.dismiss(None)


class ConfirmScreen(ModalScreen[bool]):
    """A title, what it would cost, and enter or esc."""

    BINDINGS = [Binding("enter", "go", "go"),
                Binding("escape", "stop", "stop")]

    def __init__(self, title: str, body: str, foot: str):
        super().__init__()
        self.title_text, self.body, self.foot = title, body, foot

    def compose(self) -> ComposeResult:
        t = Text()
        t.append(self.title_text + "\n\n", style="bold")
        t.append(self.body + "\n")
        with Vertical(id="threshold-box"):
            yield Static(t)
            yield Label(self.foot, id="threshold-foot")

    def action_go(self) -> None:
        self.dismiss(True)

    def action_stop(self) -> None:
        self.dismiss(False)


class ChunkRepairScreen(ModalScreen[bool]):
    """What the chunk CRCs say about flipped bits, and the repair to apply."""

    BINDINGS = [Binding("enter", "go", "apply"),
                Binding("escape", "stop", "leave it")]

    def __init__(self, result, total: int):
        super().__init__()
        self.result, self.total = result, total

    def compose(self) -> ComposeResult:
        r = self.result
        proven = [f for f in r.fixes if f.proven]
        t = Text()
        t.append("Repair with the chunk CRC\n\n", style="bold")
        if len(proven) == 1:
            t.append("ONE REPAIR, PROVEN TWICE.\n", style="bold green")
            t.append(f"{proven[0].label(self.total)}.\n\n", style="green")
            t.append("Flipping it back restores the chunk's CRC32 over the "
                     "compressed bytes and the stream's Adler-32 over the "
                     "decoded ones.\n", style="green")
            foot = "enter applies it   esc leaves the data as it is"
        else:
            t.append(r.summary(self.total) + ".\n", style="yellow")
            foot = "esc close"
        if r.candidates > len(proven):
            t.append(f"\n{r.candidates} flip(s) restore a chunk CRC; "
                     f"{len(proven)} also match the Adler-32.\n", style="dim")
        with Vertical(id="threshold-box"):
            yield Static(t)
            yield Label(foot, id="threshold-foot")

    def action_go(self) -> None:
        self.dismiss(True)

    def action_stop(self) -> None:
        self.dismiss(False)


class DictVerdict(ModalScreen[None]):
    """What searching a file for the preset dictionary found."""

    BINDINGS = [Binding("escape", "close", "close"),
                Binding("enter", "close", "close")]

    def __init__(self, result, path: str):
        super().__init__()
        self.result, self.path = result, path

    def compose(self) -> ComposeResult:
        from ..predict.corpora import tilde
        r = self.result
        t = Text()
        t.append("Find the preset dictionary\n\n", style="bold")
        t.append(f"{tilde(self.path)}\n\n", style="dim")
        if r.start is None:
            t.append(r.reason + "\n", style="yellow")
        elif r.proven:
            t.append("FOUND AND PROVEN.\n", style="bold green")
            t.append(r.summary()[0].upper() + r.summary()[1:] + ".\n\n",
                     style="green")
            t.append("The dictionary's bytes are on the tape as yours; ^z "
                     "takes them off.\n", style="dim")
        else:
            t.append("FOUND; NOTHING CONFIRMS IT.\n", style="bold yellow")
            t.append(r.summary()[0].upper() + r.summary()[1:] + ".\n",
                     style="yellow")
        if r.found and len(r.found) > 1:
            t.append(f"\n{len(r.found):,} runs of the file have that "
                     f"Adler-32; it is 32 bits, so a large file holds "
                     f"chance matches, and the stream's own Adler-32 picks "
                     f"the one that inflates.\n", style="dim")
        with Vertical(id="threshold-box"):
            yield Static(t)
            yield Label("esc close", id="threshold-foot")

    def action_close(self) -> None:
        self.dismiss(None)


class WriteChoice(ModalScreen[str]):
    """^w when there is more than one thing worth writing: one member of a
    multi-member gzip or the whole file; the decoded bytes or the repaired
    container."""

    def __init__(self, body: str, options: list[tuple[str, str, str]]):
        super().__init__()
        self.body, self.options = body, options
        for key, value, _label in options:
            self._bindings.bind(key, f"pick('{value}')", value)
        self._bindings.bind("escape", "pick('')", "cancel")

    def compose(self) -> ComposeResult:
        t = Text()
        t.append("Write which output?\n\n", style="bold")
        t.append(self.body + "\n\n")
        for key, _value, label in self.options:
            t.append(f"  {key}  {label}\n")
        foot = "   ".join(f"{k} {v}" for k, v, _ in self.options)
        with Vertical(id="threshold-box"):
            yield Static(t)
            yield Label(foot + "   esc cancel", id="threshold-foot")

    def action_pick(self, which: str) -> None:
        self.dismiss(which or None)


class DictionaryStop(ModalScreen[str]):
    """^f stopped before filling bytes of a missing preset dictionary.

    The floor was measured on running text, and a dictionary is a word list.
    Returns "find", "fill" or None.
    """

    BINDINGS = [Binding("d", "pick('find')", "find the dictionary"),
                Binding("enter", "pick('fill')", "fill anyway"),
                Binding("escape", "pick('')", "stop here")]

    def __init__(self, held: int, precision: str, done: str, floor: float):
        super().__init__()
        self.held, self.precision = held, precision
        self.done, self.floor = done, floor

    def compose(self) -> ComposeResult:
        t = Text()
        t.append("Stopped before filling the dictionary\n\n", style="bold")
        t.append(f"{self.held} class(es) clear the {pct(self.floor)} floor, "
                 f"but each holds a byte of the preset dictionary. A "
                 f"dictionary is a list of common strings, and the floor "
                 f"was measured on running text: fills there measured "
                 f"{self.precision} right at an {DICTIONARY_FLOOR} "
                 f"floor.\n\n")
        t.append("The dictionary itself proves every byte at once. The "
                 "program that wrote this stream usually holds it.\n\n")
        t.append(self.done + "\n", style="dim")
        with Vertical(id="threshold-box"):
            yield Static(t)
            yield Label("d find the dictionary in a file   enter fill anyway"
                        "   esc stop here", id="threshold-foot")

    def action_pick(self, which: str) -> None:
        self.dismiss(which or None)


class LongRunStop(ModalScreen[bool]):
    """^f stopped before filling inside a long run of unread output.

    A stop, a warning and an override in one place. The confidence floor was
    measured on short holes in decoded text; inside a lost packet's gap it does
    not hold at any setting, and the fills there feed on
    each other. Returns True to fill them anyway.
    """

    BINDINGS = [
        Binding("escape", "stop", "stop here"),
        Binding("enter", "go", "continue anyway"),
    ]

    def __init__(self, held: int, run: int, precision: str, done: str,
                 floor: float):
        super().__init__()
        self.held, self.run, self.precision = held, run, precision
        self.done, self.floor = done, floor

    def compose(self) -> ComposeResult:
        t = Text()
        t.append("Stopped before filling inside long unknown runs\n\n",
                 style="bold")
        t.append(f"{self.held} more class(es) clear the {pct(self.floor)} "
                 f"floor, but every occurrence of them sits in a run of "
                 f"{self.run} or more bytes that nothing was read from -- a "
                 f"lost packet, or a gap that size. Fills there measured "
                 f"{self.precision} right whatever the floor, because each "
                 f"one becomes context for the next and the model ends up "
                 f"agreeing with itself.\n\n")
        t.append(self.done + "\n", style="dim")
        with Vertical(id="threshold-box"):
            yield Static(t)
            yield Label("enter continue anyway   esc stop here",
                        id="threshold-foot")

    def action_go(self) -> None:
        self.dismiss(True)

    def action_stop(self) -> None:
        self.dismiss(False)


class CopyVerdict(ModalScreen[None]):
    """What laying a suspected copy over the capture showed, and the CRC's
    ruling on it. Read-only: the fills are already in, one ctrl+z from gone."""

    BINDINGS = [Binding("escape", "close", "close"),
                Binding("enter", "close", "close")]

    def __init__(self, result, path: str):
        super().__init__()
        self.result, self.path = result, path

    def compose(self) -> ComposeResult:
        from ..core.copyfill import regions
        r = self.result
        t = Text()
        t.append("Fill from a suspected copy\n\n", style="bold")
        from ..predict.corpora import tilde
        t.append(f"{tilde(self.path)}\n\n", style="dim")
        if r.offset is None:
            t.append(r.reason + "\n", style="yellow")
        elif r.window:
            # Nothing decoded to align it on; a slice of it matched the
            # recorded size and CRC on its own.
            t.append("THE COPY HOLDS THE ORIGINAL.\n", style="bold green")
            t.append(r.summary()[0].upper() + r.summary()[1:] + ".\n\n",
                     style="green")
            if r.filled:
                t.append("The capture decoded none of them, so the checksum "
                         "is the whole proof. They are on the tape as your "
                         "bytes; ^z takes them off.\n", style="dim")
            else:
                t.append("The capture decoded none of them, so the checksum "
                         "is the whole proof.\n", style="dim")
        else:
            if r.copy_matches and r.contradictions:
                # The capture's decode cannot pass on its own here: damage
                # put wrong bytes on the tape. The copy can, and when it
                # does it is the original.
                t.append("THE COPY IS THE ORIGINAL.\n", style="bold green")
                t.append(f"On its own it matches the recorded size and "
                         f"checksum. The capture differs from it at "
                         f"{len(r.contradictions):,} byte(s): that is where "
                         f"the capture is damaged.\n\n", style="green")
            elif r.proven and r.contradictions:
                # The CRC rules on the stream, and the stream takes its decoded
                # bytes from the capture, not from the copy -- so a copy that
                # differs only where the capture already has bytes still
                # passes. Calling that PROVEN outright overclaimed.
                t.append("LOST BYTES PROVEN, BUT THIS COPY DIFFERS.\n",
                         style="bold yellow")
                t.append(f"The checksum agrees on what was lost, and the copy "
                         f"contradicts the capture in "
                         f"{len(r.contradictions):,} byte(s) that did decode: "
                         f"it is a different version of the text.\n\n",
                         style="yellow")
            elif r.proven:
                t.append("PROVEN: the checksum agrees. The lost bytes were "
                         "exactly this copy's.\n\n", style="bold green")
            elif r.decode_failed and r.contradictions:
                t.append("THE CAPTURE'S DECODE IS ALREADY WRONG.\n",
                         style="bold yellow")
                t.append(f"It fails its checksum on its own. The copy "
                         f"differs from it at "
                         f"{len(r.contradictions):,} byte(s): if the copy is "
                         f"the original, those are where the capture is "
                         f"damaged. If it is not, they are where the two "
                         f"versions differ.\n\n", style="yellow")
            elif r.refuted:
                t.append("REFUTED: this copy does not match the capture.\n",
                         style="bold red")
                if r.split:
                    why = ("It holds different bytes at positions the capture "
                           "proves are equal.")
                elif r.contradictions:
                    why = ("It differs from bytes that decoded; the positions "
                           "are below.")
                else:
                    why = ("Nothing that decoded contradicts it and the "
                           "checksum does, so the difference is inside the "
                           "lost bytes.")
                t.append(why + "\n\n", style="red")
            elif r.verdict is None:
                t.append("NO CHECKSUM: the copy fits what decoded, and "
                         "nothing recorded a checksum to prove the rest.\n\n",
                         style="bold yellow")
            t.append(r.figures() + ".\n", style="white")
            if r.contradictions:
                t.append("\nwhere the copy contradicts what decoded "
                         "(output positions):\n", style="cyan")
                for a, b in regions(r.contradictions)[:8]:
                    t.append(f"  {a:,}" + (f"..{b:,}" if b != a else "") +
                             "\n")
            if r.filled:
                t.append("\nThe fills count as your guesses (yellow). ctrl+z "
                         "takes all of them back out.\n", style="dim")
        with Vertical(id="help-box"):
            yield Static(t)
            yield Label("esc close", id="help-foot")

    def action_close(self) -> None:
        self.dismiss(None)


class TrashModel(ModalScreen[object]):
    """Confirm moving one saved model to the trash, by typing its name.

    Shows what would go -- size, lineage, and for a byte model how long it
    trained -- because a model can be days of compute and the one that went by
    accident was. Returns the trash path, or None.
    """

    BINDINGS = [Binding("escape", "cancel", "keep it")]

    def __init__(self, meta: dict, in_use: bool = False):
        super().__init__()
        self.meta = meta
        self.in_use = in_use

    def compose(self) -> ComposeResult:
        from ..predict import corpora
        m = self.meta
        prov = m.get("provenance") or {}
        t = Text()
        t.append("Move this model to the trash?\n\n", style="bold")
        t.append(f"{m.get('name')}\n", style="bold white")
        t.append(f"{corpora.tilde(m.get('path'))}\n", style="dim")
        size = m.get("bytes")
        if size:
            t.append(f"{size / 1e6:,.1f} MB")
        if m.get("kind") == corpora.SLM:
            steps = prov.get("steps") or m.get("steps")
            t.append(f"  byte model, {prov.get('params', '?')} parameters"
                     f"{f', {steps:,} training steps' if steps else ''}",
                     style="yellow")
        elif m.get("trained_bytes"):
            t.append(f"  n-gram, order {m.get('order', '?')}, "
                     f"{m['trained_bytes']:,} training bytes")
        t.append("\n")
        if prov:
            t.append("trained on: " + ", ".join(
                f"{k}={_tv(v)}" for k, v in sorted(prov.items()))[:200] + "\n",
                style="dim")
        t.append(f"\nIt moves to {corpora.tilde(corpora.trash_dir())}; to "
                 f"restore it, move it back.\n", style="italic dim")
        if self.in_use:
            t.append("It is loaded right now; this session keeps using it.\n",
                     style="yellow")
        t.append("\nType its name to confirm.", style="cyan")
        with Vertical(id="threshold-box"):
            yield Static(t)
            yield Input(placeholder=str(m.get("name")), id="trash-input")
            yield Label("enter confirm   esc keep it", id="threshold-foot")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    def on_input_submitted(self, event) -> None:
        from ..predict import corpora
        if event.value.strip() != str(self.meta.get("name")):
            self.query_one("#threshold-foot", Label).update(
                "that is not its name -- typed exactly, or esc to keep it")
            return
        try:
            self.dismiss(corpora.trash_model(self.meta["path"]))
        except OSError as e:
            self.query_one("#threshold-foot", Label).update(
                f"could not move it: {e}")

    def action_cancel(self) -> None:
        self.dismiss(None)


class ThresholdPicker(ModalScreen[object]):
    """The confidence floor ^f fills down to, chosen without scoring anything.

    It lived only on f4, behind two keys the footer never drew, and f4 scores
    every open class before it opens -- minutes with a byte model loaded, to
    move one number. Returns the chosen value, or None.
    """

    BINDINGS = [Binding("escape", "close", "close")]

    # What 0.995 was measured to deliver on two carves, both short holes in
    # decoded text: 99% on a licence-text carve, 100% (21 classes) on a phrack
    # one. Inside a long unread run nothing holds.
    MEASURED = {0.995: "99-100% on two measured carves"}

    def __init__(self, current: float, calibration=None):
        super().__init__()
        self.current = current
        self.calibration = calibration

    def compose(self) -> ComposeResult:
        cal = self.calibration
        with Vertical(id="threshold-box"):
            yield Static("Confidence floor: ^f fills the surest classes "
                         "first and stops here", id="threshold-title")
            yield OptionList(id="threshold-list")
            if cal is not None:
                yield Static(f"This model is calibrated: the floor is the "
                             f"precision its fills measured on held-out text "
                             f"({cal.classes:,} classes). The best it reached "
                             f"was {pct(cal.best())}, so a floor above that "
                             f"fills nothing.", id="threshold-note")
            else:
                yield Static("This model is not calibrated, so the floor is "
                             "raw confidence, and what it delivers depends on "
                             "the model and the data. Inside a long unread "
                             "run, such as a lost packet, fills measured "
                             "19% right at any setting.",
                             id="threshold-note")
            yield Label("enter choose   esc keep "
                        f"{pct(self.current)}", id="threshold-foot")

    def on_mount(self) -> None:
        lst = self.query_one(OptionList)
        for t in reversed(THRESHOLDS):
            mark = "✓" if abs(t - self.current) < 1e-9 else " "
            note = (self._evidence(t) if self.calibration is not None
                    else self.MEASURED.get(t, ""))
            lst.add_option(Option(f" {mark} {pct(t):>6}  {note}", id=str(t)))
        here = next((i for i, t in enumerate(reversed(THRESHOLDS))
                     if abs(t - self.current) < 1e-9), 0)
        lst.highlighted = here
        lst.focus()

    def _evidence(self, floor: float) -> str:
        """How many held-out classes this floor's promise was measured on,
        in decoded text and next to a gap."""
        cal = self.calibration
        if floor <= 0:
            return ""
        short, near = cal.resting(floor), cal.resting(floor, near=True)
        if not short and not near:
            return "nothing measured this high: fills nothing"
        side = (f"{near:,} near a gap" if near
                else "none near a gap" if cal.near else "")
        return f"on {short:,} classes" + (f", {side}" if side else "")

    def on_option_list_option_selected(self, event) -> None:
        self.dismiss(float(event.option.id))

    def action_close(self) -> None:
        self.dismiss(None)


class HelpScreen(ModalScreen[None]):
    BINDINGS = [
        Binding("escape", "dismiss_help", "close"),
        Binding("question_mark", "dismiss_help", "close"),
        Binding("q", "dismiss_help", "close"),
    ]

    def compose(self) -> ComposeResult:
        # Scrolling, and focused, because the text is longer than a short
        # terminal: in a plain Vertical the tail was simply clipped away with
        # nothing on screen to say so.
        with VerticalScroll(id="help-box"):
            yield Static(HELP_TEXT, id="help-body")
            yield Label("esc to close", id="help-foot")

    def on_mount(self) -> None:
        self.query_one("#help-box").focus()

    def action_dismiss_help(self) -> None:
        self.dismiss(None)


def _clock(seconds) -> str:
    """h:mm:ss for anything a training run measures."""
    if seconds is None:
        return "—"
    s = int(seconds)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _bar(fraction: float, width: int = 40) -> str:
    done = max(0, min(width, int(round(fraction * width))))
    return "█" * done + "░" * (width - done)


class TrainingScreen(ModalScreen[object]):
    """Start a byte model, and watch one that is already going.

    The corpus screen's rule applies here and matters more: nothing acts on the
    first keystroke, because the thing being committed to is between one and four
    days of the operator's machine. Selecting a size shows what it costs -- the
    measured seconds per step on THIS box, the hours that implies, the resident
    memory -- and only a second enter starts it.

    Every option offered has been priced by a measured run. Nothing unpriced is
    listed, and the two settings
    that lost when they were measured say so on the row rather than being
    presented as equals: phrack is not every corpus, so they are not removed,
    but an operator should not have to already know the answer.
    """

    BINDINGS = [Binding("escape", "close", "back / close")]

    # d_model, layers, heads, params, steps/s, hours for 144,048 steps.
    #
    # These were measured on ONE machine on 2026-09-06.
    # They are right here and wrong anywhere else, so they are a fallback: a run
    # that has actually happened on this box measures the real figure for its own
    # shape, and `training.measured_rates()` replaces these as soon as there is
    # one. Rows say which they are showing.
    PROBED_ON = "probed here 2026-09-06"
    SIZES = [
        (256, 6, 4, "4.9M", 1.22, 32.9, "the size most measurements used"),
        (384, 6, 6, "10.9M", 0.68, 59.1, "cheaper than 256x12 and larger"),
        (512, 6, 8, "19.2M", 0.44, 91.6,
         "best measured: beat 4.9M on both test carves"),
        (384, 12, 6, "21.5M", 0.35, 113.3, "depth costs more per parameter"),
    ]
    WINDOWS = [(129, "measured best"),
               (257, "untested"),
               (513, "MEASURED WORSE on phrack")]

    def __init__(self, session=None):
        super().__init__()
        self.session = session
        self.armed: str | None = None
        # NOT `self.size` or `self.window`: both are Textual
        # properties on a screen, and shadowing them fails at
        # render time rather than here.
        self.pick_size = 2                  # 512x6, the one that won
        self.pick_window = 0                # 129
        self.steps = 144_048

    def compose(self) -> ComposeResult:
        with Vertical(id="training-box"):
            yield Label("Train a byte model. This takes days. It trains on "
                        "the corpus compressed and damaged, the way captures "
                        "arrive.", id="training-title")
            yield OptionList(id="training-list")
            yield Static("", id="training-detail")
            yield Label("↑↓ move   enter look / commit   esc back",
                        id="training-foot")

    def on_mount(self) -> None:
        self._build()
        self.query_one(OptionList).focus()

    def _build(self) -> None:
        from ..predict import training
        lst = self.query_one(OptionList)
        lst.clear_options()

        runs = training.runs()
        # Sections number from 1 whether or not there are runs to list.
        n = iter(range(1, 5))
        if runs:
            lst.add_option(Option(f"{next(n)} · RUNS", id=None, disabled=True))
            for r in runs:
                lst.add_option(Option(f"  {r.summary()}", id=f"run:{r.path}"))
            lst.add_option(Option("", id=None, disabled=True))

        lst.add_option(Option(f"{next(n)} · SIZE", id=None, disabled=True))
        for i, (d, l, _h, params, _sps, _hours, note) in enumerate(self.SIZES):
            mark = "▸" if i == self.pick_size else " "
            rate, source = self._rate(i)
            hours = self.steps / rate / 3600
            here = "*" if source == "measured on this machine" else " "
            lst.add_option(Option(
                f"  {mark} {d}x{l:<3} {params:>6}  ~{hours:>5.0f} h{here}  {note}",
                id=f"size:{i}"))
        lst.add_option(Option(
            f"    * = timed on this machine; the rest {self.PROBED_ON}",
            id=None, disabled=True))

        lst.add_option(Option("", id=None, disabled=True))
        lst.add_option(Option(f"{next(n)} · CONTEXT WINDOW", id=None,
                              disabled=True))
        for i, (w, note) in enumerate(self.WINDOWS):
            mark = "▸" if i == self.pick_window else " "
            lst.add_option(Option(f"  {mark} {w:<5} {note}", id=f"win:{i}"))

        lst.add_option(Option("", id=None, disabled=True))
        lst.add_option(Option(f"{next(n)} · START", id=None, disabled=True))
        lst.add_option(Option("  start a run with the settings above",
                              id="start:"))

    # ---- what a row would do ----------------------------------------------

    def _detail(self, text: Text) -> None:
        self.query_one("#training-detail", Static).update(text)

    def _rate(self, index=None):
        """Steps/s for a configuration, and where the number came from.

        Prefers a rate this machine actually produced over the probe's, because
        the probe measured one box and the operator may not be on it.
        """
        from ..predict import training
        i = self.pick_size if index is None else index
        d, l, _h, _p, probed, _hr, _n = self.SIZES[i]
        window = self.WINDOWS[self.pick_window][0]
        try:
            here = training.measured_rates().get((d, l, window))
        except Exception:
            here = None
        if here:
            return here, "measured on this machine"
        return probed, self.PROBED_ON

    def _cost(self) -> tuple[float, float]:
        """Hours and resident GB for the current selection, as measured."""
        steps_s, _source = self._rate()
        hours = self.steps / steps_s / 3600
        window = self.WINDOWS[self.pick_window][0]
        # Window multiplies both: 1.8x the seconds per prediction at 513 and 2.7
        # GB more resident, both measured. Applied
        # only to the probe's figure -- a rate measured on this machine was
        # measured AT this window and already carries it.
        if _source != "measured on this machine":
            if window == 257:
                hours *= 1.34
            elif window == 513:
                hours *= 1.80
        gb = 6.1 + {129: 0.0, 257: 0.5, 513: 2.7}[window]
        return hours, gb

    def on_option_list_option_highlighted(self, event) -> None:
        oid = event.option.id
        if oid is None:
            return
        t = Text()
        if oid.startswith("size:"):
            i = int(oid[5:])
            d, l, h, params, _sps, _hours, note = self.SIZES[i]
            rate, source = self._rate(i)
            t.append(f"d_model {d}, {l} layers, {h} heads: {params}\n",
                     style="bold")
            t.append(f"  {rate:.2f} steps/s, {source}\n",
                     style="dim" if source == "measured on this machine"
                     else "yellow")
            t.append(f"  {note}\n", style="dim")
            t.append("enter to select", style="cyan")
        elif oid.startswith("win:"):
            w, note = self.WINDOWS[int(oid[4:])]
            t.append(f"context window {w} bytes\n", style="bold")
            t.append(f"  {note}\n",
                     style="yellow" if "WORSE" in note else "dim")
            if w == 513:
                t.append("  41.3% / 57.9% right against 129's 49.9% / "
                         "64.6%, at the same number of training steps.\n",
                         style="dim")
            t.append("enter to select", style="cyan")
        elif oid.startswith("run:"):
            from ..predict import training
            r = training.load(oid[4:])
            p = r.progress()
            t.append(f"{r.name}: {r.state()}\n", style="bold")
            if p:
                t.append(f"  {_bar(p.fraction)} {p.fraction:>5.1%}\n")
                t.append(f"  running {_clock(p.working_s)}, "
                         f"~{_clock(r.eta_seconds())} of running left\n")
                t.append(f"  wall clock {_clock(p.elapsed_s)}"
                         + ("  (the difference is time suspended)\n"
                            if p.active_s is not None
                            and p.elapsed_s - p.active_s > 60 else "\n"),
                         style="dim")
                t.append(f"  {p.predictions:,} predictions, "
                         f"val top-1 {p.val_top1:.1f}%, "
                         f"val loss {p.val_loss:.3f}\n", style="dim")
            if r.state() == "finished" and training.checkpoint(r):
                t.append(f"  log: {r.log}\n", style="dim")
                t.append("enter to save it to the models directory, where "
                         "f5 lists it", style="cyan")
            else:
                t.append("enter to open the log", style="cyan")
        elif oid == "start:":
            hours, gb = self._cost()
            d, l, _h, params, _s, _hr, _n = self.SIZES[self.pick_size]
            w = self.WINDOWS[self.pick_window][0]
            t.append("start a training run\n", style="bold")
            t.append(f"  {params} ({d}x{l}), window {w}, "
                     f"{self.steps:,} steps\n")
            t.append(f"  about {hours:.0f} hours of running, ~{gb:.1f} GB "
                     f"resident\n", style="bold yellow")
            t.append("  It detaches: closing this app leaves it running, and a "
                     "crash resumes from the last checkpoint.\n", style="dim")
            t.append("  It carves the corpus first, which is ~23 minutes "
                     "before the first step.\n", style="dim")
            t.append("  It holds back every 20th file, up to 24, and "
                     "calibrates on them after the last step. On the 19M "
                     "model, 30 pieces of 32 KB took 4.8 hours.\n",
                     style="dim")
            t.append("\nenter again to start it", style="bold cyan")
            self.armed = oid
        self._detail(t)

    def on_option_list_option_selected(self, event) -> None:
        oid = event.option.id
        if oid is None:
            return
        if oid.startswith("size:"):
            self.pick_size = int(oid[5:])
            self._build()
        elif oid.startswith("win:"):
            self.pick_window = int(oid[4:])
            self._build()
        elif oid.startswith("run:"):
            from ..predict import training
            r = training.load(oid[4:])
            done = r.state() == "finished" and training.checkpoint(r)
            self.dismiss(("save" if done else "open", oid[4:]))
        elif oid == "start:" and self.armed == "start:":
            d, l, h, _p, _s, _hr, _n = self.SIZES[self.pick_size]
            self.dismiss(("start", dict(
                d_model=d, layers=l, heads=h,
                window=self.WINDOWS[self.pick_window][0], steps=self.steps)))

    def action_close(self) -> None:
        self.dismiss(None)
