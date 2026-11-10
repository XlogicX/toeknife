"""The two-row tape: tokens above, bytes below, one column per output byte.

The top row shows what the compressor wrote (literals and back-references) and
the bottom row shows what came out.  They share a column grid, so a match
labelled 6,35 is drawn exactly six cells wide, directly above the six bytes it
produced.  Selecting it highlights the bytes it reads from.
"""
from __future__ import annotations

from rich.segment import Segment
from rich.style import Style
from textual import events
from textual.binding import Binding
from textual.geometry import Size
from textual.message import Message
from textual.reactive import reactive
from textual.scroll_view import ScrollView
from textual.strip import Strip

from ..core.tape import UNKNOWN
from .render import (
    STYLE_CURSOR, STYLE_DIM, STYLE_EOB, STYLE_GUTTER, STYLE_GUTTER_CUR,
    STYLE_LIT_TICK, STYLE_MATCH, STYLE_MATCH_SEL, STYLE_PRE, STYLE_SOURCE,
    STYLES, UNASSIGNED_GLYPH, glyph, match_label,
)

GUTTER = 8
LINES_PER_ROW = 3          # token row, byte row, spacer


class TapeView(ScrollView):
    """Scrollable dual-row view over the whole tape, pre-fragment window first."""

    can_focus = True
    cursor = reactive(0)
    selected = reactive(-1)          # out_pos of the selected match, -1 for none

    # These are declared on the widget, and marked priority, so they beat
    # ScrollView's own arrow-key scrolling and the screen's tab-to-move-focus.
    # In this view the arrows move a byte cursor and tab walks the holes.
    BINDINGS = [
        Binding("left", "move(-1)", "left", show=False),
        Binding("right", "move(1)", "right", show=False),
        Binding("up", "move_row(-1)", "up", show=False),
        Binding("down", "move_row(1)", "down", show=False),
        Binding("home", "row_start", "row start", show=False),
        Binding("end", "row_end", "row end", show=False),
        Binding("pageup", "page(-1)", "page up", show=False),
        Binding("pagedown", "page(1)", "page down", show=False),
        Binding("tab", "hole(1)", "next hole", priority=True),
        Binding("shift+tab", "hole(-1)", "prev hole", priority=True),
    ]

    def action_move(self, delta: int) -> None:
        self.move_cursor(delta)

    def action_move_row(self, direction: int) -> None:
        self.move_cursor(direction * self.cols)

    def action_row_start(self) -> None:
        self.goto(self.lo + self.row_of(self.cursor) * self.cols)

    def action_row_end(self) -> None:
        self.goto(self.lo + self.row_of(self.cursor) * self.cols + self.cols - 1)

    def action_page(self, direction: int) -> None:
        rows = max(1, self.size.height // LINES_PER_ROW)
        self.move_cursor(direction * rows * self.cols)

    def action_hole(self, direction: int) -> None:
        self.next_hole(backwards=direction < 0)

    class CursorMoved(Message):
        def __init__(self, pos: int) -> None:
            self.pos = pos
            super().__init__()

    def __init__(self, session, **kw):
        super().__init__(**kw)
        self.session = session
        self.cols = 64
        self._span_start: dict[int, object] = {}
        self._hole_positions: set[int] = set()
        self._code_positions: dict[int, tuple[int, int]] = {}
        self._source_range: range | None = None
        self.rebuild()

    # ---- model ------------------------------------------------------------

    @property
    def tape(self):
        return self.session.tape

    @property
    def lo(self) -> int:
        return self.tape.min_pos

    @property
    def hi(self) -> int:
        return max(self.tape.length, 1)

    def rebuild(self) -> None:
        """Re-index after a decode, an alignment change, or a guess."""
        self._span_start = {s.out_pos: s for s in self.session.trace.symbols}
        # Positions that belong to a hole class, i.e. the ones actually worth
        # typing into.  The rest of the pre-fragment window is unreferenced
        # space that no back-reference reaches, so there is nothing to solve
        # there and it must not look like an invitation.
        holes = self.tape.holes()
        self._hole_positions = {p for h in holes for p in h.positions}
        # Which of those are waiting on a literal code rather than a copy chain.
        # Built here, with the hole scan already paying for itself, because a
        # per-cell lookup during rendering would walk the union-find on every
        # visible byte.
        self._code_positions = {p: h.code for h in holes if h.code is not None
                                for p in h.positions}
        self._recompute_layout()
        self._update_source_range()
        self.refresh()

    def _recompute_layout(self) -> None:
        width = max(self.size.width, GUTTER + 8)
        self.cols = max(8, width - GUTTER)
        span = self.hi - self.lo
        rows = max(1, -(-span // self.cols))
        self.virtual_size = Size(width, rows * LINES_PER_ROW)

    def on_resize(self, event: events.Resize) -> None:
        self._recompute_layout()
        self.refresh()

    # ---- cursor -----------------------------------------------------------

    def watch_cursor(self, pos: int) -> None:
        self._update_source_range()
        self.post_message(self.CursorMoved(pos))
        self.scroll_to_cursor()
        self.refresh()

    def watch_selected(self, _pos: int) -> None:
        self._update_source_range()
        self.refresh()

    def _update_source_range(self) -> None:
        sym = self.symbol_at(self.selected) if self.selected >= 0 else None
        if sym is not None and sym.kind == "match":
            src = sym.out_pos - sym.distance
            self._source_range = range(src, src + sym.length)
        else:
            self._source_range = None

    def symbol_at(self, pos: int):
        return self.session.trace.symbol_at_out(pos)

    def move_cursor(self, delta: int) -> None:
        self.cursor = max(self.lo, min(self.hi - 1, self.cursor + delta))

    def goto(self, pos: int) -> None:
        self.cursor = max(self.lo, min(self.hi - 1, pos))
        # Setting the cursor to where it already is does not fire the watcher,
        # so going to it would not bring it into view.
        self.scroll_to_cursor()

    def row_of(self, pos: int) -> int:
        return (pos - self.lo) // self.cols

    def scroll_to_cursor(self) -> None:
        row = self.row_of(self.cursor)
        top = row * LINES_PER_ROW
        _, sy = self.scroll_offset
        height = self.size.height
        if top < sy:
            self.scroll_to(y=top, animate=False)
        elif top + 2 >= sy + height:
            self.scroll_to(y=max(0, top + 3 - height), animate=False)

    # ---- navigation helpers ------------------------------------------------

    def next_hole(self, backwards: bool = False) -> bool:
        anchors = sorted(h.anchor for h in self.tape.holes())
        if not anchors:
            return False
        if backwards:
            prior = [a for a in anchors if a < self.cursor]
            self.cursor = prior[-1] if prior else anchors[-1]
        else:
            later = [a for a in anchors if a > self.cursor]
            self.cursor = later[0] if later else anchors[0]
        return True

    def next_match(self, backwards: bool = False) -> bool:
        matches = [s.out_pos for s in self.session.trace.symbols if s.kind == "match"]
        if not matches:
            return False
        if backwards:
            prior = [m for m in matches if m < self.cursor]
            target = prior[-1] if prior else matches[-1]
        else:
            later = [m for m in matches if m > self.cursor]
            target = later[0] if later else matches[0]
        self.cursor = target
        self.selected = target
        return True

    def jump_to_source(self) -> bool:
        """Follow the match under the cursor back to the bytes it copies."""
        sym = self.symbol_at(self.cursor)
        if sym is None or sym.kind != "match":
            return False
        self.selected = sym.out_pos
        self.goto(sym.out_pos - sym.distance)
        return True

    # ---- rendering ---------------------------------------------------------

    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        vy = y + scroll_y
        row, sub = divmod(vy, LINES_PER_ROW)
        start = self.lo + row * self.cols
        if start >= self.hi:
            return Strip.blank(self.size.width)
        count = min(self.cols, self.hi - start)
        if sub == 0:
            segs = self._token_segments(start, count)
        elif sub == 1:
            segs = self._byte_segments(start, count)
        else:
            return Strip.blank(self.size.width)
        strip = Strip(segs)
        return strip.crop(scroll_x, scroll_x + self.size.width)

    def _gutter(self, start: int, active: bool) -> Segment:
        style = STYLE_GUTTER_CUR if active else STYLE_GUTTER
        return Segment(f"{start:>{GUTTER - 1}} ", style)

    def _byte_segments(self, start: int, count: int) -> list[Segment]:
        tape = self.tape
        cursor_row = self.row_of(self.cursor) == self.row_of(start)
        segs = [self._gutter(start, cursor_row)]
        src = self._source_range
        for i in range(count):
            p = start + i
            v = tape.value_at(p)
            style = STYLES[tape.prov_at(p)]
            if v is None and p not in self._hole_positions:
                style = STYLE_PRE
            if src is not None and p in src:
                style = style + STYLE_SOURCE
            if p == self.cursor:
                style = style + STYLE_CURSOR
            ch = glyph(v)
            if v is None and p in self._code_positions:
                ch = UNASSIGNED_GLYPH
            segs.append(Segment(ch, style))
        return segs

    def _token_segments(self, start: int, count: int) -> list[Segment]:
        segs = [Segment(" " * GUTTER)]
        p = start
        end = start + count
        # A match may begin before this row; render only its visible remainder.
        first = self.symbol_at(p)
        if first is not None and first.kind == "match" and first.out_pos < start:
            visible = min(first.out_pos + first.length, end) - p
            style = STYLE_MATCH_SEL if first.out_pos == self.selected else STYLE_MATCH
            segs.append(Segment("─" * visible, style))
            p += visible
        while p < end:
            sym = self._span_start.get(p)
            if sym is None:
                segs.append(Segment(" "))
                p += 1
                continue
            if sym.kind == "match":
                visible = min(sym.length, end - p)
                style = STYLE_MATCH_SEL if sym.out_pos == self.selected else STYLE_MATCH
                segs.append(Segment(match_label(sym.length, sym.distance, visible), style))
                p += visible
            elif sym.kind == "eob":
                segs.append(Segment("█", STYLE_EOB))
                p += 1
            else:
                segs.append(Segment("·", STYLE_LIT_TICK))
                p += 1
        return segs

    # ---- input -------------------------------------------------------------

    def on_click(self, event: events.Click) -> None:
        # event.x/y are widget-relative and include the padding; the rendered
        # grid lives in content coordinates, so convert before doing column
        # arithmetic or every click is off by the padding width.
        offset = event.get_content_offset(self)
        if offset is None:
            return
        _, scroll_y = self.scroll_offset
        row, sub = divmod(offset.y + scroll_y, LINES_PER_ROW)
        if sub > 1:
            return
        col = offset.x - GUTTER
        if col < 0:
            return
        pos = self.lo + row * self.cols + col
        if self.lo <= pos < self.hi:
            self.cursor = pos
            sym = self.symbol_at(pos)
            self.selected = sym.out_pos if sym is not None and sym.kind == "match" else -1
