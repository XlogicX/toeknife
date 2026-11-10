"""Choosing an entry from a container before the main view opens.

An archive with several DEFLATE entries printed a table and exited, and the
interactive view could only be reached by running again with --entry
This puts the same table on screen and opens the entry
you pick.
"""
from __future__ import annotations

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.widgets import Footer, Label, OptionList, Static
from textual.widgets.option_list import Option


class EntryPicker(App):
    """The container's entries, one per row; enter opens one."""

    CSS = """
    #entries-box { padding: 1 2; }
    #entries-title { margin-bottom: 1; }
    #entries-notes { color: $text-muted; margin-bottom: 1; }
    #entries-list { height: 1fr; }
    #entries-detail { height: 4; margin-top: 1; color: $text-muted; }
    """
    BINDINGS = [Binding("escape", "quit_picker", "quit"),
                Binding("ctrl+q", "quit_picker", "quit", show=False)]
    ENABLE_COMMAND_PALETTE = False

    def __init__(self, report, path: str):
        super().__init__()
        self.report = report
        self.path = path

    def compose(self) -> ComposeResult:
        r = self.report
        n = len(r.decodable)
        with Vertical(id="entries-box"):
            yield Label(f"{self.path}: {r.kind} container, "
                        f"{len(r.streams)} entries, {n} hold DEFLATE. "
                        f"Pick one to open.", id="entries-title")
            notes = [f"  {line}" for line in r.structure]
            notes += [f"  ! {w}" for w in r.warnings]
            yield Static("\n".join(notes), id="entries-notes")
            yield OptionList(id="entries-list")
            yield Static("", id="entries-detail")
        yield Footer()

    def on_mount(self) -> None:
        self.title = "ToeKnife"
        lst = self.query_one(OptionList)
        lst.add_option(Option("     " + self.report.header_line(), id=None,
                              disabled=True))
        first = None
        for i, s in enumerate(self.report.streams):
            row = f"  {i:<2} {s.summary()}"
            if not s.decodable:
                row += f"   ({s.why_not})"
            lst.add_option(Option(row, id=str(i), disabled=not s.decodable))
            if s.decodable and first is None:
                first = lst.option_count - 1
        if first is not None:
            lst.highlighted = first
        lst.focus()

    def on_option_list_option_highlighted(self, event) -> None:
        if not event.option.id:
            return
        s = self.report.streams[int(event.option.id)]
        t = Text()
        for note in s.notes:
            t.append(f"! {note}\n", style="yellow")
        t.append("enter opens it   esc quits", style="cyan")
        self.query_one("#entries-detail", Static).update(t)

    def on_option_list_option_selected(self, event) -> None:
        if event.option.id:
            self.exit(int(event.option.id))

    def action_quit_picker(self) -> None:
        self.exit(None)


def choose_entry(report, path: str):
    """Run the picker; the chosen stream, or None if the operator quit."""
    index = EntryPicker(report, path).run()
    return None if index is None else report.streams[index]
