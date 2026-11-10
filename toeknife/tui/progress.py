"""Showing that a slow thing is alive, and letting it be called off.

The tool has several operations that run for minutes -- training on 17.9 MB of
prose is 130 seconds, and a filesystem sweep or a boundary scan is not much
better -- and every one of them used to run inside the event handler that
started it. That has two consequences the operator sees. The status line it set
first never painted, because Textual repaints when the handler returns and it had
not. And the whole app stopped answering keys, which is indistinguishable from a
hang; the only honest thing it could have said, it could not say.

So the work moves to a thread and this screen watches it. What makes it useful is
not the bar -- a bar can sit at 40% for a minute and still look stuck -- but the
elapsed clock and the note, which say what is running now and that it was running
a quarter of a second ago.
"""
from __future__ import annotations

import time

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Label, ProgressBar, Static


class Cancelled(Exception):
    """Raised inside the worker, at its next progress report, when the operator
    has asked for it to stop. Nothing is assigned to the session until a run
    finishes, so unwinding here leaves the previous state untouched."""


def human_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"


class ProgressScreen(ModalScreen[None]):
    """A bar, a note, and a clock, over work running on a thread.

    `total` of 0 means the size is not known yet, which is the honest state for
    a download with no content-length or a scan whose extent depends on what it
    finds. The bar goes indeterminate rather than pretending to a percentage.
    """

    BINDINGS = [Binding("escape", "cancel", "cancel")]

    # Rounds shown in full; older ones collapse into one line, so a run of
    # dozens of rounds cannot push the box off the screen.
    ROUNDS_SHOWN = 8
    ROUND_BAR = 24

    def __init__(self, title: str, cancellable: bool = True):
        super().__init__()
        self.title_text = title
        self.cancellable = cancellable
        self.cancelled = False
        self._started = time.monotonic()
        self._last_note = ""
        self._rounds: dict[int, tuple[int, int]] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="progress-box"):
            yield Label(self.title_text, id="progress-title")
            yield ProgressBar(total=None, show_eta=False, id="progress-bar")
            yield Static("", id="progress-note")
            yield Static("", id="progress-rounds")
            yield Label("esc cancels" if self.cancellable else
                        "this one cannot be interrupted", id="progress-foot")

    def on_mount(self) -> None:
        # The clock ticks on its own so the screen keeps moving even while the
        # worker is inside a stretch that reports nothing.
        self.set_interval(0.25, self._tick)

    def _tick(self) -> None:
        elapsed = human_time(time.monotonic() - self._started)
        note = self._last_note or "working"
        if self.cancelled:
            note = "cancelling; it stops at its next checkpoint"
        self.query_one("#progress-note", Static).update(f"{note}   {elapsed}")

    def set_progress(self, done: int, total: int, note: str = "",
                     stage=None) -> None:
        """Called from the main thread, via the app's `call_from_thread`.

        `stage`, when given, is (number, done, total) for one step of a run
        whose number of steps is not known in advance -- ^f's rounds. Each gets
        its own bar under the overall one, and a finished one stays full, so
        the stack itself shows how far the run has come.
        """
        bar = self.query_one("#progress-bar", ProgressBar)
        if total > 0:
            bar.total = total
            bar.progress = min(done, total)
        else:
            bar.total = None            # indeterminate: size not known
        if note:
            self._last_note = note
        self._overall = (done, total)
        if stage is not None:
            n, sdone, stotal = stage
            self._rounds[n] = (sdone, stotal)
            # A staged run draws its overall bar in the same style as the
            # rounds under it, so the stack reads as one picture.
            bar.display = False
            self._draw_rounds()
        self._tick()

    def _bar(self, done: int, total: int) -> tuple[str, float]:
        frac = 1.0 if total <= 0 else min(done / total, 1.0)
        full = int(frac * self.ROUND_BAR)
        return "█" * full + "░" * (self.ROUND_BAR - full), frac

    def _draw_rounds(self) -> None:
        nums = sorted(self._rounds)
        done, total = getattr(self, "_overall", (0, 0))
        if total > 0:
            bar, frac = self._bar(done, total)
            lines = [f"{'overall':<10}{bar}  {done:,}/{total:,} filled  "
                     f"{frac:.0%}", ""]
        else:
            lines = [f"{'overall':<10}{'░' * self.ROUND_BAR}  counting "
                     f"what clears the floor", ""]
        hidden = nums[:-self.ROUNDS_SHOWN]
        if hidden:
            lines.append(f"rounds {hidden[0]}–{hidden[-1]} done")
        for n in nums[-self.ROUNDS_SHOWN:]:
            done, total = self._rounds[n]
            bar, frac = self._bar(done, total)
            count = "nothing to re-score" if total <= 0 else f"{done:,}/{total:,}"
            mark = "  ✓" if frac >= 1.0 else ""
            lines.append(f"{'round ' + str(n):<10}{bar}  {count}{mark}")
        self.query_one("#progress-rounds", Static).update("\n".join(lines))

    def action_cancel(self) -> None:
        if not self.cancellable:
            return
        # The screen stays up: the worker only learns about this at its next
        # report, and a screen that vanished before the work stopped would be
        # the same lie in the other direction.
        self.cancelled = True
        self._tick()
