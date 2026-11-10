"""Shared glyph and colour vocabulary for the terminal views.

Every byte on screen has to answer two questions at a glance: what is it, and
how much do we actually know about it.  The glyph answers the first, the colour
answers the second.  Bytes recovered from the bitstream are facts; bytes the
operator typed are hypotheses.  They must never look alike.
"""
from __future__ import annotations

from rich.style import Style

from ..core.tape import (BORROWED, COPY, GAPPED, GUESS, GUESS_PROP, LITERAL,
                         RESUMED, SOLVED, SOLVED_PROP, UNKNOWN)

# Unicode Control Pictures: 0x00..0x1F render as ␀␁␂… and 0x7F as ␡, so a
# non-printable byte occupies exactly one cell and still says what it is.
UNKNOWN_GLYPH = "░"
HIGH_GLYPH = "·"
# A literal whose Huffman code is known and whose meaning is not.  It gets a
# different SHAPE, never a different colour: colour is reserved for where a byte
# came from, and this byte has not come from anywhere yet -- it is as unknown as
# any other hole.  What differs is the structure behind it, and therefore what
# filling it buys: a copy chain is usually a handful of positions, while a common
# literal code can be thousands, so one entered byte can light up a fifth of the
# block at once.
UNASSIGNED_GLYPH = "▒"


def glyph(value: int | None) -> str:
    if value is None:
        return UNKNOWN_GLYPH
    if 0x20 <= value < 0x7F:
        return chr(value)
    if value < 0x20:
        return chr(0x2400 + value)
    if value == 0x7F:
        return "␡"
    return HIGH_GLYPH


STYLES = {
    LITERAL:     Style(color="green", bold=True),
    COPY:        Style(color="cyan"),
    GUESS:       Style(color="yellow", bold=True),
    GUESS_PROP:  Style(color="yellow"),
    # A model's suggestion is a third kind of thing, and must not be mistaken
    # for either a fact from the bitstream or an assertion the operator made.
    SOLVED:      Style(color="magenta", bold=True),
    SOLVED_PROP: Style(color="magenta"),
    # Read from the bitstream, but past damage whose end nothing can locate:
    # most of these bytes are right and some are garbage carried forward, and
    # nothing on the tape says which. Not a fact's colour.
    RESUMED:     Style(color="blue", bold=True),
    BORROWED:    Style(color="blue", bold=True),
    GAPPED:      Style(color="blue", bold=True),
    UNKNOWN:     Style(color="red", dim=True),
}

STYLE_CURSOR = Style(reverse=True)
STYLE_SOURCE = Style(bgcolor="grey30")          # bytes a selected match reads from
STYLE_MATCH = Style(color="cyan", bold=True)
STYLE_MATCH_SEL = Style(color="black", bgcolor="yellow", bold=True)
STYLE_LIT_TICK = Style(color="green", dim=True)
STYLE_EOB = Style(color="magenta", bold=True)
STYLE_GUTTER = Style(color="grey42")
STYLE_GUTTER_CUR = Style(color="white", bold=True)
STYLE_DIM = Style(color="grey42")
STYLE_PRE = Style(color="grey37")               # the pre-fragment window region

LEGEND = [
    ("literal", STYLES[LITERAL]),
    ("copy", STYLES[COPY]),
    ("guess", STYLES[GUESS]),
    ("model", STYLES[SOLVED]),
    ("unproven", STYLES[RESUMED]),
    ("unknown", STYLES[UNKNOWN]),
]


def _centre(label: str, width: int, fill: str = "─") -> str:
    pad = width - len(label)
    left = pad // 2
    return fill * left + label + fill * (pad - left)


def match_label(length: int, distance: int, width: int) -> str:
    """A back-reference drawn across the bytes it produces.

    The span is the length, so the token row lines up with the byte row cell for
    cell.  That also means the length is already on screen as the width of the
    span, which is what makes the narrow fallbacks affordable: when the full
    label will not fit, the distance is the part worth keeping.

    Arrow caps appear only when there is room for a separator on each side.
    Letting a solid triangle sit flush against a digit makes both hard to read,
    and on real data a bit over 40% of matches land in exactly that range.
    """
    if width <= 0:
        return ""
    label = f"{length},{distance}"
    if width >= len(label) + 4:
        return "◀" + _centre(label, width - 2) + "▶"
    if width >= len(label):
        return _centre(label, width)
    # Too narrow for length,distance.  Fall back to the distance alone, marked
    # with a back-arrow so a bare number is never mistaken for the pair.
    back = f"←{distance}"
    if width >= len(back):
        return _centre(back, width)
    short = str(distance)
    if width >= len(short):
        return _centre(short, width)
    return "─" * width


def ascii_preview(tape, start: int, count: int) -> str:
    return "".join(glyph(tape.value_at(p)) for p in range(start, start + count))
