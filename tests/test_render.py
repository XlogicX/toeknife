"""Match labels must fit their span exactly, at every width."""
import random

import pytest

from toeknife.tui.render import glyph, match_label


@pytest.mark.parametrize("length", range(3, 40))
@pytest.mark.parametrize("distance", [1, 9, 35, 258, 4521, 32768])
def test_label_always_fills_its_span_exactly(length, distance):
    out = match_label(length, distance, length)
    assert len(out) == length, f"{out!r} is {len(out)} cells, span is {length}"


def test_label_never_puts_an_arrow_against_a_digit():
    """The whole point of the tiering: a solid cap flush against a digit is
    unreadable, so caps only appear when a separator fits on each side."""
    rng = random.Random(0)
    for _ in range(2000):
        length = rng.randrange(3, 259)
        distance = rng.randrange(1, 32769)
        out = match_label(length, distance, length)
        for cap, neighbour in (("◀", 1), ("▶", -2)):
            if out.startswith(cap) or out.endswith(cap):
                idx = 1 if cap == "◀" else len(out) - 2
                assert out[idx] == "─", f"{out!r} has {cap} flush against {out[idx]!r}"


def test_label_degrades_predictably():
    # roomy: caps, dashes, full pair
    assert match_label(6, 35, 12) == "◀───6,35───▶"
    # tight: full pair survives, caps go
    assert match_label(6, 35, 6) == "─6,35─"
    assert match_label(6, 35, 4) == "6,35"
    # too narrow for the pair: the distance is what is worth keeping, since the
    # length is already on screen as the width of the span
    assert match_label(3, 4521, 5) == "←4521"
    assert match_label(3, 4521, 4) == "4521"
    assert match_label(3, 4521, 3) == "───"


def test_a_comma_always_means_length_distance():
    """The disambiguation rule the help text promises."""
    for length in range(3, 60):
        for distance in (1, 35, 4521, 32768):
            out = match_label(length, distance, length)
            if "," in out:
                assert out.strip("◀▶─") == f"{length},{distance}"
            elif out.strip("◀▶─←"):
                assert out.strip("◀▶─←") == str(distance)


def test_control_bytes_get_one_cell_glyphs():
    for v in range(256):
        assert len(glyph(v)) == 1
    assert glyph(None) == "░"
    assert glyph(0x00) == "␀"
    assert glyph(0x0A) == "␊"
    assert glyph(0x7F) == "␡"
    assert glyph(ord("A")) == "A"
