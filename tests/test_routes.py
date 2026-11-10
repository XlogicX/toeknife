"""The route board says what applies, and says why when it does not."""
import zlib

import pytest

from toeknife.core import ledger as ledger_mod
from toeknife.core import routes
from toeknife.core.session import Session

TEXT = b"the quick brown fox jumps over the lazy dog. " * 200


def raw_deflate(data, flush_every=0):
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9)
    if not flush_every:
        return co.compress(data) + co.flush()
    parts = []
    for i in range(0, len(data), flush_every):
        parts.append(co.compress(data[i:i + flush_every]))
        parts.append(co.flush(zlib.Z_SYNC_FLUSH))
    parts.append(co.flush())
    return b"".join(parts)


@pytest.fixture
def healthy():
    return Session(raw_deflate(TEXT), mode="stream")


def test_every_route_returns_a_verdict(healthy):
    board = routes.board(healthy)
    assert len(board) == len(routes.ROUTES)
    assert all(isinstance(v.reason, str) and v.reason for v in board)


def test_unavailable_routes_always_carry_a_reason(healthy):
    """The whole point: a route that does not apply must say why."""
    for v in routes.board(healthy):
        if not v.ok:
            assert v.reason.strip(), f"{v.route.key} refused without a reason"


def test_reading_routes_are_always_available(healthy):
    by_key = {v.route.key: v for v in routes.board(healthy)}
    assert by_key["decode"].ok
    assert by_key["resync"].ok


def test_recovery_routes_stand_down_when_the_table_is_present(healthy):
    by_key = {v.route.key: v for v in routes.board(healthy)}
    for key in ("dict-head", "dict-full", "midheader"):
        assert not by_key[key].ok
        assert "already present" in by_key[key].reason


def test_bitrot_stands_down_on_a_capture_that_decodes(healthy):
    """Its precondition is not "the table is missing" -- on a rotted header the
    table is right there, which is the whole distinction between correcting a
    header and completing one. What rules the route out is a parse that landed."""
    v = {x.route.key: x for x in routes.board(healthy)}["bitrot"]
    assert not v.ok
    assert "decodes cleanly" in v.reason


def test_bitrot_applies_once_the_parse_stops_landing():
    """The case it exists for: every bit present, and the parse still fails."""
    import zlib
    from toeknife.core import bitrot as br
    from toeknife.core.session import Session
    raw = b"forensic analysis of compressed evidence, at length. " * 80
    co = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    comp = co.compress(raw) + co.flush()
    span = br.header_span(Session(comp, mode="stream"))
    damaged = br.flip(comp, [span[0] + 30])
    v = {x.route.key: x for x in
         routes.board(Session(damaged, mode="stream"))}["bitrot"]
    assert v.ok, v.reason
    assert "header bits" in v.reason
    assert v.route.status == routes.WIRED, "it is wired now, not research"


def test_research_routes_say_so_rather_than_vanishing():
    """A route that is measured but unwired must be visible as unwired.

    Left off the board it would be indistinguishable from one that does not
    apply, which is the failure the board exists to prevent.
    """
    s = Session(raw_deflate(TEXT)[40:])          # no header: recovery applies
    by_key = {v.route.key: v for v in routes.board(s)}
    unwired = [v for v in by_key.values() if v.blocked_by_status]
    assert unwired, "expected some route to be applicable but unwired"
    for v in unwired:
        assert v.route.status == routes.RESEARCH
        assert "not in this release" in v.reason


def test_alphabet_gate_is_sound_only_from_the_streams_start():
    """Off the first block the gate rejects the TRUE table 92% of
    the time, so its precondition is a check and not a comment."""
    s = Session(raw_deflate(TEXT), mode="stream")
    gate = {v.route.key: v for v in routes.board(s)}["alphabet-gate"]
    # Sound here: the decode starts at the data, on a block at output 0.
    assert gate.ok or gate.blocked_by_status


def test_alphabet_gate_refuses_after_a_resync():
    """The trap: a resynced view's first block reports out_start == 0 while
    real output precedes it, which is exactly where the gate is unsound."""
    comp = raw_deflate(TEXT * 4, flush_every=16384)
    s = Session(comp, mode="stream")
    later = [b for b in s.trace.blocks if b.btype == 2 and b.out_start > 0]
    if not later:
        pytest.skip("no mid-stream dynamic block in this build of zlib")
    s.resync_to(later[0].bit_start)
    dyn = [b for b in s.trace.blocks if b.btype == 2]
    assert dyn and dyn[0].out_start == 0, "the trap needs out_start to read 0"
    gate = {v.route.key: v for v in routes.board(s)}["alphabet-gate"]
    assert not gate.ok and not gate.blocked_by_status
    assert "92%" in gate.reason


def test_a_broken_check_reads_as_unavailable_not_available():
    def boom(_s):
        raise RuntimeError("nope")

    r = routes.Route("x", "X", routes.READ, "-", "-", "-", routes.WIRED, boom)
    v = r.verdict(object())
    assert not v.ok and "check failed" in v.reason


def test_render_names_every_route(healthy):
    text = routes.render(healthy)
    for r in routes.ROUTES:
        assert r.name in text


class TestLedger:
    def test_empty_ledger_says_nothing_has_run(self):
        assert "Nothing has been run" in ledger_mod.Ledger().render()

    def test_abstention_is_not_the_same_as_finding_nothing(self):
        """Every dictionary false accept was stopped by abstention, so the
        two outcomes must not render as the same sentence."""
        lg = ledger_mod.Ledger()
        lg.abstained("dict-full", have=3, need=20)
        lg.add("dict-head", ledger_mod.NOTHING)
        text = lg.render()
        assert "REFUSED to rule" in text
        assert "nothing found" in text
        assert "3 assertions available, 20 required" in text
        assert "REFUSED" in text.split("nothing found")[0]

    def test_parameters_are_recorded_for_replay(self):
        lg = ledger_mod.Ledger()
        lg.add("kp-climb", ledger_mod.FOUND, "a table",
               params={"seed": 41, "rho": 0.4}, counts={"climbs": 30},
               seconds=1.5)
        line = lg.entries[0].line()
        assert "seed=41" in line and "rho=0.4" in line and "climbs 30" in line

    def test_ledger_is_append_only(self):
        lg = ledger_mod.Ledger()
        lg.add("a", ledger_mod.NOTHING)
        lg.add("b", ledger_mod.FAILED)
        assert len(lg) == 2 and [e.route for e in lg.entries] == ["a", "b"]

    def test_session_starts_with_an_empty_ledger(self, healthy):
        assert len(healthy.ledger) == 0
