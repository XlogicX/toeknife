"""A table from a related capture, for a carve whose own table is gone."""
import zlib

import pytest

from toeknife.core import borrow
from toeknife.core.inflate import Decoder
from toeknife.core.session import Session
from toeknife.core.tape import BORROWED, UNKNOWN

from tests.test_resume import text


def two_blocks(raw):
    """memLevel 7 halves zlib's symbol buffer, so 60 KB comes out in two
    dynamic blocks, the shape GNU gzip gives the Phrack article."""
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 7)
    return co.compress(raw) + co.flush()


@pytest.fixture(scope="module")
def case():
    raw = text()
    blob = two_blocks(raw)
    blocks = Decoder(blob).stream(0).blocks
    assert len(blocks) >= 2 and blocks[1].btype == 2
    lose = blocks[0].bit_end // 8 // 3            # a third of block 0's bytes
    return raw, blob, blob[lose:]


def resynced(carve):
    s = Session(carve, mode="fragment")
    s.resync_to(borrow.boundary(s))
    return s


def test_the_capture_s_own_table_lands_on_the_next_header(case):
    raw, blob, carve = case
    s = resynced(carve)
    r = borrow.borrow(s, blob)
    assert r.boundary == s.bit_offset
    assert [c.label for c in r.landings] == ["its own block 0"]
    assert r.landings[0].starts > 1


def test_an_unrelated_text_lands_nowhere(case):
    raw, blob, carve = case
    other = two_blocks(text(seed=99))
    r = borrow.borrow(resynced(carve), other)
    assert not r.landings and "none lands there" in r.summary()


def test_adopting_it_joins_both_halves_and_marks_the_borrowed_part(case):
    raw, blob, carve = case
    s = resynced(carve)
    before = s.tape.stats()
    r = borrow.borrow(s, blob)
    borrow.apply(s, r.landings[0], r.boundary)
    tape = s.tape
    assert tape.stats()["holes"] < before["holes"]
    assert tape.stats()["out_len"] > before["out_len"]
    n = len(tape)
    off = len(raw) - n
    assert not [p for p in range(n) if tape.value_at(p) is not None
                and tape.value_at(p) != raw[off + p]]
    first = s.trace.blocks[1].out_start
    assert all(tape.prov_at(p) in (BORROWED, UNKNOWN) for p in range(first))


def test_the_true_table_s_seam_reads_like_the_text(case):
    """The next block's reads through the borrowed bytes score about as well
    as its own text. (A near miss, which lands and reads worse, is measured
    in the manual: -5.39 against -4.52 on the Phrack carve.)"""
    from toeknife.predict.context import ContextModel
    raw, blob, carve = case
    model = ContextModel(order=4)
    model.train(raw)
    s = resynced(carve)
    truth = borrow.borrow(s, blob, model=model)
    assert truth.landings[0].score >= truth.baseline - 0.5


def test_no_header_to_test_against_says_so():
    raw = text(4000)
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    blob = co.compress(raw) + co.flush()
    s = Session(blob[200:], mode="fragment")
    r = borrow.borrow(s, blob)
    assert r.boundary is None and "no block header" in r.reason


def test_an_assumed_fixed_table_that_chokes_is_unproven(case):
    raw, blob, carve = case
    s = Session(carve, mode="fragment")
    if s.trace.status in ("end", "eob"):
        pytest.skip("this carve happens to read cleanly with the fixed table")
    assert all(s.tape.prov_at(p) in (BORROWED, UNKNOWN)
               for p in range(len(s.tape)))
