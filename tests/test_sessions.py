"""A saved session replays its steps and comes back as it was saved."""
import json
import random
import zlib

import pytest

from toeknife.containers.base import Stream
from toeknife.core import bitrot, copyfill, resume
from toeknife.core.session import Session

from test_resume import burst, text


def gz_stream(raw, blob):
    return Stream(name="t", container="gzip", offset=0, length=len(blob),
                  crc32=zlib.crc32(raw) & 0xFFFFFFFF,
                  uncompressed_size=len(raw))


def deflate(raw):
    co = zlib.compressobj(6, zlib.DEFLATED, -15)
    return co.compress(raw) + co.flush()


@pytest.fixture(scope="module")
def rotted():
    """A header bit flipped so the decode fails, and the flip that undoes it."""
    raw = text(20000, seed=11)
    good = deflate(raw)
    st = gz_stream(raw, good)
    for bit in range(17, 300):
        blob = bytearray(good)
        blob[bit >> 3] ^= 1 << (bit & 7)
        s = Session(bytes(blob), stream=st)
        if s.verify() is not None and s.verify().ok:
            continue
        r = bitrot.repair(s, k=1)
        if len(r.matching) == 1:
            return bytes(blob), st, r.matching[0].bits
    pytest.skip("no single-match flip in range")


def test_a_repair_survives_save_and_load(rotted, tmp_path):
    blob, st, bits = rotted
    s = Session(blob, stream=st)
    s.flip_bits(bits)
    assert s.verify().ok is True
    s.ledger.add("bitrot", "found", "test")
    s.save(tmp_path / "s.json")
    again = Session(blob, stream=st)
    again.load(tmp_path / "s.json")
    assert again.verify().ok is True
    assert [e.route for e in again.ledger.entries] == ["bitrot"]
    assert [x["op"] for x in again.steps] == ["flip"]


def test_a_different_input_of_the_same_size_is_refused(rotted, tmp_path):
    blob, st, bits = rotted
    s = Session(blob, stream=st)
    s.save(tmp_path / "s.json")
    other = bytearray(blob)
    other[-5] ^= 0xFF
    with pytest.raises(ValueError, match="SHA-256"):
        Session(bytes(other), stream=st).load(tmp_path / "s.json")


def test_reading_on_past_damage_and_a_copy_come_back(tmp_path):
    raw = text()
    blob = burst(deflate(raw))
    st = gz_stream(raw, blob)
    s = Session(blob, stream=st)
    resume.apply(s, resume.resume(s))
    copyfill.fill(s, raw)
    s.save(tmp_path / "s.json")
    again = Session(blob, stream=st)
    again.load(tmp_path / "s.json")
    assert again.fingerprint() == s.fingerprint()
    assert again.resumed_from is not None
    assert len(again.guesses) == len(s.guesses)


def test_a_tape_that_replays_differently_is_refused(rotted, tmp_path):
    blob, st, bits = rotted
    s = Session(blob, stream=st)
    s.flip_bits(bits)
    s.save(tmp_path / "s.json")
    saved = json.loads((tmp_path / "s.json").read_text())
    saved["steps"] = []                      # drop the repair
    (tmp_path / "s.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="different tape"):
        Session(blob, stream=st).load(tmp_path / "s.json")


def test_a_version_3_file_still_loads(tmp_path):
    raw = text(4000, seed=2)
    blob = deflate(raw)
    s = Session(blob)
    (tmp_path / "v3.json").write_text(json.dumps({
        "version": 3, "size": len(blob), "mode": s.mode,
        "bit_offset": s.bit_offset, "agreed": False,
        "guesses": {}, "provenance": {}}))
    Session(blob).load(tmp_path / "v3.json")


def test_cli_saves_and_loads(rotted, tmp_path, capsys):
    from toeknife import cli
    blob, st, bits = rotted
    src, sess = tmp_path / "rot.bin", tmp_path / "s.json"
    src.write_bytes(blob)
    assert cli.main([str(src), "--raw", "--stream", "--bitrot",
                     "--save-session", str(sess), "--list"]) == 0
    assert "saved to" in capsys.readouterr().err
    cli.main([str(src), "--raw", "--stream", "--session", str(sess),
              "--list"])
    assert "the tape matches the one saved" in capsys.readouterr().err


def test_a_guess_past_the_output_is_refused_cleanly(rotted, tmp_path):
    from toeknife import cli
    blob, _st, _bits = rotted
    src = tmp_path / "rot.bin"
    src.write_bytes(blob)
    assert cli.main([str(src), "--raw", "--stream", "--guess",
                     "999999=65", "--list"]) == 1


def test_a_byte_from_a_copy_names_the_copy(tmp_path):
    """--why said "guess" for a byte a copy filled."""
    from toeknife.core import provenance
    raw = text()
    blob = burst(deflate(raw))
    st = gz_stream(raw, blob)
    s = Session(blob, stream=st)
    resume.apply(s, resume.resume(s))
    copyfill.fill(s, raw, "copy.txt")
    pos = next(iter(s.guesses))
    assert s.source_of(pos) == "copy copy.txt"
    assert "came from copy copy.txt" in provenance.render(s, pos)
    s.undo()
    s.redo()
    assert s.source_of(pos) == "copy copy.txt", "undo and redo keep it"
    s.save(tmp_path / "s.json")
    again = Session(blob, stream=st)
    again.load(tmp_path / "s.json")
    assert again.source_of(pos) == "copy copy.txt"


def test_a_typed_byte_has_no_source():
    raw = text()
    blob = burst(deflate(raw))
    s = Session(blob, stream=gz_stream(raw, blob))
    resume.apply(s, resume.resume(s))
    hole = s.tape.holes()[0].anchor
    s.set_guess(hole, ord("a"))
    assert s.source_of(hole) == ""


def test_a_raw_stream_that_lost_its_tail_is_read_as_a_stream():
    """A raw stream cut short decodes cleanly from bit 0 until the data runs
    out. Auto mode ranked it as a headerless fragment instead, read it with
    the fixed table and marked 61 garbage bytes known, where the stream it is
    gives all 43,694 (wrap-up #2's matrix)."""
    import json, random, zlib
    from toeknife.core.session import Session
    rng = random.Random(7)
    raw = json.dumps([{"id": i, "v": rng.random()} for i in range(3000)],
                     indent=1).encode()[:65536]
    co = zlib.compressobj(6, zlib.DEFLATED, -15, 9, zlib.Z_HUFFMAN_ONLY)
    stream = co.compress(raw) + co.flush()
    cut = stream[:2 * len(stream) // 3]
    s = Session(cut)
    assert s.mode == "stream"
    buf, mask = s.tape.recovered()
    assert sum(mask) > 10_000
    assert all(buf[i] == raw[i] for i in range(len(buf)) if mask[i])
