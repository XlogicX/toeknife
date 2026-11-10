"""Regressions for what the pre-release bug hunt found."""
import gzip
import struct
import zlib
from pathlib import Path

import pytest

from toeknife import cli
from toeknife import containers

D = Path("samples/damaged")


def test_a_walked_trailer_with_an_impossible_size_is_not_trusted(tmp_path):
    """A burst over member 1's trailer read as 4,185,609,946 bytes; it was
    trusted, and --entry all wrote a 4.2 GB export."""
    a, b = b"first member " * 300, b"second member " * 300
    blob = bytearray(gzip.compress(a, mtime=0) + gzip.compress(b, mtime=0))
    blob[-4:] = struct.pack("<I", 4_185_609_946)      # the size field, damaged
    report = containers.parse(bytes(blob))
    last = report.streams[-1]
    assert last.uncompressed_size is None and last.crc32 is None
    assert any("more than DEFLATE can produce" in n for n in last.notes)
    f = tmp_path / "two.gz"
    f.write_bytes(bytes(blob))
    out = tmp_path / "out.bin"
    cli.main([str(f), "--entry", "all", "--export", str(out)])
    assert out.stat().st_size < 10 * (len(a) + len(b))


def test_a_resync_saved_from_the_command_line_replays(tmp_path):
    """--resync opened the session at the boundary without recording the
    step, so the saved session refused its own input."""
    sess, a, b = tmp_path / "s.session", tmp_path / "a.bin", tmp_path / "b.bin"
    src = str(D / "lost-table.bin")
    assert cli.main([src, "--resync", "--save-session", str(sess),
                     "--export", str(a)]) == 0
    assert cli.main([src, "--session", str(sess), "--export", str(b)]) == 0
    assert a.read_bytes() == b.read_bytes()


def test_a_total_size_smaller_than_the_decode_is_refused(capsys):
    """Clamped to a 0-byte gap, it joined the pieces with no gap and placed
    every byte after it at the wrong position as known."""
    cli.main([str(D / "capture-head.bin"), "--gap",
              str(D / "capture-tail.bin"), "--total-size", "10", "--list"])
    err = capsys.readouterr().err
    assert "is less than the" in err and "is not used" in err
    assert "output gap of 0 bytes, from the recorded" not in err


@pytest.mark.parametrize("argv, says", [
    (["--export", "{tmp}"], "is a directory"),
    (["--scan", "--scan-kinds", "banana"], "unknown kind banana"),
    (["--bit", "999999", "--list"], "is outside the data"),
    (["--tables"], "reserved"),
])
def test_the_command_line_refuses_clearly(tmp_path, capsys, argv, says):
    argv = [a.replace("{tmp}", str(tmp_path)) for a in argv]
    try:
        cli.main(["samples/fragment.bin", *argv])
    except SystemExit:
        pass
    got = capsys.readouterr()
    assert says in got.out + got.err
    assert "Traceback" not in got.out + got.err


@pytest.mark.parametrize("argv", [
    ["--order", "x"], ["--solve-confidence", "7"]])
def test_bad_numbers_are_refused_by_the_parser(argv):
    with pytest.raises(SystemExit) as e:
        cli.main(["samples/fragment.bin", "--corpus", "samples", *argv,
                  "--list"])
    assert e.value.code == 2


def test_a_model_that_cannot_load_stops_the_run(tmp_path):
    bad = tmp_path / "not.tkm"
    bad.write_bytes(b"not a model")
    with pytest.raises(SystemExit) as e:
        cli.main(["samples/fragment.bin", "--model", str(bad), "--list"])
    assert e.value.code == 1


def test_an_action_on_a_container_with_several_entries_names_what_is_missing(
        capsys):
    assert cli.main([str(D / "evidence.zip"), "--chunk-repair"]) == 1
    assert "pick it with --entry N" in capsys.readouterr().err


@pytest.mark.parametrize("cut", [4, 5, 6, 7])
def test_a_tiff_cut_inside_its_header_says_so(cut):
    """Fuzzing cut a TIFF inside its 8-byte header and the parser crashed."""
    report = containers.parse(b"II\x2a\x00\x08\x00\x00\x00"[:cut], "tiff")
    assert report is not None
    assert any("cut short" in w for w in report.warnings)


def test_a_parser_that_fails_reports_instead_of_raising(monkeypatch):
    def boom(data):
        raise ValueError("an unforeseen layout")
    kind = "tiff"
    monkeypatch.setitem(containers.PARSERS, kind,
                        (containers.PARSERS[kind][0], boom))
    report = containers.parse(b"II\x2a\x00" + bytes(20), kind)
    assert any("parser stopped" in w and "unforeseen" in w
               for w in report.warnings)


def test_the_woff_flavour_is_named():
    from toeknife.containers import woff
    assert woff._flavor(b"\x00\x01\x00\x00") == "TrueType"
    assert woff._flavor(b"OTTO") == "OpenType (CFF)"
    assert woff._flavor(b"\x01\x02\x03\x04") == "flavour 01020304"
