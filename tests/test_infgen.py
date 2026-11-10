"""Differential test against Mark Adler's infgen, the reference for this job.

Skipped automatically when infgen is not on PATH.
"""
import random
import shutil
import subprocess
import zlib

import pytest

from toeknife.cli import cmd_dump
from toeknife.core.session import Session

INFGEN = shutil.which("infgen")
pytestmark = pytest.mark.skipif(INFGEN is None, reason="infgen not installed")

STRUCTURE = ("last", "fixed", "dynamic", "stored", "match", "end")


def structural(lines):
    return [l.split("!")[0].strip() for l in lines
            if l.split("!")[0].strip().startswith(STRUCTURE)]


def infgen_lines(data: bytes, tmp_path):
    p = tmp_path / "in.deflate"
    p.write_bytes(data)
    out = subprocess.run([INFGEN, "-d", str(p)], capture_output=True, text=True)
    return structural(out.stdout.splitlines())


def toeknife_lines(data: bytes, capsys):
    cmd_dump(Session(data, mode="stream"), verbose=0)
    return structural(capsys.readouterr().out.splitlines())


@pytest.mark.parametrize("level,strategy", [
    (0, zlib.Z_DEFAULT_STRATEGY),
    (6, zlib.Z_FIXED),
    (9, zlib.Z_DEFAULT_STRATEGY),
    (6, zlib.Z_RLE),
])
def test_symbol_stream_matches_infgen(level, strategy, tmp_path, capsys):
    rng = random.Random(20240729)
    chunks = [b"alpha ", b"beta ", b"gamma delta ", b"\x00\x01\xff",
              b"the quick brown fox ", b"forensic analysis "]
    for size in (40, 900, 12000):
        data = b"".join(rng.choice(chunks) for _ in range(size))
        co = zlib.compressobj(level, zlib.DEFLATED, -15, 9, strategy)
        comp = co.compress(data) + co.flush()
        assert toeknife_lines(comp, capsys) == infgen_lines(comp, tmp_path), (
            f"diverged from infgen at level={level} strategy={strategy} size={size}")


def test_multi_block_stream_matches_infgen(tmp_path, capsys):
    rng = random.Random(11)
    data = b"".join(rng.choice([b"one ", b"two ", b"three "]) for _ in range(6000))
    co = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    comp = b""
    step = len(data) // 4
    for i in range(0, len(data), step):
        comp += co.compress(data[i:i + step]) + co.flush(zlib.Z_FULL_FLUSH)
    comp += co.flush()
    ours, theirs = toeknife_lines(comp, capsys), infgen_lines(comp, tmp_path)
    assert ours == theirs
    assert theirs.count("stored") + theirs.count("fixed") + theirs.count("dynamic") > 1
