"""Finding a zlib preset dictionary inside a file (manual workflow H)."""
import random
import zlib

import pytest

from toeknife.containers import parse
from toeknife.core import dictfill
from toeknife.core.session import Session

WORDS = (b"stack buffer overflow shellcode address return pointer function "
         b"execve memory bytes program exploit offset string") * 3
DICT = b" ".join(sorted(set(WORDS.split()), key=len))


def text(n=20000, seed=5):
    rng = random.Random(seed)
    words = WORDS.split() + [b"the", b"a", b"of", b"we", b"is"]
    out = []
    while sum(len(w) + 1 for w in out) < n:
        out.append(rng.choice(words))
    return b" ".join(out)[:n]


def stream(raw=None, zdict=DICT):
    raw = raw if raw is not None else text()
    co = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY,
                          zdict=zdict)
    blob = co.compress(raw) + co.flush()
    st = parse(blob).streams[0]
    return raw, Session(st.slice(blob), stream=st)


def test_the_stream_names_its_dictionary_and_reaches_into_it():
    _raw, s = stream()
    assert s.stream.dictid == zlib.adler32(DICT)
    assert 0 < dictfill.need(s) <= len(DICT)
    assert s.tape.holes()


def test_the_dictionary_as_a_file_of_its_own_proves_the_stream():
    raw, s = stream()
    got = dictfill.fill(s, DICT)
    assert (got.start, got.length, got.proven) == (0, len(DICT), True)
    assert not s.tape.holes()
    assert s.verify().ok is True
    buf, mask, start = s.full()
    assert (buf, start) == (raw, 0), "the dictionary is not part of the output"


def test_the_dictionary_is_found_inside_a_larger_file():
    raw, s = stream()
    rng = random.Random(1)
    noise = bytes(rng.randrange(256) for _ in range(40000))
    blob = noise[:17000] + DICT + noise[17000:]
    got = dictfill.fill(s, blob)
    assert (got.start, got.length, got.proven) == (17000, len(DICT), True)
    assert s.verify().ok is True


def test_a_different_dictionary_is_not_taken():
    _raw, s = stream()
    other = DICT.replace(b"stack", b"stark")
    got = dictfill.fill(s, other)
    assert got.start is None
    assert s.tape.holes()


def test_windows_agree_with_a_brute_force_search():
    rng = random.Random(3)
    for _ in range(60):
        blob = bytes(rng.randrange(256) for _ in range(rng.randrange(2, 120)))
        i = rng.randrange(len(blob) - 1)
        n = rng.randrange(1, len(blob) - i + 1)
        target = zlib.adler32(blob[i:i + n])
        brute = {(a, b - a) for a in range(len(blob))
                 for b in range(a + 1, len(blob) + 1)
                 if zlib.adler32(blob[a:b]) == target}
        assert set(dictfill.windows(blob, target)) == brute


def test_cli_dict(tmp_path, capsys):
    from toeknife import cli
    raw, _s = stream()
    co = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY,
                          zdict=DICT)
    z, d, out = tmp_path / "s.zlib", tmp_path / "prog.c", tmp_path / "out"
    z.write_bytes(co.compress(raw) + co.flush())
    d.write_bytes(b'static const char dict[] = "' + DICT + b'";\n')
    assert cli.main([str(z), "--dict", str(d), "--export", str(out)]) == 0
    assert out.read_bytes() == raw
    assert "this is the dictionary" in capsys.readouterr().err


def test_a_fill_holds_back_dictionary_bytes_until_asked(tmp_path, capsys):
    """The floor was measured on text; a dictionary is a word list."""
    from toeknife import cli
    raw, _s = stream(text(4000))
    co = zlib.compressobj(6, zlib.DEFLATED, 15, 9, zlib.Z_DEFAULT_STRATEGY,
                          zdict=DICT)
    z, out = tmp_path / "s.zlib", tmp_path / "out"
    z.write_bytes(co.compress(raw) + co.flush())
    cli.main([str(z), "--solve", "--solve-confidence", "0.0",
              "--export", str(out)])
    err = capsys.readouterr().err
    assert "hold bytes of the preset dictionary" in err
    assert "--fill-dictionary fills them anyway" in err
    assert "solver: filled 0 of" in err
    cli.main([str(z), "--solve", "--solve-confidence", "0.0",
              "--fill-dictionary", "--export", str(out)])
    assert "solver: filled 0 of" not in capsys.readouterr().err
