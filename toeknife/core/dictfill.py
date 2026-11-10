"""Filling a missing zlib preset dictionary from a file.

A stream compressed against a preset dictionary copies from bytes that sit in
front of its own output: the dictionary's tail. Without it those copies are
holes at negative positions on the tape, -1 being the dictionary's last byte,
and the most negative position says how long the dictionary has to be.

Two numbers prove a dictionary. The zlib header's DICTID is the Adler-32 of
the whole dictionary, which picks candidates out of a file; the stream's own
Adler-32 over the output then confirms one, because a wrong dictionary decodes
to wrong bytes. The dictionary usually lives in the program that wrote the
stream -- a string constant in its source or binary, or a file it ships -- so a
candidate is any run of a file, and the whole file is only the first one
tried.

Fills go in as the operator's bytes (yellow), as a suspected copy's do: the
file is something the operator supplied, and the checksum is the proof.
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import dataclass

from .tape import GUESS

MOD = 65521             # Adler-32's modulus
MAX_TRIED = 1_000_000   # candidates inflated before giving up; each is under 1 ms


@dataclass
class DictFill:
    dictid: int | None = None
    need: int = 0                       # bytes of dictionary the stream reaches
    found: list = None                  # (start, length) whose Adler-32 matched
    start: int | None = None            # the one used
    length: int = 0
    proven: bool | None = None          # the stream's Adler-32 agreed
    filled: int = 0
    positions: int = 0
    reason: str = ""
    size: int = 0                       # of the file searched

    def summary(self) -> str:
        if self.start is None:
            return self.reason
        where = (f"the whole file ({self.length:,} bytes)" if self.start == 0
                 and self.length == self.size else
                 f"bytes {self.start:,}-{self.start + self.length - 1:,} of "
                 f"the file")
        verb = "has" if where.startswith("the whole") else "have"
        head = (f"{where} {verb} the DICTID's Adler-32 {self.dictid:08x}; "
                f"filled {self.filled:,} classes ({self.positions:,} "
                f"positions)")
        if self.proven:
            return head + "; the stream's Adler-32 agrees: this is the dictionary"
        return head + ("; the stream records no Adler-32, so nothing confirms "
                       "it beyond the DICTID")


def need(session) -> int:
    """How many bytes of dictionary the stream reaches back into."""
    return max(0, -session.tape.min_pos)


def windows(blob: bytes, target: int, minlen: int = 1,
            progress=None) -> list[tuple[int, int]]:
    """Every (start, length) of `blob` at least `minlen` long whose Adler-32 is
    `target`, the whole file first.

    Adler-32 over bytes i..j-1 comes from two prefix sums: A is 1 + S[j] - S[i]
    and B is the length plus sum (j - k) * b[k], which is j * (S[j] - S[i]) -
    (T[j] - T[i]) with T the prefix sum of k * b[k]. Indexing the ends by S mod
    65521 means each start visits only the ends whose A already matches, about
    n / 65521 of them: 1.1 s over 300 KB.
    """
    n = len(blob)
    ta, tb = target & 0xFFFF, target >> 16
    out = []
    if n >= minlen and zlib.adler32(blob) & 0xFFFFFFFF == target:
        out.append((0, n))
    S, T = [0] * (n + 1), [0] * (n + 1)
    s = t = 0
    for k, b in enumerate(blob):
        s += b
        t += k * b
        S[k + 1], T[k + 1] = s, t
    ends = defaultdict(list)
    for j in range(minlen, n + 1):
        ends[S[j] % MOD].append(j)
    starts = n - minlen + 1
    for i in range(starts):
        if progress is not None and not i % 4096:
            progress(i, starts, f"{i:,} of {starts:,} starting bytes, "
                                f"{len(out):,} Adler-32 match(es)")
        for j in ends.get((ta - 1 + S[i]) % MOD, ()):
            length = j - i
            if length < minlen or (i == 0 and j == n):
                continue
            if (length + j * (S[j] - S[i]) - (T[j] - T[i])) % MOD == tb:
                out.append((i, length))
    return out


def _inflates(session, dictionary: bytes) -> bool | None:
    """Whether the stream, inflated against this dictionary, matches its own
    Adler-32: None when it records none."""
    adler = getattr(session.stream, "adler32", None)
    try:
        out = zlib.decompressobj(-15, zdict=dictionary).decompress(session.data)
    except zlib.error:
        return False
    if adler is None:
        return None
    return zlib.adler32(out) & 0xFFFFFFFF == adler


def fill(session, blob: bytes, name: str = "") -> DictFill:
    """Find the dictionary in `blob`, prove it, and fill the holes from it."""
    out = find(session, blob)
    if out.start is not None:
        apply(session, out, blob, name)
    return out


def find(session, blob: bytes, progress=None) -> DictFill:
    """Find and prove the dictionary without touching the tape, so it can run
    off the interface's event loop."""
    stream = getattr(session, "stream", None)
    out = DictFill(dictid=getattr(stream, "dictid", None), need=need(session),
                   found=[], size=len(blob))
    if out.dictid is None:
        out.reason = "this stream names no preset dictionary"
        return out
    if not out.need:
        out.reason = "nothing in the stream reaches into the dictionary"
        return out
    out.found = windows(blob, out.dictid, out.need, progress)
    if not out.found:
        out.reason = (f"no run of {out.need:,} bytes or more in the file has "
                      f"the DICTID's Adler-32 {out.dictid:08x}")
        return out
    pick = None
    for start, length in out.found[:MAX_TRIED]:
        verdict = _inflates(session, blob[start:start + length])
        if verdict is True:
            pick, out.proven = (start, length), True
            break
        if verdict is None and len(out.found) == 1:
            pick, out.proven = (start, length), None
    if pick is None:
        tried = min(len(out.found), MAX_TRIED)
        out.reason = (f"{len(out.found):,} run(s) of the file have the "
                      f"DICTID's Adler-32 {out.dictid:08x}"
                      + (f"; none of the {tried:,} tried inflates to the "
                         f"stream's Adler-32" if stream.adler32 is not None
                         else "; the stream records no Adler-32 to tell them "
                              "apart"))
        return out
    out.start, out.length = pick
    return out


def apply(session, out: DictFill, blob: bytes, name: str = "") -> None:
    """Fill every hole that reaches into the dictionary from the run `find`
    chose. One undo group: a wrong file comes back out with one ctrl+z."""
    dictionary = blob[out.start:out.start + out.length]
    source = (f"dictionary {name} bytes {out.start:,}-"
              f"{out.start + out.length - 1:,}").replace("  ", " ")
    tape = session.tape
    session.begin_group()
    try:
        for hole in tape.holes():
            before = [p for p in hole.positions if p < 0]
            if not before:
                continue
            session.set_guess(hole.anchor, dictionary[len(dictionary) + before[0]],
                              GUESS, source=source)
            out.filled += 1
            out.positions += hole.size
    finally:
        session.end_group()
