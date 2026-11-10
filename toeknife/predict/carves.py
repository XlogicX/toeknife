"""Training data for a byte filler: real carves, with truth known by construction.

The damage has to be the damage the tool actually sees.  Random masking is what
ByT5 was pretrained on and it is why the raw checkpoint scores 5 of 40 -- holes
in a carve are not scattered, they arrive as one lost region plus every equality
class whose back-references reach into it, and the density around them is nothing
like uniform.  So each example comes from an actual DEFLATE carve: compress with
a sync flush, destroy the first block's table, resync, and read the tape.

Sync flush, not full flush.  A full flush resets the window, so nothing after the
damage refers back into it and every later byte decodes exactly -- clean, and
missing the entire structure worth learning.  A sync flush lets back-references
reach into the lost bytes, which is what creates the classes.

**This is why "train a byte model on this corpus" is not what it sounds like.**
An n-gram trains on the corpus bytes.  This trains on the corpus *compressed and
then damaged*, once per file, which takes about a second and a half each -- 23
minutes before the first gradient step on a corpus the size of phrack.  A corpus
that does not compress into the right shape produces a confident, useless model
several days later.

The tool builds its own training data from here.
"""
from __future__ import annotations

import random
import zlib
from pathlib import Path

CHUNK = 1024


def carve(payload: bytes, seed: int = 0, chunk: int = CHUNK) -> bytes:
    """Sync-flushed raw deflate with the first block's Huffman table destroyed.

    Only the bits between BTYPE and the first symbol are touched, so the block
    still announces itself as dynamic: the failure is a bad table rather than a
    fragment the decoder walks past, which is the case the tool is for.
    """
    from toeknife.core.inflate import Decoder

    co = zlib.compressobj(9, zlib.DEFLATED, -15, 9)
    parts = []
    for i in range(0, len(payload), chunk):
        parts.append(co.compress(payload[i:i + chunk]))
        parts.append(co.flush(zlib.Z_SYNC_FLUSH))
    parts.append(co.flush())
    intact = b"".join(parts)

    trace = Decoder(intact).stream(0)
    first_body = min(s.bit_start for s in trace.symbols if s.block == 0)
    rng = random.Random(seed)
    broken = bytearray(intact)
    for bit in range(trace.blocks[0].bit_start + 3, first_body):
        if rng.random() < 0.5:
            broken[bit >> 3] ^= 1 << (bit & 7)
    return bytes(broken)


def resynced(data: bytes):
    """A session resynced past the ruined table, as f3 would leave it."""
    from toeknife.core.scan import scan
    from toeknife.core.session import Session

    found = scan(data)
    if not found:
        return None
    s = Session(data)
    s.resync_to(found[0].bit_offset)
    return s


def clean_runs(session, min_run: int = 16):
    """Every contiguous run of recovered tape of at least `min_run` bytes."""
    buf, mask = session.tape.recovered()
    out, cur, start = [], bytearray(), None
    for i, (b, m) in enumerate(zip(buf, mask)):
        if m:
            if start is None:
                start = i
            cur.append(b)
        else:
            if len(cur) >= min_run:
                out.append((start, bytes(cur)))
            cur, start = bytearray(), None
    if len(cur) >= min_run:
        out.append((start, bytes(cur)))
    return out


def align(session, truth: bytes, min_run: int = 16,
          min_known: int = 256) -> int | None:
    """Offset AT such that tape position p holds truth[AT + p].

    Resync lands on a block boundary whose output offset is not known in advance,
    so the offset is found rather than computed.

    Propose, then verify -- and the verification is what makes it safe.  The
    longest clean run must occur in the article at exactly `AT + start`, so every
    occurrence of that run is a candidate and the true offset is guaranteed to be
    among them.  Each candidate is then checked against **every known byte on the
    tape**, thousands of them, and a wrong offset dies within a few.  An offset
    that agrees everywhere is right; there is nothing to trust about the choice
    of anchor, because the anchor only nominates.

    This replaces a first-run-must-be-unique rule that refused 177 of 409
    articles, 43% of everything tried.  115 of those refusals were a first clean
    run made of `-----` or `_____`, which is the least unique string in a Phrack
    article; the other 62 simply had no clean run reaching 64 bytes, while their
    tapes were 35-78% known.  Neither is a reason to throw the article away.

    Still refuses rather than guessing, in two cases: too little known tape to
    verify against, and two offsets that both explain every known byte, which
    means the article really is self-similar over everything that was recovered.
    """
    runs = clean_runs(session, min_run)
    if not runs:
        return None
    buf, mask = session.tape.recovered()
    known = [(i, b) for i, (b, m) in enumerate(zip(buf, mask)) if m]
    if len(known) < min_known:
        return None                     # too little evidence to verify against

    start, anchor = max(runs, key=lambda r: len(r[1]))
    candidates = []
    where = truth.find(anchor)
    while where >= 0:
        candidates.append(where - start)
        where = truth.find(anchor, where + 1)

    survivors = []
    for at in candidates:
        if all(0 <= at + i < len(truth) and truth[at + i] == b
               for i, b in known):
            survivors.append(at)
            if len(survivors) > 1:
                return None             # genuinely ambiguous: refuse to guess
    return survivors[0] if survivors else None



class Carve:
    """One damaged stream, its tape, and the truth lined up against it."""

    def __init__(self, path, seed: int, limit: int):
        self.path = Path(path)
        self.truth = self.path.read_bytes()[:limit]
        self.data = carve(self.truth, seed=seed)
        self.session = resynced(self.data)
        self.at = align(self.session, self.truth) if self.session else None

    @property
    def usable(self) -> bool:
        return self.session is not None and self.at is not None

    def stats(self) -> dict:
        tape = self.session.tape
        holes = tape.holes()
        unknown = sum(len(h.positions) for h in holes)
        known = tape.stats().get("known", 0)
        return dict(article=self.path.name, bytes=len(self.truth),
                    known=known, unknown=unknown, classes=len(holes),
                    masked=unknown / max(known + unknown, 1))


def build(paths, count: int | None = None, limit: int = 48000,
          seed0: int = 0, progress=None):
    """Carve the given files.  Unusable ones are reported, never silently kept.

    Takes explicit paths rather than globbing a corpus root, because the tool
    lets an operator choose the corpus and the caller is the one that knows what
    was chosen.  `seed0 + index` seeds each carve, and the index is over the list
    as given so the same list always produces the same carves -- though the seed
    turns out not to reach the tape at all.
    """
    out, skipped = [], []
    for i, path in enumerate(paths):
        if count is not None and len(out) >= count:
            break
        path = Path(path)
        try:
            if path.stat().st_size < 8192:
                continue
        except OSError:
            continue
        c = Carve(path, seed=seed0 + i, limit=limit)
        if c.usable:
            out.append(c)
        else:
            skipped.append(path.name)
        if progress is not None:
            progress(i + 1, len(out), len(skipped))
    return out, skipped


def feasible(paths, sample: int = 12, limit: int = 48000):
    """Would this corpus make a byte model, or waste four days finding out?

    Training carves every file, which is about a second and a half each -- 23
    minutes before the first gradient step on a corpus the size of phrack, and
    90 hours before the answer. Almost everything that can go wrong is visible
    in a handful of files, so a handful is carved first.

    Three ways a corpus fails here, and only the first is obvious:

      * **it will not carve.** Files too small to hold two DEFLATE blocks, or
        that will not resync, or whose tape cannot be aligned. A corpus where
        most files drop out trains on far less than the operator thinks.
      * **it carves but leaves nothing to learn.** The maskable positions are
        the training data; a corpus that compresses to almost nothing yields
        almost none of them, however many files it has.
      * **it is the wrong KIND.** Not visible here at all -- that is
        `bytemodel.sanity_check`'s job, after the fact, against a real capture.
        This function measures whether training can happen, never whether the
        result would be any good.

    Returns (usable, tried, positions, verdict).
    """
    paths = list(paths)
    if not paths:
        return 0, 0, 0, "nothing to train on"
    step = max(1, len(paths) // sample)
    picked = paths[::step][:sample]
    made, _skipped = build(picked, limit=limit)
    positions = sum(c.stats()["unknown"] for c in made)
    tried = len(picked)
    if not made:
        return 0, tried, 0, (
            f"none of {tried} sampled file(s) could be carved. Either they are "
            f"too small -- under 8 KB is skipped -- or they do not compress "
            f"into a stream this can damage. This corpus cannot train a byte "
            f"model")
    rate = len(made) / tried
    per_file = positions / len(made)
    projected = int(per_file * rate * len(paths))
    if rate < 0.5:
        return len(made), tried, positions, (
            f"only {len(made)} of {tried} sampled file(s) carved. The run would "
            f"train on roughly {rate:.0%} of this corpus, about "
            f"{projected:,} maskable positions")
    if per_file < 500:
        return len(made), tried, positions, (
            f"carves, but thinly: {per_file:,.0f} maskable positions per file "
            f"against phrack's ~12,000. Projected {projected:,} positions for "
            f"the whole corpus, where the measured runs used 9.3M")
    return len(made), tried, positions, ""


def contamination(tape, paths, needles: int = 12, min_run: int = 64,
                  threshold: float = 0.5):
    """Which of these files contain the capture that is being recovered.

    Training on the file being recovered makes every number afterwards fiction,
    and an operator pointing at a corpus has no way to know whether their capture
    came out of it. The recovered tape is the evidence: if the corpus holds the
    source, the bytes already read off the tape are sitting in one of these files
    verbatim.

    **The obvious version of this does not work**, and it fails the way the
    aligner used to. Short runs off the tape are boilerplate -- mastheads,
    separator bars, stock phrases -- and searching for a handful of 48-byte
    needles matched 270 of phrack's 969 files. Uniqueness is the whole problem
    and it is not free.

    What works is whole runs, longest first, scored by the FRACTION of them a
    file contains rather than by a count of hits. The source file holds nearly
    all of them; a file that merely shares a masthead holds one or two. Measured
    on the p38-5 capture against the phrack corpus: p38-5 at 100%, the next best
    at 33%, one file over the threshold and it is the right one, 30 MB in 0.02 s.

    Returns [(fraction, path)] over `threshold`, worst offender first. An empty
    list is not proof of innocence -- a capture whose tape is too holed to yield
    long runs cannot be looked for at all, which `enough_evidence` reports.
    """
    runs = sorted(clean_runs_of(tape, min_run), key=len, reverse=True)
    probes = runs[:needles]
    if not probes:
        return []
    out = []
    for path in paths:
        try:
            blob = Path(path).read_bytes()
        except OSError:
            continue
        found = sum(1 for p in probes if p in blob)
        frac = found / len(probes)
        if frac > threshold:
            out.append((frac, str(path)))
    return sorted(out, reverse=True)


def clean_runs_of(tape, min_run: int = 64):
    """Contiguous recovered runs on a tape, as bytes."""
    buf, mask = tape.recovered()
    out, cur = [], bytearray()
    for b, m in zip(buf, mask):
        if m:
            cur.append(b)
        else:
            if len(cur) >= min_run:
                out.append(bytes(cur))
            cur = bytearray()
    if len(cur) >= min_run:
        out.append(bytes(cur))
    return out


def enough_evidence(tape, min_run: int = 64, min_bytes: int = 192) -> bool:
    """Whether the tape holds enough to search the corpus with.

    Measured in BYTES of long run, not in a count of them. A tape recovered
    whole is a single run and is the best possible evidence; counting runs
    called that insufficient and would have cleared the very captures easiest to
    find. What actually disqualifies a tape is being so holed that nothing
    contiguous survives, and then "no contamination found" would be a lie about
    a search that never happened.
    """
    runs = clean_runs_of(tape, min_run)
    return bool(runs) and sum(len(r) for r in runs) >= min_bytes
