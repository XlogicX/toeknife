"""What to train on, and where to find it.

A corpus is not an optional nicety here and picking the wrong one is worse than
picking none. `profiles.sanity_check` records the measurement: an English-trained
order-4 model scores -3.9 bits per byte on English and **-17.2 on ELF**, against
-8.0 for knowing nothing at all. Wrong-type training does not degrade to
ignorance, it degrades past it, and the solver reads confidence as a threshold --
so a confidently wrong model actively fills holes with rubbish.

That makes this module's job less "make picking easy" and more "make picking
wrong hard, then measure it anyway".

**The trap this exists to avoid.** For `.txt`, `.html`, `.c`, the file's bytes
are the content, and training on the files is right. For `.png`, `.pdf`, `.zip`,
`.gz` they are not: a PNG's DEFLATE stream carries *filtered scanlines*, a PDF's
carries content streams, a ZIP member carries whatever that member happens to be.
Scanning for `*.png` and training on the file bytes trains on compressed noise --
which is the -17.2 case, reached while appearing to do exactly the right thing.
Types that need it say so, and the scan decompresses before it trains.

Sizes are small on purpose. The docs measure order 3-4 beating order 6 on 285 KB,
so these want a few hundred KB to a few MB. A gigabyte of corpus is not the goal
and mostly is not better.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# How the training bytes are obtained from a file the scan found.
AS_IS = "as-is"              # the file's bytes are the content
INFLATE = "inflate"          # the content is inside a compressed container


@dataclass(frozen=True)
class Source:
    """An external place to get this kind of data. Never fetched implicitly."""
    name: str
    url: str
    licence: str
    approx: str


@dataclass(frozen=True)
class CorpusType:
    key: str
    name: str
    category: str
    contains: str                       # what the DEFLATE stream actually holds
    extensions: tuple[str, ...]
    extract: str = AS_IS
    roots: tuple[str, ...] = ()         # suggested places to look
    sources: tuple[Source, ...] = ()
    note: str = ""

    @property
    def needs_inflating(self) -> bool:
        return self.extract == INFLATE


# Roots are suggestions the operator confirms or edits. They are deliberately
# ordinary locations: a forensic tool should not surprise anyone about what it
# read, and on a machine that is itself under examination, reading is not free.
_TEXT_ROOTS = ("~/Documents", "/usr/share/doc", "/usr/share/common-licenses")
_CODE_ROOTS = ("~/code", "~/src", "/usr/share")
_BIN_ROOTS = ("/usr/bin", "/usr/lib", "/bin")
_IMG_ROOTS = ("~/Pictures", "/usr/share/icons", "/usr/share/pixmaps")

TYPES: tuple[CorpusType, ...] = (
    CorpusType(
        "text-en", "Text, English", "text",
        "the prose itself, byte for byte",
        (".txt", ".md", ".rst", ".log"), AS_IS, _TEXT_ROOTS,
        (Source("Project Gutenberg", "https://www.gutenberg.org/",
                "public domain", "a few MB is plenty"),),
    ),
    CorpusType(
        "text-es", "Text, Spanish", "text",
        "prose, with accented characters as multi-byte UTF-8",
        (".txt", ".md"), AS_IS, _TEXT_ROOTS,
        (Source("Project Gutenberg (Spanish)",
                "https://www.gutenberg.org/browse/languages/es",
                "public domain", "a few MB"),),
        note="Accented text shifts the byte distribution enough to matter: "
             "the UTF-8 lead bytes are common here and absent in English.",
    ),
    CorpusType(
        "text-de", "Text, German", "text",
        "prose, with umlauts as multi-byte UTF-8",
        (".txt", ".md"), AS_IS, _TEXT_ROOTS,
        (Source("Project Gutenberg (German)",
                "https://www.gutenberg.org/browse/languages/de",
                "public domain", "a few MB"),),
    ),
    CorpusType(
        "markup", "Markup and source", "text",
        "HTML, XML, SVG, JSON, CSS, JS or source code, as written",
        (".html", ".htm", ".xml", ".svg", ".json", ".css", ".js", ".py",
         ".c", ".h", ".go", ".rs", ".java"), AS_IS, _CODE_ROOTS,
        (Source("Tranco homepages", "https://tranco-list.eu/",
                "the pages' own terms", "~200 KB a page, hundreds is plenty"),),
        note="The commonest thing wild DEFLATE actually carries.",
    ),
    CorpusType(
        "elf", "ELF binaries", "binary",
        "machine code and string tables",
        (".so", ".o", ".elf"), AS_IS, _BIN_ROOTS,
        (Source("Debian package pool", "https://deb.debian.org/debian/pool/",
                "per package", "one package is enough"),),
        note="The measured worst case for a wrong corpus: an English model "
             "scores -17.2 bits per byte here against -8.0 for no model at "
             "all. If the stream is a binary, this is the type that matters.",
    ),
    CorpusType(
        "png", "PNG pixel data", "image",
        "FILTERED SCANLINES -- not the PNG file's bytes",
        (".png",), INFLATE, _IMG_ROOTS,
        (Source("Wikimedia Commons", "https://commons.wikimedia.org/",
                "per image, mostly free", "a few hundred images"),),
        note="The file is already compressed. Training on PNG files as bytes "
             "trains on noise; the scan inflates the IDAT data first.",
    ),
)

BY_KEY = {t.key: t for t in TYPES}
CATEGORIES = ("text", "binary", "image")


def expand(root: str) -> Path:
    return Path(os.path.expanduser(root))


# Order by corpus size, as measured.
#
# Below a few MB this is a small dial and the ways of measuring it disagree.
# Held-out bits per byte prefers order 2 under 200 KB and 3 up to 1 MB, and
# never prefers 4 or more; filling holes on a stream whose answers are known
# prefers 5 from 128 KB up. Prediction and recovery are not the same job -- bits
# per byte averages over every byte, while the solver only acts where it clears
# 99.5%, so what matters there is the sharpest predictions rather than the mean.
# Below 4 MB the carve track's columns are noise, one or two fills apart.
#
# The kind of data in the corpus matters far more than this, and its size more
# than that: the same carve went from 0.17 to 0.42 precision on corpus size
# alone, while order moved it by one to three points.
ORDER_TIERS = ((32 << 10, 3), (128 << 10, 4), (8 << 20, 5))
# Above 8 MB the top order wins, and ziprec's default of 6 is vindicated: on the
# carve track, precision by order was
#
#    8 MB   0.367  0.381  0.384  0.389      (orders 3, 4, 5, 6)
#   15 MB   0.392  0.407  0.415  0.419
#
# monotonic in order at both sizes, and 52 correct fills at order 6 against 50
# at order 4. An earlier version of this table capped the order at 4 for large
# corpora on memory grounds, which would have thrown those away.
ORDER_MAX = 6

# What memory a model will need, which is not a function of the corpus size
# alone: it is the number of DISTINCT contexts, and that depends on what the
# data is. High-entropy data makes almost every long context unique, so the
# order steps down only when the corpus at hand is the expensive kind, rather
# than whenever it is large.
# Measured peak, in bytes of model per byte of BINARY corpus, at each order,
# for the pair of models the tool actually builds (forward and reverse), with
# `ContextModel.train_corpus`'s storage-time filtering in place.
#
# 10.9 MB of ELF at order 6: 3.4 GB, against 7.8 GB before the filter. One
# direction alone is 2.15 GB against 7.8/2. Text is far cheaper -- 15 MB of
# English prose at order 6 keeps 1.0 million contexts and peaks at 0.66 GB per
# direction -- because prose reuses its contexts and machine code does not.
#
# The cap now fires only on genuinely large binary corpora: 11 MB of ELF steps
# from 6 to 5, and the corpus readers stop at 8 MB in any case.
#
# A sample-based estimate of distinct contexts was tried first and abandoned: a
# Heaps-law fit from 512 KB overshot prose by 1.5x and undershot ELF by 3x, at
# which point it was predicting nothing useful. What the byte values are is a
# far better predictor than how their variety grows, and it costs one pass.
BINARY_COST = {3: 52, 4: 98, 5: 199, 6: 325}

# The same measurement for TEXT, which used to have no cost figure at all: the
# order was chosen from the corpus size and the memory check was skipped
# entirely for anything that looked textual. That was survivable only because the
# corpus readers silently stopped at 8 MB, so a text corpus never got large
# enough to matter. They no longer stop, so this table has to exist.
#
# Peak RSS per byte of corpus, for the forward/reverse pair, measured on 16 MB of
# Phrack (prose, C, hex dumps, tables -- a realistic mixed corpus rather than a
# novel). Text is roughly half the cost of binary at every order, not the sixth
# an earlier note here implied from a single Austen figure.
#
# Calibrated on MIXED text, and the distinction matters more than expected.
# Measured on the whole 30 MB Phrack corpus:
#
#     order 3   0.57 GB    20 B/byte
#     order 4   2.20 GB    78
#     order 5   4.32 GB   143
#     order 6   7.62 GB   271
#
# A first version of this table used the alphabetically-first 16 MB of the same
# corpus and got 13/36/81/160 -- roughly half. The later issues carry far more
# distinct context than the early ones, so a table calibrated on part of a corpus
# under-read the whole of it by 90%, and the order step-down it fed then chose an
# order believing it fit a budget it did not. Sampling the front of a corpus is
# not sampling the corpus.
#
# Even so this is one number for a range that spans 3x: 15 MB of plain English
# prose costs 88 B/byte at order 6 against phrack's 271, because a novel reuses
# its contexts and an article full of C, hex dumps and tables does not. These are
# the expensive-text figures, so the estimate runs high on plain prose. That is
# the safe direction for something whose only job is to warn.
TEXT_COST = {3: 20, 4: 78, 5: 143, 6: 271}

# The floor on what a model is allowed to cost before the operator is asked about
# it. Not a cap on what can be trained: a step down in order is reported with the
# override next to it, and TOEKNIFE_MEMORY_BUDGET moves the line.
MEMORY_BUDGET = 3 << 30


def memory_budget() -> int:
    """What to keep a model under, unless told otherwise.

    Half of what this machine has free, floored at the 3 GB this used to assume
    unconditionally. A fixed figure was the wrong shape once the cost estimate
    became realistic: mixed text at order 6 costs 271 bytes per corpus byte, so a
    3 GB ceiling shortens the context on any corpus past about 11 MB -- which is
    precisely the size range where the measurements say a long context pays.
    Timid on a 32 GB workstation, and still too generous on a 4 GB VM.

    The consequence is that the same corpus can earn a different order on a
    different machine. That is a real cost, paid deliberately: the order is
    reported before training, recorded in the model's provenance, and pinnable
    with --order when a run has to be reproducible.
    """
    raw = os.environ.get("TOEKNIFE_MEMORY_BUDGET", "").strip().lower()
    if raw:
        mult = {"k": 1 << 10, "m": 1 << 20, "g": 1 << 30}.get(raw[-1:], 1)
        try:
            return max(1 << 20, int(float(raw.rstrip("kmgb")) * mult))
        except ValueError:
            pass
    return max(MEMORY_BUDGET, available_memory() // 2)


def available_memory() -> int:
    """What the OS says is free right now, or 0 if it will not say."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def cost_per_byte(order: int, sample: bytes | None = None) -> int:
    table = TEXT_COST if (sample is not None and looks_textual(sample)) \
        else BINARY_COST
    return table.get(order, table[max(table)])


def estimate_memory(n_bytes: int, order: int,
                    sample: bytes | None = None) -> int:
    """Roughly what training this corpus at this order will peak at.

    Measured, and still an estimate with a factor of two or three in it: what a
    model costs is the number of DISTINCT contexts in the data, and no formula
    over a corpus's size and printable share can know that. It is calibrated to
    run high rather than low. Good enough to warn with, never a promise.
    """
    return n_bytes * cost_per_byte(order, sample)


def looks_textual(sample: bytes) -> bool:
    """Whether this corpus is the cheap kind to model at a long context."""
    if not sample:
        return False
    printable = sum(1 for b in sample if 0x20 <= b < 0x7F or b in (9, 10, 13))
    return printable / len(sample) > 0.9


def auto_order(n_bytes: int, sample: bytes | None = None,
               budget: int | None = None) -> int:
    """The context order to use for this corpus.

    Rises with the corpus, because that is what the measurements say: at 8 MB
    and 15 MB the highest order wins on the carve track, monotonically. The only
    thing that pulls it back down is memory.

    That check now runs on text as well as binary. It used to skip anything
    textual, which was safe only while the readers stopped at 8 MB; a 30 MB text
    corpus at order 6 wants about 4.8 GB. When it does pull the order down,
    `order_note` says so and names the override, because a silently shortened
    context is exactly the kind of quiet degradation this tool exists to avoid.
    """
    budget = memory_budget() if budget is None else budget
    order = ORDER_MAX
    for limit, tier in ORDER_TIERS:
        if n_bytes < limit:
            order = tier
            break
    if sample is None or order <= 3:
        return order
    while order > 3 and estimate_memory(n_bytes, order, sample) > budget:
        order -= 1
    return order


def _si(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def order_note(n_bytes: int, sample: bytes | None = None,
               chosen=None) -> tuple[int, str]:
    """(order, one line saying where it came from and what it will cost).

    The line is the whole point: an operator who is about to spend minutes and
    gigabytes should be told the figure first, and told when the tool has quietly
    picked a shorter context than the size alone would earn.
    """
    budget = memory_budget()
    if chosen not in (None, "auto"):
        order = int(chosen)
        why = "you set it"
    else:
        order = auto_order(n_bytes, sample)
        by_size = ORDER_MAX
        for limit, tier in ORDER_TIERS:
            if n_bytes < limit:
                by_size = tier
                break
        why = ("from its size" if order == by_size else
               f"lowered from {by_size} to stay inside {_si(budget)}; "
               f"--order {by_size} overrides, TOEKNIFE_MEMORY_BUDGET moves it")
    want = estimate_memory(n_bytes, order, sample)
    note = (f"context order {order} ({why}); roughly {_si(want)} of memory, "
            f"give or take a factor of two")
    if want > budget:
        note += f", over the {_si(budget)} budget"
    note += "; " + training_time_note(n_bytes, order)
    return order, note


# ---- how long an n-gram takes to train, on this machine --------------------

# One measured run, for a machine that has not trained anything yet: 13
# Project Gutenberg books, 17,926,231 bytes at order 6, 9 min 3 s with the
# calibration (2026-10-30). Every run after it records its own rate.
REFERENCE_RATE = (6, 33_000)


def rates_path() -> Path:
    return expand(os.environ.get("TOEKNIFE_RATES", "~/.toeknife/rates.json"))


def _rates() -> dict:
    import json
    try:
        return json.loads(rates_path().read_text())
    except (OSError, ValueError):
        return {}


def record_training_rate(order: int, n_bytes: int, seconds: float) -> None:
    """Keep what this run measured: bytes per second at this order, training
    and calibration together, the newest run winning."""
    import json
    if seconds <= 0 or n_bytes <= 0:
        return
    rates = _rates()
    rates[f"ngram-o{order}"] = {"bytes_per_second": n_bytes / seconds,
                                "bytes": n_bytes, "seconds": seconds}
    p = rates_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rates, indent=1))
    except OSError:
        pass                            # an estimate is not worth failing for


def _duration(seconds: float) -> str:
    if seconds < 90:
        return f"{max(1, round(seconds))} s"
    if seconds < 90 * 60:
        return f"{round(seconds / 60)} min"
    return f"{seconds / 3600:.1f} h"


def training_time_note(n_bytes: int, order: int) -> str:
    """How long training this many bytes at this order should take, and where
    that figure comes from."""
    rates = _rates()
    mine = rates.get(f"ngram-o{order}")
    if mine:
        rate = mine["bytes_per_second"]
        return (f"about {_duration(n_bytes / rate)} at this machine's "
                f"measured {_si(rate)}/s for order {order}")
    measured = sorted((int(k[7:]), v["bytes_per_second"]) for k, v in
                      rates.items() if k.startswith("ngram-o"))
    if measured:
        near_order, rate = min(measured, key=lambda m: abs(m[0] - order))
        return (f"about {_duration(n_bytes / rate)} at this machine's "
                f"{_si(rate)}/s measured at order {near_order}; this run "
                f"measures order {order}")
    ref_order, rate = REFERENCE_RATE
    return (f"about {_duration(n_bytes / rate)} at a reference "
            f"{_si(rate)}/s measured at order {ref_order} on another machine; "
            f"this run measures yours")


def resolve_order(order, n_bytes: int, sample: bytes | None = None) -> int:
    """Whatever the operator asked for, or the measured default for this data."""
    if order in (None, "auto"):
        return auto_order(n_bytes, sample)
    return int(order)


def sample_of(source, limit: int = 64 << 10) -> bytes:
    """A slice of a corpus to judge its kind by: bytes, or paths to read."""
    if isinstance(source, (bytes, bytearray)):
        return bytes(source[:limit])
    out = b""
    for path in source or ():
        try:
            with open(path, "rb") as fh:
                out += fh.read(limit - len(out))
        except OSError:
            continue
        if len(out) >= limit:
            break
    return out


def suggested_roots(t: CorpusType) -> list[Path]:
    """The roots for this type that actually exist on this machine."""
    return [p for p in (expand(r) for r in t.roots) if p.is_dir()]


def find(t: CorpusType, roots, limit: int = 20000,
         max_bytes: int | None = None) -> list[Path]:
    """Files of this type under `roots`, oldest-fashioned way: by extension.

    Extension matching is crude and deliberately so. The alternative is sniffing
    every file on a disk, which on a machine under examination means reading
    everything -- a cost the operator has not agreed to. A wrong guess here is
    cheap because `sanity_check` measures the result against the real stream
    afterwards.

    No byte cap by default. This used to stop at 8 MB and additionally skip any
    single file larger than 8 MB, both silently, so a corpus the operator had
    assembled on purpose was quietly sampled instead of read -- and the largest
    file in it, usually the most substantial, was the one guaranteed to be
    dropped. `limit` remains as a backstop on enumeration rather than on content,
    and `Plan.truncated` says out loud when it bites.
    """
    want = {e.lower() for e in t.extensions}
    out, total = [], 0
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if len(out) >= limit or (max_bytes is not None
                                     and total >= max_bytes):
                return out
            try:
                if not p.is_file() or p.suffix.lower() not in want:
                    continue
                size = p.stat().st_size
            except OSError:
                continue
            if size == 0:
                continue
            out.append(p)
            total += size
    return out


def inflate_streams(raw: bytes) -> bytes:
    """Every DEFLATE stream inside `raw`, decompressed and joined.

    Used wherever the content is inside a container rather than being the file:
    a PNG's scanlines, a gzip's payload, a fetched `.gz`. Guessing the wrapper
    wrong yields no bytes rather than wrong ones, so each is tried in turn.
    """
    import zlib

    from ..containers import GZIP, ZLIB, parse
    try:
        report = parse(raw)
    except Exception:
        return b""
    if report is None:
        return b""
    out = []
    for s in report.streams:
        end = (s.offset + s.length) if s.length else len(raw)
        blob = raw[s.offset:end]
        if not blob:
            continue
        wbits = 47 if s.encoding == GZIP else 15 if s.encoding == ZLIB else -15
        for w in (wbits, -15, 15, 47):
            try:
                out.append(zlib.decompressobj(w).decompress(blob))
                break
            except zlib.error:
                continue
    return b"".join(out)


def training_bytes(t: CorpusType, path: Path) -> bytes:
    """What to actually train on, which is not always the file.

    For an already-compressed type this inflates first. Failing to do that is
    the quiet way to end up with a model that is confidently wrong.
    """
    raw = Path(path).read_bytes()
    return inflate_streams(raw) if t.needs_inflating else raw


@dataclass
class Plan:
    """What a scan proposes to do, before it does any of it."""
    type: CorpusType
    roots: list[Path] = field(default_factory=list)
    files: list[Path] = field(default_factory=list)
    whole_system: bool = False
    # The enumeration hit its backstop, so this is a sample of what is there
    # rather than all of it. Said out loud because a silently sampled corpus
    # reads as a complete one.
    truncated: bool = False

    @property
    def total_bytes(self) -> int:
        n = 0
        for p in self.files:
            try:
                n += p.stat().st_size
            except OSError:
                pass
        return n

    def render(self) -> str:
        where = ("the whole filesystem" if self.whole_system else
                 ", ".join(tilde(r) for r in self.roots) or "nowhere")
        out = [f"{self.type.name}  ({self.type.category})",
               f"  the stream holds: {self.type.contains}",
               f"  looking in: {where}",
               f"  found {len(self.files)} file(s), "
               f"{self.total_bytes:,} bytes"]
        if self.truncated:
            out.append("  the scan stopped at its file-count backstop: this is "
                       "a sample of what is there")
        if self.type.needs_inflating:
            out.append("  these will be INFLATED before training -- the file "
                       "bytes are already compressed")
        if self.type.note:
            out.append(f"  note: {self.type.note}")
        if not self.files:
            out.append("  nothing found. Name a root that has this kind of "
                       "data, or pick another type.")
        return "\n".join(out)


def plan(key: str, roots=None, whole_system: bool = False,
         limit: int = 20000) -> Plan:
    """Propose a scan. Nothing is read beyond directory listings until the
    operator accepts it."""
    t = BY_KEY[key]
    if whole_system:
        use = [Path("/")]
    elif roots:
        use = [expand(str(r)) for r in roots]
    else:
        use = suggested_roots(t)
    files = find(t, use, limit)
    return Plan(t, use, files, whole_system, truncated=len(files) >= limit)


def sources_for(category: str | None = None) -> list[tuple[CorpusType, Source]]:
    """The external list, by category. Nothing here is fetched implicitly."""
    out = []
    for t in TYPES:
        if category and t.category != category:
            continue
        for s in t.sources:
            out.append((t, s))
    return out


# A model is written wherever `--save-model` was pointed, which is usually not
# the models directory: the README's own example writes `english.tkm` into the
# working directory. Discovery therefore looks where models actually land, and
# accepts the extensions people actually use, rather than making the operator
# remember a path.
MODEL_EXTS = (".tkm", ".model", ".pkl", ".tks")
MODEL_DIRS = ("~/.toeknife/models",)

# Two kinds of model can be sitting in that directory and they are not
# interchangeable.  An n-gram answers logprob(byte, left) and plugs into a
# Mixture; a byte model reads the TAPE, where UNKNOWN is a token it was trained
# on, and answers a whole equality class at once.  Splicing the known text
# around a hole to make the second look like the first is a measured failure -- "for" + hole + "forensic" reads as "forforensic" -- so the kinds
# have to stay apart all the way to the picker.
NGRAM, SLM = "ngram", "slm"
SLM_EXT = ".tks"


def _cache_path(p: Path) -> Path:
    return p.with_suffix(p.suffix + ".json")


def _read_cache(p: Path):
    """Metadata cached beside a model, or None to say read the model itself.

    Invalidated on size and mtime rather than on content, because the point of
    the cache is not to open the model.  A cache that disagrees with the file
    beside it is discarded rather than trusted, and a corrupt one is treated the
    same way: a bad cache is not a bad model.
    """
    side = _cache_path(p)
    if not side.is_file():
        return None
    try:
        import json
        blob = json.loads(side.read_text())
        st = p.stat()
        if blob.get("_bytes") != st.st_size or blob.get("_mtime") != int(
                st.st_mtime):
            return None
        return {k: v for k, v in blob.items() if not k.startswith("_")}
    except Exception:
        return None


def _write_cache(p: Path, info: dict) -> None:
    try:
        import json
        st = p.stat()
        blob = {k: v for k, v in info.items() if k not in ("path", "name")}
        blob["_bytes"], blob["_mtime"] = st.st_size, int(st.st_mtime)
        _cache_path(p).write_text(json.dumps(blob, indent=1, default=str))
    except Exception:
        pass                            # caching is a convenience, never a duty


def _slm_meta(p: Path) -> dict:
    """What a byte-model checkpoint says about itself.

    Read from a JSON sidecar when one is there, and only from the checkpoint
    otherwise.  The checkpoint is the canonical copy -- provenance travels
    inside it, the way it does for an n-gram -- but reading it means importing
    torch and unpickling tens of megabytes of tensors, and the corpus picker
    should not pay two seconds and a torch import to draw a list for an operator
    who only wants an n-gram.  The sidecar is a cache: if it is missing it is
    rebuilt from the checkpoint, and if the checkpoint is gone the sidecar is
    worthless and ignored.
    """
    info = {"path": str(p), "name": p.name, "bytes": p.stat().st_size,
            "kind": SLM}
    cached = _read_cache(p)
    if cached is not None:
        info.update(cached)
        info["path"], info["name"] = str(p), p.name
        return info
    try:
        import torch
        blob = torch.load(p, weights_only=False, map_location="cpu")
    except Exception as e:
        info["error"] = type(e).__name__
        return info
    if not isinstance(blob, dict) or "cfg" not in blob:
        info["error"] = "not a ToeKnife byte model"
        return info
    cfg = blob.get("cfg") or {}
    info.update(provenance=blob.get("provenance") or {},
                d_model=cfg.get("d_model"), layers=cfg.get("layers"),
                window=cfg.get("window"), steps=blob.get("steps"),
                predictions=blob.get("predictions"),
                val_acc=blob.get("val_acc"))
    from .calibration import Calibration
    cal = Calibration.from_dict(blob.get("calibration"), kind="slm")
    if cal is not None:
        info["calibrated"] = {"classes": cal.classes, "best": cal.best()}
    _write_cache(p, info)
    return info


def _model_meta(p: Path) -> dict:
    """What a saved model says about itself, without instantiating it."""
    if p.suffix == SLM_EXT:
        return _slm_meta(p)
    info = {"path": str(p), "name": p.name, "bytes": p.stat().st_size,
            "kind": NGRAM}
    # An n-gram carries no cheap header: reading its order means unpickling the
    # whole model, and a corpus-sized one runs to hundreds of megabytes. The
    # picker drew one row per model by unpickling every byte of every model.
    cached = _read_cache(p)
    if cached is not None:
        info.update(cached)
        info["path"], info["name"] = str(p), p.name
        return info
    try:
        import pickle
        blob = pickle.loads(p.read_bytes())
    except Exception as e:
        info["error"] = type(e).__name__
        return info
    if not isinstance(blob, dict):
        info["error"] = "not a ToeKnife model"
        return info
    info["provenance"] = blob.get("provenance") or {}
    cal = blob.get("calibration")
    if cal:
        best = max((b[1] for b in cal.get("short") or []), default=0.0)
        info["calibrated"] = {"classes": cal.get("classes"), "best": best}
    if blob.get("bidirectional"):
        # A bidirectional model keeps its two halves pickled inside itself, so
        # order and size live one level down.
        info["direction"] = "forward+reverse"
        try:
            fwd = pickle.loads(blob["forward"])
            info["order"] = fwd.get("order")
            info["trained_bytes"] = fwd.get("trained_bytes")
        except Exception:
            pass
    elif "counts" in blob:
        info["direction"] = "forward only"
        info["order"] = blob.get("order")
        info["trained_bytes"] = blob.get("trained_bytes")
    else:
        info["error"] = "not a ToeKnife model"
    if "error" not in info:
        _write_cache(p, info)
    return info


def calibrated_name(path) -> str:
    """The file name a calibrated copy of this byte model is written to."""
    p = Path(str(path))
    return p.stem + "-cal" + p.suffix


def register_slm(src, name: str, provenance: dict,
                 directory=None) -> Path:
    """Copy a byte-model checkpoint into the models directory with its lineage.

    The n-gram's rule applies unchanged: what a model was trained on travels
    with it or the operator is holding a hypothesis generator of unknown
    parentage.  A research checkpoint carries its config and its step count and
    nothing about the corpus behind it, so registering one means supplying that,
    and the caller is expected to have it from the write-up rather than to guess.

    Refuses to overwrite.  These cost days.
    """
    import shutil
    import torch

    src = Path(os.path.expanduser(str(src)))
    d = Path(os.path.expanduser(str(directory or model_dir())))
    d.mkdir(parents=True, exist_ok=True)
    dest = d / (name if name.endswith(SLM_EXT) else name + SLM_EXT)
    if dest.exists():
        raise FileExistsError(f"{dest} already exists")
    blob = torch.load(src, weights_only=False, map_location="cpu")
    if not isinstance(blob, dict) or "cfg" not in blob:
        raise ValueError(f"{src} is not a ToeKnife byte model")
    blob["provenance"] = dict(provenance)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    torch.save(blob, tmp)
    shutil.move(str(tmp), str(dest))
    side = dest.with_suffix(dest.suffix + ".json")
    if side.exists():
        side.unlink()               # stale cache from a previous file of the name
    return dest


def saved_models(directory: str | Path) -> list[dict]:
    """Models the operator built earlier, with whatever lineage they carry.

    Pre-trained corpora are operator-built, so a saved model IS the
    distribution format. What it was trained on travels with it or the operator
    is holding a hypothesis generator of unknown parentage.
    """
    d = Path(os.path.expanduser(str(directory)))
    if not d.is_dir():
        return []
    seen, out = set(), []
    for ext in MODEL_EXTS:
        for p in sorted(d.glob("*" + ext)):
            if p in seen:
                continue
            seen.add(p)
            out.append(_model_meta(p))
    return out


# Directory names that hold prose rather than build output, checked under the
# places the operator already is. Offering the working directory itself is not
# on: on a project checkout that means the virtualenv, and a corpus of compiled
# wheels is the -17.2 case.
DOC_DIRS = ("docs", "doc", "documentation", "text", "notes")


def _texty() -> set[str]:
    """Extensions whose bytes ARE the content, so training on the files is
    training on the thing itself."""
    out = set()
    for t in TYPES:
        if t.category == "text" and not t.needs_inflating:
            out |= {e.lower() for e in t.extensions}
    return out


def _count_files(root: Path, limit: int = 400) -> tuple[int, int, int, int]:
    """(files, bytes, texty files, texty bytes) under `root`, stopping at
    `limit`. Directory listings only, which is all the operator has agreed to
    by opening the screen."""
    want = _texty()
    files = total = tfiles = ttotal = 0
    for p in root.rglob("*"):
        if files >= limit:
            break
        try:
            if not p.is_file():
                continue
            size = p.stat().st_size
        except OSError:
            continue
        files += 1
        total += size
        if p.suffix.lower() in want:
            tfiles += 1
            ttotal += size
    return files, total, tfiles, ttotal


def nearby_corpora(extra=()) -> list[tuple[Path, int, int]]:
    """Prose you already have to hand, as (path, files, bytes).

    The corpus that fits a capture is usually the one sitting next to it or in
    the project it came from, and the difference is not small: on the preset
    dictionary sample this repo's own `docs/` scores -3.1 bits per byte where
    2 MB of general English scores -4.4 and 8 MB of markup scores -5.1. That
    corpus was reachable only by typing its path from memory, which is a poor
    place to keep the best answer.
    """
    roots = []
    for base in list(extra) + [os.getcwd()]:
        b = Path(base)
        if not b.is_dir():
            continue
        roots.append(b if b.name in DOC_DIRS else None)
        roots += [b / name for name in DOC_DIRS]
        # A capture's own directory is usually small and usually relevant.
        if b != Path(os.getcwd()):
            roots.append(b)
    out, seen = [], set()
    for p in roots:
        if p is None or not p.is_dir():
            continue
        try:
            key = p.resolve()
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        files, total, tfiles, ttotal = _count_files(p)
        # Committing one of these rows trains on everything under it, so only
        # offer directories that are mostly prose. A capture's own directory
        # often holds the compressed samples themselves, and training on those
        # is the -17.2 bits-per-byte case wearing a helpful label.
        if tfiles >= 3 and total and ttotal >= 0.8 * total:
            out.append((p, tfiles, ttotal))
    return out


def model_search_paths(extra=()) -> list[Path]:
    """Directories to look in for saved models, in order, deduplicated."""
    raw = []
    env = os.environ.get("TOEKNIFE_MODELS", "")
    raw += [w for w in env.split(os.pathsep) if w]
    raw += list(MODEL_DIRS)
    raw += [str(e) for e in extra if e]
    raw.append(os.getcwd())
    out, seen = [], set()
    for r in raw:
        p = expand(str(r))
        try:
            key = p.resolve()
        except OSError:
            continue
        if key in seen or not p.is_dir():
            continue
        seen.add(key)
        out.append(p)
    return out


def find_models(extra=()) -> list[dict]:
    """Every saved model on the search path, newest first within a directory."""
    out, seen = [], set()
    for d in model_search_paths(extra):
        for info in saved_models(d):
            key = os.path.realpath(info["path"])
            if key in seen:
                continue
            seen.add(key)
            info["dir"] = str(d)
            out.append(info)
    return out


def model_dir() -> Path:
    """Where a model written from inside the tool goes.

    The first entry of the search path, so anything saved here is found again
    without the operator recording where it went. Not created until something
    is actually written.
    """
    return expand(MODEL_DIRS[0])


def tilde(path) -> str:
    """A path as the operator would type it: under their home, as ~/...

    Shorter on a screen that has little room, and it keeps a username out of
    every screenshot and shared session.
    """
    s = str(path)
    home = os.path.expanduser("~")
    if home and home != "/" and (s == home or s.startswith(home + os.sep)):
        return "~" + s[len(home):]
    return s


def trash_dir() -> Path:
    """Where a model goes when the operator removes it: beside the models,
    never gone. Some of these cost days of compute, and the only undo a plain
    delete offers is training it again."""
    base = expand(MODEL_DIRS[0]).parent if MODEL_DIRS else expand("~/.toeknife")
    return base / "trash"


def trash_model(path) -> Path:
    """Move a saved model, and its metadata cache, into the trash.

    Returns where it went. Timestamped so two models of one name never
    collide, and moved rather than copied-then-deleted so a model of hundreds
    of megabytes costs a rename. Restoring is `mv` back.
    """
    import shutil
    import time
    src = Path(os.path.expanduser(str(path)))
    if not src.is_file():
        raise FileNotFoundError(f"{src} is not there")
    d = trash_dir()
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{time.strftime('%Y%m%d-%H%M%S')}-{src.name}"
    shutil.move(str(src), str(dest))
    side = _cache_path(src)
    if side.exists():
        shutil.move(str(side), str(_cache_path(dest)))
    return dest


def _url_stem(url: str) -> str:
    """Enough of a URL to recognise a corpus by: the host and the file."""
    from urllib.parse import urlparse
    try:
        u = urlparse(url)
    except ValueError:
        return ""
    host = (u.netloc or "").split(":")[0]
    host = host[4:] if host.startswith("www.") else host
    tail = Path(u.path or "").stem
    return "-".join(w for w in (host, tail) if w)


def _path_stem(path: str) -> str:
    """What a corpus read off this machine should be called: the file's own
    name, or the directory the sweep started from."""
    p = Path(path)
    name = p.stem if p.suffix else p.name
    return name or p.parent.name


def _slug(s: str) -> str:
    out = "".join(c if (c.isalnum() or c in "-_.") else "-"
                  for c in str(s).strip().lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-.") or "corpus"


def suggest_model_name(provenance: dict | None = None) -> str:
    """A filename that says what the model is, from what training recorded.

    This is shown to the operator as a default they will mostly accept, so it
    carries the two things that decide whether a saved model is the right one to
    reuse later: what it read, and how long a context it kept. The rest of the
    lineage travels inside the file.
    """
    prov = provenance or {}
    stem = (prov.get("download") or prov.get("type")
            or _url_stem(str(prov.get("url") or ""))
            or _path_stem(str(prov.get("path") or "")) or "corpus")
    order = prov.get("order")
    return f"{_slug(stem)}{f'-o{order}' if order else ''}.tkm"


def suggest_model_path(provenance: dict | None = None) -> Path:
    """Where to offer to write it, without proposing to overwrite anything.

    Two models off the same corpus at the same order differ by what was
    truncated or which files were swept, and neither name can say which. So the
    second one gets a suffix rather than the first one's bytes.
    """
    d = model_dir()
    p = d / suggest_model_name(provenance)
    stem, ext, n = p.stem, p.suffix, 2
    while p.exists():
        p = d / f"{stem}-{n}{ext}"
        n += 1
    return p


# ---- getting it from outside -------------------------------------------------

CACHE = "~/.toeknife/corpora"
USER_AGENT = "ToeKnife/2 (DEFLATE forensics; corpus fetch)"
MAX_FETCH = 32 << 20


@dataclass(frozen=True)
class Download:
    """One specific file that can be fetched.

    `Source` names a library; this names a book on its shelf. The difference is
    the whole reason the old screen did not help: an operator who picked
    "Project Gutenberg" off a list still had no corpus, because a landing page
    cannot be trained on.

    `approx` is the size measured when the entry was written, so the operator
    knows what they are agreeing to before anything is fetched.
    """
    key: str
    type_key: str
    name: str
    url: str
    approx: int
    licence: str
    note: str = ""


DOWNLOADS: tuple[Download, ...] = (
    Download("gutenberg-en-pride", "text-en",
             "Project Gutenberg: Pride and Prejudice",
             "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
             772_386, "public domain"),
    Download("gutenberg-en-moby", "text-en",
             "Project Gutenberg: Moby-Dick",
             "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
             1_276_263, "public domain",
             "More of the same kind; order 4 wants hundreds of KB, not GB."),
    Download("gutenberg-es-quijote", "text-es",
             "Project Gutenberg: Don Quijote",
             "https://www.gutenberg.org/cache/epub/2000/pg2000.txt",
             2_225_803, "public domain"),
    Download("gutenberg-de-verwandlung", "text-de",
             "Project Gutenberg: Die Verwandlung",
             "https://www.gutenberg.org/cache/epub/22367/pg22367.txt",
             145_983, "public domain"),
    Download("gutenberg-de-faust", "text-de",
             "Project Gutenberg: Faust I",
             "https://www.gutenberg.org/cache/epub/2229/pg2229.txt",
             218_440, "public domain"),
    Download("zlib-inflate-c", "markup",
             "zlib source: inflate.c",
             "https://raw.githubusercontent.com/madler/zlib/develop/inflate.c",
             52_247, "zlib licence",
             "Small. Good for a stream that carries C, thin for anything else."),
    Download("wikipedia-deflate", "markup",
             "Wikipedia: DEFLATE, as served",
             "https://en.wikipedia.org/wiki/DEFLATE",
             232_851, "CC BY-SA",
             "Served HTML, which is what wild DEFLATE mostly carries."),
    Download("wikimedia-png", "png",
             "Wikimedia: PNG transparency demonstration",
             "https://upload.wikimedia.org/wikipedia/commons/4/47/"
             "PNG_transparency_demonstration_1.png",
             224_566, "free licence, see the file page",
             "Inflated to filtered scanlines before training. One image is a "
             "thin corpus; this is a demonstration of the right shape."),
)


def downloads_for(type_key: str | None = None) -> list[Download]:
    return [d for d in DOWNLOADS if type_key is None or d.type_key == type_key]


class FetchError(RuntimeError):
    """A fetch that did not happen, with a reason worth reading."""


def cache_dir() -> Path:
    d = expand(CACHE)
    d.mkdir(parents=True, exist_ok=True)
    return d


def cache_path(url: str) -> Path:
    """A stable, readable filename for a URL, host first."""
    import hashlib
    from urllib.parse import urlparse
    u = urlparse(url)
    stem = os.path.basename(u.path) or "index"
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in stem)[:60]
    host = (u.netloc or "elsewhere").replace(":", "_")
    digest = hashlib.sha256(url.encode()).hexdigest()[:8]
    return cache_dir() / f"{host}-{digest}-{stem}"


def fetch(url: str, max_bytes: int = MAX_FETCH, timeout: int = 20,
          reuse_cache: bool = True, progress=None) -> tuple[Path, int, bool]:
    """Download `url` into the cache. Returns (path, bytes, came_from_cache).

    Nothing in this module calls it on its own: the screens show the URL and
    the size first and only get here once the operator has said yes. A forensic
    tool reaching the network is an event, so it is one the operator causes.
    """
    import urllib.error
    import urllib.request
    from urllib.parse import urlparse

    if urlparse(url).scheme not in ("http", "https"):
        raise FetchError("only http and https are fetched; that is not one")
    dest = cache_path(url)
    if reuse_cache and dest.is_file() and dest.stat().st_size:
        return dest, dest.stat().st_size, True

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if urlparse(resp.geturl()).scheme not in ("http", "https"):
                raise FetchError("redirected somewhere that is not http")
            declared = resp.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise FetchError(
                    f"{int(declared):,} bytes is past the {max_bytes:,} limit")
            # Content-Length when the server gives one, so the bar is real; 0
            # when it does not, which the caller renders as "size unknown"
            # rather than inventing a percentage.
            expected = int(declared) if declared and declared.isdigit() else 0
            got, chunks = 0, []
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                got += len(chunk)
                if got > max_bytes:
                    raise FetchError(
                        f"stopped at the {max_bytes:,} byte limit")
                chunks.append(chunk)
                if progress is not None:
                    progress(got, expected, f"downloading {url}")
    except FetchError:
        raise
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"could not reach it: {e.reason}") from e
    except OSError as e:
        raise FetchError(str(e)) from e
    if not got:
        raise FetchError("the server sent nothing")
    dest.write_bytes(b"".join(chunks))
    return dest, got, False


def corpus_bytes(path: Path) -> tuple[bytes, str]:
    """Training bytes from a file of unknown kind, and how they were obtained.

    A fetched `.png` or `.gz` is a container, and training on its bytes trains
    on compressed noise. This inflates when the file turns out to hold DEFLATE
    and says which happened, because the difference decides whether the model
    is any use.
    """
    raw = Path(path).read_bytes()
    inflated = inflate_streams(raw)
    if inflated:
        return inflated, "inflated"
    return raw, "as-is"
