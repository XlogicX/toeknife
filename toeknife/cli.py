"""Command line entry point.

With no output flag it opens the interactive view.  Every other flag makes it
behave like a normal unix filter so it can be scripted, diffed against infgen,
or dropped into a carving pipeline.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core import align, ledger, plaintext, provenance, related, routes
from .core.bitreader import bit_string
from .core.huffman import HuffmanTable
from .core.session import Session, load_input, tables_from_header
from .core.tables import BTYPE_STORED, DISTANCE_BASE, LENGTH_BASE
from .core.tape import UNKNOWN


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="toeknife",
        description="Forensic carver and bit-level inspector for DEFLATE streams "
                    "and fragments.",
        epilog="With no output option, opens the interactive view.")
    src = p.add_argument_group("input")
    src.add_argument("file", nargs="?", help="file containing the DEFLATE data")
    src.add_argument("--data", metavar="HEX",
                     help="raw data as hex on the command line")
    src.add_argument("--header", metavar="FILE",
                     help="separate fragment that starts at a block header, used "
                          "to recover the Huffman tables for a body fragment")
    src.add_argument("--header-bit", type=int, default=0, metavar="N",
                     help="bit offset of the block header inside --header")
    src.add_argument("--entry", metavar="N|NAME|all",
                     help="which stream inside a container to open, by index or "
                          "by name; with no --entry a container's entries are "
                          "listed, unless only one can be decoded. 'all' joins "
                          "the members of a multi-member gzip into one file, "
                          "as gunzip does, for --verify, --fill-from and "
                          "--export")
    src.add_argument("--raw", action="store_true",
                     help="treat the input as bare DEFLATE, ignoring any container")

    mode = p.add_argument_group("decoding")
    g = mode.add_mutually_exclusive_group()
    g.add_argument("--fragment", action="store_const", const="fragment", dest="mode",
                   help="treat input as headerless mid-stream bits")
    g.add_argument("--stream", action="store_const", const="stream", dest="mode",
                   help="treat input as a complete stream with block headers")
    mode.add_argument("--bit", type=int, metavar="N",
                      help="start at this bit offset and skip the ranking")
    mode.add_argument("--profile", choices=("auto", "text", "binary"), default="auto",
                      help="what the plaintext is expected to look like (default auto)")
    mode.add_argument("--corpus", nargs="+", metavar="PATH",
                      help="files or directories of similar data to train the "
                           "n-gram model on; sharpens alignment ranking a "
                           "great deal")
    mode.add_argument("--corpus-type", metavar="TYPE",
                      help="train on a KIND of data found on this machine, "
                           "found by type.  --corpus-types lists "
                           "them.  A type that is already compressed (png) is "
                           "inflated first, because training on the file's own "
                           "bytes there trains on noise")
    mode.add_argument("--corpus-root", nargs="+", metavar="PATH",
                      help="where to look for --corpus-type (default: "
                           "suggested places for that type that exist here)")
    mode.add_argument("--corpus-scan-all", action="store_true",
                      help="look for --corpus-type across the whole "
                           "filesystem.  Slow, and on a machine under "
                           "examination it reads everything")
    mode.add_argument("--corpus-types", action="store_true",
                      help="list the corpus types, what each one's DEFLATE "
                           "stream actually contains, and where to get it")
    mode.add_argument("--order", default="auto", metavar="N|auto",
                      type=_order_arg,
                      help="context model order, per corpus (default auto: 3 "
                           "under 32 KB, 4 under 128 KB, 5 under 8 MB, 6 above, "
                           "stepped down on binary corpora where a long context "
                           "costs gigabytes). Below a few MB the kind of data "
                           "matters far more than the order; above 8 MB the "
                           "longest context wins")
    mode.add_argument("--forward-only", action="store_true",
                      help="train only a forward model. Both directions are "
                           "built by default, as ziprec does: a hole at the edge "
                           "of the window has no left context and can only be "
                           "judged by what follows it")
    mode.add_argument("--predictor", default="ngram", metavar="SPEC",
                      help="ngram (default, no dependencies), or hf:NAME for a "
                           "HuggingFace causal LM, which needs "
                           "pip install 'toeknife[ml]' and is for text only")
    mode.add_argument("--save-model", metavar="FILE",
                      help="train on --corpus, write the model out, and exit")
    mode.add_argument("--model", metavar="FILE",
                      help="load a saved model (skips training)")
    mode.add_argument("--calibrate-slm", metavar="FILE.tks",
                      help="measure a byte model on text it never trained on "
                           "and write FILE-cal.tks with its curve inside, then "
                           "exit. The text is --held-out, or the files the "
                           "training run held back, found under --corpus. "
                           "--model's n-gram builds the candidate pool, as "
                           "it would at fill time. The 19M model took 5 "
                           "minutes for 2 articles and 4.8 hours for 30 "
                           "pieces of 32 KB")
    mode.add_argument("--held-out", nargs="+", metavar="FILE[@OFFSET]",
                      help="for --calibrate-slm: text the model never trained "
                           "on, 32 KB from OFFSET (default 0). The trainer "
                           "reads the first 48,000 bytes of each file, so "
                           "FILE@48000 is unseen text of a file it trained on")
    mode.add_argument("--max-offset", type=int, default=48, metavar="N",
                      help="how many starting bit offsets to rank (default 48)")

    solve = p.add_argument_group("filling holes with a model")
    solve.add_argument("--solve", action="store_true",
                       help="fill unknown byte classes with the context model, "
                            "then continue with whatever output option follows")
    solve.add_argument("--solve-confidence", type=_probability, default=0.995,
                       metavar="P",
                       help="the floor the fill descends to, surest class "
                            "first. With a calibrated model it is a measured "
                            "precision: 0.9 fills only what the model got right "
                            "90%% of the time on text held back from it. "
                            "Uncalibrated, it is the model's own confidence, "
                            "which overstates. Lower it to trade accuracy for "
                            "coverage (default 0.995)")
    solve.add_argument("--solve-batch", type=int, default=32, metavar="N",
                       help="classes filled per round before re-scoring what they "
                            "touched. Staging is worth 8 points of precision over "
                            "filling everything above the floor at once; 1 is no "
                            "better than 32 and 4.7x slower (default 32)")
    solve.add_argument("--solve-propose", action="store_true",
                       help="list what the model would fill, and stop")
    solve.add_argument("--byte-model", metavar="FILE.tks",
                       help="fill with a byte model as well (needs torch). "
                            "It reads the tape, holes included, and answers a "
                            "whole class at once. Workflow A's fill took it "
                            "81-89 s against the n-gram's 11")
    solve.add_argument("--fill-mode", choices=("both", "slm", "ngram"),
                       help="with --byte-model: both (the byte model's answer, "
                            "the default), slm (the byte model alone) or "
                            "ngram (ignore the byte model)")
    solve.add_argument("--fill-dictionary", action="store_true",
                       help="also fill classes holding bytes of a missing "
                            "zlib preset dictionary. Held back by default: "
                            "a dictionary is a word list, and fills there "
                            "measured 48%% right at a 0.8 floor; --dict "
                            "finds the real one")
    solve.add_argument("--fill-long-runs", action="store_true",
                       help="also fill classes that sit only inside runs of "
                            "256+ bytes nothing was read from (a lost packet, "
                            "a bad sector). Held back by default: fills there "
                            "measured 19%% right whatever the "
                            "confidence floor")
    solve.add_argument("--solve-loop", action="store_true",
                       help="alternate the model and the container's checksum "
                            "until neither makes progress. Each class the model "
                            "fills is one fewer unknown in the checksum's linear "
                            "system; each the checksum pins is new context for "
                            "the model")
    solve.add_argument("--checksum-solve", "--crc-solve", action="store_true",
                       dest="checksum_solve",
                       help="use the container's checksum to pin the remaining "
                            "unknown bytes. CRC32 is strong; Adler-32 is two "
                            "sums and routinely admits wrong answers, so "
                            "surviving candidates are counted and every one "
                            "reported")
    solve.add_argument("--shortlist", type=int, default=16, metavar="N",
                       help="model candidates per unknown class to combine and "
                            "test against the checksum (default 16)")
    solve.add_argument("--crc-limit", type=int, default=1 << 21, metavar="N",
                       help="most combinations to test (default 2097152)")
    solve.add_argument("--self-train", action="store_true",
                       help="also train on the stream's own recovered bytes, "
                            "which is often the best available sample of the "
                            "part that is missing")

    gap = p.add_argument_group("two fragments with something missing between them")
    gap.add_argument("--gap", metavar="FILE",
                     help="a later fragment of the same stream. The first file "
                          "supplies the Huffman tables, and the output produced "
                          "by the missing bytes becomes a run of unknowns")
    gap.add_argument("--gap-size", type=int, metavar="N",
                     help="the gap itself: how many output bytes are missing")
    gap.add_argument("--total-size", type=int, metavar="N",
                     help="the whole stream's decompressed size; with it the "
                          "gap is arithmetic. Read from the container "
                          "automatically when it records one")
    gap.add_argument("--gap-max", type=int, default=8192, metavar="N",
                     help="largest output gap to search for (default 8192). "
                          "Searching only works while something still reaches "
                          "across the gap: reliable to about 4K, useless past "
                          "24K, because the window is 32K")
    gap.add_argument("--gap-workers", type=int, metavar="N",
                     help="processes that score gaps (default: one per "
                          "physical core). Each adds 0.22 GB to the "
                          "model's memory; 1 scores in this process")

    resync = p.add_argument_group("resynchronising a fragment with no table")
    resync.add_argument("--scan", action="store_true",
                        help="find candidate block boundaries and exit.  Use this "
                             "when a fragment starts inside a dynamic block: its "
                             "Huffman table is gone, but the next block carries a "
                             "fresh one")
    resync.add_argument("--resync", action="store_true",
                        help="decode from the best block boundary found by "
                             "scanning")
    resync.add_argument("--scan-kinds", default="sync,stored,dynamic", metavar="LIST",
                        help="signatures to look for: sync, stored, dynamic, fixed "
                             "(default sync,stored,dynamic; fixed is three bits and "
                             "fires constantly)")
    resync.add_argument("--scan-limit", type=int, metavar="N",
                        help="only scan the first N bytes")

    codeset = p.add_argument_group("a recovered code set, without its meanings")
    codeset.add_argument("--codeset", metavar="FILE",
                         help="JSON code set {a,e,L,D} to decode a fragment "
                              "with.  Recovering a table's shape gives the "
                              "parse but never says which byte a literal "
                              "stands for; this decodes with what is known and "
                              "reads the meanings back off whatever you can "
                              "pin down.  Use --bit for the body offset and "
                              "--guess POS=VAL to identify occurrences")

    rel = p.add_argument_group("repairs, and evidence from outside the capture")
    rel.add_argument("--related", metavar="FILE",
                     help="another capture of the same or similar content.  "
                          "Its token overlap with what has already been "
                          "recovered here decides the odds, and those odds are "
                          "printed before anything is tried")
    rel.add_argument("--related-kind", default="html",
                     choices=("html", "css", "js", "other"),
                     help="what the content is, because the odds are "
                          "calibrated per content type (default html)")
    rel.add_argument("--plaintext", metavar="FILE",
                     help="bytes you already know appear in the output, used "
                          "to judge whatever table is in play.  Where a copy's "
                          "own position and its source are both covered, its "
                          "claim can be tested without knowing what any "
                          "literal means")
    rel.add_argument("--chunk-repair", action="store_true",
                     help="find one or two flipped bits in a PNG chunk that "
                          "fails its CRC, and apply the repair when the "
                          "stream's Adler-32 agrees")
    rel.add_argument("--write-repaired", metavar="FILE",
                     help="after --chunk-repair, write the input file with "
                          "the repaired bits flipped back: a PNG whose chunk "
                          "CRCs and Adler-32 all pass")
    rel.add_argument("--dict", metavar="FILE",
                     help="a file that holds the zlib preset dictionary, whole "
                          "or anywhere inside it (the program that wrote the "
                          "stream, say): found by the header's DICTID, proven "
                          "by the stream's Adler-32, and filled in")
    rel.add_argument("--fill-from", metavar="FILE",
                     help="a copy you suspect holds this content: aligned to "
                          "what decoded, checked against it, used to fill "
                          "every open class, and ruled on by the container's "
                          "checksum. Combine with --verify or --dump")
    rel.add_argument("--bitrot", nargs="?", const=1, type=int, metavar="K",
                     help="when a block header's bits are wrong, try every "
                          "K-bit flip over it (default 1) and list the ones "
                          "that parse, each checked against the recorded size "
                          "and checksum. Applied only when exactly one "
                          "matches. 2 tries every pair: about 165,000 over a "
                          "575-bit header, minutes")
    rel.add_argument("--resume", action="store_true",
                     help="when the decode stops on damage part way through "
                          "a block, read on past it and place the rest by the "
                          "recorded size. What it reads is unproven until "
                          "--fill-from gives a copy that matches the checksum")
    rel.add_argument("--plaintext-at", type=int, default=0, metavar="POS",
                     help="output position the plaintext starts at (default 0)")
    rel.add_argument("--why", type=int, metavar="POS",
                     help="explain one output position: the chain of copies "
                          "behind it and what it finally rests on")

    out = p.add_argument_group("output")
    out.add_argument("--routes", action="store_true",
                     help="print every route to a lost table, which of them "
                          "apply to this capture and WHY the others do not, "
                          "with what each costs, then exit.  Nothing runs "
                          "unless you ask for it")
    out.add_argument("--ledger", action="store_true",
                     help="print what has been run against this capture and "
                          "what came back, refusals included: a check that "
                          "declined to rule is a different answer from one "
                          "that found nothing")
    out.add_argument("--list", action="store_true",
                     help="print ranked starting offsets with previews, then exit")
    out.add_argument("--dump", action="store_true",
                     help="print the symbol stream, infgen style, then exit")
    out.add_argument("--tables", action="store_true",
                     help="print the Huffman tables, then exit")
    out.add_argument("--assignment", action="store_true",
                     help="with --codeset, print what each literal code has "
                          "turned out to mean, then exit")
    out.add_argument("--export", metavar="FILE",
                     help="write recovered bytes (window + output) and exit")
    out.add_argument("--fill", default="?", metavar="CH",
                     help="byte to stand in for unknowns on export (default '?')")
    out.add_argument("--verify", action="store_true",
                     help="check the decode against the container's recorded size "
                          "and CRC, then exit")
    out.add_argument("-v", "--verbose", action="count", default=0,
                     help="with --dump, annotate each symbol with its bits")

    p.add_argument("--guess", nargs="+", metavar="POS=VAL", default=[],
                   help="fill holes non-interactively, e.g. --guess -3=f -2=o")
    p.add_argument("--session", metavar="FILE",
                   help="load a saved session: its steps replay onto this "
                        "input and its bytes go back. Give the same input "
                        "and entry it was saved from")
    p.add_argument("--save-session", metavar="FILE",
                   help="save the session after everything else on this "
                        "command line has run")
    return p


def _iter_corpus(paths):
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    yield child
        elif p.is_file():
            yield p


def corpus_catalogue() -> str:
    """Every corpus type, what its stream really holds, and where to get it."""
    from .predict import corpora
    out = ["Corpus types. Pick the one matching the DECOMPRESSED content,",
           "not the file the stream came out of.", ""]
    for cat in corpora.CATEGORIES:
        types = [t for t in corpora.TYPES if t.category == cat]
        if not types:
            continue
        out.append(f"  [{cat}]")
        for t in types:
            here = corpora.suggested_roots(t)
            out.append(f"    {t.key:<9} {t.name}")
            out.append(f"              the stream holds: {t.contains}")
            out.append(f"              on this machine: "
                       + (", ".join(str(p) for p in here) if here
                          else "no suggested root exists here"))
            for s in t.sources:
                out.append(f"              external: {s.name} <{s.url}> "
                           f"({s.licence}, {s.approx})")
            if t.note:
                out.append(f"              note: {t.note}")
    out += ["", "A wrong-type corpus is worse than none: an English model "
            "scores -17.2 bits",
            "per byte on ELF against -8.0 for no model at all, and the solver "
            "reads",
            "confidence as a threshold. Whatever you pick is measured against "
            "the",
            "real stream afterwards."]
    return "\n".join(out)


def _corpus_size(files) -> int:
    """How much there is to train on, which is what the order scales with."""
    total = 0
    for f in files:
        try:
            total += Path(f).stat().st_size
        except OSError:
            pass
    return total


def _announce_corpus(args, files, corpora) -> int:
    """Say what training is about to cost, then do it anyway.

    Nothing is capped and nothing is refused: the operator named these files, so
    the decision has been made. What was missing is the sentence before it -- the
    whole corpus is read now, and an order-6 model over 30 MB of text wants
    several gigabytes, which is worth knowing before it is allocated rather than
    after the machine starts swapping.
    """
    size = _corpus_size(files)
    order, note = corpora.order_note(size, corpora.sample_of(files), args.order)
    print(f"corpus: {len(files)} file(s), {size:,} bytes; {note}",
          file=sys.stderr)
    if "over the" in note:
        print("corpus: training anyway, because you asked for these files. "
              "--order lowers it, TOEKNIFE_MEMORY_BUDGET moves the line.",
              file=sys.stderr)
    return order


def _session_order(session, args) -> int:
    """The order for a model built from the stream's own recovered bytes.

    Its corpus is those bytes, so it is usually tiny and wants a low order --
    the same rule, applied to a much smaller corpus.
    """
    from .predict import corpora
    known = session.tape.stats().get("known", 0)
    return corpora.resolve_order(args.order, known)


def model_from_type(args):
    """Train on a KIND of data found locally, rather than a named path."""
    from .predict import corpora
    from .predict.context import Bidirectional, ContextModel
    if args.corpus_type not in corpora.BY_KEY:
        print(f"toeknife: unknown corpus type {args.corpus_type!r}; "
              f"--corpus-types lists them", file=sys.stderr)
        raise SystemExit(1)
    plan = corpora.plan(args.corpus_type, args.corpus_root,
                        args.corpus_scan_all)
    print(plan.render(), file=sys.stderr)
    if not plan.files:
        raise SystemExit(1)
    blob = b"".join(corpora.training_bytes(plan.type, p) for p in plan.files)
    if not blob:
        print("toeknife: those files yielded no training bytes",
              file=sys.stderr)
        raise SystemExit(1)
    prov = {"type": plan.type.key, "files": len(plan.files),
            "bytes": len(blob),
            "roots": "whole filesystem" if plan.whole_system
            else ",".join(str(r) for r in plan.roots)}
    order = corpora.resolve_order(args.order, len(blob),
                                  corpora.sample_of(blob))
    if args.forward_only:
        model = ContextModel(order=order).train(blob)
    else:
        model = Bidirectional.train_on(blob, order=order)
    model.provenance = prov
    print(f"trained order-{order} on {len(blob):,} bytes from "
          f"{len(plan.files)} file(s)", file=sys.stderr)
    return model


def load_model(args):
    from .predict.context import Bidirectional, ContextModel

    if args.predictor.startswith("hf:"):
        from .predict.hf import HFPredictor, MissingBackend, available
        if not available():
            print("toeknife: --predictor hf: needs torch and transformers: "
                  "pip install 'toeknife[ml]'", file=sys.stderr)
            return None
        name = args.predictor[3:] or "distilgpt2"
        print(f"loading {name} (text only; the byte model is better on binary)",
              file=sys.stderr)
        try:
            return HFPredictor(name=name).load()
        except Exception as e:
            first = str(e).strip().splitlines()[0] if str(e).strip() else ""
            print(f"toeknife: cannot load {name}: {type(e).__name__}: {first}",
                  file=sys.stderr)
            raise SystemExit(1)

    if args.model:
        try:
            model = ContextModel.load(args.model)
        except Exception as e:
            print(f"toeknife: cannot load model: {args.model} is not a saved "
                  f"ToeKnife model ({type(e).__name__}: {e})", file=sys.stderr)
            raise SystemExit(1)
        print(f"loaded order-{model.order} model, {model.trained_bytes} bytes "
              f"of training data", file=sys.stderr)
        prov = getattr(model, "provenance", {}) or {}
        # A saved model travels between machines. Where it came from is part of
        # what it is, and "unknown" is worth saying rather than leaving blank.
        print("  trained on: " + (", ".join(f"{k}={v}" for k, v in
                                            sorted(prov.items()))
                                  if prov else
                                  "unknown -- saved before provenance was "
                                  "recorded"), file=sys.stderr)
        return model

    if args.corpus_type:
        return model_from_type(args)

    if not args.corpus:
        return None
    files = list(_iter_corpus(args.corpus))
    if not files:
        print(f"toeknife: no readable files in {args.corpus}", file=sys.stderr)
        raise SystemExit(1)
    from .predict import calibration, corpora
    order = _announce_corpus(args, files, corpora)
    model, cal, train = calibration.train_files(
        files, order, forward_only=args.forward_only, progress=_cli_progress)
    print(f"trained order-{order} "
          f"{'forward-only' if args.forward_only else 'forward+reverse'} model on "
          f"{model.trained_bytes} bytes from {len(train)} file(s)",
          file=sys.stderr)
    _report_calibration(cal, len(files) - len(train))
    return model


def cmd_save_model(args) -> int:
    from .predict.context import Bidirectional, ContextModel
    if not args.corpus:
        print("toeknife: --save-model needs --corpus to train from",
              file=sys.stderr)
        return 1
    files = list(_iter_corpus(args.corpus))
    if not files:
        print(f"toeknife: no readable files in {args.corpus}", file=sys.stderr)
        return 1
    from .predict import calibration, corpora
    order = _announce_corpus(args, files, corpora)
    model, cal, train = calibration.train_files(
        files, order, forward_only=args.forward_only,
        progress=_cli_progress)
    _report_calibration(cal, len(files) - len(train))
    model.save(args.save_model, {"corpus": " ".join(str(f) for f in args.corpus),
                                 "files": len(train), "order": order,
                                 "bytes": model.trained_bytes,
                                 # What calibration was measured on: whole
                                 # files, or with under 4 the corpus's tail
                                 # in pieces, when no file was left out.
                                 "held_out": (len(cal.held_out) if cal
                                              else len(files) - len(train))})
    size = Path(args.save_model).stat().st_size
    print(f"trained order-{order} on {model.trained_bytes} bytes from "
          f"{len(train)} file(s); wrote {size} bytes to {args.save_model}")
    return 0


def cmd_calibrate_slm(args) -> int:
    """Measure a byte model on held-out text and write a calibrated copy."""
    import time
    from .predict import calibration

    src = Path(args.calibrate_slm).expanduser()
    if calibration.calibrated_path(src).exists():
        print(f"toeknife: {calibration.calibrated_path(src)} already exists; "
              "move it aside first", file=sys.stderr)
        return 1
    specs = list(args.held_out or [])
    if not specs:
        try:
            names = calibration.recorded_held_out(src)
        except Exception as e:
            print(f"toeknife: cannot load byte model: {e}", file=sys.stderr)
            return 1
        if not names:
            print("toeknife: this model does not record what it held out; "
                  "give --held-out with text it never trained on",
                  file=sys.stderr)
            return 1
        specs, missing = calibration.find_held_out(names, args.corpus or ())
        if missing:
            print(f"toeknife: {len(missing)} held-out file(s) not found"
                  + ("" if args.corpus else "; give --corpus where they live")
                  + f": {', '.join(Path(m).name for m in missing[:4])}",
                  file=sys.stderr)
            return 1
    ngram = None
    if args.model:
        from .predict.context import ContextModel
        try:
            ngram = ContextModel.load(args.model)
        except Exception as e:
            print(f"toeknife: cannot load n-gram: {e}", file=sys.stderr)
            return 1
    t0 = time.monotonic()

    def progress(done, total, _note=""):
        if done:
            left = (time.monotonic() - t0) / done * (total - done)
            print(f"toeknife: calibrating: {done} of {total} documents, "
                  f"~{left / 60:.0f} min left", file=sys.stderr)
        else:
            print(f"toeknife: calibrating {src.name} on {total} held-out "
                  f"document(s)", file=sys.stderr)

    try:
        dst, cal = calibration.calibrate_checkpoint(src, specs, ngram,
                                                    progress)
    except (OSError, ValueError) as e:
        print(f"toeknife: {e}", file=sys.stderr)
        return 1
    if cal is None:
        print("toeknife: not calibrated -- too few classes in the held-out "
              "text to fit a curve", file=sys.stderr)
        return 1
    print(f"toeknife: {cal.summary()}", file=sys.stderr)
    print(f"toeknife: {_near_evidence(cal)}", file=sys.stderr)
    print(f"wrote {dst} in {(time.monotonic() - t0) / 60:.0f} min")
    return 0


def _cli_progress(done, total, note=""):
    """Calibration and training are minutes; say which stage is running."""
    if note.startswith("calibrating") and done == 0:
        print(f"toeknife: {note.split(':')[0]} on held-out documents...",
              file=sys.stderr)


def _near_evidence(cal) -> str:
    """What the near-gap floors rest on, which at 80% and up is little."""
    if not cal.near:
        return "near a gap: no curve; fills there read the long-run figure"
    from .predict.solver import pct
    cells = [f"{pct(f)} on {cal.resting(f, near=True):,}"
             for f in (0.7, 0.8, 0.9)]
    return ("near a gap, held-out classes under each floor: "
            + ", ".join(cells))


def _report_calibration(cal, held: int) -> None:
    if cal is not None:
        print(f"toeknife: {cal.summary()}", file=sys.stderr)
        print(f"toeknife: {_near_evidence(cal)}", file=sys.stderr)
        return
    print("toeknife: not calibrated -- too little held-out text to measure "
          "the model on, so its confidence floor stays raw confidence",
          file=sys.stderr)


def parse_guess(text: str) -> tuple[int, int]:
    if "=" not in text:
        raise ValueError(f"expected POS=VAL, got {text!r}")
    pos_s, val_s = text.rsplit("=", 1)
    pos = int(pos_s, 0)
    if len(val_s) == 1 and not val_s.isdigit():
        return pos, ord(val_s)
    try:
        return pos, int(val_s, 0) & 0xFF
    except ValueError:
        if len(val_s) != 1:
            raise ValueError(f"{val_s!r} is not a single byte") from None
        return pos, ord(val_s)


# ---- headless renderers ----------------------------------------------------

def cmd_list(session) -> int:
    cands = session.candidates()
    if not cands:
        print("stream mode: no alignment search needed")
        return 0
    width = 56
    print(f"{'bit':>4} {'score':>7} {'lit':>4} {'mch':>4} {'out':>6} "
          f"{'holes':>6}  {'status':<12} recovered")
    for c in cands[:20]:
        t = c.trace
        st = t.tape.stats()
        from .tui.render import glyph
        prev = "".join(glyph(t.tape.value_at(p))
                       for p in range(min(t.tape.length, width)))
        status = t.status + (f" ⊂{c.subset_of}" if c.subset_of is not None else "")
        print(f"{c.bit_offset:>4} {c.score:>7.2f} {t.n_literals:>4} {t.n_matches:>4} "
              f"{st['out_len']:>6} {st['holes']:>6}  {status:<12} {prev}")
    print(f"\n{len(cands)} offsets ranked. Ranking is advisory: several offsets can "
          f"be individually valid.\nRead the previews.  ⊂N means this decode is a "
          f"tail of offset N and recovers less.")
    return 0


WRAP = 68


def _emit_run(run: list[int | None], keyword: str) -> None:
    """Print a literal run the way infgen does: quoted text, decimal otherwise.

    A literal decoded with a recovered code set has no value yet, and printing
    one as a byte would invent a meaning the bitstream never gave.  Those are
    emitted as `?` runs instead, which is what the rest of the tool renders an
    unknown byte as.
    """
    i = 0
    while i < len(run):
        if run[i] is None:
            j = i
            while j < len(run) and run[j] is None:
                j += 1
            for k in range(i, j, WRAP):
                print(f"{keyword} ?x{min(WRAP, j - k)}")
            i = j
            continue
        printable = 0x20 <= run[i] < 0x7F
        j = i
        while j < len(run) and run[j] is not None \
                and (0x20 <= run[j] < 0x7F) == printable:
            j += 1
        chunk = run[i:j]
        if printable:
            text = "".join(chr(b) for b in chunk)
            for k in range(0, len(text), WRAP):
                print(f"{keyword} '{text[k:k + WRAP]}")
        else:
            parts = [str(b) for b in chunk]
            line = keyword
            for part in parts:
                if len(line) + len(part) + 1 > WRAP:
                    print(line)
                    line = keyword
                line += " " + part
            print(line)
        i = j
    run.clear()


def cmd_dump(session, verbose: int) -> int:
    tape = session.tape
    trace = session.trace
    print("! toeknife")
    print(f"! {len(session.data)} bytes, {session.mode}, start bit {session.bit_offset}")
    print("!")

    by_block: dict[int, list] = {b.index: [] for b in trace.blocks}
    for sym in trace.symbols:
        by_block.setdefault(sym.block, []).append(sym)

    for block in trace.blocks:
        if not trace.headerless:
            if block.bfinal:
                print("last")
            print(block.btype_name)
            if block.btype != BTYPE_STORED and block.hlit:
                print(f"count {block.hlit} {block.hdist} {block.hclen}")
        keyword = "data" if block.btype == BTYPE_STORED else "literal"
        run: list[int] = []
        for sym in by_block.get(block.index, ()):
            if sym.kind == "literal":
                run.append(sym.literal)
                continue
            _emit_run(run, keyword)
            if sym.kind == "match":
                known = all(tape.value_at(p) is not None
                            for p in range(sym.out_pos, sym.out_pos + sym.length))
                line = f"match {sym.length} {sym.distance}"
                if not known:
                    line += "   ! unresolved"
                if verbose:
                    line += f"   ! bit {sym.bit_start} " + " ".join(
                        bit_string(session.data, f.start, f.end) for f in sym.fields)
                print(line)
            elif trace.headerless:
                print("end")          # a fragment's only block boundary marker
        _emit_run(run, keyword)
        if not trace.headerless:
            # A stored block has no end-of-block symbol, but it still ends.
            print("end")

    st = tape.stats()
    if st["holes"]:
        print(f"! {st['holes']} unknown byte classes covering "
              f"{st['hole_positions']} output positions")
        for h in tape.holes():
            print(f"!   hole at {h.anchor}: also at "
                  f"{', '.join(str(p) for p in h.positions[1:]) or '(nowhere else)'}")
    return 0


def run_scan(data: bytes, args):
    from .core.scan import scan
    kinds = tuple(k.strip() for k in args.scan_kinds.split(",") if k.strip())
    kinds = tuple({"sync": "sync-flush"}.get(k, k) for k in kinds)
    known = ("sync-flush", "stored", "dynamic", "fixed")
    bad = [k for k in kinds if k not in known]
    if bad or not kinds:
        print(f"toeknife: --scan-kinds: unknown kind {', '.join(bad) or '(none)'}"
              f"; the kinds are sync, stored, dynamic and fixed",
              file=sys.stderr)
        raise SystemExit(1)
    max_bit = args.scan_limit * 8 if args.scan_limit else None
    return scan(data, kinds=kinds, max_bit=max_bit)


def cmd_scan(data: bytes, args) -> int:
    points = run_scan(data, args)
    if not points:
        print("no block boundaries found.")
        print("A fragment with no recoverable boundary cannot be decoded: the "
              "Huffman table that\ndefines its symbols is behind the start of "
              "what you have, and reconstructing it from\nthe body alone is not "
              "tractable.  Supply the table with --header if you have it.")
        return 1
    print(f"{'position':<28} {'kind':<11} {'output':>8}       {'blocks':>3}  "
          f"{'holes':>5}  {'status':<10} note")
    for p in points[:25]:
        print(p.summary())
    print(f"\n{len(points)} candidate boundaries. Decode from one with "
          f"--stream --bit N, or take the best with --resync.")
    if points[0].kind == "sync-flush":
        print("A sync-flush marker is a compressor flush point: near certain, and "
              "byte-aligned.")
    return 0


def _interactive(args) -> bool:
    """Whether this run ends in the interactive view: no flag asks for output
    instead, and there is a terminal to draw on."""
    output = ("solve_propose", "routes", "ledger", "verify", "list", "tables",
              "assignment", "dump", "export", "related", "plaintext", "scan")
    if any(getattr(args, a, None) for a in output):
        return False
    if getattr(args, "why", None) is not None:
        return False
    return sys.stdout.isatty() and sys.stdin.isatty()


def cmd_entries(report, path: str) -> int:
    print(f"{path}: {report.kind} container, {len(report.streams)} entr"
          f"{'y' if len(report.streams) == 1 else 'ies'}")
    for line in report.structure:
        print(f"  {line}")
    for w in report.warnings:
        print(f"  ! {w}")
    print()
    print("  #  " + report.header_line())
    for i, s in enumerate(report.streams):
        print(f"  {i:<2} {s.summary()}")
        for note in s.notes:
            print(f"       ! {note}")
    n = len(report.decodable)
    print(f"\n{n} of {len(report.streams)} entries hold DEFLATE. "
          f"Open one with --entry N (or --entry NAME).")
    print("Flags: E encrypted, D decodable.")
    return 0


def cmd_all(data: bytes, report, args) -> int:
    """--entry all: every member of a multi-member gzip, written as one file."""
    from .core import members
    if not members.multi(report):
        print("toeknife: --entry all joins the members of a multi-member "
              "gzip; this input has none to join", file=sys.stderr)
        return 1
    sessions = members.open_all(data, report, model=load_model(args))
    copy = Path(args.fill_from).read_bytes() if args.fill_from else None
    for stream, s in zip(report.streams, sessions):
        for note in stream.notes:
            print(f"toeknife: {stream.name}: {note}", file=sys.stderr)
        if s is None:
            print(f"toeknife: {stream.name}: {stream.why_not}", file=sys.stderr)
            continue
        v = s.verify()
        if copy is not None and not (v is not None and v.ok):
            from .core import copyfill
            got = copyfill.fill(s, copy, Path(args.fill_from).name)
            print(f"copy: {stream.name}: {got.summary()}",
                  file=sys.stderr)
    fill = ord(args.fill[0]) if args.fill else 0x3F
    buf, mask, parts = members.join(report, sessions, fill)
    proven = 0
    for p in parts:
        v = p.verdict
        mark = ("no record" if v is None else "ok" if v.ok else
                "incomplete" if p.known < p.length else
                "FAILED" if v.ok is False else "inconclusive")
        proven += v is not None and v.ok is True
        print(f"{p.name}: output {p.start:,}-{p.start + p.length - 1:,}, "
              f"{p.known:,} of {p.length:,} known, checksum {mark}")
    print(f"all: {len(buf):,} bytes, {sum(mask):,} known; {proven} of "
          f"{len(parts)} members match their trailers")
    if args.export:
        Path(args.export).write_bytes(buf)
        print(f"wrote {len(buf):,} bytes to {args.export} "
              f"({sum(mask):,} known, {len(buf) - sum(mask):,} filled with "
              f"{chr(fill)!r})", file=sys.stderr)
    if args.verify:
        return 0 if proven == len(parts) else 1 if any(
            p.verdict is not None and p.verdict.ok is False
            for p in parts) else 2
    return 0


def _has_checksum(session) -> bool:
    st = getattr(session, "stream", None)
    return st is not None and (st.crc32 is not None
                               or getattr(st, "adler32", None) is not None)


def _duration(seconds: float) -> str:
    from .predict.corpora import _duration as d
    return d(seconds)


def pick_entry(report, spec: str):
    """Select a container entry by index or by name."""
    if spec.isdigit() or (spec.startswith("-") and spec[1:].isdigit()):
        i = int(spec)
        if not -len(report.streams) <= i < len(report.streams):
            raise ValueError(f"no entry {i}; the container has "
                             f"{len(report.streams)}")
        return report.streams[i]
    exact = [s for s in report.streams if s.name == spec]
    if exact:
        return exact[0]
    partial = [s for s in report.streams if spec.lower() in s.name.lower()]
    if len(partial) == 1:
        return partial[0]
    if not partial:
        raise ValueError(f"no entry matching {spec!r}")
    names = ", ".join(s.name for s in partial[:6])
    raise ValueError(f"{spec!r} matches {len(partial)} entries: {names}")


def run_solver(session, args):
    from .predict.profiles import build, detect, sanity_check
    from .predict.solver import HoleSolver

    corpus_model = session.model            # from --corpus / --model, or None
    profile = detect(session.tape, session.trace)
    print(f"data looks like: {profile.name} ({profile.why})", file=sys.stderr)

    # Only the corpus model can be the wrong KIND of data, and it is the only
    # part this score is honest about. A self-trained model is measured on the
    # very bytes it was trained on, so a good number there means it memorised
    # them, not that it can predict anything.
    if corpus_model is not None and hasattr(corpus_model, "score"):
        score, warning = sanity_check(corpus_model, session.tape)
        if warning:
            print(f"WARNING: {warning}", file=sys.stderr)
        else:
            print(f"corpus model scores {score:.1f} bits per byte on what is "
                  f"already recovered (-8.0 would be no better than guessing)",
                  file=sys.stderr)

    # --self-train ADDS the stream's own bytes to whatever corpus model exists
    # rather than replacing it, so a bidirectional corpus model stays
    # bidirectional. With no corpus at all it is the only source there is.
    model, profile = build(session, corpus_model=corpus_model,
                           self_train=args.self_train or corpus_model is None,
                           profile=profile, order=_session_order(session, args))
    print(f"predictor: {model.describe()}", file=sys.stderr)
    if not any(p.weight and p.label != "empty" for p in model.parts):
        hint = profile.corpus_hint or "data of the same kind"
        print(f"note: nothing to predict from: too few recovered bytes to "
              f"self-train and no usable table prior. Pass --corpus {hint}",
              file=sys.stderr)
    cal = getattr(corpus_model, "calibration", None)
    filler, mode = _byte_model(session, args)
    from .predict.solver import floor_calibration
    floor = floor_calibration(mode, cal, filler)
    whose = ("the byte model's" if floor is not None and floor is not cal
             else "the n-gram's")
    print("confidence floor: " + (f"measured precision, {whose} curve "
                                  f"({floor.summary()})"
                                  if floor is not None else
                                  "raw confidence; this model was never "
                                  "calibrated, so the floor ranks guesses "
                                  "without promising a precision"),
          file=sys.stderr)
    return HoleSolver(session, model, min_confidence=args.solve_confidence,
                      batch=getattr(args, "solve_batch", 32), calibration=cal,
                      filler=filler, mode=mode)


def _byte_model(session, args):
    """--byte-model and --fill-mode -> (filler or None, mode). Exits on error.

    Loaded once per run and kept on args, since --solve-loop and --solve can
    both build a solver. The fill mode defaults as in the TUI: both with a
    byte model, the n-gram without one.
    """
    path = getattr(args, "byte_model", None)
    want = getattr(args, "fill_mode", None)
    if not path:
        if want in ("both", "slm"):
            print(f"toeknife: --fill-mode {want} needs --byte-model",
                  file=sys.stderr)
            raise SystemExit(1)
        return None, "ngram"
    filler = getattr(args, "_filler", None)
    if filler is None:
        try:
            from .predict.bytemodel import ClassFiller, sanity_check
            filler = ClassFiller.load(path)
        except Exception as e:
            print(f"toeknife: cannot load byte model: {path} is not a ToeKnife "
                  f"byte model ({type(e).__name__}: {e})",
                  file=sys.stderr)
            raise SystemExit(1)
        bits, top1, warning = sanity_check(filler, session.tape)
        print(f"{filler.describe()}: {bits:.1f} bits/byte, "
              f"{top1:.0%} of known bytes right"
              + (f". WARNING: {warning}" if warning else ""), file=sys.stderr)
        args._filler = filler
    return filler, want or "both"


def cmd_solve_propose(session, args) -> int:
    solver = run_solver(session, args)
    proposals = solver.propose()
    if not proposals:
        print("no holes to fill.")
        return 0
    print(f"{'anchor':>8} {'seen':>4} {'proposal':<16} {'conf':>7}  alternatives")
    for p in proposals[:40]:
        print(p.summary())
    above = sum(1 for p in proposals if p.gate >= args.solve_confidence)
    print(f"\n{len(proposals)} classes, {above} at or above "
          f"{args.solve_confidence:.1%} confidence.")
    print("Confidence ranks proposals; a calibrated model's floor is a "
          "measured precision. See --help.")
    return 0


def cmd_related(session, args) -> int:
    """Odds first, then the attempt, then what it actually bought.

    The order matters. The measured overlap curve is a usable prediction
    before any work is done, so an operator can decline a route that is not
    worth trying rather than reading a failure afterwards.
    """
    blob = Path(args.related).read_bytes()
    # Only the bytes actually determined here. Including unknown positions would
    # feed placeholder values into the overlap and quietly corrupt the one
    # number this route asks the operator to decide on.
    tape = session.tape
    have = bytes(tape.out[i] for i in range(tape.length)
                 if tape.prov[i] != UNKNOWN)
    rep = related.report(have, blob, args.related_kind)
    print(f"related capture: {Path(args.related).name}, "
          f"{rep['related_bytes']:,} bytes of content")
    if rep["overlap"] is None:
        print("token overlap: not measurable")
    else:
        print(f"token overlap with what is recovered here: "
              f"{rep['overlap']:.3f}")
    print(f"  {rep['expectation']}")
    from .core import borrow
    result = borrow.borrow(session, blob, model=session.model)
    print(f"\n{result.summary()}")
    for n, c in enumerate(result.landings, 1):
        print(f"  {n}. {c.summary(result.baseline)}")
    if not result.landings:
        session.ledger.add("related", ledger.NOTHING, result.summary(),
                           params={"overlap": rep["overlap"]})
        return 1
    print("\nThe TUI adopts one (f7, related); its bytes stay unproven until "
          "a copy matches them.")
    session.ledger.add("related", ledger.FOUND, result.summary(),
                       counts={"landed": len(result.landings)},
                       params={"overlap": rep["overlap"]})
    return 0



def capture_session(data: bytes, stream, args):
    """A session assembled from a capture with packets missing.

    The container has already done the hard part: TCP numbered every byte, so
    the hole's position and width are read rather than searched for. What is left
    is the output gap -- how much the missing compressed bytes would have
    produced -- and that comes from a recorded decompressed size where one
    exists, which for a gzip body is its ISIZE.
    """
    import struct

    from . import containers
    from .core import gap as gapmod
    from .core.session import Session

    joined = stream.slice(data, allow_gaps=True)
    head_skip = trail_skip = 0
    total = stream.uncompressed_size
    if stream.encoding == containers.GZIP and joined[:2] == b"\x1f\x8b":
        # Fixed-header gzip only; anything with FEXTRA or a filename needs the
        # header walked, and saying so beats guessing an offset.
        if joined[3] & 0x1E:
            print("toeknife: gzip header carries optional fields; not handled "
                  "here", file=sys.stderr)
            return None
        head_skip, trail_skip = 10, 8
        if len(joined) > 8:
            _crc, isize = struct.unpack("<II", joined[-8:])
            total = isize
            print(f"toeknife: the gzip trailer records {isize:,} bytes of "
                  f"original, which is what makes the output gap arithmetic",
                  file=sys.stderr)
    lost = sum(n for _p, n in stream.missing)
    print(f"toeknife: {lost:,} compressed byte(s) missing in "
          f"{len(stream.missing)} hole(s)", file=sys.stderr)
    asm = gapmod.from_holes(joined, stream.missing, head_skip, trail_skip,
                            total)
    for note in asm.notes:
        print(f"capture: {note}", file=sys.stderr)
    return Session.from_trace(asm.trace, stream=stream, model=load_model(args))


def cmd_verify(session) -> int:
    v = session.verify()
    if v is None:
        print("nothing to verify against: no container metadata for these bytes.",
              file=sys.stderr)
        return 2
    name = session.stream.name if session.stream else "stream"
    print(f"{name}: {v.summary()}")
    if v.ok is True:
        print("verified: the recovered bytes are the original bytes.")
        return 0
    if v.ok is False:
        print("FAILED: the decode does not match what the container recorded.")
        st = session.tape.stats()
        if not st["holes"] and session.trace.status not in ("end", "eob"):
            print(f"The decode stopped at output {st['out_len']:,} "
                  f"({session.trace.status}): everything after it is missing.")
        return 1
    st = session.tape.stats()
    if session.resumed_from is not None:
        print("inconclusive: the bytes decoded near the damage are unproven and "
              "some hold its garbage, so the checksum cannot pass on this "
              "decode. --fill-from checks a suspected copy against it.")
        return 2
    if not st["holes"] and session.trace.status not in ("end", "eob"):
        print(f"inconclusive: the decode stops at {st['out_len']:,} bytes "
              f"({session.trace.status}), and nothing recorded a size or "
              f"checksum to check it against.")
        return 2
    if not st["holes"]:
        # Complete output and nothing recorded to check it against is a
        # different situation from a partial recovery, and telling the operator
        # to "fill them and check again" when there is nothing to fill is the
        # tool talking nonsense at them.
        print("inconclusive: the decode is complete, but this container "
              "recorded nothing to check it against.")
        return 2
    print(f"inconclusive: {st['holes']} unknown byte classes remain "
          f"({st['hole_positions']} positions). Fill them and check again.")
    return 2


def codeset_session(data: bytes, args, stream=None, model=None):
    """A session decoded with a recovered code set, meanings still unknown.

    Built by carving rather than by the ordinary alignment path: a session
    normally ranks bit offsets and assumes a table it can name symbols with, and
    here the meanings are exactly what is missing.  Wrapping the carve in a
    Session is what lets the TUI, the guess machinery and the undo stack work on
    it unchanged -- an unassigned literal is just a hole whose class happens to
    be a Huffman code.
    """
    import json

    from .core.huffman import structural_tables
    from .core.inflate import carve
    from .core.session import Session

    try:
        spec = json.loads(Path(args.codeset).read_text())
        litlen, dist, ident = structural_tables(
            list(spec["a"]), int(spec["e"]), list(spec["L"]), list(spec["D"]))
    except Exception as e:
        print(f"toeknife: --codeset: {args.codeset} is not a code set: it "
              f"needs \"a\" (16 literal counts), \"e\", \"L\" (29 lengths) "
              f"and \"D\" (30 lengths) ({type(e).__name__}: {e})",
              file=sys.stderr)
        return None

    # Whoever recovered the code set knew where the block body started, and a
    # file that carries the table but makes the operator re-derive the offset is
    # only most of a handover.  An explicit --bit still wins.
    bit = args.bit
    if bit is None and isinstance(spec.get("bit"), int):
        bit = spec["bit"]
        print(f"codeset: starting at bit {bit} (from the file)", file=sys.stderr)
    elif bit is None:
        # Not a harmless default. Decoding from the wrong offset with a valid
        # code set can still reach a clean end-of-block -- measured: bit 0 on a
        # test stream parses 39 symbols and stops, against the true 89 -- so the
        # result would look like a successful recovery and be nothing of the
        # kind. Say so rather than let a guess pass for a fact.
        print("codeset: no --bit and none in the file, starting at 0. A wrong "
              "offset can still parse cleanly, so treat the result as "
              "unanchored.", file=sys.stderr)
    trace = carve(data, bit or 0, litlen, dist, identity=ident)
    return Session.from_trace(trace, stream=stream, model=model)


def cmd_assignment(session) -> int:
    """Read off what the literal codes have turned out to mean."""
    from .core.tape import PROV_NAMES

    tape = session.tape
    trace = session.trace
    st = tape.stats()
    print(f"parse:  {trace.status}, {len(trace.symbols)} symbols, "
          f"{st['out_len']} bytes out")
    print(f"codes:  {st['codes_known']} of {st['codes']} literal codes have a "
          f"meaning; {st['known']} of {st['out_len']} bytes follow from that "
          f"({st['known'] / max(st['out_len'], 1) * 100:.1f}%)")

    clashes = tape.assignment_conflicts()
    if clashes:
        # Two literal codes are two byte values, so this cannot happen unless
        # the code set being decoded with is the wrong one.
        print(f"\nCONTRADICTION: {len(clashes)} pair(s) of distinct literal "
              f"codes are forced to the same byte. This code set does not fit "
              f"this data.")
        for x, y in clashes[:8]:
            print(f"  {x[0]}/{x[1]} and {y[0]}/{y[1]}")

    sizes = {h.code: h.size for h in tape.holes() if h.code is not None}
    print(f"\n  {'code':>7} {'fired':>7} {'class':>7} {'byte':>6}  where it came from")
    for code, info in sorted(tape.assignment().items(),
                             key=lambda kv: -kv[1]["occurrences"]):
        v = info["value"]
        shown = f"{v:#04x}" if v is not None else "--"
        ch = f" {chr(v)!r}" if v is not None and 0x20 <= v < 0x7F else ""
        # A settled code has no class left to report; printing 0 would read as
        # "accounts for nothing" when it is the opposite.
        cls = "--" if v is not None else str(sizes.get(code, 0))
        print(f"  {code[0]:>3}/{code[1]:<3} {info['occurrences']:>7} "
              f"{cls:>7} {shown:>6}{ch}  "
              f"{PROV_NAMES[info['provenance']]}")
    unknown = [c for c, i in tape.assignment().items() if i["value"] is None]
    if unknown:
        print(f"\n  {len(unknown)} code(s) still unassigned, covering "
              f"{sum(sizes.get(c, 0) for c in unknown)} byte positions. "
              f"Identify one occurrence of each and the rest follow.")
    return 0


def cmd_tables(session) -> int:
    for block in session.trace.blocks:
        print(f"block #{block.index}  {block.btype_name}"
              + ("  [final]" if block.bfinal else ""))
        for f in block.fields:
            print(f"  {f.label:<10} {f.detail}")
        for name, table in (("literal/length", block.litlen), ("distance", block.dist)):
            if table is None or not table.codes:
                continue
            used = table.symbols()
            print(f"  {name} code: {len(used)} of {len(table.lengths)} symbols used, "
                  f"max {table.max_bits} bits, "
                  f"{'complete' if table.complete else 'INCOMPLETE'}")
            for sym in used:
                if name == "distance":
                    meaning = (f"dist {DISTANCE_BASE[sym]}" if sym < len(DISTANCE_BASE)
                               else "INVALID")
                elif sym < 256:
                    meaning = repr(chr(sym)) if 0x20 <= sym < 0x7F else f"0x{sym:02x}"
                elif sym == 256:
                    meaning = "end-of-block"
                elif sym - 257 < len(LENGTH_BASE):
                    meaning = f"length {LENGTH_BASE[sym - 257]}"
                else:
                    # 286 and 287: the fixed table gives them codes and
                    # DEFLATE never uses them.
                    meaning = "reserved"
                print(f"    {sym:>4}  {meaning:<14} {table.code_str(sym)}")
        print()
    return 0


def _actions_asked(args) -> list[str]:
    """The options given that act on one stream, named as typed."""
    flags = ("solve", "solve_loop", "checksum_solve", "chunk_repair",
             "write_repaired", "dict", "fill_from", "bitrot", "resume",
             "export", "verify", "dump", "tables", "scan", "resync", "why",
             "guess", "save_session", "solve_propose", "routes")
    return ["--" + f.replace("_", "-") for f in flags
            if getattr(args, f, None) not in (None, False, [], 0)
            or (f == "bitrot" and getattr(args, f, None) == 0)]


def _recorded_checksum(session) -> bool:
    st = getattr(session, "stream", None)
    return bool(st and (getattr(st, "crc32", None) is not None
                        or getattr(st, "adler32", None) is not None))


def _probability(value: str) -> float:
    """--solve-confidence: a share between 0 and 1."""
    try:
        p = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not a number")
    if not 0.0 <= p <= 1.0:
        raise argparse.ArgumentTypeError(f"{value} is outside 0 to 1 "
                                         f"(0.9 means 90%)")
    return p


def _order_arg(value: str):
    """--order: auto, or a context length of 1 or more."""
    if value == "auto":
        return value
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not auto or a number")
    if n < 1:
        raise argparse.ArgumentTypeError("the order is 1 or more")
    return value


def _check_outputs(args) -> str | None:
    """A problem with a path this run will write to, before any work starts.
    A directory given to --export used to end the run in a traceback."""
    for flag in ("export", "save_session", "write_repaired", "save_model"):
        path = getattr(args, flag, None)
        if not path:
            continue
        p = Path(path).expanduser()
        name = "--" + flag.replace("_", "-")
        if p.is_dir():
            return f"{name} {path} is a directory; give a file name"
        if not p.parent.exists() and flag != "save_model":
            return f"{name} {path}: the folder {p.parent} does not exist"
    return None


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    problem = _check_outputs(args)
    if problem:
        print(f"toeknife: {problem}", file=sys.stderr)
        return 1
    # A catalogue of what could be trained on, which is a question about this
    # machine rather than about any capture.
    if args.corpus_types:
        print(corpus_catalogue())
        return 0
    if args.save_model:
        return cmd_save_model(args)
    if args.calibrate_slm:
        return cmd_calibrate_slm(args)
    if not args.file and not args.data:
        build_parser().print_help()
        return 2
    try:
        data = load_input(args.file, args.data)
    except Exception as e:
        print(f"toeknife: cannot read input: {e}", file=sys.stderr)
        return 1
    if not data:
        print("toeknife: input is empty", file=sys.stderr)
        return 1

    # ---- container layer: locate the stream before decoding anything --------
    stream = None
    whole = None            # (file, report) when the entry is one gzip member
    file_data = None        # the whole input, when the stream is inside it
    if not args.raw and not args.data:
        from . import containers
        report = containers.parse(data)
        if report is not None and report.streams:
            if args.entry == "all":
                return cmd_all(data, report, args)
            if args.entry is not None:
                try:
                    stream = pick_entry(report, args.entry)
                except ValueError as e:
                    print(f"toeknife: {e}", file=sys.stderr)
                    return 1
            elif len(report.decodable) == 1:
                stream = report.decodable[0]
                print(f"opening the only DEFLATE entry: {stream.name}",
                      file=sys.stderr)
            elif _interactive(args):
                # Pick from a list on screen; scripted use keeps the table.
                from .tui.entries import choose_entry
                from .predict.corpora import tilde
                stream = choose_entry(report, tilde(args.file) if args.file
                                      else "input")
                if stream is None:
                    return 0
            else:
                code = cmd_entries(report, args.file or "input")
                asked = _actions_asked(args)
                if asked:
                    print(f"toeknife: {', '.join(asked)} works on one entry; "
                          f"pick it with --entry N", file=sys.stderr)
                    return 1
                return code
            if not stream.decodable:
                print(f"toeknife: {stream.name}: {stream.why_not}", file=sys.stderr)
                return 1
            for note in stream.notes:
                print(f"toeknife: {stream.name}: {note}", file=sys.stderr)
            if stream.missing:
                # A capture that lost packets. The hole's position and width came
                # off the wire format, so this assembles rather than decodes:
                # nothing here is searched for except the bit the tail resumes on.
                session = capture_session(data, stream, args)
                if session is None:
                    return 1
                return _finish(session, args)
            from .core import members
            if members.multi(report):
                whole = (data, report)
            file_data = data
            data = stream.slice(data)

    litlen = dist = None
    if args.header:
        try:
            litlen, dist, block = tables_from_header(
                load_input(args.header), args.header_bit)
            print(f"tables from {args.header}: {block.btype_name}, "
                  f"HLIT={block.hlit} HDIST={block.hdist}", file=sys.stderr)
        except Exception as e:
            print(f"toeknife: cannot build tables from --header: {e}", file=sys.stderr)
            return 1

    if args.scan:
        return cmd_scan(data, args)

    mode, bit = args.mode or "auto", args.bit
    if args.resync:
        points = run_scan(data, args)
        if not points:
            print("toeknife: no block boundary found to resync on; try --scan",
                  file=sys.stderr)
            return 1
        best = points[0]
        mode, bit = "stream", best.bit_offset
        print(f"resynced at bit {best.bit_offset} ({best.kind}), "
              f"recovering {best.out_bytes} bytes", file=sys.stderr)

    if args.bit is not None and not 0 <= args.bit < len(data) * 8:
        print(f"toeknife: --bit {args.bit} is outside the data: it holds "
              f"{len(data) * 8:,} bits", file=sys.stderr)
        return 1

    model = load_model(args)

    if args.codeset:
        session = codeset_session(data, args, stream=stream, model=model)
        if session is None:
            return 1
        return _finish(session, args)

    if args.gap:
        import dataclasses
        from .containers.base import Stream as _Stream
        from .core import gap as gapmod
        try:
            tail = load_input(args.gap)
        except Exception as e:
            print(f"toeknife: cannot read --gap file: {e}", file=sys.stderr)
            return 1
        total = args.total_size
        if total is None and stream is not None:
            total = stream.uncompressed_size
        # Read once with no gap: what the pieces produce, and whether the
        # tail ends on a final block with a gzip trailer after it. A head cut
        # short has no trailer of its own; the tail carries the real one.
        zero = gapmod.from_holes(data + tail, [(len(data), 0)])
        trailer = gapmod.tail_trailer(zero)
        if total is None and trailer is not None and trailer[1] >= len(zero.tape):
            crc, total = trailer
            print(f"gap: the tail ends on a final block with 8 bytes after "
                  f"it, a gzip trailer: CRC32 {crc:08x}, {total:,} bytes",
                  file=sys.stderr)
            base = stream or _Stream(name=Path(args.file).name,
                                     container="gzip", offset=0)
            stream = dataclasses.replace(base, crc32=crc,
                                         uncompressed_size=total,
                                         source="the tail's trailer")
        try:
            if args.gap_size is not None:
                gaps = [args.gap_size] + [0] * (len(zero.pieces) - 2)
                asm = gapmod.assemble(zero.pieces, gaps, zero.litlen,
                                      zero.dist)
                print(f"gap: using the {args.gap_size:,} bytes you specified",
                      file=sys.stderr)
            else:
                asm = gapmod.two_pieces(data, tail, total, model=model,
                                        max_gap=args.gap_max,
                                        workers=args.gap_workers)
        except ValueError as e:
            print(f"toeknife: {e}", file=sys.stderr)
            return 1
        for note in asm.notes:
            print(f"gap: {note}", file=sys.stderr)
        session = Session.from_trace(asm.trace, stream=stream, model=model)
        return _finish(session, args)

    try:
        session = Session(data, mode=mode, litlen=litlen, dist=dist,
                          profile=args.profile, bit_offset=bit,
                          max_offset=args.max_offset, model=model, stream=stream)
    except Exception as e:
        print(f"toeknife: {e}", file=sys.stderr)
        return 1
    if session.tape.length == 0 and data and session.trace is not None:
        how = "from the start" if bit is None else f"from bit {bit}"
        print(f"toeknife: nothing decoded {how} "
              f"({session.trace.status}); "
              + ("--stream asserts a block header there and there is none; "
                 "without it the tool ranks fragment offsets"
                 if args.mode == "stream" else "--scan looks for a block "
                 "boundary further in"), file=sys.stderr)
    if getattr(session, "assumed_note", ""):
        print(f"toeknife: {session.assumed_note}", file=sys.stderr)
    if args.resync:
        # The session opened at the boundary, so it holds the resynced tape;
        # without the step a saved session could not replay to it and refused
        # its own file (wrap-up #2). f3 records the same step.
        session.record_step("resync", bit=bit)
    session.whole = whole
    session.file_data = file_data

    return _finish(session, args)


def _finish(session, args) -> int:
    """Everything after a session exists, whichever way it was built."""
    if args.session:
        # First, so its steps replay onto the session exactly as built, and
        # anything asked for on this command line goes on top of them.
        try:
            session.load(args.session)
        except Exception as e:
            print(f"toeknife: cannot load session: {e}", file=sys.stderr)
            return 1
        st = session.tape.stats()
        print(f"session: {len(session.steps)} step(s) replayed, "
              f"{len(session.guesses):,} byte(s) put back, {st['holes']:,} "
              f"class(es) open; the tape matches the one saved",
              file=sys.stderr)
    if args.bitrot:
        from .core import bitrot
        result = bitrot.repair(session, k=args.bitrot)
        print(f"bitrot: {result.summary()}", file=sys.stderr)
        session.ledger.add("bitrot", ledger.FOUND if result.matching
                           else ledger.NOTHING, result.summary(),
                           counts={"survivors": len(result.candidates)})
        for n, c in enumerate(result.candidates[:10], 1):
            print(f"bitrot: {n:>3}. {c.label}", file=sys.stderr)
        if len(result.candidates) > 10:
            print(f"bitrot: ... {len(result.candidates) - 10} more",
                  file=sys.stderr)
        if len(result.matching) == 1:
            pick = result.matching[0]
            session.flip_bits(pick.bits)
            print(f"bitrot: applied {', '.join(f'bit {b:,}' for b in pick.bits)}",
                  file=sys.stderr)
        elif not result.matching and result.k < 2 and _has_checksum(session):
            flips, secs = result.next_cost()
            print(f"bitrot: --bitrot {result.k + 1} tries all {flips:,} "
                  f"{result.k + 1}-bit flips: about {_duration(secs)} at this "
                  f"search's rate", file=sys.stderr)

    if args.resume:
        from .core import resume
        got = resume.resume(session)
        print(f"resume: {got.summary()}", file=sys.stderr)
        session.ledger.add("resume", ledger.NOTHING if got.trace is None
                           else ledger.FOUND, got.summary())
        if got.trace is None:
            return 1
        resume.apply(session, got)

    for g in args.guess:
        try:
            pos, val = parse_guess(g)
            session.set_guess(pos, val)
        except (ValueError, KeyError) as e:
            print(f"toeknife: --guess {g}: {e}", file=sys.stderr)
            return 1

    if args.solve_propose:
        return cmd_solve_propose(session, args)
    if args.solve_loop:
        from .predict.pipeline import solve_loop
        solver = run_solver(session, args)
        result = solve_loop(session, solver.model,
                            min_confidence=args.solve_confidence,
                            long_runs=args.fill_long_runs,
                            dictionary=args.fill_dictionary,
                            calibration=solver.calibration,
                            filler=solver.filler, mode=solver.mode,
                            shortlist=args.shortlist,
                            max_combinations=args.crc_limit,
                            on_round=lambda r: print(f"  {r.summary()}",
                                                     file=sys.stderr))
        print(f"loop: {result.summary()}", file=sys.stderr)
    if args.solve:
        solver = run_solver(session, args)
        before = session.tape.stats()
        session.begin_group()
        result = solver.solve(long_runs=args.fill_long_runs,
                              dictionary=args.fill_dictionary)
        session.end_group()
        after = session.tape.stats()
        print(f"solver: filled {len(result.filled)} of {before['holes']} classes "
              f"({result.positions_filled} output positions) in {result.rounds} "
              f"rounds; stopped because {result.stopped}. "
              f"{after['holes']} classes still open.", file=sys.stderr)
        if result.held_dictionary:
            print("solver: --dict finds the real dictionary; "
                  "--fill-dictionary fills them anyway", file=sys.stderr)
        elif result.held:
            print("solver: --fill-long-runs fills them anyway", file=sys.stderr)

    if args.checksum_solve:
        from .core.tape import SOLVED
        from .predict.crcsolve import (apply as crc_apply, solve_by_shortlist,
                                       solve_from_crc)
        before = session.tape.stats()["holes"]
        model = getattr(session, "model", None)
        result = None
        assignment = None

        # A small enough system is determined outright by CRC32 alone, with no
        # model involved and nothing to choose between.
        linear = solve_from_crc(session)
        if linear.ok and not linear.free_bits:
            result, assignment = linear, linear.assignment
            print(f"checksum: {linear.summary()}", file=sys.stderr)
        elif model is None:
            print(f"checksum: {linear.summary()}", file=sys.stderr)
            if _recorded_checksum(session):
                print("checksum: supply --corpus so the model can shortlist "
                      "the candidates the checksum has to choose between",
                      file=sys.stderr)
        else:
            result = solve_by_shortlist(session, model, per_class=args.shortlist,
                                        max_combinations=args.crc_limit)
            assignment = result.assignment if result.ok else None
            print(f"checksum: {result.summary()}", file=sys.stderr)
            if result.ambiguous:
                print("checksum: more than one answer fits, so this is the "
                      "model's pick and the checksum cannot settle it",
                      file=sys.stderr)

        if assignment:
            session.begin_group()
            for anchor, value in sorted(assignment.items()):
                if session.tape.value_at(anchor) is None:
                    session.set_guess(anchor, value, SOLVED)
            session.end_group()
            after = session.tape.stats()["holes"]
            print(f"checksum: filled {before - after} of {before} classes",
                  file=sys.stderr)

    if args.chunk_repair:
        from .core import chunkcrc
        total = len(session.stream.segments or []) if session.stream else 0
        got = chunkcrc.repair(session, getattr(session, "file_data", None)
                              or b"")
        print(f"chunks: {got.summary(total)}", file=sys.stderr)
        session.ledger.add("chunk-crc", ledger.FOUND if any(
            f.proven for f in got.fixes) else ledger.NOTHING, got.summary(total),
            counts={"candidates": got.candidates})
        proven = [f for f in got.fixes if f.proven]
        if len(proven) == 1:
            chunkcrc.apply(session, proven[0])
            print(f"chunks: applied {proven[0].label(total)}", file=sys.stderr)
    if args.write_repaired:
        from .core import chunkcrc
        fixed = chunkcrc.repaired_file(session)
        if fixed is None:
            print("toeknife: --write-repaired: no repair has been applied "
                  "(--chunk-repair applies a proven one)", file=sys.stderr)
            return 1
        Path(args.write_repaired).write_bytes(fixed)
        print(f"wrote {len(fixed):,} bytes to {args.write_repaired}: the "
              f"input with {len(session.file_fixes)} bit(s) flipped back",
              file=sys.stderr)

    if args.dict:
        from .core import dictfill
        got = dictfill.fill(session, Path(args.dict).read_bytes(),
                            Path(args.dict).name)
        print(f"dict: {got.summary()}", file=sys.stderr)
        session.ledger.add("dictionary", ledger.NOTHING if got.start is None
                           else ledger.FOUND, got.summary(),
                           counts={"filled": got.filled},
                           params={"path": args.dict})
        if got.start is None:
            return 1

    if args.fill_from:
        from .core import copyfill
        result = copyfill.fill(session, Path(args.fill_from).read_bytes(),
                               Path(args.fill_from).name)
        print(f"copy: {result.summary()}", file=sys.stderr)
        session.ledger.add("copy", ledger.FOUND if result.proven
                           or result.copy_matches else ledger.CONTRADICTED
                           if result.refuted else ledger.NOTHING,
                           result.summary(),
                           counts={"filled": result.filled,
                                   "contradicted": len(result.contradictions)},
                           params={"path": args.fill_from})
        for a, b in copyfill.regions(result.contradictions)[:8]:
            print(f"copy: contradicts decoded output at {a:,}"
                  + (f"..{b:,}" if b != a else ""), file=sys.stderr)
        if result.offset is None:
            return 1

    if args.save_session:
        session.save(args.save_session)
        print(f"session: saved to {args.save_session} ({len(session.steps)} "
              f"step(s), {len(session.guesses):,} byte(s), "
              f"{len(session.ledger)} ledger entr"
              f"{'y' if len(session.ledger) == 1 else 'ies'})",
              file=sys.stderr)

    # Record the evidence the operator supplied before anything dispatches, so
    # the route board reflects what is actually available rather than what was
    # available a moment ago.
    if args.related:
        session.related = args.related
    if args.plaintext:
        blob = Path(args.plaintext).read_bytes()
        session.known_plaintext = {args.plaintext_at + i: b
                                   for i, b in enumerate(blob)}
    if args.why is not None:
        buf, _mask, lo = session.full(0x3F)
        hi = lo + len(buf) - 1
        if not lo <= args.why <= hi:
            print(f"toeknife: --why {args.why}: the output runs from {lo} to "
                  f"{hi}", file=sys.stderr)
            return 1
        print(provenance.render(session, args.why))
        return 0
    if args.related and not args.routes:
        return cmd_related(session, args)
    if args.plaintext and not args.routes:
        print(plaintext.render(session, session.known_plaintext))
        v = plaintext.judge(session, session.known_plaintext)
        if not v.ruled:
            session.ledger.abstained("plaintext", v.testable, plaintext.FLOOR)
            return 2
        session.ledger.add(
            "plaintext",
            ledger.NOTHING if v.bad else ledger.FOUND,
            v.summary().split(":")[0],
            counts={"tested": v.testable, "failed": v.bad})
        return 1 if v.bad else 0
    if args.routes:
        print(routes.render(session))
        return 0
    if args.ledger:
        print(session.ledger.render())
        return 0
    if args.verify:
        return cmd_verify(session)
    if args.list:
        return cmd_list(session)
    if args.tables:
        return cmd_tables(session)
    if args.assignment:
        return cmd_assignment(session)
    if args.dump:
        return cmd_dump(session, args.verbose)
    if args.export:
        fill = ord(args.fill[0]) if args.fill else 0x3F
        buf, mask, start = session.full(fill)
        Path(args.export).write_bytes(buf)
        known = sum(mask)
        print(f"wrote {len(buf)} bytes to {args.export} "
              f"({known} known, {len(buf) - known} filled with {args.fill!r}); "
              f"first byte is output position {start}", file=sys.stderr)
        return 0

    from .tui.app import run
    from .predict.corpora import tilde
    name = tilde(args.file) if args.file else "hex input"
    stream = getattr(session, "stream", None)
    if stream is not None and getattr(stream, "container", None) not in (
            None, "pcap") and stream.name:
        name = f"{name}: {stream.name}"
    run(session, f"ToeKnife: {name}", solve_confidence=args.solve_confidence,
        self_train=args.self_train, order=args.order,
        shortlist=args.shortlist, crc_limit=args.crc_limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
