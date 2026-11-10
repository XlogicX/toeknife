"""Train a byte filler on real carves, as a run that can be supervised.

    python -m toeknife.predict.slmtrain --out ~/.toeknife/training/mine \
        --corpus ~/Documents/phrackarchive --steps 144048

The design and its reasons, all measured:

  * 256 byte values plus one UNKNOWN token.  "A byte is here and I do not know
    it" is a symbol the model trains on, not an improvisation in a prompt.  Every
    off-the-shelf checkpoint measured lacked exactly that, and it was what the
    42-to-72 gap was made of.
  * every masked position in the window is predicted from one forward pass.
    Predicting only the centre costs a factor of 50 in training throughput for
    identical FLOPs -- and note that this is also why a matched PREDICTION budget
    is not a matched OPTIMISATION budget once the window moves, which
    is why --steps exists and is the safer currency.
  * examples come from real carves, never random masking.  Holes in a carve
    arrive as one lost region plus every equality class reaching into it, and the
    density around them is nothing like uniform.
  * capacity is what was short.  19.2M parameters beat 4.9M on both held-out
    carves where four times the data and four times the context did not.

Two operational rules this file exists to enforce, both learned expensively:

  * **Checkpoints go in the run directory.**  The research trainer wrote to one
    fixed name beside itself, so a three-minute smoke test destroyed a 34-hour
    model.  `--out` makes that impossible rather than merely archived.
  * **The recovery target is never trained on.**  `--exclude` defaults to the two
    phrack articles this project holds out; point it at whatever the operator is
    actually trying to recover, or the numbers afterwards are fiction.
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import torch
import torch.nn as nn

from toeknife.predict import carves as carvegen
from toeknife.predict.bytemodel import IGNORE, ByteFiller, Windows


def evaluate(model, windows, batches=20, size=32):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    total, hit, n = 0.0, 0, 0
    with torch.no_grad():
        for _ in range(batches):
            ids, tgt = windows.batch(size)
            out = model(ids)
            total += loss_fn(out.reshape(-1, 256), tgt.reshape(-1)).item()
            keep = tgt != IGNORE
            hit += (out.argmax(-1)[keep] == tgt[keep]).sum().item()
            n += keep.sum().item()
    model.train()
    return total / batches, hit / max(n, 1)


def paths(out: Path):
    """Where this run's checkpoints go.

    Beside the run, not beside the script. The research trainer wrote to one
    fixed filename next to itself, which meant two runs contended for it and a
    smoke test could destroy a 34-hour model -- which is exactly what happened
    once. A run directory makes that impossible rather than merely archived.
    """
    return out / "bytemodel.pt", out / "bytemodel-best.pt"


def archive_existing(resuming: bool, CKPT, BEST):
    """Move any existing checkpoint aside before a fresh run can overwrite it.

    Written immediately after a three-minute smoke test destroyed a 34-hour
    model, because a fresh run and the run before it share one filename and
    nothing stood between them.  Archiving rather than refusing: a refusal in the
    middle of a launch gets --force-d out of habit, and the whole failure mode
    here is a habit.  20 MB is cheaper than the hours.
    """
    if resuming:
        return                          # --resume is meant to load and continue
    for path in (CKPT, BEST):
        if not path.is_file():
            continue
        try:
            n = torch.load(path, weights_only=True).get("predictions", 0)
        except Exception:
            n = 0
        dest = path.with_name(f"{path.stem}-prev-{n}{path.suffix}")
        i = 2
        while dest.is_file():
            dest = path.with_name(f"{path.stem}-prev-{n}-{i}{path.suffix}")
            i += 1
        path.rename(dest)
        print(f"  existing {path.name} ({n:,} predictions) moved to "
              f"{dest.name}", flush=True)


def corpus_files(root, exclude=()):
    """Every plausible training file under `root`, sorted for reproducibility.

    Sorted because the carve seed is the index into this list, so an unstable
    order would mean a different training set each run for no stated reason.
    """
    root = Path(root).expanduser()
    if root.is_file():
        return [root]
    out = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if exclude and p.name in exclude:
            continue
        out.append(p)
    return out


def hold_out(files, args, ckpt):
    """Files to train on, and held-out text to calibrate on at the end.

    Held back before carving, so nothing of them reaches a training window or
    a validation one. A resume holds back what the run it continues held
    back: a checkpoint written before runs held anything out trained on every
    file, so it gets text past --limit instead, which it never read either.
    """
    from toeknife.predict import calibration
    if args.no_calibrate:
        return files, []
    if args.resume and ckpt.is_file():
        try:
            had = torch.load(ckpt, weights_only=False,
                             map_location="cpu").get("held_out")
        except Exception:
            had = None
        if had is None:
            print("  resuming a run that held nothing out: calibrating on text "
                  f"past byte {args.limit:,} of its files", flush=True)
            return files, calibration.tail_windows(
                [str(f) for f in files], args.limit)
    train, held = calibration.filler_holdout(files, args.limit)
    if not held:
        print("  nothing held back for calibration: no file is long enough",
              flush=True)
        return [Path(p) for p in train], held
    whole = "@" not in held[0][0]
    print(f"  holding back {len(held)} "
          + ("file(s) for calibration: "
             + ", ".join(Path(n).name for n, _ in held[:6])
             + ("..." if len(held) > 6 else "") if whole else
             f"piece(s) of text past byte {args.limit:,} for calibration "
             "(too few files to hold whole ones back)"), flush=True)
    return [Path(p) for p in train], held


def calibrate_run(target: Path, held) -> None:
    """Measure the finished model on its held-out text and write the curve in.

    The same damage and the same lookup as an n-gram's calibration
    (predict/calibration.py), on the byte model's own confidence. On the 19.2M
    model this took 4.8 hours for 30 pieces of 32 KB, so each document's line
    says how long it took.
    """
    from toeknife.predict import calibration
    from toeknife.predict.bytemodel import ClassFiller
    if not held:
        print("not calibrated: no held-out text (--no-calibrate, or no file "
              "long enough)", flush=True)
        return
    filler = ClassFiller.load(target)
    t0 = time.monotonic()
    last = [t0]

    def progress(done, total, _note=""):
        now = time.monotonic()
        if done:
            per = (now - t0) / done
            print(f"  calibrating: {done} of {total} documents, last "
                  f"{now - last[0]:.0f} s, ~{per * (total - done) / 60:.0f} min "
                  "left", flush=True)
        else:
            print(f"calibrating {target.name} on {total} held-out "
                  "document(s)", flush=True)
        last[0] = now

    cal = calibration.calibrate_filler(filler, held, progress=progress)
    if cal is None:
        print("not calibrated: too few classes in the held-out text to fit "
              "a curve", flush=True)
        return
    calibration.save_filler_calibration(target, target, cal)
    print(f"calibration: {cal.summary()}, "
          f"{(time.monotonic() - t0) / 60:.0f} min -> {target.name}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="~/Documents/phrackarchive",
                    help="directory of training files, or one file")
    ap.add_argument("--exclude", nargs="*", default=["p38-5.txt", "p35-12.txt"],
                    help="file NAMES never to train on -- the recovery target "
                         "and anything reserved to confirm a number chosen on it")
    ap.add_argument("--out", default=".",
                    help="run directory; checkpoints are written here")
    ap.add_argument("--limit", type=int, default=48000,
                    help="bytes per file; 48,000 is where equality classes "
                         "were measured to saturate")
    # The budget is PREDICTIONS, not hours.  Two reasons, both learned the hard
    # way.  A wall-clock budget measures the machine as much as the training --
    # the same work takes a different number of hours under a different load --
    # so "the same training effort as the 34-hour run" is only expressible in
    # predictions.  And a wall-clock cosine schedule is broken by suspend: the
    # clock advances while the process is frozen, so the run wakes believing it
    # spent budget it did not, and anneals early at a learning rate meant for the
    # end.  Scheduling on predictions makes suspend a non-event.
    ap.add_argument("--predictions", type=int, default=269_659_138,
                    help="training budget (default: the 34-hour run's exact "
                         "count, for a matched-effort comparison)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="optional wall-clock guard; stops early, does not "
                         "change the schedule")
    # A matched PREDICTION budget is not a matched OPTIMISATION budget
    # whenever the window moves, because a wider window harvests more predictions
    # per forward pass.  At 269.7M predictions, window 129 takes 144,048 steps and
    # window 513 takes 35,433 -- a quarter of the gradient updates for the same
    # nominal effort, which owned a bit under half of that result's apparent loss.
    # --steps removes the question: it drives both the stopping condition and the
    # cosine, and it means the same thing regardless of window.
    ap.add_argument("--steps", type=int, default=None,
                    help="budget in optimiser steps instead of predictions; "
                         "the schedule follows it too")
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--carves", type=int, default=120)
    ap.add_argument("--val-carves", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--window", type=int, default=129)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="train on every file and skip the calibration at the "
                         "end; the model's confidence floor stays raw")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    CKPT, BEST = paths(out)
    files = corpus_files(args.corpus, args.exclude)
    files, held = hold_out(files, args, CKPT)
    want = args.carves + args.val_carves
    print(f"building up to {want} carves from {len(files)} file(s) in "
          f"{args.corpus}"
          + (f", excluding {args.exclude}" if args.exclude else "")
          + "...", flush=True)
    t0 = time.time()
    every, skipped = carvegen.build(files, count=want, limit=args.limit)
    if skipped:
        print(f"  {len(skipped)} would not align and were dropped", flush=True)
    train_c, val_c = every[:-args.val_carves], every[-args.val_carves:]
    print(f"  {len(train_c)} train / {len(val_c)} val carves in "
          f"{time.time()-t0:.0f}s", flush=True)

    tr = Windows(train_c, args.window, random.Random(1))
    va = Windows(val_c, args.window, random.Random(2))
    print(f"  {tr.total:,} maskable train positions, {va.total:,} val\n",
          flush=True)

    archive_existing(args.resume, CKPT, BEST)
    # What a later calibration from the CLI needs to find this run's held-out
    # text again (toeknife --calibrate-slm).
    held_names = [name for name, _ in held]

    model = ByteFiller(d_model=args.d_model, layers=args.layers,
                       heads=args.heads, window=args.window)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    resumed = None
    resumed_steps = 0
    if args.resume and CKPT.is_file():
        # A true continuation, not a warm restart.  An earlier run avoided --resume
        # deliberately because it restarted the optimiser and re-warmed the
        # schedule over a fresh budget, which makes the run a different
        # experiment from the one it claims to continue.  Restoring the
        # optimiser, both RNGs and the schedule position removes that objection:
        # a run interrupted at prediction N and resumed continues the same cosine
        # from N, with Adam's moments intact.
        state = torch.load(CKPT, weights_only=False)
        model.load_state_dict(state["model"])
        done = state["predictions"]
        if "opt" in state:
            opt.load_state_dict(state["opt"])
        else:
            print("  WARNING: checkpoint has no optimiser state -- this is a "
                  "warm restart, not a continuation", flush=True)
        if state.get("torch_rng") is not None:
            torch.set_rng_state(state["torch_rng"])
        if state.get("sampler_rng") is not None:
            tr.rng.setstate(state["sampler_rng"])
        if state.get("target") and state["target"] != (args.steps
                                                       or args.predictions):
            print(f"  WARNING: checkpoint was scheduled against "
                  f"{state['target']:,} predictions, this run says "
                  f"{args.predictions:,} -- the cosine will not line up",
                  flush=True)
        resumed = done
        resumed_steps = state.get("steps", 0)
        print(f"resumed from {CKPT.name} at {done:,} predictions "
              f"({done/args.predictions:.1%} of budget)\n", flush=True)
    else:
        done = 0
    params = sum(p.numel() for p in model.parameters()) / 1e6
    budget_says = (f"budget {args.steps:,} steps" if args.steps
                   else f"budget {args.predictions:,} predictions")
    print(f"{params:.1f}M params (d_model {args.d_model}, {args.layers} layers), "
          f"window {args.window}, batch {args.batch}, {budget_says}"
          + (f", guard {args.max_hours:.1f} h" if args.max_hours else "")
          + "\n", flush=True)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    model.train()

    def lr_at(fraction: float) -> float:
        """Warm up over the first 2%, then cosine down to a twentieth.

        The smoke run reached 41% in one minute at a constant rate and then sat
        there oscillating, which is what too high a rate looks like once the easy
        structure is learned.  `fraction` is predictions done over predictions
        budgeted -- see --predictions for why it is not wall clock.
        """
        if fraction < 0.02:
            return args.lr * fraction / 0.02
        t = (fraction - 0.02) / 0.98
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t)))

    base_loss, base_acc = evaluate(model, va, batches=6)
    print(f"{'elapsed':>8} {'predictions':>13} {'train loss':>11} "
          f"{'val loss':>9} {'val top-1':>10}   {'lr':>10}")
    print(f"{'0:00':>8} {done:>13,} {'-':>11} {base_loss:>9.3f} "
          f"{base_acc:>9.1%}   (random = 0.4%)", flush=True)

    started = time.time()
    active0 = time.monotonic()
    guard = started + args.max_hours * 3600 if args.max_hours else float("inf")
    taken = 0                       # optimiser steps THIS leg
    step0 = resumed_steps or 0      # steps before it, if resuming
    if args.steps:
        report_every = max(args.steps // 20, 1)
        next_report = step0 + report_every
    else:
        report_every = max(args.predictions // 20, 1)
        next_report = done + report_every
    # Snapshots at fixed fractions, kept regardless of validation, because
    # val top-1 was measured moving a tenth as far as the carve scores it is
    # meant to stand in for.  keep_best() is selecting close to blind, and no
    # earlier run kept the alternatives that would show what it passed over.
    snap_at = [f for f in (0.25, 0.50, 0.75)]
    last_report = started
    running, steps = 0.0, 0
    best_acc = 0.0

    def save(vl, vacc):
        """The latest checkpoint, with everything a resume needs to continue.

        Weights alone are not enough: without Adam's moments the resume is a warm
        restart, and without `target` the cosine cannot be placed.  Written on
        every report, so a crash or a power loss costs at most one report
        interval.
        """
        torch.save(dict(model=model.state_dict(), cfg=model.cfg,
                        opt=opt.state_dict(), predictions=done,
                        steps=step0 + taken,
                        target=args.steps or args.predictions,
                        val_loss=vl, val_acc=vacc, held_out=held_names,
                        torch_rng=torch.get_rng_state(),
                        sampler_rng=tr.rng.getstate()), CKPT)

    def keep_best(vacc, vl=None):
        """Save the best checkpoint, selected on val top-1 rather than val loss.

        Both halves of that sentence were paid for by the 34-hour run.  Only the
        latest checkpoint was written, so its peak -- hour 30, 58.3% -- was
        overwritten by hour 35 at 57.6%, and the model that got scored was the
        worse one.  And val loss bottomed at hour 9 and never returned while top-1
        climbed for another 21 hours, so loss is the wrong thing to select on:
        top-1 is what tracked the carve.

        A candidate is re-measured on 3x the batches before it is kept.  The
        periodic estimate is 10 batches and it swings a point and a half between
        reports, so taking its running maximum would select the luckiest
        evaluation as often as the best model."""
        nonlocal best_acc
        if vacc <= best_acc:
            return
        loss, acc = evaluate(model, va, batches=30)
        if acc <= best_acc:
            print(f"{'':>8} {'':>13} {'':>11} {'':>9} {'':>10}   "
                  f"candidate {vacc:.1%} not confirmed ({acc:.1%})", flush=True)
            return
        best_acc = acc
        # val_loss too, so the best checkpoint carries the same fields as the
        # latest one -- bytemodel-eval.py prints `val loss nan` without it.
        torch.save(dict(model=model.state_dict(), cfg=model.cfg,
                        predictions=done, val_acc=acc, val_loss=loss,
                        held_out=held_names), BEST)
        print(f"{'':>8} {'':>13} {'':>11} {'':>9} {'':>10}   "
              f"best {acc:.1%} -> {BEST.name}", flush=True)
    def progress():
        """Fraction of budget spent, in whichever currency was asked for."""
        if args.steps:
            return (step0 + taken) / args.steps
        return done / args.predictions

    while progress() < 1.0 and time.time() < guard:
        lr = lr_at(progress())
        for g in opt.param_groups:
            g["lr"] = lr
        ids, tgt = tr.batch(args.batch)
        opt.zero_grad(set_to_none=True)
        out = model(ids)
        loss = loss_fn(out.reshape(-1, 256), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item()
        steps += 1
        taken += 1
        done += int((tgt != IGNORE).sum())

        here = (step0 + taken) if args.steps else done
        if here >= next_report or time.time() - last_report > 1800:
            vl, vacc = evaluate(model, va, batches=10)
            el = time.time() - started
            # Two clocks, because they diverge and only one can answer "how
            # much longer". time.time() keeps counting through a suspend, so an
            # ETA extrapolated from it after a ten-hour hibernation is wrong by
            # ten hours. CLOCK_MONOTONIC -- time.monotonic() on Linux -- stops
            # while the machine is suspended, so it measures time actually
            # spent running. Not process_time(), which sums across threads and
            # reads 8x wall on this box: that answers "how much CPU", not "how
            # much longer".
            active = time.monotonic() - active0
            print(f"{int(el)//60:>5}:{int(el)%60:02d} {done:>13,} "
                  f"{running/max(steps,1):>11.3f} {vl:>9.3f} {vacc:>9.1%} "
                  f"  lr {lr:.2e}  {progress():>5.1%}"
                  f"  active {int(active)//60:>5}:{int(active)%60:02d}",
                  flush=True)
            save(vl, vacc)
            keep_best(vacc)
            while snap_at and progress() >= snap_at[0]:
                frac = snap_at.pop(0)
                dest = CKPT.with_name(f"{CKPT.stem}-at{int(frac*100):02d}"
                                      f"{CKPT.suffix}")
                torch.save(dict(model=model.state_dict(), cfg=model.cfg,
                                predictions=done, steps=step0 + taken,
                                val_loss=vl, val_acc=vacc,
                                held_out=held_names), dest)
                print(f"{'':>8} {'':>13} {'':>11} {'':>9} {'':>10}   "
                      f"snapshot at {frac:.0%} -> {dest.name}", flush=True)
            running, steps = 0.0, 0
            next_report = here + report_every
            last_report = time.time()

    vl, vacc = evaluate(model, va, batches=20)
    save(vl, vacc)
    keep_best(vacc)
    # Before the completion line, which the app reads as "this run ended":
    # calibration is hours, and a run that printed it would sit in
    # "finishing" for all of them.
    if progress() >= 1.0:
        calibrate_run(BEST if BEST.is_file() else CKPT, held)
    else:
        print("not calibrated: stopped by the wall-clock guard before its "
              "budget; --resume finishes it and calibrates", flush=True)
    stopped = "budget reached" if progress() >= 1.0 else "wall-clock guard"
    spent = (f"{step0 + taken:,} of {args.steps:,} steps ({done:,} predictions)"
             if args.steps else
             f"{done:,} of {args.predictions:,} predictions")
    print(f"\ndone ({stopped}): {spent}, val loss {vl:.3f}, "
          f"val top-1 {vacc:.1%} (was {base_acc:.1%} at init)")
    if resumed is not None:
        print(f"  this leg added {done - resumed:,} predictions "
              f"over {taken:,} steps")
    print(f"saved {CKPT}")
    if BEST.is_file():
        print(f"saved {BEST} at {best_acc:.1%} val top-1 -- score THIS one")


if __name__ == "__main__":
    main()
