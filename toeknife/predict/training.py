"""Supervising a byte-model training run that outlives the process that started it.

An n-gram trains in seconds, inside the app, on the thread that asked for it. A
byte model takes 30 to 90 hours, which changes the problem from "show a progress
bar" to "do not lose four days of somebody's machine". The run that produced the
current best model took 90 hours across several hibernations.

So the app must not own the process. A run lives in its own directory under
`~/.toeknife/training/`, holding the spec it was launched with, its log, and its
checkpoints. Starting one detaches it; quitting the app leaves it running;
opening the app again finds it. A crash or a reboot loses at most the interval
between checkpoints, because the trainer's `--resume` is a true continuation --
same cosine position, optimiser moments restored.

**Liveness is not a pid.** A pid can be recycled, a process can be alive and
wedged, and a run can have finished while the app was closed. Three signals are
checked and disagreement is reported rather than resolved: whether the pid exists
and is still the process that was launched, whether the log has advanced
recently, and whether the trainer wrote its own completion line.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

RUNS_DIR = "~/.toeknife/training"
SPEC = "run.json"
LOG = "train.log"

# The trainer prints one of these per report. Parsed rather than shared through
# a side channel, because the log is the thing that survives the app closing and
# a second source of truth would be one that can disagree.
PROGRESS = re.compile(
    r"^\s*(\d+):(\d\d)\s+([\d,]+)\s+\S+\s+([\d.]+)\s+([\d.]+)%"
    r"[^%]*?([\d.]+)%(?:\s+active\s+(\d+):(\d\d))?\s*$")
DONE = re.compile(r"^done \(([^)]+)\):")
# A stall is not an error, but a run that has said nothing for this long while
# claiming to be alive is worth surfacing rather than drawing as healthy.
QUIET_SECONDS = 3600


@dataclass
class Progress:
    elapsed_s: int = 0                 # wall clock, hibernation included
    predictions: int = 0
    val_loss: float = 0.0
    val_top1: float = 0.0
    fraction: float = 0.0
    active_s: int | None = None        # time actually running, suspend excluded

    @property
    def working_s(self) -> int:
        """The clock to extrapolate from. Falls back to wall time for runs
        started before the trainer logged an active clock."""
        return self.active_s if self.active_s is not None else self.elapsed_s


@dataclass
class Run:
    """One training run, as it can be known from disk alone."""
    path: Path
    spec: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def log(self) -> Path:
        return self.path / LOG

    @property
    def pid(self) -> int | None:
        return self.spec.get("pid")

    # ---- what the log says -------------------------------------------------

    def tail(self, lines: int = 40) -> list[str]:
        try:
            return self.log.read_text(errors="replace").splitlines()[-lines:]
        except OSError:
            return []

    def progress(self) -> Progress | None:
        """The last report line, or None if it has not made one yet."""
        for line in reversed(self.tail(400)):
            m = PROGRESS.match(line)
            if m:
                active = (int(m.group(7)) * 60 + int(m.group(8))
                          if m.group(7) else None)
                return Progress(
                    elapsed_s=int(m.group(1)) * 60 + int(m.group(2)),
                    predictions=int(m.group(3).replace(",", "")),
                    val_loss=float(m.group(4)), val_top1=float(m.group(5)),
                    fraction=float(m.group(6)) / 100.0, active_s=active)
        return None

    def finished(self) -> str | None:
        """The trainer's own completion line, which is the only claim that a run
        ended on purpose rather than by being killed."""
        for line in reversed(self.tail(60)):
            m = DONE.match(line)
            if m:
                return m.group(1)
        return None

    def quiet_for(self) -> float | None:
        try:
            return time.time() - self.log.stat().st_mtime
        except OSError:
            return None

    # ---- whether it is actually running ------------------------------------

    def process_alive(self) -> bool:
        """Whether the pid is still the process this run launched.

        A pid alone is not enough: they are recycled, and on a box that has been
        up for weeks a stale one will eventually name somebody else's work. The
        command line is checked too, so the worst case is reporting a live run
        as dead rather than killing a stranger.
        """
        pid = self.pid
        if not pid:
            return False
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return False
        want = self.spec.get("marker") or self.name
        return want.encode() in cmdline

    def state(self) -> str:
        """One word, and it never guesses when the signals disagree."""
        done = self.finished()
        alive = self.process_alive()
        if done and not alive:
            return "finished"
        if done and alive:
            return "finishing"          # wrote its line, has not exited yet
        if not alive:
            return "stopped"            # died, was killed, or the box rebooted
        quiet = self.quiet_for()
        if quiet is not None and quiet > QUIET_SECONDS:
            return "quiet"              # alive by pid, saying nothing
        return "running"

    def eta_seconds(self) -> float | None:
        """Seconds of RUNNING left, from the run's own measured rate.

        Extrapolated from the active clock, which stops while the machine is
        suspended. Wall time would fold every hibernation into the estimate and
        report a run as slower than it is -- one that spent ten hours asleep at
        the halfway mark would claim ten extra hours of work remaining.

        It is therefore an answer to "how much more computing", not "at what
        time will this be done": the latter depends on how often the machine
        gets suspended next, which nothing here can know.
        """
        p = self.progress()
        if not p or p.fraction <= 0.0 or p.fraction >= 1.0:
            return None
        return p.working_s * (1.0 - p.fraction) / p.fraction

    def summary(self) -> str:
        p = self.progress()
        st = self.state()
        if p is None:
            return f"{self.name}: {st}, no reports yet"
        eta = self.eta_seconds()
        when = f", ~{eta / 3600:.1f} h left" if eta else ""
        return (f"{self.name}: {st}, {p.fraction:.1%}, "
                f"val top-1 {p.val_top1:.1f}%{when}")

    # ---- acting on it ------------------------------------------------------

    def steps_per_second(self) -> float | None:
        """This run's own measured rate, or None if it cannot be known.

        Derived rather than logged: the log carries predictions and a fraction
        of budget, and the spec carries the step budget, so steps done is
        fraction x budget. Divided by the ACTIVE clock, which stops while the
        machine is suspended -- dividing by wall time would report a run that
        hibernated as a slower machine, and this figure is used to size the next
        run.

        Only for runs budgeted in steps. A prediction-budgeted run's fraction is
        a fraction of predictions, and steps cannot be recovered from it.
        """
        cfg = self.spec.get("config") or {}
        budget = cfg.get("steps")
        p = self.progress()
        if not budget or not p or p.fraction <= 0 or p.working_s <= 0:
            return None
        done = p.fraction * budget
        if done < 200:                  # too little to be a rate rather than noise
            return None
        return done / p.working_s

    def resume_command(self) -> list[str]:
        return list(self.spec.get("command", [])) + ["--resume"]

    def stop(self) -> bool:
        """Ask it to stop. Never SIGKILL: the trainer checkpoints on a report
        and a hard kill can cost the interval since the last one."""
        if not self.process_alive():
            return False
        os.kill(self.pid, signal.SIGTERM)
        return True


def runs_dir() -> Path:
    return Path(os.path.expanduser(
        os.environ.get("TOEKNIFE_TRAINING", RUNS_DIR)))


def load(path) -> Run:
    path = Path(path)
    try:
        spec = json.loads((path / SPEC).read_text())
    except Exception:
        spec = {}
    return Run(path, spec)


def runs() -> list[Run]:
    """Every run on disk, newest first, whatever state it is in.

    A finished run is still listed: its checkpoints are the product, and the
    operator arriving after a reboot needs to see that the four days happened.
    """
    d = runs_dir()
    if not d.is_dir():
        return []
    found = [load(p) for p in d.iterdir() if (p / SPEC).is_file()]
    return sorted(found, key=lambda r: r.spec.get("started", 0), reverse=True)


def launch(name: str, command, *, spec: dict | None = None,
           resume: bool = False) -> Run:
    """Start a run detached, so closing the app does not take it with you.

    `setsid` puts it in its own session: no controlling terminal, no SIGHUP when
    the parent goes away. stdin comes from /dev/null so it can never block
    waiting for input nobody is there to give.

    The run directory becomes the process's cwd. Every path in `command` after
    the program name must therefore be absolute; only the program name is fixed
    up here.
    """
    d = runs_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    log = d / LOG
    argv = list(command) + (["--resume"] if resume else [])
    # The run's own directory is its cwd, so anything it writes relatively lands
    # with it rather than wherever the app happened to be started from. That
    # makes a relative program name resolve against the wrong directory and fail
    # as a bare FileNotFoundError several frames from the cause, so it is made
    # absolute here, while the caller's cwd still means something.
    #
    # abspath and NOT resolve(): a virtualenv's bin/python is a symlink to the
    # system interpreter, and resolving it launches the system python instead --
    # same binary, different sys.prefix, none of the venv's packages. Following
    # that symlink turns "train a model" into ModuleNotFoundError: torch.
    #
    # Only argv[0] is touched. Which of the remaining arguments are paths is the
    # caller's business, so THEY MUST BE ABSOLUTE ALREADY.
    if argv and os.sep in argv[0]:
        argv[0] = os.path.abspath(argv[0])
    with open(log, "ab") as out, open(os.devnull, "rb") as devnull:
        out.write(f"\n=== {'resumed' if resume else 'started'} "
                  f"{time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode())
        out.flush()
        proc = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT,
                                stdin=devnull, start_new_session=True,
                                cwd=str(d))
    body = dict(spec or {})
    body.update(name=name, command=argv, pid=proc.pid, started=time.time(),
                marker=body.get("marker") or name,
                shell=" ".join(shlex.quote(a) for a in argv))
    (d / SPEC).write_text(json.dumps(body, indent=1, default=str))
    return load(d)


def measured_rates() -> dict:
    """Steps per second for each model shape, from runs on THIS machine.

    The training screen's costs come from a probe on one machine and are wrong
    on any other. Every run that has made progress has already measured the real
    figure for its own configuration, so the screen can stop quoting somebody
    else's hardware as soon as the operator has trained anything at all.

    Keyed by (d_model, layers, window), which is what changes the cost. Newest
    run wins: a machine that got faster or slower since should be believed about
    now rather than about then.
    """
    out = {}
    for r in runs():                    # newest first
        cfg = r.spec.get("config") or {}
        key = (cfg.get("d_model"), cfg.get("layers"), cfg.get("window"))
        if None in key or key in out:
            continue
        rate = r.steps_per_second()
        if rate:
            out[key] = rate
    return out


def checkpoint(run: Run) -> Path | None:
    """The checkpoint a finished run is used by: the best one, calibrated at
    the end of the run, or the latest when no best was ever kept."""
    for name in ("bytemodel-best.pt", "bytemodel.pt"):
        if (run.path / name).is_file():
            return run.path / name
    return None


def run_provenance(run: Run) -> dict:
    """What a saved byte model says about the run that made it.

    From the run's spec and its checkpoint, so f5's lineage reads the corpus,
    the shape and how far it trained without anyone typing it in.
    """
    import torch
    ckpt = checkpoint(run)
    blob = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = dict(blob.get("cfg") or {})
    params = sum(t.numel() for t in (blob.get("model") or {}).values())
    conf = run.spec.get("config") or {}
    prov = {"corpus": Path(str(run.spec.get("corpus") or "?")).name,
            "run": run.name,
            "params": f"{params / 1e6:.1f}M",
            "arch": (f"d_model {cfg.get('d_model')}, {cfg.get('layers')} "
                     f"layers, {cfg.get('heads')} heads"),
            "window": cfg.get("window"),
            "steps": conf.get("steps") or blob.get("steps"),
            "val_top1": (f"{blob['val_acc']:.1%}"
                         if blob.get("val_acc") is not None else None),
            "held_out": len(blob.get("held_out") or [])}
    return {k: v for k, v in prov.items() if v is not None}


def suggest_name(run: Run) -> Path:
    """`<corpus>-slm-<params>.tks` in the models directory, never one taken."""
    from .corpora import _slug, model_dir
    prov = run_provenance(run)
    d = model_dir()
    stem = f"{_slug(prov['corpus'])}-slm-{prov['params']}"
    p, n = d / f"{stem}.tks", 2
    while p.exists():
        p = d / f"{stem}-{n}.tks"
        n += 1
    return p


def save_run(run: Run, dest) -> Path:
    """Copy a finished run's checkpoint into the models directory as a .tks,
    with its lineage, so f5 lists it. Refuses to overwrite."""
    from .corpora import register_slm
    ckpt = checkpoint(run)
    if ckpt is None:
        raise FileNotFoundError(f"{run.name} has no checkpoint")
    dest = Path(os.path.expanduser(str(dest)))
    return register_slm(ckpt, dest.name, run_provenance(run),
                        directory=dest.parent)
