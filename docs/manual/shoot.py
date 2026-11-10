#!/usr/bin/env python3
"""Regenerate the manual's screenshots by driving the real app.

Every image in docs/manual/shots/ comes from here: the same entry point as
`toeknife FILE ...`, run headless through Textual's Pilot, keys pressed, and the
screen saved as SVG. A step in a workflow that cannot be scripted here is a
step the manual should not claim works.

    .venv/bin/python docs/manual/shoot.py            # every scene
    .venv/bin/python docs/manual/shoot.py A B        # some workflows

Each scene runs from the directory that holds its specimen and names it
relatively, so screenshots show `stack.bin`, not somebody's home directory.
Scenes whose specimen is not on this machine are skipped and say why. The
phrack specimens are made by the recipes on the workflow pages; set
TOEKNIFE_SPECIMENS to where you keep them (default ~/.toeknife/specimens). The
rest are in samples/.

Steps:
  KEY                press it (textual key names: f5, ctrl+k, enter, ...)
  text:STRING        type it
  shot:NAME          save shots/<workflow>/NAME.svg
  wait               let background work (training, scoring, ^f) finish
  highlight:SUFFIX   in an option list, highlight the row whose id ends so
  goto:POS           move the tape cursor to an output position
  firstfill          move the cursor to the first model-filled byte
  gapstart           move the cursor to the start of the longest unknown run
  pick:NAME:N        (first step only) shoot the archive's entry picker as
                     NAME, then open entry N from it
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHOTS = Path(__file__).resolve().parent / "shots"
SIZE = (140, 40)
SPEC = Path(os.environ.get("TOEKNIFE_SPECIMENS",
                           "~/.toeknife/specimens")).expanduser()
MODELS = Path("~/.toeknife/models").expanduser()

# workflow -> [(directory to run in, toeknife arguments, steps)]
SCENES: dict[str, list[tuple[Path, list[str], list[str]]]] = {
    # Workflow A: a carved fragment whose start was chopped off.
    "A": [
        (SPEC, ["stack.bin"],
         ["shot:01-open", "f2", "wait", "shot:02-align", "escape",
          "f7", "shot:03-next", "escape",
          "f5", "highlight:phrackarchive-o6-cal.tkm", "shot:04-models",
          "enter", "enter", "wait", "shot:05-loaded",
          "ctrl+k", "text:threshold", "enter", "shot:06-floor",
          "highlight:0.8", "enter",
          "ctrl+f", "wait", "shot:07-filled",
          "firstfill", "shot:08-fillnote",
          "f4", "wait", "shot:09-proposals", "escape",
          "f8", "shot:10-checksum", "escape"]),
    ],
    # Workflow B: a capture that lost a packet.
    "B": [
        (SPEC, ["stack-http-drop17.pcap"],
         ["shot:01-open",
          "f7", "shot:03-next", "escape",
          "f5", "highlight:phrackarchive-o6-cal.tkm", "enter", "enter", "wait",
          "ctrl+k", "text:threshold", "enter", "highlight:0.9", "enter",
          "ctrl+f", "wait", "shot:04-fill", "escape",
          "f8", "shot:05-checksum", "escape",
          "ctrl+z",
          "ctrl+k", "text:suspected", "enter", "text:suspected-copy.txt",
          "enter", "shot:06-proven", "escape", "shot:07-complete"]),
        (SPEC, ["stack-http-drop17.pcap"],
         ["ctrl+k", "text:suspected", "enter", "text:edited-copy.txt",
          "enter", "shot:08-refuted"]),
    ],
    # Workflow C: a damaged zip.
    "C": [
        (SPEC, ["stack.zip"], ["pick:01-picker:0", "f8", "shot:02-intact"]),
        (SPEC, ["stack.zip"],
         ["pick:-:1", "shot:03-rotted", "f8", "shot:04-rotted-check",
          "escape", "f7", "shot:05-rotted-next", "escape",
          "ctrl+k", "text:suspected", "enter", "text:suspected-copy.txt",
          "enter", "shot:06-rotted-copy"]),
        (SPEC, ["stack.zip"],
         ["pick:-:2", "shot:07-shredded", "f8", "shot:08-shredded-check",
          "escape", "f7", "highlight:act:resume", "shot:09-shredded-next",
          "enter", "wait", "goto:25400", "shot:10-shredded-resumed",
          "f8", "shot:11-shredded-resumed-check", "escape",
          "ctrl+k", "text:suspected", "enter",
          "text:suspected-copy.txt", "enter", "shot:12-shredded-copy"]),
        (SPEC, ["stack-noeocd.zip"], ["pick:13-noeocd:0"]),
        (SPEC, ["stack-sfx.zip"], ["pick:14-sfx:0"]),
    ],
    # Workflow D: bit rot in a header.
    "D": [
        (SPEC, ["stack-rot.gz"],
         ["shot:01-open", "f8", "shot:02-check", "escape",
          "f7", "highlight:act:bitrot", "shot:03-next",
          "enter", "wait", "shot:04-survivors",
          "enter", "shot:05-repaired", "f8", "shot:06-verified"]),
    ],
    # Workflow E: a carve whose table is gone.
    "E": [
        (SPEC, ["stack-carve.bin"],
         ["shot:01-open", "f7", "shot:02-next", "escape",
          "f3", "wait", "shot:03-resync", "enter", "shot:04-resynced",
          "f5", "highlight:phrackarchive-o6-cal.tkm", "enter", "enter", "wait",
          "f7", "highlight:act:related", "shot:05-next", "enter",
          "text:edited-copy.txt.gz", "enter", "wait", "shot:06-edited",
          "escape",
          "f7", "highlight:act:related", "enter",
          "text:mirror.txt.gz", "enter", "wait", "shot:07-mirror",
          "enter", "shot:08-borrowed",
          "ctrl+k", "text:suspected", "enter", "text:suspected-copy.txt",
          "enter", "shot:09-copy"]),
    ],
    # Workflow F: two pieces of one gzip file.
    "F": [
        (SPEC, ["stack-part1.gz", "--gap", "stack-part2.bin"],
         ["gapstart", "shot:01-open", "f8", "shot:02-check", "escape",
          "ctrl+k", "text:suspected", "enter", "text:suspected-copy.txt",
          "enter", "shot:03-proven"]),
    ],
    # Workflow G: three gzip members, the middle one's table gone.
    "G": [
        (SPEC, ["stack-members.gz"], ["pick:01-picker:0"]),
        (SPEC, ["stack-members.gz"],
         ["pick:-:1", "shot:02-bad", "f8", "shot:03-bad-check", "escape",
          "f7", "shot:04-bad-next", "enter", "wait", "shot:05-bad-bitrot",
          "escape", "f7", "highlight:act:fill_from_copy", "enter",
          "text:suspected-copy.txt", "enter", "shot:06-bad-copy",
          "escape", "f8", "shot:07-bad-proven", "escape",
          "ctrl+w", "shot:08-write"]),
    ],
    # Workflow H: a zlib stream whose preset dictionary is not in the file.
    "H": [
        (SPEC, ["stack-dict.zlib"],
         ["shot:01-open", "f8", "shot:02-check", "escape",
          "f7", "shot:03-next", "enter", "text:stackpack.c", "enter", "wait",
          "shot:04-found", "escape", "goto:-40", "shot:05-dictionary",
          "f8", "shot:06-verified"]),
        (SPEC, ["stack-dict.zlib"],
         ["f5", "highlight:phrackarchive-o6-cal.tkm", "enter", "enter", "wait",
          "ctrl+k", "text:threshold", "enter", "highlight:0.9", "enter",
          "ctrl+f", "wait", "shot:07-held"]),
    ],
    # Workflow I: DEFLATE inside a PNG and a PDF.
    "I": [
        (SPEC, ["stack.png"],
         ["shot:01-png", "f8", "shot:02-png-check", "escape",
          "f7", "shot:03-png-next", "enter", "wait", "shot:04-png-repair",
          "enter", "f8", "shot:05-png-repaired", "escape",
          "ctrl+w", "shot:06-png-write"]),
        (SPEC, ["stack-cut.pdf"], ["pick:07-pdf-picker:15", "shot:08-pdf-cut"]),
    ],
    # Models: choosing, loading, saving and removing one. books.tkm is a
    # throwaway model saved next to the specimens.
    "M": [
        (SPEC, ["stack.bin"],
         ["f5", "shot:01-f5", "highlight:books.tkm", "shot:02-saved",
          "enter", "shot:03-lineage", "enter", "wait", "shot:04-loaded",
          "ctrl+k", "text:save the loaded", "enter", "shot:05-save", "escape",
          "f5", "highlight:books.tkm", "ctrl+d", "shot:06-trash", "escape",
          "escape", "ctrl+k", "text:train a byte", "enter", "home", "shot:07-train",
          "escape",
          "f5", "highlight:phrack-slm-19M-cal.tks", "enter", "enter", "wait",
          "f5", "highlight:mode:ngram", "shot:08-fillmode"]),
    ],
    # Sessions: save one, open it again, and read what has run.
    "S": [
        (SPEC, ["stack-http-drop17.pcap"],
         ["ctrl+k", "text:suspected", "enter", "text:suspected-copy.txt",
          "enter", "escape", "ctrl+s", "shot:01-save",
          "home", "ctrl+k", "text:stack-http.session", "enter"]),
        (SPEC, ["stack-http-drop17.pcap", "--session", "stack-http.session"],
         ["shot:02-reopened"]),
    ],
}


def need(cwd: Path, argv) -> str | None:
    """The first file argument that does not exist, or None."""
    if not cwd.is_dir():
        return str(cwd)
    for a in argv:
        if not a.startswith("-") and not (cwd / Path(a).expanduser()).exists():
            return str(cwd / a)
    return None


def shoot(workflow: str, cwd: Path, argv: list[str], steps: list[str]) -> None:
    import toeknife.tui.app as tuiapp
    from toeknife import cli

    out = SHOTS / workflow
    out.mkdir(parents=True, exist_ok=True)
    captured = {}

    def fake_run(session, title="ToeKnife", *a, **kw):
        captured["app"] = tuiapp.ToeKnifeApp(session, title, *a, **kw)

    import toeknife.tui.entries as entries
    real_pick = entries.choose_entry
    if steps and steps[0].startswith("pick:"):
        _, pick_name, pick_index = steps.pop(0).split(":")

        def scripted_pick(report, path):
            picker = entries.EntryPicker(report, path)

            async def go_pick():
                if pick_name == "-":
                    return
                async with picker.run_test(size=SIZE) as pilot:
                    await pilot.pause()
                    picker.save_screenshot(str(out / f"{pick_name}.svg"))
                    print(f"  {workflow}/{pick_name}.svg  (EntryPicker)")
            asyncio.run(go_pick())
            return report.streams[int(pick_index)]

        entries.choose_entry = scripted_pick
    real = tuiapp.run
    tuiapp.run = fake_run
    os.chdir(cwd)
    # Headless, so there is no terminal; the CLI only opens the picker for
    # one, and a scene is the interactive path by definition.
    ttys = (sys.stdout.isatty, sys.stdin.isatty)
    sys.stdout.isatty = sys.stdin.isatty = lambda: True
    try:
        cli.main(argv)
    finally:
        tuiapp.run = real
        entries.choose_entry = real_pick
        sys.stdout.isatty, sys.stdin.isatty = ttys
    app = captured["app"]

    async def go():
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            for s in steps:
                if s.startswith("shot:"):
                    await pilot.pause()
                    app.save_screenshot(str(out / f"{s[5:]}.svg"))
                    print(f"  {workflow}/{s[5:]}.svg  ({type(app.screen).__name__})")
                elif s == "wait":
                    await app.workers.wait_for_complete()
                    await pilot.pause()
                elif s.startswith("text:"):
                    await pilot.press(*s[5:])
                elif s.startswith("highlight:"):
                    from textual.widgets import OptionList
                    lst = app.screen.query_one(OptionList)
                    want = s[len("highlight:"):]
                    lst.highlighted = next(
                        i for i in range(lst.option_count)
                        if (lst.get_option_at_index(i).id or "").endswith(want))
                    await pilot.pause()
                elif s.startswith("goto:"):
                    pos = int(s[5:])
                    app.tape_view.goto(pos)
                    app.update_inspector(pos)
                    await pilot.pause()
                elif s == "gapstart":
                    # The first byte of the longest unknown run: where a lost
                    # packet's bytes would have gone.
                    from toeknife.predict.solver import evidence_runs
                    runs = evidence_runs(app.session.tape)
                    longest = max(runs.values())
                    pos = min(p for p, n in runs.items() if n == longest)
                    app.tape_view.goto(pos)
                    app.update_inspector(pos)
                    await pilot.pause()
                elif s == "firstfill":
                    from toeknife.core.tape import SOLVED
                    pos = min(p for p, r in app.session._records.items()
                              if r.provenance == SOLVED)
                    app.tape_view.goto(pos)
                    app.update_inspector(pos)
                    await pilot.pause()
                else:
                    await pilot.press(s)
                    await pilot.pause()

    asyncio.run(go())


def main(which: list[str]) -> int:
    for workflow in which or sorted(SCENES):
        for cwd, argv, steps in SCENES[workflow]:
            os.chdir(ROOT)
            missing = need(Path(cwd), argv)
            if missing:
                print(f"skip {workflow}: {missing} is not on this machine")
                continue
            print(f"{workflow}: toeknife {' '.join(argv)}")
            shoot(workflow, Path(cwd), argv, steps)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
