"""What was tried, with what, and what came back -- including the refusals.

A recovery tool that only reports its successes is telling the operator half of
what happened, and in this project it is the wrong half. A sweep of 89,247
dictionary tables against 300 blocks: 55 tables got past the rigid verifier, and
**every one of them was stopped by the reconstruction test ABSTAINING** -- there
was too little evidence to rule, so it refused -- rather than by catching a
contradiction. An operator who sees "nothing found" cannot tell that apart from
"never looked", and the thing that actually protected them is invisible.

So a refusal is a first-class outcome here, and it carries its own numbers.

The other half is reproducibility. Every entry records the parameters it ran
with, seeds included, because one measured find cannot be replayed: adding a file
to a directory shifted a glob position, which changed a seed, which changed the
search. A forensic result that cannot be re-derived is a story.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

# Outcome kinds. The two refusals are deliberately distinct: one means the
# evidence contradicted the candidate, the other means there was not enough
# evidence to have an opinion, and they say completely different things about
# how much the operator should trust the silence.
FOUND = "found"                  # a table, and what vouched for it
NOTHING = "nothing"              # ran, ruled on candidates, none passed
ABSTAINED = "abstained"          # ran, could not rule: too little evidence
CONTRADICTED = "contradicted"    # ran, ruled against: the evidence disagreed
SKIPPED = "skipped"              # did not run, and why
FAILED = "failed"                # errored

_TEXT = {
    FOUND: "found",
    NOTHING: "nothing found",
    ABSTAINED: "REFUSED to rule",
    CONTRADICTED: "ruled against",
    SKIPPED: "not run",
    FAILED: "failed",
}


@dataclass
class Entry:
    route: str
    outcome: str
    detail: str = ""
    params: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    seconds: float = 0.0
    when: float = field(default_factory=time.time)

    def line(self) -> str:
        bits = [f"{self.route}: {_TEXT.get(self.outcome, self.outcome)}"]
        if self.detail:
            bits.append(self.detail)
        if self.counts:
            bits.append(", ".join(f"{k} {v:,}" if isinstance(v, int) else
                                  f"{k} {v}" for k, v in self.counts.items()))
        if self.params:
            bits.append("(" + ", ".join(f"{k}={v}" for k, v in
                                        sorted(self.params.items())) + ")")
        if self.seconds:
            bits.append(f"{self.seconds:.1f}s")
        return "  ".join(bits)


class Ledger:
    """Append-only. Nothing is removed, because a run that was undone still ran."""

    def __init__(self):
        self.entries: list[Entry] = []

    def add(self, route: str, outcome: str, detail: str = "", **kw) -> Entry:
        e = Entry(route=route, outcome=outcome, detail=detail,
                  params=kw.pop("params", {}) or {},
                  counts=kw.pop("counts", {}) or {},
                  seconds=kw.pop("seconds", 0.0))
        self.entries.append(e)
        return e

    def abstained(self, route: str, have: int, need: int, **kw) -> Entry:
        """The refusal that matters most, with the numbers that explain it."""
        return self.add(route, ABSTAINED,
                        f"{have:,} assertions available, {need:,} required",
                        **kw)

    def __len__(self) -> int:
        return len(self.entries)

    def render(self) -> str:
        if not self.entries:
            return ("Nothing has been run against this capture yet.\n"
                    "Every route is listed on the route board, including the "
                    "ones that do not apply.")
        out = ["What has been run, in order:", ""]
        out += [f"  {i + 1}. {e.line()}" for i, e in enumerate(self.entries)]
        refused = sum(1 for e in self.entries if e.outcome == ABSTAINED)
        if refused:
            out += ["", f"  {refused} of these REFUSED to rule. That is a",
                    "  different statement: it means the evidence was too thin "
                    "to have an opinion,",
                    "  not that the candidate was checked and rejected."]
        return "\n".join(out)

    def to_json(self) -> str:
        return json.dumps([{
            "route": e.route, "outcome": e.outcome, "detail": e.detail,
            "params": e.params, "counts": e.counts,
            "seconds": round(e.seconds, 3), "when": e.when,
        } for e in self.entries], indent=1)
