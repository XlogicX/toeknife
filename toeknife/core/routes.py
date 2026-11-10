"""Every route to a lost table, what it needs, and what it costs.

The tool's rule is that facts and hypotheses never look alike.  This is that rule
applied one level up, to the *methods*: an operator should be able to see which
routes exist, which ones apply to the capture in front of them, and **why the
others do not** -- before anything runs.

Two failures this file exists to prevent, both of which have already happened:

  A route that silently does not apply.  The alphabet gate's precondition is
  a hard check rather than a comment, because off the first
  block that gate rejects the TRUE table 92% of the time, and a rejected truth
  looks exactly like a filtered candidate.  Silence there costs the operator the
  right answer with no way to notice.

  A route that is missing rather than refused.  Some routes were measured
  during development and are not in this release.  Left out of this list they
  would be indistinguishable from routes that do not apply, which is the same
  failure wearing different clothes.  They are listed, and they say so.

Costs are measured on a capture of roughly 100-200 KB with 12 workers where the
route is parallel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# Where a route stands relative to the shipped package.
WIRED = "wired"           # runs from the CLI or TUI today
RESEARCH = "research"     # measured during development, not in this release

# What kind of leverage a route uses, which is what an operator is really
# choosing between.
READ = "read"             # get more out of what is already here
BORROW = "borrow"         # take the table from somewhere else
REPAIR = "repair"         # the header is present but damaged
SEARCH = "search"         # look for the table, with plaintext to score against
FILTER = "filter"         # not a route: something that removes candidates


@dataclass(frozen=True)
class Route:
    key: str
    name: str
    family: str
    needs: str                       # the evidence it requires, in one line
    cost: str                        # measured, per capture
    source: str                      # where the numbers came from, or ""
    status: str
    check: Callable[[object], tuple[bool, str]]

    def verdict(self, session) -> "Verdict":
        try:
            ok, why = self.check(session)
        except Exception as e:                      # a broken check must not
            ok, why = False, f"check failed: {e}"   # look like inapplicability
        if ok and self.status == RESEARCH:
            return Verdict(self, False, "applies here, but this route is not "
                                        "in this release",
                           blocked_by_status=True)
        return Verdict(self, ok, why)


@dataclass(frozen=True)
class Verdict:
    route: Route
    ok: bool
    reason: str
    blocked_by_status: bool = False


# ---- the checks -------------------------------------------------------------
#
# Each returns (applicable, reason).  The reason is written for the operator and
# is expected to be shown when it is negative -- it is the whole point.

def _blocks(session):
    tr = getattr(session, "trace", None)
    return list(getattr(tr, "blocks", []) or [])


def _dynamic(session):
    return [b for b in _blocks(session) if getattr(b, "btype", -1) == 2]


def _always(_session):
    return True, "always available"


def _needs_resync(session):
    tr = getattr(session, "trace", None)
    status = getattr(tr, "status", "")
    if status != "end":
        return True, f"the stream ends as {status!r}, so a better start may exist"
    holes = 0
    tape = getattr(session, "tape", None)
    if tape is not None:
        try:
            holes = tape.stats().get("unknown", 0)
        except Exception:
            holes = 0
    if holes:
        return True, f"{holes:,} positions are still unknown"
    return True, "decodes cleanly, but an earlier boundary may recover more"


def _has_dynamic(session):
    dyn = _dynamic(session)
    if not dyn:
        kinds = {getattr(b, "btype_name", "?") for b in _blocks(session)}
        return False, ("no dynamic block here"
                       + (f" (found {', '.join(sorted(kinds))})" if kinds else ""))
    return True, f"{len(dyn)} dynamic block(s)"


def _table_missing(session):
    """A route that recovers a table is pointless when one is already present."""
    ok, why = _has_dynamic(session)
    if ok:
        intact = [b for b in _dynamic(session) if getattr(b, "litlen", None)]
        if intact:
            return False, (f"the table is already present and readable on "
                           f"{len(intact)} block(s); nothing to recover")
        return True, why
    if getattr(session, "litlen", None) is not None:
        return False, "a table was supplied on the command line"
    return True, "no readable table in this capture"


def _first_block_only(session):
    """The alphabet gate's precondition, and why it is load-bearing.

    `out_start == 0` is NOT enough on its own, and the trap is worth spelling
    out: after a resync the decode begins mid-stream, so its first block reports
    out_start 0 while real output precedes it that this view never saw. Bytes
    copied from that output need not be in the block's alphabet -- which is
    precisely the case the gate is unsound on. So the decode must also have
    started where the data does.
    """
    dyn = _dynamic(session)
    if not dyn:
        return False, "no dynamic block to gate"
    started_at = getattr(session, "bit_offset", 0)
    container = getattr(session, "container", None)
    origin = getattr(container, "bit_offset", 0) or 0
    if started_at != origin:
        return False, (f"this view starts {started_at - origin:,} bits into the "
                       f"data, so its first block is not the stream's -- output "
                       f"before it can supply bytes the block's own table "
                       f"cannot code, and the gate rejects the TRUE table on "
                       f"92% of such blocks")
    first = [b for b in dyn if getattr(b, "out_start", 0) == 0]
    if not first:
        starts = ", ".join(f"{getattr(b, 'out_start', 0):,}" for b in dyn[:3])
        return False, (f"unsound off the first block -- it rejects the TRUE "
                       f"table on 92% of mid-stream blocks, and these start at "
                       f"output {starts}")
    return True, "the decode starts at the data, on a block at output 0"


def _needs_second_capture(session):
    if getattr(session, "related", None):
        return True, "a related capture was supplied"
    return False, ("needs a second capture of the same or similar content; "
                   "supply one and its token overlap decides the odds")


def _needs_plaintext(session):
    if getattr(session, "known_plaintext", None):
        return True, "plaintext was supplied"
    return False, ("needs known plaintext to score candidates against; without "
                   "it there is nothing to tell a right table from a wrong one")


def _needs_block_end(session):
    """Repair routes need the block's end, which is what C1 tests against."""
    ok, why = _table_missing(session)
    if not ok:
        return False, why
    dyn = _dynamic(session)
    if dyn and any(getattr(b, "bit_end", 0) for b in dyn):
        return True, "the block's end is known, so a candidate can be tested"
    return False, ("the block's end is not known here; every repair route tests "
                   "candidates by landing exactly on it")


def _decode_failed(session):
    """Bit rot applies when the bits are all present and the parse still fails.

    Not `_table_missing`: on a rotted header the table is right there, which is
    the whole distinction -- a correction problem rather than a completion one.
    What says rot is that the decode did not land, and what makes the search
    testable is that there is a header to enumerate flips over.
    """
    tr = getattr(session, "trace", None)
    status = getattr(tr, "status", "")
    if status in ("end", "eob"):
        return False, (f"this capture already decodes cleanly (ends {status!r}), "
                       f"so there is no damage here to undo")
    if not _blocks(session):
        return False, ("nothing parsed at all, so there is no header to "
                       "enumerate flips over -- f2 or f3 first")
    if all(getattr(b, "btype", -1) == -1 for b in _blocks(session)):
        return False, ("a fragment read without its header has no header "
                       "here to repair; f3 finds the next one")
    from . import bitrot
    if not bitrot.spans(session):
        return False, "no block header could be bounded to search"
    lo, hi = bitrot.spans(session)[0]
    return True, (f"the parse ends as {status!r} with {hi - lo:,} header bits "
                  f"to try one-bit flips over")


def _stopped_on_damage(session):
    from . import resume
    return resume.applies(session)


ROUTES: tuple[Route, ...] = (
    Route("decode", "Decode and tape what is here", READ,
          "nothing", "milliseconds", "the engine", WIRED, _always),
    Route("resume", "Read on past the damage", READ,
          "a recorded decompressed size", "well under a second",
          "", WIRED, _stopped_on_damage),
    Route("resync", "Scan for a better block boundary", READ,
          "nothing", "4.6 s bounded, 14.6 s exhaustive", "", WIRED,
          _needs_resync),
    Route("related", "Take the table from a related capture", BORROW,
          "a second capture", "1-2 s", "", WIRED,
          _needs_second_capture),
    Route("dict-head", "Sweep the 1,828-table dictionary", BORROW,
          "the table census", "1.3 s", "", RESEARCH,
          _table_missing),
    Route("dict-full", "Sweep the full 89,247-table dictionary", BORROW,
          "the census plus its tail", "83 s", "", RESEARCH,
          _table_missing),
    Route("midheader", "Fill a header lost from the front", REPAIR,
          "the block's end", "seconds at 17 bits, ~30 s at 24",
          "", RESEARCH, _needs_block_end),
    # Wired. The rates were measured by classifying each candidate against the
    # true output, which is exactly what an operator does not have;
    # core/bitrot.py is the same enumeration with the answer key removed. Measured on 250 single-bit flips over a real header:
    # the true repair survives every time, it is alone 129 times, and it is the
    # first of the survivors 78% of the time. So the list is ordered and never
    # picked from -- 22% of the time the right answer is not at the top.
    Route("bitrot", "Repair a header whose bits are wrong", REPAIR,
          "a parse that fails with its bits present",
          "~0.1 s at one flip over a few hundred bits", "",
          WIRED, _decode_failed),
    # When this one is wired, its output must NOT carry a single reliability
    # figure. The grounded class was measured over two independent
    # populations and found the error rate is a property of the block, not of
    # the method: 0 wrong in 321,007 grounded bytes on one page, 1 in 2,338 on
    # another, with nothing predicting which you are on. Any badge quoting one
    # number would be a lie on the first page and needlessly pessimistic on the
    # second. Report the classes and their counts; if a rate is shown at all,
    # show the measured range and say it varies by block.
    Route("kp-judge", "Judge this table against known plaintext", SEARCH,
          "operator-supplied plaintext", "instant",
          "", WIRED, _needs_plaintext),
    # Measured and closed rather than left pending. Three seedings
    # have been measured -- a real sibling capture, fresh near misses, and a
    # dictionary near-hit -- and none pay. The reason is not the seeding: the
    # only score an operator can compute correlates +0.37 with being FURTHER
    # from the truth, so the search cannot be steered. It stays listed because
    # an operator asking "why not just search for it" deserves the answer.
    Route("kp-climb", "Search for a table against known plaintext", SEARCH,
          "operator-supplied plaintext", "2.5 min per climb, dozens wanted",
          "measured, does not pay", RESEARCH,
          _needs_plaintext),
    Route("alphabet-gate", "Drop candidates the block's bytes rule out", FILTER,
          "a first block", "instant, removes 97-99% of a sweep",
          "", RESEARCH, _first_block_only),
)

BY_KEY = {r.key: r for r in ROUTES}


def board(session) -> list[Verdict]:
    """Every route, judged against this capture, in the order they'd be tried."""
    return [r.verdict(session) for r in ROUTES]


def render(session, width: int = 78) -> str:
    """The board as text, for the CLI and for the TUI modal to reuse."""
    out = ["Routes to a lost table, for this capture:", ""]
    fam_seen = set()
    for v in board(session):
        r = v.route
        if r.family not in fam_seen:
            fam_seen.add(r.family)
            out.append(f"  [{r.family}]")
        mark = "OK " if v.ok else ("..." if v.blocked_by_status else "no ")
        out.append(f"    {mark} {r.name}")
        out.append(f"        needs {r.needs} | {r.cost}"
                   + (f" | {r.source}" if r.source else ""))
        if not v.ok:
            out.append(f"        {v.reason}")
    out += ["", "  OK = available now   ... = not in this release   "
                "no = does not apply here",
            "  A route that does not apply says why. None of them run "
            "without being asked."]
    return "\n".join(out)
