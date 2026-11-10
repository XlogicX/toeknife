"""Judging a decode against plaintext the operator already has.

This is the half that decides whether to BELIEVE a table. The search that
proposes tables is a separate and
much more expensive thing; this is what makes any of them safe to look at, and it
applies just as well to a table that arrived by another route entirely -- from a
related capture, from a dictionary, or straight out of an intact header.

The whole mechanism rests on one observation. A match says out[a] == out[a-d].
That is a claim about the *parse*, and where the operator's plaintext covers both
ends it can be tested without knowing what any literal means. So the strongest
evidence here is assignment-free, and that half is kept separate.

Three findings shape what this reports, and none of them are decoration:

  **It abstains rather than guessing.** Below `FLOOR` testable assertions there
  is not enough evidence to have an opinion. A sweep of 89,247 dictionary
  tables and every one of the 55 that beat the rigid check was stopped by this
  refusal rather than by catching a contradiction -- so the refusal is the
  mechanism, not a footnote, and it is reported as its own verdict.

  **A byte is only as good as its chain.** A copy inherits the standing of what
  it copied. `classify` walks that through, which is why a byte can be `floating`
  while sitting in the middle of well-corroborated territory.

  **No single rate is attached to any of it.** Measuring the grounded
  class over two independent populations found the error rate is a property
  of the block: 0 wrong in 321,007 bytes on one page, 1 in 2,338 on another, with
  nothing predicting which. Counts are reported. A percentage is not.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# Provenance classes.
GIVEN = "given"          # the operator already had this byte
GROUNDED = "grounded"    # derived, and every link of its chain sits in reach
FLOATING = "floating"    # the parse implies it; its chain leaves the evidence
PRE = "pre"              # the chain reaches behind the start of what we hold
UNFILLED = "unfilled"    # no value: nothing determined it

# A bucket is the resolution at which reach is decided, and `SUPPORT` is how many
# tested assertions a bucket needs before it counts as reached. TOL is zero
# because a candidate is being judged rather than described: one violation in a
# bucket is reason enough to stop trusting the parse there.
BUCKET, SUPPORT, TOL = 1024, 8, 0.0

# Fewer testable assertions than this and the verdict is "cannot rule", never
# "looks fine". The floor was measured, and an 89,247-table sweep showed it is
# what actually stops false accepts.
FLOOR = 20


@dataclass
class Verdict:
    testable: int = 0            # assertions the plaintext could check at all
    ok: int = 0                  # of those, the ones that held
    bad: int = 0                 # of those, the ones that did not
    counts: dict = field(default_factory=dict)   # class -> byte count

    @property
    def ruled(self) -> bool:
        """Whether there was enough evidence to have an opinion at all."""
        return self.testable >= FLOOR

    @property
    def consistent(self) -> bool:
        return self.ruled and self.bad == 0

    def summary(self) -> str:
        if not self.ruled:
            return (f"cannot rule: {self.testable} testable assertion(s), "
                    f"{FLOOR} required. Too little plaintext here to have an "
                    f"opinion either way.")
        if self.bad:
            return (f"contradicted: {self.bad:,} of {self.testable:,} tested "
                    f"assertions fail. The parse disagrees with plaintext you "
                    f"supplied, so this table is wrong somewhere.")
        return (f"consistent: {self.ok:,} of {self.ok:,} tested assertions "
                f"hold. Consistent is not correct -- two tables have passed "
                f"this and still decoded wrong.")


def assertions(session, known: dict):
    """Every match assertion the plaintext can actually test.

    Yields (position, held) for each copied byte whose own position and whose
    source both fall inside the plaintext. Both ends must be known: with only
    the near end there is nothing to compare against but the candidate's own
    decode, which is the candidate vouching for itself.
    """
    for sym in session.trace.symbols:
        if sym.kind != "match" or not sym.length:
            continue
        d = sym.distance or 0
        for i in range(sym.length):
            a = sym.out_pos + i
            here, there = known.get(a), known.get(a - d)
            if here is None or there is None:
                continue
            yield a, here == there


def reach_map(session, known: dict, bucket: int = BUCKET,
              support: int = SUPPORT, tol: float = TOL) -> bytearray:
    """Which output positions the evidence corroborates, by neighbourhood.

    Grounding only the positions the plaintext already covers would be true and
    useless -- it would never extend past what the operator handed over. What is
    worth saying is regional: this neighbourhood carries enough tested assertions
    and they all hold, so a parse landing here has been corroborated.
    """
    ok, bad = defaultdict(int), defaultdict(int)
    for a, held in assertions(session, known):
        (ok if held else bad)[a // bucket] += 1
    end = session.tape.length
    reach = bytearray(max(end, 0))
    for k in set(ok) | set(bad):
        o, x = ok[k], bad[k]
        if o + x < support or x > tol * (o + x):
            continue
        for i in range(max(k * bucket, 0), min((k + 1) * bucket, len(reach))):
            reach[i] = 1
    return reach


def source_reach(session, known: dict, bucket: int = BUCKET,
                 support: int = SUPPORT, tol: float = TOL) -> bytearray:
    """Evidence carried BACKWARD along a match, into output no plaintext covers.

    `reach_map` can only mark territory the plaintext itself covers, because an
    assertion needs both ends known -- and an operator's plaintext is usually a
    window at one end, so on a large block that leaves most of the output with no
    evidence of any kind. This is the only thing here that reaches past it.

    A match landing at a known position says what the byte at its SOURCE must be,
    wherever that source sits, and the decode's own value there is something to
    compare it against. Where they agree in bulk, the parse at the source has
    been corroborated by evidence from somewhere else entirely.
    """
    tape = session.tape
    ok, bad = defaultdict(int), defaultdict(int)
    for sym in session.trace.symbols:
        if sym.kind != "match" or not sym.length:
            continue
        d = sym.distance or 0
        for i in range(sym.length):
            a = sym.out_pos + i
            here = known.get(a)
            if here is None:
                continue
            src = a - d
            if src < 0:
                continue                    # behind what we hold
            val = tape.value_at(src)
            if val is None:
                continue                    # nothing decoded there to compare
            (ok if val == here else bad)[src // bucket] += 1
    reach = bytearray(max(tape.length, 0))
    for k in set(ok) | set(bad):
        o, x = ok[k], bad[k]
        if o + x < support or x > tol * (o + x):
            continue
        for i in range(max(k * bucket, 0), min((k + 1) * bucket, len(reach))):
            reach[i] = 1
    return reach


def combined_reach(session, known: dict) -> bytearray:
    """Both directions. Forward marks what the plaintext covers; backward
    carries assertions into output it never touches, and the union is what a
    chain is checked against."""
    fwd = reach_map(session, known)
    bak = source_reach(session, known)
    n = max(len(fwd), len(bak))
    out = bytearray(n)
    for i in range(n):
        if (i < len(fwd) and fwd[i]) or (i < len(bak) and bak[i]):
            out[i] = 1
    return out


def _in(reach, a) -> bool:
    return 0 <= a < len(reach) and reach[a] == 1


def classify(session, known: dict, reach: bytearray | None = None) -> list[str]:
    """What each output byte is worth, walking the chain the way a copy does.

    A literal inside reach is grounded. A copy is grounded only if its SOURCE is
    grounded or given AND it lands in reach itself -- the measured false accept
    was exactly a corroborated position whose source was not, so the chain is
    checked rather than the destination.
    """
    if reach is None:
        reach = combined_reach(session, known)
    tape = session.tape
    n = tape.length
    out = [UNFILLED] * n
    for sym in session.trace.symbols:
        if sym.kind == "literal":
            p = sym.out_pos
            if 0 <= p < n:
                out[p] = (GIVEN if p in known else
                          GROUNDED if _in(reach, p) else FLOATING)
        elif sym.kind == "match" and sym.length:
            d = sym.distance or 0
            for i in range(sym.length):
                p = sym.out_pos + i
                if not (0 <= p < n):
                    continue
                if p in known:
                    out[p] = GIVEN
                    continue
                src = p - d
                if src < 0:
                    out[p] = PRE          # behind what we hold: not judged here
                    continue
                rooted = out[src] in (GIVEN, GROUNDED)
                out[p] = GROUNDED if (rooted and _in(reach, p)) else FLOATING
    return out


def judge(session, known: dict) -> Verdict:
    """The whole verdict: does the parse survive the plaintext, and what is
    each byte worth if it does."""
    v = Verdict()
    for _a, held in assertions(session, known):
        v.testable += 1
        if held:
            v.ok += 1
        else:
            v.bad += 1
    if v.ruled:
        classes = classify(session, known)
        counts = defaultdict(int)
        for c in classes:
            counts[c] += 1
        v.counts = dict(counts)
    return v


def render(session, known: dict) -> str:
    """The verdict as text, with counts and never a rate."""
    v = judge(session, known)
    out = [f"known plaintext: {len(known):,} byte(s) supplied", "",
           v.summary()]
    if v.counts:
        out += ["", "what each recovered byte is worth:"]
        for name in (GIVEN, GROUNDED, FLOATING, PRE, UNFILLED):
            if v.counts.get(name):
                out.append(f"  {name:<10} {v.counts[name]:>10,}")
        out += ["",
                "  given     you supplied it",
                "  grounded  every link of its chain sits where the evidence "
                "reaches",
                "  floating  the parse implies it, and nothing corroborates it",
                "",
                "  A grounded byte is very probably right. How "
                "often it is",
                "  wrong is a property of the block and not of the method -- "
                "measured from",
                "  0 wrong in 321,007 on one page to 1 in 2,338 on another, "
                "with nothing",
                "  known that predicts which case you are in."]
    return "\n".join(out)
