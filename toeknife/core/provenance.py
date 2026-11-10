"""Why is this byte what it is: the chain behind one output position.

The tape already colours a byte by how it was obtained, and the tool's rule is
that facts and hypotheses never share a colour. This answers the next question,
which is the one an operator actually asks when a byte looks wrong: *what is it
resting on?*

A literal rests on bits in the stream. A copy rests on an earlier byte, which may
itself be a copy, which may bottom out in a literal, a guess the operator typed,
a model fill, or nothing at all. That chain is the whole story of the byte, and
it is the same structure that decides whether a byte may be vouched for: a copy
is only ever as good as its source.

This shows what was *checked* rather than a verdict because of one measured
failure. Its cause C is a byte where every piece of visible evidence agrees and the answer
is still wrong -- so a confident-looking verdict would have been worth nothing,
while the chain at least shows how thin the support is.
"""
from __future__ import annotations

from dataclasses import dataclass

from .tape import PROV_NAMES, UNKNOWN

# A chain can be long on repetitive data, and an operator reading a pane does not
# want three thousand links. The tail matters less than the root, so walk from
# the byte, keep this many, and say how many were skipped.
DEFAULT_LIMIT = 12


@dataclass(frozen=True)
class Link:
    pos: int
    value: int | None
    prov: int
    kind: str                    # "literal", "copy", "guess", "model", "unknown"
    detail: str = ""
    distance: int = 0            # for a copy, how far back it reads
    source: str = ""             # for a guess: the file it came from

    @property
    def prov_name(self) -> str:
        return PROV_NAMES.get(self.prov, "?")


@dataclass(frozen=True)
class Chain:
    links: list[Link]
    truncated: int = 0           # links not shown, between the tail and the root
    cycle: bool = False

    @property
    def root(self) -> Link | None:
        return self.links[-1] if self.links else None

    def rests_on(self) -> str:
        """One line: what the byte ultimately depends on."""
        r = self.root
        if r is None:
            return "nothing selected"
        if len(self.links) == 1:
            if r.kind == "unknown":
                return "not recovered: nothing here has determined it yet"
            if r.kind == "window":
                return "window space, from before the fragment begins"
            if r.kind == "unproven":
                from .tape import BORROWED, GAPPED
                if r.prov == GAPPED:
                    return ("copied across a gap whose width was searched "
                            "for: unproven until a copy matches it")
                if r.prov == BORROWED:
                    return ("decoded with a table this stream's bits did not "
                            "supply: unproven until a copy matches it")
                return ("decoded near damage, so it may be garbage: unproven "
                        "until a copy matches the checksum")
            if r.kind == "guess" and r.source:
                return f"put in from {r.source}, not read from the stream"
            if r.kind in ("guess", "model"):
                return f"set by a {r.kind}, not read from the stream"
            return f"read directly from the stream as a {r.kind}"
        hops = len(self.links) - 1 + self.truncated
        what = (f"put in from {r.source}" if r.kind == "guess" and r.source
                else r.kind)
        return (f"copied through {hops} hop(s) from out {r.pos}, "
                f"which is {what}")


def _kind_for(sym, prov: int) -> str:
    from .tape import GUESS, GUESS_PROP, SOLVED, SOLVED_PROP, UNPROVEN
    if prov in UNPROVEN:
        return "unproven"
    if prov in (GUESS, GUESS_PROP):
        return "guess"
    if prov in (SOLVED, SOLVED_PROP):
        return "model"
    if prov == UNKNOWN:
        return "unknown"
    if sym is not None and getattr(sym, "kind", "") == "match":
        return "copy"
    return "literal"


def chain(session, pos: int, limit: int = DEFAULT_LIMIT) -> Chain:
    """Walk back from `pos` to whatever the byte finally rests on."""
    tape = session.tape
    trace = session.trace
    links: list[Link] = []
    seen: set[int] = set()
    cur = pos
    truncated = 0
    cycle = False
    while True:
        if cur in seen:                 # a match cannot legally read itself, so
            cycle = True                # this means a damaged parse, not a bug
            break
        seen.add(cur)
        sym = trace.symbol_at_out(cur) if cur >= 0 else None
        prov = tape.prov_at(cur)
        val = tape.value_at(cur)
        kind = _kind_for(sym, prov)
        dist = 0
        detail = ""
        if cur < 0:
            kind = "window" if val is not None else "unknown"
            detail = "before the fragment starts"
        elif sym is not None and getattr(sym, "kind", "") == "match":
            dist = sym.distance
            detail = f"copies {sym.length} byte(s) from {dist} back"
        elif sym is not None:
            detail = f"literal at bit {sym.bit_start}"
        links.append(Link(cur, val, prov, kind, detail, dist))
        if kind != "copy" or dist <= 0:
            break
        if len(links) >= limit:
            # Keep going far enough to name the root, but do not keep the middle.
            src = cur - dist
            hops = 0
            while hops < 4096:
                s2 = trace.symbol_at_out(src) if src >= 0 else None
                if src < 0 or s2 is None or getattr(s2, "kind", "") != "match":
                    break
                src -= s2.distance
                hops += 1
            truncated = hops
            p2 = tape.prov_at(src)
            v2 = tape.value_at(src)
            s2 = trace.symbol_at_out(src) if src >= 0 else None
            links.append(Link(src, v2, p2, _kind_for(s2, p2),
                              "the root of the copy chain", 0))
            break
        cur = cur - dist
    if links and links[-1].kind == "guess" and hasattr(session, "source_of"):
        from dataclasses import replace
        links[-1] = replace(links[-1], source=session.source_of(links[-1].pos))
    return Chain(links, truncated, cycle)


def render(session, pos: int, limit: int = DEFAULT_LIMIT) -> str:
    """The chain as text, for the inspector and for `--why`."""
    c = chain(session, pos, limit)
    if not c.links:
        return "nothing at that position"
    out = [f"out {pos}: {c.rests_on()}", ""]
    for i, ln in enumerate(c.links):
        arrow = "    " if i == 0 else " <- "
        val = "--" if ln.value is None else f"{ln.value:02x}"
        out.append(f"{arrow}out {ln.pos:<9} {val}  {ln.kind:<8} "
                   f"[{ln.prov_name}]")
        if ln.detail:
            out.append(f"        {ln.detail}")
        if i == limit - 1 and c.truncated:
            out.append(f"        ... {c.truncated} further hop(s) not shown")
    if c.cycle:
        out.append("\n  the chain loops, which means the parse is damaged here")
    root = c.root
    if root is not None and root.kind in ("guess", "model", "unknown"):
        what = (f"came from {root.source}" if root.source
                else f"is {root.prov_name}")
        out.append(f"\n  This byte is only as good as out {root.pos}, which "
                   f"{what}.")
        out.append("  Change that one and this byte changes with it.")
    return "\n".join(out)
