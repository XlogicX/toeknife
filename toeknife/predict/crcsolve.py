"""Solving unknown bytes from the container's CRC32, with linear algebra.

CRC32 is affine over GF(2). For messages of one length,

    CRC(a xor b xor c) == CRC(a) xor CRC(b) xor CRC(c)

so unknown bytes enter the checksum *linearly*, and recovering them is a linear
system rather than a search. Thirty-two equations means eight unknown bytes'
worth of bits is already overdetermined and four is generically exact.

The unknowns are hole classes, not positions, which is what makes this worth
doing. A class appearing at three hundred places contributes one variable, not
three hundred, because every occurrence holds the same byte. A stream with
eleven thousand unknown positions across thirty-five classes is a system in two
hundred and eighty unknowns, and thirty-two equations bite hard on that.

Classes living entirely before output position zero never enter the checksum,
because the CRC covers the decompressed output and nothing behind it. They come
back as free variables, which is correct: the checksum genuinely says nothing
about them.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

CRC_POLY_REV = 0xEDB88320
# Past this many open classes the system is not built at all; the model has to
# close some first. The what-next board quotes it, so it is named once here.
MAX_CLASSES = 512
GF2_DIM = 32


# ---- the linear part of CRC32, and shifting it in O(log n) -----------------
#
# Write L(m) = CRC(m) xor CRC(zeros of the same length).  L is linear, and it
# has two properties that make the basis cheap to build:
#
#   L(zeros) = 0
#   L(A || B) = shift(L(A), len(B)) xor L(B)
#
# So for a message that is all zeros except one byte v at position p, every
# zero-run term vanishes and the whole thing reduces to
#
#   L = shift(L(v), length - p - 1)
#
# one shift of a constant, with no dependence on p except through the shift
# distance.  Shifting is a linear operator applied by repeated squaring, which
# is the machinery inside zlib's crc32_combine that CPython does not expose.
#
# The alternative, CRC-ing a full-length probe buffer per unknown bit, is
# O(message) each and hopeless once a stream has thousands of unknown positions.

def _gf2_times(mat: list[int], vec: int) -> int:
    out = 0
    i = 0
    while vec and i < len(mat):
        if vec & 1:
            out ^= mat[i]
        vec >>= 1
        i += 1
    return out


def _gf2_square(mat: list[int]) -> list[int]:
    return [_gf2_times(mat, mat[n]) for n in range(GF2_DIM)]


_ZERO_OPS: list[list[int]] = []      # _ZERO_OPS[k] appends 2**k zero BYTES


def _ensure_ops(upto: int) -> None:
    """Operators for appending 2**k zero bytes, built once and kept.

    The operators depend only on the power of two, never on the CRC being
    shifted, so rebuilding them per call (which is what a literal transcription
    of crc32_combine does) wastes the entire saving.
    """
    if not _ZERO_OPS:
        odd = [CRC_POLY_REV] + [1 << (n - 1) for n in range(1, GF2_DIM)]  # 1 bit
        even = _gf2_square(odd)                                           # 2 bits
        odd = _gf2_square(even)                                           # 4 bits
        _ZERO_OPS.append(_gf2_square(odd))                                # 8 bits
    while len(_ZERO_OPS) <= upto:
        _ZERO_OPS.append(_gf2_square(_ZERO_OPS[-1]))


def shift(crc: int, nbytes: int) -> int:
    """Apply the append-nbytes-of-zeros operator to a CRC register.

    This is the linear operator alone.  It is NOT CRC(message || zeros), which
    additionally xors in the CRC of the zero run itself; in the L formulation
    above that term always cancels, so it is deliberately not included here.
    """
    if nbytes <= 0:
        return crc
    _ensure_ops(nbytes.bit_length())
    i = 0
    while nbytes:
        if nbytes & 1:
            crc = _gf2_times(_ZERO_OPS[i], crc)
        nbytes >>= 1
        i += 1
    return crc


_ZERO_BYTE_CRC = zlib.crc32(b"\x00")


def linear_of_byte(value: int) -> int:
    """L of a one-byte message."""
    return zlib.crc32(bytes([value])) ^ _ZERO_BYTE_CRC


def basis(value: int, pos: int, length: int) -> int:
    """L of a message that is all zeros except `value` at `pos`."""
    return shift(linear_of_byte(value), length - pos - 1)


# ---- the linear system -----------------------------------------------------

@dataclass
class CrcSystem:
    anchors: list[int] = field(default_factory=list)   # one per class, in column order
    columns: list[int] = field(default_factory=list)   # 8 per class, low bit first
    target: int = 0
    length: int = 0
    free_classes: list[int] = field(default_factory=list)   # never touch the output

    @property
    def unknown_bits(self) -> int:
        return len(self.columns)


def build_system(tape, expected_crc: int, holes=None) -> CrcSystem:
    """Express "the CRC comes out right" as a GF(2) system in the unknown bits."""
    length = len(tape)
    holes = list(holes if holes is not None else tape.holes())
    sysm = CrcSystem(length=length)

    known = bytearray(length)
    for p in range(length):
        v = tape.value_at(p)
        if v is not None:
            known[p] = v
    # CRC(full) == CRC(known) xor L(unknown), because the linear part of the two
    # messages differs only where the unknown bytes sit.
    sysm.target = expected_crc ^ (zlib.crc32(bytes(known)) & 0xFFFFFFFF)

    for hole in holes:
        inside = [p for p in hole.positions if 0 <= p < length]
        if not inside:
            sysm.free_classes.append(hole.anchor)
            continue
        sysm.anchors.append(hole.anchor)
        for bit in range(8):
            col = 0
            for p in inside:
                col ^= basis(1 << bit, p, length)
            sysm.columns.append(col)
    return sysm


def _reduce(sysm: CrcSystem):
    """Row-reduce to (particular solution, null-space basis), or None if
    inconsistent."""
    ncol = len(sysm.columns)
    rows = [[(sysm.columns[c] >> r) & 1 for c in range(ncol)] + [(sysm.target >> r) & 1]
            for r in range(GF2_DIM)]
    pivots: list[int] = []
    r = 0
    for c in range(ncol):
        pick = next((i for i in range(r, GF2_DIM) if rows[i][c]), None)
        if pick is None:
            continue
        rows[r], rows[pick] = rows[pick], rows[r]
        for i in range(GF2_DIM):
            if i != r and rows[i][c]:
                rows[i] = [a ^ b for a, b in zip(rows[i], rows[r])]
        pivots.append(c)
        r += 1
        if r == GF2_DIM:
            break
    for i in range(r, GF2_DIM):
        if rows[i][ncol] and not any(rows[i][:ncol]):
            return None

    particular = [0] * ncol
    for i, c in enumerate(pivots):
        particular[c] = rows[i][ncol]

    # One null-space vector per free column: set that column, and cancel what it
    # forces on each pivot column.
    pivot_set = set(pivots)
    nullspace = []
    for f in range(ncol):
        if f in pivot_set:
            continue
        vec = [0] * ncol
        vec[f] = 1
        for i, c in enumerate(pivots):
            vec[c] = rows[i][f]
        nullspace.append(vec)
    return particular, nullspace


def _to_assignment(sysm: CrcSystem, bits: list[int]) -> dict:
    return {anchor: sum(bits[i * 8 + b] << b for b in range(8))
            for i, anchor in enumerate(sysm.anchors)}


def solve(sysm: CrcSystem):
    """Gaussian elimination over GF(2).

    Returns (assignment, free_bits), or (None, 0) when the system is
    inconsistent, which means a byte already treated as known is wrong.
    """
    ncol = len(sysm.columns)
    if ncol == 0:
        return ({}, 0) if sysm.target == 0 else (None, 0)
    reduced = _reduce(sysm)
    if reduced is None:
        return None, 0
    particular, nullspace = reduced
    return _to_assignment(sysm, particular), len(nullspace)


def enumerate_solutions(sysm: CrcSystem, limit: int = 1 << 20):
    """Every assignment consistent with the CRC, when few enough remain.

    Thirty-two equations rarely pin an answer outright, but they cut the space
    hard: five unknown classes are 2^40 possibilities before the checksum and
    2^8 after it. Enumerating what survives and letting a model choose is a very
    different problem from searching the original space.
    """
    reduced = _reduce(sysm)
    if reduced is None:
        return
    particular, nullspace = reduced
    if len(nullspace) > 30 or (1 << len(nullspace)) > limit:
        return
    ncol = len(sysm.columns)
    for mask in range(1 << len(nullspace)):
        bits = list(particular)
        m, i = mask, 0
        while m:
            if m & 1:
                vec = nullspace[i]
                for c in range(ncol):
                    bits[c] ^= vec[c]
            m >>= 1
            i += 1
        yield _to_assignment(sysm, bits)


@dataclass
class CrcResult:
    ok: bool = False
    reason: str = ""
    assignment: dict = field(default_factory=dict)
    free_bits: int = 0
    unknown_bits: int = 0
    classes: int = 0
    free_classes: int = 0

    def summary(self) -> str:
        if not self.ok:
            return self.reason
        if self.free_bits:
            return (f"{self.classes} classes / {self.unknown_bits} unknown bits "
                    f"against 32 equations: {self.free_bits} bits undetermined, "
                    f"2^{self.free_bits} solutions remain")
        return (f"solved {self.classes} classes exactly from the CRC "
                f"({self.unknown_bits} unknown bits, 32 equations)")


def unproven(session) -> str | None:
    """Why the checksum must not be solved against this tape, or None.

    Bytes decoded near damage are not facts, and some of them are garbage. Solve
    the open bytes against the checksum around them and the solver picks
    whatever makes the garbage add up: the checksum then passes on bytes that
    are wrong, and reads as proof.
    """
    from ..core.tape import UNPROVEN
    if any(code in session.tape.prov for code in UNPROVEN):
        return ("bytes decoded near the damage are unproven, and solving the "
                "checksum around them would make it agree with whatever they "
                "hold; a suspected copy (ctrl+k) can prove them")
    return None


def solve_from_crc(session, max_classes: int = MAX_CLASSES) -> CrcResult:
    """Try to pin the remaining unknown bytes using the container's CRC32."""
    why = unproven(session)
    if why:
        return CrcResult(reason=why)
    stream = getattr(session, "stream", None)
    if stream is None or stream.crc32 is None:
        return CrcResult(reason="no CRC32 recorded for this stream")
    tape = session.tape
    holes = tape.holes()
    if not holes:
        return CrcResult(reason="nothing unknown to solve for")
    if stream.uncompressed_size is not None and len(tape) != stream.uncompressed_size:
        return CrcResult(
            reason=f"output is {len(tape)} bytes but the container recorded "
                   f"{stream.uncompressed_size}; the CRC is over different data")
    if len(holes) > max_classes:
        return CrcResult(
            reason=f"{len(holes)} classes is past the {max_classes} the solver "
                   f"will attempt; fill some with the model first")

    sysm = build_system(tape, stream.crc32, holes)
    assignment, free_bits = solve(sysm)
    result = CrcResult(unknown_bits=sysm.unknown_bits,
                       classes=len(sysm.anchors),
                       free_classes=len(sysm.free_classes))
    if assignment is None:
        result.reason = ("the CRC contradicts the bytes already decoded: an "
                         "earlier guess is wrong, or the data is damaged")
        return result
    result.ok = True
    result.free_bits = free_bits
    result.assignment = assignment if free_bits == 0 else {}
    if free_bits:
        result.reason = result.summary()
    return result


def solve_with_model(session, model, max_classes: int = MAX_CLASSES,
                     limit: int = 1 << 20, sample: int = 12):
    """Let the CRC narrow the field, then let the model pick from what is left.

    Neither alone is enough on a real stream. The checksum removes 32 bits of
    freedom but rarely determines an answer; the model ranks well but has no way
    to be certain. Applied in that order the model only ever chooses between
    candidates that already produce the right checksum.
    """
    base = solve_from_crc(session, max_classes)
    if not base.ok:
        return base, None
    if not base.free_bits:
        return base, base.assignment

    stream = session.stream
    sysm = build_system(session.tape, stream.crc32, session.tape.holes())
    tape = session.tape
    order = getattr(model, "order", 4)

    # Where each class lands, computed once.  A class can occupy thousands of
    # positions, and scoring every one of them for every candidate is the whole
    # cost; a sample of each class is enough to rank candidates apart.
    placement: dict[int, list[int]] = {}
    for anchor in sysm.anchors:
        hole = tape.hole_at(anchor)
        if hole is not None:
            placement[anchor] = list(hole.positions[:sample])
    scored_positions = sorted({p for ps in placement.values() for p in ps})

    best, best_score = None, None
    count = 0
    for assignment in enumerate_solutions(sysm, limit):
        count += 1
        trial = {p: assignment[a] for a, ps in placement.items() for p in ps}
        total = 0.0
        for p in scored_positions:
            ctx = bytearray()
            for q in range(p - order, p):
                v = trial.get(q, tape.value_at(q))
                if v is None:
                    ctx.clear()
                else:
                    ctx.append(v)
            total += model.logprob(trial[p], bytes(ctx))
        if best_score is None or total > best_score:
            best, best_score = assignment, total
    base.reason = (f"{count} assignments satisfy the CRC; the model picked one"
                   if count else base.reason)
    return base, best


def _checksums(session):
    stream = getattr(session, "stream", None)
    if stream is None:
        return []
    out = []
    if stream.crc32 is not None:
        out.append(("crc32", stream.crc32, lambda b: zlib.crc32(b) & 0xFFFFFFFF))
    if stream.adler32 is not None:
        out.append(("adler32", stream.adler32, lambda b: zlib.adler32(b) & 0xFFFFFFFF))
    return out


@dataclass
class ShortlistResult:
    ok: bool = False
    reason: str = ""
    assignment: dict = field(default_factory=dict)
    tried: int = 0
    checked: str = ""
    survivors: int = 0
    truncated: bool = False       # stopped collecting survivors at the cap

    @property
    def ambiguous(self) -> bool:
        return self.survivors > 1

    def summary(self) -> str:
        if not self.ok:
            return self.reason
        more = "+" if self.truncated else ""
        if self.ambiguous:
            return (f"{self.survivors}{more} of {self.tried} combinations satisfy "
                    f"{self.checked}; the model picked among them, and the "
                    f"checksum did not decide")
        return (f"exactly one of {self.tried} combinations satisfies "
                f"{self.checked}")


def solve_by_shortlist(session, model, per_class: int = 4,
                       max_combinations: int = 200000) -> ShortlistResult:
    """Let the model shortlist each unknown byte, then let the checksum decide.

    The pure linear solve needs a CRC32 and gets nowhere on Adler-32, which is
    what zlib streams actually carry. This works with either.

    A surviving combination is not proof. CRC32 spreads 32 bits over the whole
    message and a wrong answer surviving it is unlikely. Adler-32 does not: it
    is two sums modulo 65521, and across k unknown bytes the reachable sums
    number in the low thousands, not the tens of thousands, so its effective
    strength is far below 32 bits. Measured here, five unknown bytes under
    Adler-32 alone produced a wrong five-byte answer that checksummed correctly,
    because `etesa` and `ensic` have the same byte sum and the same weighted
    sum. So every survivor is collected, the count is reported, and the model
    only breaks ties the checksum left open.
    """
    why = unproven(session)
    if why:
        return ShortlistResult(reason=why)
    checks = _checksums(session)
    if not checks:
        return ShortlistResult(reason="no checksum recorded for this stream")
    tape = session.tape
    holes = tape.holes()
    if not holes:
        return ShortlistResult(reason="nothing unknown to solve for")
    stream = session.stream
    if stream.uncompressed_size is not None and len(tape) != stream.uncompressed_size:
        return ShortlistResult(
            reason=f"output is {len(tape)} bytes but the container recorded "
                   f"{stream.uncompressed_size}; the checksum is over other data")

    total = per_class ** len(holes)
    if total > max_combinations:
        return ShortlistResult(
            reason=f"{len(holes)} classes at {per_class} candidates each is "
                   f"{total} combinations, past the {max_combinations} limit; "
                   f"fill some with the model first")

    from .solver import HoleSolver
    solver = HoleSolver(session, model)
    ranked = [(hole, solver.rank_values(hole, per_class)) for hole in holes]
    if any(not opts for _, opts in ranked):
        return ShortlistResult(reason="the model produced no candidates")

    n = len(tape)
    base = bytearray(n)
    for p in range(n):
        v = tape.value_at(p)
        if v is not None:
            base[p] = v
    slots = [[p for p in hole.positions if 0 <= p < n] for hole, _ in ranked]

    # Both checksums are linear in the bytes, so a candidate's effect can be
    # precomputed per (class, value) and combined in O(classes).  Recomputing a
    # whole checksum per combination is O(message) and makes anything past a few
    # thousand combinations impossible.
    deltas = []
    targets = []
    if stream.crc32 is not None:
        cols = []
        for positions in slots:
            per_bit = [0] * 8
            for bit in range(8):
                acc = 0
                for p in positions:
                    acc ^= basis(1 << bit, p, n)
                per_bit[bit] = acc
            cols.append(per_bit)

        def crc_delta(ci, value, _cols=cols):
            acc = 0
            for bit in range(8):
                if value >> bit & 1:
                    acc ^= _cols[ci][bit]
            return acc

        deltas.append(("crc32", crc_delta, lambda a, b: a ^ b, 0,
                       stream.crc32 ^ (zlib.crc32(bytes(base)) & 0xFFFFFFFF)))
    if stream.adler32 is not None:
        MOD = 65521
        counts = [len(ps) % MOD for ps in slots]
        weights = [sum(n - p for p in ps) % MOD for ps in slots]
        a_base = zlib.adler32(bytes(base)) & 0xFFFFFFFF
        a_want, b_want = stream.adler32 & 0xFFFF, stream.adler32 >> 16
        a_have, b_have = a_base & 0xFFFF, a_base >> 16

        def adler_delta(ci, value, _c=counts, _w=weights):
            return ((value * _c[ci]) % MOD, (value * _w[ci]) % MOD)

        deltas.append(("adler32", adler_delta,
                       lambda x, y: ((x[0] + y[0]) % MOD, (x[1] + y[1]) % MOD),
                       (0, 0),
                       ((a_want - a_have) % MOD, (b_want - b_have) % MOD)))

    import itertools
    tried = 0
    survivors = []
    cap = 256
    for combo in itertools.product(*[opts for _, opts in ranked]):
        tried += 1
        for _name, delta, combine, ident, want in deltas:
            acc = ident
            for ci, value in enumerate(combo):
                acc = combine(acc, delta(ci, value))
            if acc != want:
                break
        else:
            # Collect rather than return on the first hit.  A checksum with
            # fewer effective bits than the unknowns has many solutions, and
            # returning the first one found presents a guess as a certainty.
            survivors.append(combo)
            if len(survivors) >= cap:
                break

    checked = " and ".join(name for name, _, _, _, _ in deltas)
    if not survivors:
        return ShortlistResult(
            reason=f"none of {tried} combinations reproduce {checked}; the true "
                   f"byte is outside the model's top {per_class} somewhere",
            tried=tried, checked=checked)

    if len(survivors) == 1:
        best = survivors[0]
    else:
        order = getattr(model, "order", 4)
        placement = [[p for p in hole.positions[:12] if 0 <= p < n]
                     for hole, _ in ranked]

        def score(combo):
            trial = {p: v for v, ps in zip(combo, placement) for p in ps}
            total = 0.0
            for p in sorted(trial):
                ctx = bytearray()
                for q in range(p - order, p):
                    v = trial.get(q, tape.value_at(q))
                    if v is None:
                        ctx.clear()
                    else:
                        ctx.append(v)
                total += model.logprob(trial[p], bytes(ctx))
            return total

        best = max(survivors, key=score)

    return ShortlistResult(
        ok=True, tried=tried, checked=checked, survivors=len(survivors),
        truncated=len(survivors) >= cap,
        assignment={hole.anchor: value
                    for (hole, _), value in zip(ranked, best)})


def apply(session, result, provenance=None) -> list[int]:
    """Write a solved assignment onto the tape."""
    from ..core.tape import SOLVED
    prov = SOLVED if provenance is None else provenance
    touched = []
    for anchor, value in sorted(result.assignment.items()):
        if session.tape.value_at(anchor) is not None:
            continue
        touched.extend(session.set_guess(anchor, value, prov))
    return touched
