"""The stateful object the UI and the CLI both drive.

Holds the data, the chosen alignment, the decode, and the operator's guesses,
and keeps them consistent when any one of them changes.
"""
from __future__ import annotations

import binascii
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import align
from .events import Trace
from .huffman import HuffmanError, HuffmanTable
from .inflate import Decoder
from .ledger import Ledger
from .tape import GUESS, SOLVED, SOLVED_PROP, Tape

GZIP_MAGIC = b"\x1f\x8b"


@dataclass
class Container:
    """What wrapper, if any, the DEFLATE bits were found inside."""
    kind: str                  # 'raw' | 'zlib' | 'gzip'
    bit_offset: int
    note: str = ""


# How far a fragment read with the assumed fixed table must decode cleanly
# before its bytes count as read rather than assumed: measured, real
# fixed-table fragments 11,383 bytes and up, dynamic data read as fixed 1,336
# at most.
ASSUMED_CONFIRMS = 4096


def detect_container(data: bytes) -> Container:
    if data[:2] == GZIP_MAGIC and len(data) > 10 and data[2] == 8:
        flg = data[3]
        pos = 10
        if flg & 0x04 and pos + 2 <= len(data):        # FEXTRA
            xlen = data[pos] | (data[pos + 1] << 8)
            pos += 2 + xlen
        for bit in (0x08, 0x10):                        # FNAME, FCOMMENT
            if flg & bit:
                end = data.find(b"\x00", pos)
                pos = len(data) if end < 0 else end + 1
        if flg & 0x02:                                  # FHCRC
            pos += 2
        return Container("gzip", pos * 8, "gzip member header skipped")
    if len(data) >= 2:
        cmf, flg = data[0], data[1]
        if (cmf & 0x0F) == 8 and ((cmf << 8) | flg) % 31 == 0 and not (flg & 0x20):
            return Container("zlib", 16, "zlib CMF/FLG header skipped")
    return Container("raw", 0)


def tables_from_header(data: bytes, bit_pos: int = 0):
    """Build Huffman tables from a fragment that begins at a block header.

    This is the old --dynamic workflow: the table survived in one place and the
    compressed body in another.  Returns (litlen, dist, BlockInfo).
    """
    trace = Decoder(data, strict_window=False).stream(bit_pos, max_symbols=0)
    if not trace.blocks:
        raise ValueError("no block header found")
    block = trace.blocks[0]
    if block.litlen is None:
        raise ValueError(
            f"block at bit {bit_pos} is {block.btype_name}, which carries no Huffman table")
    return block.litlen, block.dist, block


@dataclass
class FillNote:
    """How a model fill came to be, kept for the operator to read later.

    Magenta says "a model suggested this" and nothing about how strongly, and
    the number the floor reads is not a probability. So
    each fill keeps what was known when it was made, and the side pane shows
    it beside what that confidence band was measured to deliver.
    """
    how: str                            # "n-gram", "byte model", "both", "f4", "checksum"
    confidence: float | None = None
    agreement: bool | None = None       # both modes: did the two name the same byte
    runners: tuple[int, ...] = ()       # values it beat, best first
    grounded: float | None = None       # share of its context that was evidence
    run: int = 0                        # unread run its best occurrence sat in
    floor: float | None = None          # the floor in force when it was filled
    precision: float | None = None      # measured on held-out text, when calibrated
    near: bool = False                  # filled next to an interior gap


@dataclass
class GuessRecord:
    pos: int
    value: int
    touched: list[int] = field(default_factory=list)
    provenance: int = GUESS


class Session:
    def __init__(self, data: bytes, mode: str = "auto",
                 litlen: HuffmanTable | None = None,
                 dist: HuffmanTable | None = None,
                 profile: str = "auto",
                 bit_offset: int | None = None,
                 max_offset: int = 48,
                 model=None,
                 stream=None):
        self.data = data
        self.profile = profile
        self.max_offset = max_offset
        self.model = model
        # The container entry these bytes came from, when there was one.  Carries
        # the recorded uncompressed size and checksum, which is what lets a
        # recovery be confirmed rather than eyeballed.
        self.stream = stream
        self.litlen = litlen
        self.dist = dist
        self.header_block = None
        self.container = detect_container(data)
        self.mode = self._resolve_mode(mode)
        self._fresh_state()
        self.bit_offset = 0
        self.trace: Trace = None            # type: ignore[assignment]

        if self.mode == "stream":
            # An explicit offset wins: it is how a caller resumes from a sync
            # point found part way into a damaged stream.
            self.bit_offset = (self.container.bit_offset if bit_offset is None
                               else bit_offset)
            self._decode()
        else:
            if bit_offset is not None:
                self.select_alignment(bit_offset)
            else:
                cands = self.candidates()
                # The tool's own pick reads only what every reading agrees on;
                # a start the operator names is honoured as named.
                self.select_alignment(cands[0].bit_offset if cands else 0,
                                      agreed=True)

    def _fresh_state(self) -> None:
        """Everything a session carries that is not derived from its input.

        Both constructors call this. It was written out twice, and the copy in
        `from_trace` was missing three fields -- so a gap-mode or code-set
        session had no ledger and no evidence slots, and any action that
        recorded itself crashed on one while working fine on an ordinary
        capture. `routes.py` still reaches for `related` through `getattr`,
        which is the scar from the same wound.
        """
        self._candidates: list[align.Candidate] | None = None
        self.guesses: dict[int, int] = {}
        self._records: dict[int, GuessRecord] = {}
        self.fill_notes: dict[int, FillNote] = {}
        # Where a fragment's decode really starts: the first boundary every
        # ranked reading agrees on (see _agreed_start). None when not a fragment.
        self.agreed_from: int | None = None
        self._use_agreed = False
        self._undo: list[tuple] = []
        self._redo: list[tuple] = []
        self._group: int | None = None
        self._group_seq = 0
        self.note = ""
        # Set when a fragment's bytes rest on the assumed fixed table.
        self.assumed_note = ""
        # What has been run against this capture. Append-only, and it records
        # refusals as loudly as finds -- see core/ledger.py for why that
        # distinction is the one an operator most needs.
        self.ledger = Ledger()
        # Evidence the operator has supplied. Routes check for these rather
        # than assuming: a route that cannot run says which of them is missing.
        self.related = None                 # a second capture of similar content
        self.known_plaintext = None         # {output position: byte}
        # Set when the decode was read on past damage (core/resume.py).
        self.resumed_from = None
        # Where the unproven run before a damage stop starts (core/headcheck.py).
        self.headcheck = None
        # The input as it arrived, before any repair, which is what a saved
        # session has to be loaded against.
        self.input_sha256 = hashlib.sha256(self.data).hexdigest()
        # Every step that changed the input or rebuilt the decode, in order,
        # with what it needs to run again: a saved session replays them.
        self.steps: list[dict] = []
        # Where an operator's byte came from, by the position it was set at.
        self.guess_sources: dict[int, str] = {}

    @classmethod
    def from_trace(cls, trace, stream=None, model=None, mode: str = "assembled"):
        """Wrap an already-built decode, as gap mode produces.

        An assembly is stitched from several fragments, so there is no single
        bit offset to re-decode from and no alignment to reconsider: the trace
        is the input, not something derived from one.
        """
        self = cls.__new__(cls)
        self.data = trace.data
        self.profile = "auto"
        self.max_offset = 48
        self.model = model
        self.stream = stream
        self.litlen = self.dist = None
        self.header_block = None
        # Where these bytes came from, when the caller knows. An assembly built
        # out of a packet capture is not raw, and the pane that reports the
        # container is the one place an operator looks to find out.
        kind = getattr(stream, "container", None) or "raw"
        self.container = Container(kind, 0)
        self.mode = mode
        self._fresh_state()
        # Not None, which would mean "not ranked yet": an assembly has no
        # alignment to reconsider, so the answer is that there are none.
        self._candidates = []
        self.bit_offset = trace.bit_start
        self.trace = trace
        return self

    # ---- mode ---------------------------------------------------------------

    def _resolve_mode(self, mode: str) -> str:
        if mode in ("stream", "fragment"):
            return mode
        if self.container.kind != "raw":
            return "stream"
        if self.stream is not None and self.stream.decodable:
            # A container entry starts where its header says, whatever happens
            # further in. Falling back to fragment ranking because damage
            # stopped the strict decode threw away a clean prefix: on a zip
            # entry with a 200-byte burst it recovered 125 bytes where the
            # stream recovers 24,853.
            return "stream"
        # A complete raw stream decodes from bit 0 all the way to a final block.
        # Allow outside references while testing, or a stream compressed against
        # a preset dictionary looks like a fragment and gets sent through
        # alignment ranking it does not need.
        strict = not bool(getattr(self.stream, "references_outside", False))
        try:
            t = Decoder(self.data, strict_window=strict).stream(0)
            if t.status == "end":
                return "stream"
            # A stream that lost its tail decodes cleanly from bit 0 until the
            # data runs out. Ranked as a fragment instead, a JSON stream cut
            # at two thirds gave 61 "known" bytes read with the fixed table,
            # every one wrong, where reading it as the stream it is gives all
            # 43,694 (wrap-up #2's matrix). A real headerless fragment does
            # not get this far: a reference before its start fails the strict
            # window, and garbage hits an invalid code long before the end.
            if (t.status == "truncated" and t.tape.length >= 64
                    and all(b.status in ("eob", "ok", "truncated")
                            for b in t.blocks)):
                return "stream"
        except Exception:
            pass
        return "fragment"

    # ---- alignment ----------------------------------------------------------

    def candidates(self) -> list[align.Candidate]:
        if self._candidates is None:
            if self.mode == "stream":
                self._candidates = []
            else:
                self._candidates = align.rank(
                    self.data, self.litlen, self.dist,
                    max_offset=self.max_offset, profile=self.profile,
                    model=self.model)
        return self._candidates

    def select_alignment(self, bit_offset: int, agreed: bool | None = False) -> None:
        """Decode the fragment from `bit_offset`.

        `agreed=True` starts at the first boundary every ranked reading agrees
        on instead (see `_agreed_start`); False honours the offset exactly, for
        an operator who knows the start; None keeps whichever was in force, so
        a re-decode after adopting a table does not change the rule under them.
        """
        self.bit_offset = bit_offset
        if agreed is not None:
            self._use_agreed = agreed
        self._decode()

    def agreement_point(self) -> int | None:
        """The first bit every ranked reading agrees on, or None."""
        from .gap import agreed_boundaries
        common = agreed_boundaries([c.trace for c in (self._candidates or [])])
        return common[0] if common else None

    # ---- steps a saved session replays --------------------------------------

    def record_step(self, op: str, **params) -> None:
        self.steps.append({"op": op, **params})

    def choose_start(self, bit_offset: int, agreed: bool | None = False) -> None:
        """`select_alignment` on the operator's say-so, recorded."""
        self.select_alignment(bit_offset, agreed)
        self.record_step("align", bit=bit_offset, agreed=agreed)

    def flip_bits(self, bits, file_fixes=None) -> None:
        """Invert these bits of the input and decode again: a repair. With
        `file_fixes` ((file offset, bit) pairs) the input file itself can be
        written back repaired."""
        data = bytearray(self.data)
        for b in bits:
            data[b >> 3] ^= 1 << (b & 7)
        self.data = bytes(data)
        self.trace = None
        self.select_alignment(self.bit_offset, agreed=None)
        if file_fixes:
            fixes = getattr(self, "file_fixes", None) or []
            fixes.extend(tuple(f) for f in file_fixes)
            self.file_fixes = fixes
        self.record_step("flip", bits=list(bits),
                         file_fixes=[list(f) for f in file_fixes or []])

    def _replay(self, step: dict) -> None:
        op = step.get("op")
        if op == "align":
            self.choose_start(step["bit"], step.get("agreed", False))
        elif op == "resync":
            self.resync_to(step["bit"])
        elif op == "flip":
            self.flip_bits(step["bits"], step.get("file_fixes"))
        elif op == "resume":
            from . import resume
            got = resume.resume(self)
            if got.trace is None:
                raise ValueError(f"reading on past the damage no longer "
                                 f"applies: {got.reason}")
            resume.apply(self, got)
        elif op == "borrow":
            from types import SimpleNamespace
            from . import borrow
            landing = SimpleNamespace(
                start=step["start"], label=step.get("label", ""),
                litlen=HuffmanTable(tuple(step["litlen"])),
                dist=HuffmanTable(tuple(step["dist"])))
            borrow.apply(self, landing, step["end"])
        else:
            raise ValueError(f"unknown step {op!r}")

    def source_of(self, pos: int) -> str:
        """Where the guess that set this position came from: "" when typed,
        or when no guess set it."""
        if pos in self.guess_sources and pos in self._records:
            return self.guess_sources[pos]
        for anchor, rec in self._records.items():
            if pos in rec.touched:
                return self.guess_sources.get(anchor, "")
        return ""

    def fingerprint(self) -> str:
        """A hash of the tape as it stands: every byte and whether it is
        known, which a replayed session has to reproduce."""
        buf, mask, start = self.full(0)
        return hashlib.sha256(f"{start}:".encode() + buf + mask).hexdigest()

    def resync_to(self, bit_offset: int) -> None:
        """Re-open as a stream from a block boundary found by scanning.

        Not the same thing as `select_alignment`, and the difference is the
        whole point of resyncing.  That one picks a bit offset to start reading
        a headerless fragment with assumed tables; this one asserts that a block
        *header* begins here, so every block from that point carries its own
        table and nothing has to be guessed at.  On a mid-stream capture it is
        the difference between recovering a couple of thousand bytes and
        recovering all of them.

        Guesses are re-seated by `_decode`, which drops any that no longer land
        on a hole and says so -- output positions do not survive a change of
        starting point.
        """
        self.mode = "stream"
        self.bit_offset = bit_offset
        self._candidates = None
        # The fragment's agreed start means nothing to a stream that begins
        # at a header; left set, the pane said `start: bit 2 (agreed)`.
        self.agreed_from = None
        self._use_agreed = False
        self._decode()
        self.record_step("resync", bit=bit_offset)

    @property
    def resumed(self) -> bool:
        """True when decoding starts part way into a stream rather than at its head."""
        return self.mode == "stream" and self.bit_offset != self.container.bit_offset

    @property
    def expects_outside_refs(self) -> bool:
        """Whether back-references may legitimately reach outside the output.

        True when resuming mid-stream, and true for a stream compressed against
        a preset dictionary that is not in the file.  Both are the same
        situation as far as the tape is concerned: the referenced bytes are
        unknown rather than wrong.
        """
        return self.resumed or bool(getattr(self.stream, "references_outside", False))

    def _decode(self) -> None:
        # Only a fragment read with the assumed table sets it again; a
        # resync to a real header, or anything else, leaves it clear.
        self.assumed_note = ""
        if self.mode == "assembled":
            return                      # the trace is the input, not derived
        if self.mode == "stream":
            # Starting at the head of a self-contained stream, a distance
            # reaching behind the output is corruption and should stop the
            # decode.  When the stream is resumed, or was compressed against a
            # dictionary we do not have, it is the normal case and must not.
            self.trace = Decoder(self.data,
                                 strict_window=not self.expects_outside_refs
                                 ).stream(self.bit_offset)
            # A decode that stopped on damage read garbage for a while first;
            # those bytes are unproven from where the encoder replay says the
            # damage starts.
            from . import headcheck
            self.headcheck = headcheck.check(self.trace)
            if self.headcheck is not None:
                self.trace.tape = headcheck.tape_with_cut(
                    self.trace.symbols, self.headcheck.cut)
        else:
            self.agreed_from = (self._agreed_start(self.bit_offset)
                                if self._use_agreed else self.bit_offset)
            self.trace = Decoder(self.data, strict_window=False).fragment(
                self.agreed_from, self.litlen, self.dist)
            clean = self.trace.status in ("end", "eob", "truncated")
            if self.litlen is None and not (
                    clean and self.trace.tape.length >= ASSUMED_CONFIRMS):
                # No table was supplied, so the fixed one was assumed. A
                # fragment it reads cleanly for a long way (stack.bin: 60,472
                # bytes to an end-of-block) is a fixed block; one it chokes on
                # is not, and what it read before choking went on the tape as
                # facts: `|=?!` in green on a dynamic carve. Nor is a short
                # clean read: random bits hit the 7-bit
                # end-of-block code about once in 128 symbols, and 130 of 418
                # dynamic carves stopped on one, putting 15,452 garbage bytes
                # on the tape as facts. Real fixed-table fragments read
                # 11,383 bytes or more, readings of dynamic data at most 1,336.
                self.assumed_note = (
                    f"read with the fixed Huffman table assumed, which "
                    f"{'stopped on ' + self.trace.status if not clean else 'read only ' + format(self.trace.tape.length, ',') + ' bytes'}"
                    f": these bytes are unproven. f3 (--resync) finds the "
                    f"next block header, which brings its own table")
                from .gap import replay
                from .tape import BORROWED, Tape
                tape = Tape()
                tape.taint = BORROWED
                replay(self.trace.symbols, tape)
                tape.taint = None
                self.trace.tape = tape
        self._reapply_guesses()

    def _agreed_start(self, bit: int) -> int:
        """Where to really start a fragment chosen to start at `bit`.

        The ranked readings disagree for their first few symbols and then land
        on the same boundaries for good. Before that point which bytes came out
        depends on which start is true, which nothing here knows; read from
        there and they went onto the tape as facts -- on stack.bin, starting at
        bit 0 put 6 misread bytes in green, and the picker made bit 0 look
        better than the right answer. So the decode starts
        at the first boundary every reading agrees on, at or after the one
        chosen. With no agreement it starts where it was asked to.
        """
        from .gap import agreed_boundaries
        cands = self._candidates or []
        common = agreed_boundaries([c.trace for c in cands])
        return next((b for b in common if b >= bit), bit)

    # ---- guesses ------------------------------------------------------------

    @property
    def tape(self) -> Tape:
        return self.trace.tape

    def hole_positions(self) -> list[int]:
        return [h.anchor for h in self.tape.holes()]

    def set_guess(self, pos: int, value: int, provenance: int = GUESS,
                  source: str = "") -> list[int]:
        """Assign a byte to a hole.  Propagates to every equal position.

        `source` says where an operator's byte came from when it was not
        typed: "copy suspected-copy.txt". Kept beside the guess, so --why can
        name the file a byte rests on.
        """
        if source:
            self.guess_sources[pos] = source
        else:
            self.guess_sources.pop(pos, None)
        if pos in self._records:
            self.clear_guess(pos, _record_undo=False)
        touched = self.tape.resolve(pos, value, provenance)
        self.guesses[pos] = value
        self._records[pos] = GuessRecord(pos, value, touched, provenance)
        # The provenance travels in the entry. Looking it up at undo time reads
        # `_records`, which by then no longer holds the position being restored,
        # so a model fill came back as an operator's guess -- the tool
        # misattributing authorship to the person using it.
        self._undo.append(("set", pos, value, touched, self._group, provenance))
        self._redo.clear()
        return touched

    def begin_group(self) -> int:
        """Start an undo group.  A solver run is one action, not a hundred."""
        self._group_seq += 1
        self._group = self._group_seq
        return self._group

    def end_group(self) -> None:
        self._group = None

    def clear_guess(self, pos: int, _record_undo: bool = True) -> list[int]:
        rec = self._records.pop(pos, None)
        if rec is None:
            return []
        self.tape.unresolve(rec.touched)
        self.guesses.pop(pos, None)
        if _record_undo:
            self._undo.append(("clear", pos, rec.value, rec.touched, self._group,
                               rec.provenance))
            self._redo.clear()
        return rec.touched

    def retract_model_fills(self) -> list[int]:
        """Take back every byte the model filled, and keep every byte you typed.

        A fill resolves its class, so the solver never looks at it again: an early
        wrong guess is permanent, and context that arrives afterwards -- the
        operator reading a word two bytes away -- cannot dislodge it.  Retracting
        returns those classes to the pool so the next fill scores them against
        everything now known, while the operator's own bytes stay put as the
        evidence they are.

        Provenance is what makes this separable, and it is why the tool keeps it.
        Returns the anchors it took back.  A caller already inside an undo group
        keeps it, so retract-then-refill is one action and one ctrl+z; otherwise
        this opens one of its own.
        """
        mine = [pos for pos, rec in self._records.items()
                if rec.provenance in (SOLVED, SOLVED_PROP)]
        if not mine:
            return []
        owned = self._group is None
        if owned:
            self.begin_group()
        try:
            for pos in mine:
                self.clear_guess(pos)
        finally:
            if owned:
                self.end_group()
        return mine

    def note_fill(self, anchor: int, note: FillNote) -> None:
        self.fill_notes[anchor] = note

    def fill_note(self, pos: int) -> FillNote | None:
        """The note for whichever fill put the byte at `pos` there."""
        owner = self.guess_owner(pos)
        return self.fill_notes.get(pos if owner is None else owner)

    def guess_owner(self, pos: int) -> int | None:
        """Which guess, if any, is responsible for the byte at pos."""
        for p, rec in self._records.items():
            if pos in rec.touched:
                return p
        return None

    def _reapply_guesses(self) -> None:
        """Re-seat guesses onto a freshly decoded tape.

        Guesses are stored by output position, which is stable for a given
        alignment but not across alignments.  Any guess that no longer lands on
        a hole is dropped and reported rather than silently ignored.
        """
        pending = [(pos, value, self._records.get(pos, GuessRecord(pos, value)).provenance)
                   for pos, value in sorted(self.guesses.items())]
        self.guesses.clear()
        self._records.clear()
        dropped = []
        for pos, value, prov in pending:
            try:
                touched = self.tape.resolve(pos, value, prov)
            except (ValueError, KeyError):
                dropped.append(pos)
                continue
            self.guesses[pos] = value
            self._records[pos] = GuessRecord(pos, value, touched, prov)
        self.note = (f"{len(dropped)} guess(es) dropped: no longer holes at "
                     f"{', '.join(str(p) for p in dropped)}") if dropped else ""

    def _apply(self, op: str, pos: int, value: int, touched: list[int],
               prov: int) -> None:
        if op == "set":
            t = self.tape.resolve(pos, value, prov)
            self.guesses[pos] = value
            self._records[pos] = GuessRecord(pos, value, t, prov)
        else:
            self.tape.unresolve(touched)
            self.guesses.pop(pos, None)
            self._records.pop(pos, None)

    def _invert(self, op: str) -> str:
        return "clear" if op == "set" else "set"

    def _pop_step(self, stack: list, other: list) -> bool:
        """Move one action across.  A grouped action moves as a whole."""
        if not stack:
            return False
        group = stack[-1][4]
        moved = 0
        while stack:
            if moved and (group is None or stack[-1][4] != group):
                break
            op, pos, value, touched, g, prov = stack.pop()
            self._apply(self._invert(op), pos, value, touched, prov)
            other.append((op, pos, value, touched, g, prov))
            moved += 1
        return moved > 0

    def undo(self) -> bool:
        return self._pop_step(self._undo, self._redo)

    def redo(self) -> bool:
        if not self._redo:
            return False
        group = self._redo[-1][4]
        moved = 0
        while self._redo:
            if moved and (group is None or self._redo[-1][4] != group):
                break
            op, pos, value, touched, g, prov = self._redo.pop()
            self._apply(op, pos, value, touched, prov)
            self._undo.append((op, pos, value, touched, g, prov))
            moved += 1
        return moved > 0

    # ---- output -------------------------------------------------------------

    def recovered(self, fill: int = 0x3F) -> tuple[bytes, bytes]:
        return self.tape.recovered(fill)

    def window(self, fill: int = 0x3F) -> tuple[bytes, bytes, int]:
        return self.tape.window_bytes(fill)

    @property
    def has_dictionary(self) -> bool:
        """The bytes before output 0 are a zlib preset dictionary's tail."""
        return getattr(self.stream, "dictid", None) is not None

    def full(self, fill: int = 0x3F) -> tuple[bytes, bytes, int]:
        """Pre-fragment window followed by the decoded output, as one buffer.

        A preset dictionary's tail is not part of what the stream holds, so
        it is left out: the file written is the stream's output.
        """
        if self.has_dictionary:
            obuf, omask = self.recovered(fill)
            return obuf, omask, 0
        wbuf, wmask, start = self.window(fill)
        obuf, omask = self.recovered(fill)
        return wbuf + obuf, wmask + omask, start

    def verify(self):
        """Check the decode against the container's size and checksum.

        Returns None when the bytes did not come from a container, or the
        container recorded nothing to check against.
        """
        if self.stream is None:
            return None
        from ..containers.base import verify as _verify
        return _verify(self.stream, self.tape)

    def stats(self) -> dict:
        s = self.tape.stats()
        s.update(mode=self.mode, container=self.container.kind,
                 bit_offset=self.bit_offset, status=self.trace.status,
                 blocks=len(self.trace.blocks), symbols=len(self.trace.symbols),
                 guesses=len(self.guesses))
        return s

    # ---- persistence --------------------------------------------------------

    def save(self, path: str | Path) -> None:
        from dataclasses import asdict
        Path(path).write_text(json.dumps({
            "version": 4,
            # The input as it arrived. Version 3 kept its length and its first
            # 16 bytes and checked only the length, so a session loaded onto
            # any other file of the same size.
            "input_sha256": self.input_sha256,
            "sha_prefix": binascii.hexlify(self.data[:16]).decode(),
            "size": len(self.data),
            "mode": self.mode,
            "profile": self.profile,
            "bit_offset": self.bit_offset,
            # Whether the decode started at the agreed boundary rather than at
            # bit_offset itself. Output positions differ between the two, so a
            # reload without it seats every guess in the wrong place.
            "agreed": self._use_agreed,
            # Repairs, resyncs, reading on, borrowed tables: replayed in order
            # before any guess goes back. Without them a repaired stream came
            # back unrepaired, and a resumed one crashed on reload.
            "steps": self.steps,
            "guesses": {str(k): v for k, v in sorted(self.guesses.items())},
            # Who asserted each byte, saved alongside it. Version 2 wrote only
            # the values, so reloading turned every model fill into an operator's
            # guess -- the file could not tell the difference and neither could
            # the colour on screen afterwards.
            "provenance": {str(k): rec.provenance
                           for k, rec in sorted(self._records.items())},
            "sources": {str(k): v for k, v in sorted(self.guess_sources.items())
                        if k in self.guesses},
            "ledger": [asdict(e) for e in self.ledger.entries],
            "fingerprint": self.fingerprint(),
        }, indent=2))

    def load(self, path: str | Path) -> None:
        from .ledger import Entry
        blob = json.loads(Path(path).read_text())
        want = blob.get("input_sha256")
        if want is not None and want != self.input_sha256:
            raise ValueError("session file was saved against a different "
                             "input: its SHA-256 differs")
        if blob.get("size") != len(self.data):
            raise ValueError("session file was saved against a different input")
        guesses = {int(k): v for k, v in blob.get("guesses", {}).items()}
        # Seed the records so _reapply_guesses can read each byte's provenance
        # back off them. A version 2 file has none, and GUESS is the honest
        # reading there: it cannot say otherwise, so it must not claim to.
        prov = {int(k): v for k, v in blob.get("provenance", {}).items()}
        if blob.get("version", 0) < 4:
            self.guesses = guesses
            self._records = {pos: GuessRecord(pos, value, [],
                                              prov.get(pos, GUESS))
                             for pos, value in guesses.items()}
            self.select_alignment(blob.get("bit_offset", self.bit_offset),
                                  agreed=blob.get("agreed", False))
            return
        self.steps = []
        for n, step in enumerate(blob.get("steps", []), 1):
            try:
                self._replay(step)
            except Exception as e:
                raise ValueError(f"step {n} of the session ({step.get('op')}) "
                                 f"did not replay: {e}") from e
        # A fragment's start can be set without a recorded step (a caller
        # using select_alignment directly); the saved start is the final word.
        bit, agreed = blob.get("bit_offset"), blob.get("agreed", False)
        if (self.mode == "fragment" and bit is not None
                and (bit, agreed) != (self.bit_offset, self._use_agreed)):
            self.select_alignment(bit, agreed=agreed)
        self.guesses = guesses
        self._records = {pos: GuessRecord(pos, value, [], prov.get(pos, GUESS))
                         for pos, value in guesses.items()}
        self._reapply_guesses()
        self.ledger.entries = [Entry(**e) for e in blob.get("ledger", [])]
        self.guess_sources = {int(k): v for k, v in
                              blob.get("sources", {}).items()}
        saved = blob.get("fingerprint")
        if saved is not None and saved != self.fingerprint():
            raise ValueError("the session replayed to a different tape than "
                             "the one saved; "
                             + (self.note or "every step ran"))


def load_input(file: str | None = None, hexdata: str | None = None) -> bytes:
    if file:
        return Path(file).read_bytes()
    if hexdata:
        cleaned = "".join(hexdata.split()).replace("0x", "").replace(",", "")
        return binascii.unhexlify(cleaned)
    raise ValueError("supply either a file or hex data")
