"""Left pane (block and table structure) and right pane (bit-level inspector)."""
from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import VerticalScroll
from textual.message import Message
from textual.widgets import Static, Tree

from ..core.bitreader import bit_string
from ..core.events import BitField, BlockInfo, Symbol
from ..core.tables import BTYPE_STORED, DISTANCE_BASE, LENGTH_BASE
from ..predict.solver import confidence_text, measured_text


class StatsLink:
    """Tree-node payload meaning "open the statistics for this block".

    A distinct type rather than a string, because the tree's selection handler
    dispatches on what a node carries and every other node carries something the
    tape can be sent to.
    """

    __slots__ = ("block",)

    def __init__(self, block: BlockInfo):
        self.block = block


def _sym_name(sym: int) -> str:
    if sym < 256:
        return f"{sym:>3} {chr(sym)!r}" if 0x20 <= sym < 0x7F else f"{sym:>3} 0x{sym:02x}"
    if sym == 256:
        return "256 end-of-block"
    # The fixed table assigns codes to 286 and 287, but no length base exists
    # for them: they cannot legally appear, and seeing one means the decode has
    # gone wrong.
    if sym - 257 >= len(LENGTH_BASE):
        return f"{sym:>3} INVALID"
    return f"{sym:>3} length {LENGTH_BASE[sym - 257]}"


class BlocksPane(VerticalScroll):
    """Block headers, Huffman tables, and how the tables were encoded.

    This is the part that makes a well-formed stream worth opening: the whole
    header hierarchy, expandable, with every field still pointing at its bits.

    It is also tied to the tape both ways.  Moving the cursor over a byte walks
    this tree to the block that produced it; selecting a block here jumps the
    tape to its first output byte.  Without that link the two panes show the
    same stream at two scales with nothing connecting them, and on a stream of
    any size it stops being obvious which block you are looking at.
    """

    class BlockSelected(Message):
        """The user picked a block; the tape should go there."""

        def __init__(self, block: BlockInfo) -> None:
            super().__init__()
            self.block = block

    class StatsRequested(Message):
        """The user picked the statistics row under a block."""

        def __init__(self, block: BlockInfo) -> None:
            super().__init__()
            self.block = block

    def __init__(self, session, **kw):
        super().__init__(**kw)
        self.session = session
        self._block_nodes: dict[int, object] = {}
        self._showing: int | None = None      # block index the cursor is in
        self._auto_expanded = None            # node this pane opened by itself
        self._at: int | None = None           # output position last asked about

    def compose(self) -> ComposeResult:
        tree: Tree = Tree("stream")
        tree.show_root = False
        tree.guide_depth = 2
        yield tree

    def on_mount(self) -> None:
        self.rebuild()

    def rebuild(self) -> None:
        try:
            tree = self.query_one(Tree)
        except Exception:
            return
        tree.clear()
        self._block_nodes.clear()
        self._showing = None
        self._auto_expanded = None
        s = self.session
        root = tree.root

        meta = root.add("input", expand=True)
        meta.add_leaf(f"{len(s.data)} bytes")
        stream = getattr(s, "stream", None)
        if stream is not None and getattr(stream, "container", None):
            # The sliced bytes sniff as raw; the entry came out of a container
            # and the pane should say which.
            meta.add_leaf(f"container: {stream.container}")
            if stream.name:
                meta.add_leaf(f"entry: {stream.name}")
        else:
            meta.add_leaf(f"container: {s.container.kind}")
        meta.add_leaf(f"mode: {s.mode}")
        start = getattr(s, "agreed_from", None)
        if start is not None and start != s.bit_offset:
            # The decode starts where every ranked reading agrees, not at the
            # top-ranked offset itself.
            meta.add_leaf(f"start: bit {start} (agreed)")
        else:
            meta.add_leaf(f"start: bit {s.bit_offset}")
        meta.add_leaf(f"status: {s.trace.status}")
        st = s.tape.stats()
        meta.add_leaf(f"output: {st['out_len']} B")
        if st["holes"]:
            meta.add_leaf(f"holes: {st['holes']} classes / {st['hole_positions']} slots")
        hc = getattr(s, "headcheck", None)
        if hc is not None:
            meta.add_leaf(f"unproven: from out {hc.cut} ({hc.how})")
        self._add_evidence(meta, s)

        for block in s.trace.blocks:
            self._add_block(root, block)
        # Blocks start collapsed and the one under the cursor opens itself, so
        # the tree stays readable on a stream with hundreds of them.  A lone
        # block has nothing to disambiguate, so it just opens.
        if len(s.trace.blocks) == 1:
            self._block_nodes[s.trace.blocks[0].index].expand()
        # A rebuild throws away the nodes, so the tape's position has to be
        # applied again.  This pane mounts after the app does, so on startup
        # this is the call that puts the tree on the right block at all.
        if self._at is not None:
            self.highlight(self._at)

    def _add_evidence(self, meta, s) -> None:
        """What the container says, on screen rather than only on stderr.

        A capture knows exactly what it lost and whether a checksum survived
        to prove a fill; both were printed to the terminal before the
        interface took it over and were never seen again. Lost ranges come first, then what can check the result.
        """
        stream = getattr(s, "stream", None)
        if stream is None:
            return
        missing = getattr(stream, "missing", None) or []
        if missing:
            lost = sum(n for _p, n in missing)
            node = meta.add(f"lost: {lost:,} B in {len(missing)} gap(s)",
                            expand=True)
            for pos, n in missing[:6]:
                where = ("HTTP body byte" if s.container.kind == "pcap"
                         else "stream byte")
                node.add_leaf(f"{n:,} B at {where} {pos:,}")
            if len(missing) > 6:
                node.add_leaf(f"... {len(missing) - 6} more")
            node.add_leaf("measured from TCP sequence numbers"
                          if s.container.kind == "pcap" else
                          "measured from the container")
        crc = getattr(stream, "crc32", None)
        size = getattr(stream, "uncompressed_size", None)
        if crc is not None:
            node = meta.add(f"checks: CRC32 {crc:08x}", expand=True)
            if size:
                node.add_leaf(f"over {size:,} B")
        elif getattr(stream, "adler32", None) is not None:
            meta.add_leaf(f"checks: Adler-32 {stream.adler32:08x}")
        elif size:
            meta.add_leaf(f"checks: size {size:,} B only")
        else:
            meta.add_leaf("checks: none recorded")
        if getattr(stream, "dictid", None) is not None:
            meta.add_leaf(f"dictionary: Adler-32 {stream.dictid:08x}, "
                          f"absent")
        from ..core import chunkcrc
        bad = chunkcrc.bad_segments(s)
        fixed = getattr(s, "file_fixes", None)
        if bad:
            meta.add_leaf(f"chunk CRC: {', '.join(str(k + 1) for k in bad)} "
                          f"of {len(stream.segments)} fail")
        elif fixed:
            meta.add_leaf(f"chunk CRC: {len(fixed)} bit(s) repaired")

    def _add_block(self, root, block: BlockInfo) -> None:
        title = (f"block #{block.index}  {block.btype_name}"
                 + ("  [final]" if block.bfinal else ""))
        node = root.add(title, data=block, expand=False)
        self._block_nodes[block.index] = node
        size = block.out_end - block.out_start
        span = (f"output: {block.out_start}..{block.out_end - 1}  ({size} B)"
                if size > 0 else "output: none")
        node.add_leaf(span, data=block)
        for f in block.fields[:8]:
            node.add_leaf(f"{f.label} = {f.detail}", data=f)
        if len(block.fields) > 8:
            more = node.add(f"… {len(block.fields) - 8} more header fields")
            for f in block.fields[8:]:
                more.add_leaf(f"{f.label} = {f.detail}", data=f)

        node.add_leaf(f"literals: {block.n_literals}   matches: {block.n_matches}")
        if block.btype == BTYPE_STORED:
            node.add_leaf(f"stored bytes: {block.stored_len}")
            return
        if block.litlen is not None:
            # Above the tables rather than below them: the tables are hundreds of
            # rows and this is the thing that summarises them, so it should not be
            # reachable only by scrolling past what it summarises.
            node.add_leaf("▸ table statistics: how the codes are spent",
                          data=StatsLink(block))
        if block.codelen is not None:
            self._add_codelen(node, block)
        if block.litlen is not None:
            self._add_table(node, "literal/length code", block.litlen, _sym_name)
        if block.dist is not None and block.dist.codes:
            self._add_table(node, "distance code", block.dist,
                            lambda s: f"{s:>3} dist {DISTANCE_BASE[s]}"
                            if s < len(DISTANCE_BASE) else f"{s:>3} invalid")

    def _add_codelen(self, node, block: BlockInfo) -> None:
        cl = node.add(f"code-length code ({len(block.codelen.codes)} symbols)")
        for sym in block.codelen.symbols():
            cl.add_leaf(f"{sym:>2} → {block.codelen.code_str(sym)}")
        ops = node.add(f"code-length stream ({len(block.clen_ops)} ops)")
        for op in block.clen_ops:
            if op.op == "lens":
                ops.add_leaf(f"lens {op.value}  @{op.index}", data=op)
            else:
                ops.add_leaf(f"{op.op} {op.value} ×{op.count}  @{op.index}", data=op)

    def _add_table(self, node, title: str, table, namer) -> None:
        used = table.symbols()
        t = node.add(f"{title} ({len(used)} of {len(table.lengths)} used)")
        for sym in used:
            code = table.code_str(sym)
            t.add_leaf(f"{namer(sym)}  {len(code)}b  {code}")

    # ---- staying in step with the tape --------------------------------------

    def highlight(self, pos: int) -> BlockInfo | None:
        """Walk the tree to the block that produced output position pos."""
        self._at = pos
        block = self.session.trace.block_at_out(pos)
        if block is None or block.index == self._showing:
            return block
        node = self._block_nodes.get(block.index)
        if node is None:
            return block
        self._showing = block.index
        # Only this pane's own expansion is undone, so a block the user opened
        # by hand stays open as the cursor moves past it.
        if self._auto_expanded is not None and self._auto_expanded is not node:
            self._auto_expanded.collapse()
        node.expand()
        self._auto_expanded = node
        tree = self.query_one(Tree)
        # Expanding and collapsing invalidate the tree's cached line numbers,
        # and the cursor is addressed by line.  Reading a node back forces the
        # rebuild.  Without it move_cursor lands on whatever has since moved
        # into the node's old line, or on line 0, because a node that has never
        # been laid out reports line -1 and the cursor validator clamps it.
        tree.get_node_at_line(0)
        # move_cursor does not post NodeSelected, so this cannot bounce back
        # into the tape and fight the cursor that caused it.
        tree.move_cursor(node)
        return block

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Choosing a block sends the tape to the first byte it produced."""
        data = event.node.data
        if isinstance(data, StatsLink):
            self.post_message(self.StatsRequested(data.block))
        elif isinstance(data, BlockInfo):
            self.post_message(self.BlockSelected(data))


class BitsPane(VerticalScroll):
    """What the thing under the cursor looks like as bits."""

    def __init__(self, session, **kw):
        super().__init__(**kw)
        self.session = session
        self._suggestion = None

    def compose(self) -> ComposeResult:
        yield Static("", id="bits-body")

    def _append_suggestion(self, t: Text) -> None:
        """What the model would put here, if anything.

        Shown, never applied. The figure ranks candidates well and is not a
        probability, so it belongs next to the alternatives rather than on its
        own.
        """
        p = self._suggestion
        if p is None:
            return
        t.append("\nmodel suggests\n", style="magenta")
        t.append(f"  {_byte(p.value)}", style="bold magenta")
        if p.precision is not None:
            t.append(f"   measured {measured_text(p.precision)} right\n",
                     style="white")
            t.append(f"  (confidence {confidence_text(p.confidence)}, on "
                     f"held-out text for this model"
                     f"{', next to a long gap' if p.near else ''})\n",
                     style="grey42")
        else:
            t.append(f"   {confidence_text(p.confidence)}\n", style="white")
            bare = getattr(self.session, "model", None) is None
            t.append("  (no corpus loaded: from this stream's own bytes)\n"
                     if bare else
                     "  (raw confidence: this model is not calibrated)\n",
                     style="grey42")
        if p.runners:
            t.append("  then " + " ".join(_byte(v) for v, _ in p.runners[:4]) + "\n",
                     style="grey42")
        t.append("  f4 to review, ^f to fill\n", style="grey42")

    def _body(self) -> Static:
        return self.query_one("#bits-body", Static)

    def clear(self) -> None:
        self._body().update(Text("nothing selected", style="grey42"))

    def show_symbol(self, sym: Symbol | None, pos: int | None = None,
                    suggestion=None) -> None:
        self._suggestion = suggestion
        if sym is None:
            self.show_position(pos)
            return
        data = self.session.data
        t = Text()
        t.append(sym.label() + "\n", style="bold white")
        t.append(f"bit {sym.bit_start}..{sym.bit_end}  ({sym.nbits} bits)\n",
                 style="grey42")
        if sym.kind == "match":
            src = sym.out_pos - sym.distance
            t.append(f"out {sym.out_pos}..{sym.out_pos + sym.length - 1}\n", style="grey42")
            t.append(f"reads {src}..{src + sym.length - 1}\n\n", style="grey42")
        else:
            t.append(f"out {sym.out_pos}\n\n", style="grey42")

        for f in sym.fields:
            t.append(f"{f.label}\n", style="cyan")
            t.append(f"  {bit_string(data, f.start, f.end)}\n", style="bold yellow")
            t.append(f"  {f.detail}\n", style="white")
            t.append(f"  @bit {f.start}\n\n", style="grey42")

        raw = bit_string(data, sym.bit_start, sym.bit_end)
        t.append("raw\n", style="cyan")
        t.append(_wrap(raw, 16), style="yellow")
        if pos is not None:
            t.append(f"\n\ncursor at out {pos}\n", style="grey42")
            self._append_byte_detail(t, pos)
        self._body().update(t)

    def show_position(self, pos: int | None) -> None:
        t = Text()
        if pos is None:
            self.clear()
            return
        t.append(f"out {pos}\n", style="bold white")
        if pos < 0 and getattr(self.session, "has_dictionary", False):
            t.append("preset dictionary\n", style="grey42")
            t.append(f"(byte {-pos:,} from its end)\n\n", style="grey42")
        elif pos < 0:
            t.append("pre-fragment window\n", style="grey42")
            t.append("(before the first decoded byte)\n\n", style="grey42")
        else:
            t.append("no symbol here\n\n", style="grey42")
        self._append_byte_detail(t, pos)
        self._body().update(t)

    def _append_byte_detail(self, t: Text, pos: int) -> None:
        tape = self.session.tape
        v = tape.value_at(pos)
        from ..core.tape import PROV_NAMES
        t.append(f"provenance: {PROV_NAMES[tape.prov_at(pos)]}\n", style="grey42")
        source = (self.session.source_of(pos)
                  if hasattr(self.session, "source_of") else "")
        if source:
            t.append(f"from: {source}\n", style="grey42")
        if v is None:
            hole = tape.hole_at(pos)
            t.append("value: unknown\n", style="red")
            if hole is not None:
                t.append(f"\nthis hole is one byte\nseen at {hole.size} positions:\n",
                         style="cyan")
                shown = hole.positions[:12]
                t.append("  " + ", ".join(str(p) for p in shown), style="white")
                if hole.size > len(shown):
                    t.append(f"  …+{hole.size - len(shown)}", style="grey42")
                t.append("\n\ntype a character to fill\nall of them at once.\n",
                         style="grey42")
                self._append_suggestion(t)
        else:
            ch = chr(v) if 0x20 <= v < 0x7F else ""
            t.append(f"value: 0x{v:02x}  {v:>3}  {ch}\n", style="white")
            t.append(f"bits:  {v:08b}\n", style="yellow")
            owner = self.session.guess_owner(pos)
            if owner is not None and owner != pos:
                t.append(f"\nfrom a guess at {owner}\n", style="grey42")
            self._append_fill_note(t, pos)

    def _append_fill_note(self, t: Text, pos: int) -> None:
        """How a magenta byte got there, and what its confidence is worth.

        Magenta only says a model suggested it. The confidence it cleared is
        not a probability -- on one measured carve the 0.995+ band was right
        57% of the time -- so it is shown beside what that band delivered,
        and the regime it was filled in is named when it was a long gap.
        """
        from ..core.tape import SOLVED, SOLVED_PROP
        from ..predict.solver import (LONG_RUN, LONG_RUN_PRECISION,
                                      confidence_text, measured_band, pct)
        if self.session.tape.prov_at(pos) not in (SOLVED, SOLVED_PROP):
            return
        note = getattr(self.session, "fill_note", lambda _p: None)(pos)
        t.append("\nfilled by ", style="magenta")
        if note is None:
            t.append("a model (no details kept: restored from a saved "
                     "session)\n", style="magenta")
            return
        if note.how == "checksum":
            t.append("the checksum\n", style="magenta")
            t.append("pinned by CRC32 arithmetic\n",
                     style="grey42")
            return
        who = {"n-gram": "the n-gram", "byte model": "the byte model",
               "both": "the byte model, n-gram agreeing",
               "f4": "you, from f4"}.get(note.how, note.how)
        if note.how == "both" and note.agreement is False:
            # A calibrated byte model fills on its own curve in "both" mode,
            # disagreements included.
            who = "the byte model; the n-gram named another byte"
        t.append(who + "\n", style="magenta")
        if note.confidence is not None:
            t.append(f"confidence {confidence_text(note.confidence)}",
                     style="white")
            if note.floor is not None:
                t.append(f"  (floor {pct(note.floor)})", style="grey42")
            t.append("\n")
            band = measured_band(note.confidence)
            if note.precision is not None:
                t.append(f"measured {measured_text(note.precision)} right for this "
                         f"model at this confidence"
                         f"{' next to a long gap' if note.near else ''}, on "
                         f"held-out text\n", style="grey62")
            elif note.how in ("n-gram", "both", "f4") and band:
                t.append(f"model not calibrated; that band was right "
                         f"{band[0]} of the time on a measured phrack carve "
                         f"({band[1]} classes)\n", style="grey42")
            elif note.how == "byte model":
                t.append("the byte model's confidence is not calibrated\n",
                         style="yellow")
        if note.runners:
            t.append("beat " + " ".join(_byte(v) for v in note.runners[:4])
                     + "\n", style="grey42")
        if note.grounded is not None:
            t.append("context: all read off the capture\n"
                     if note.grounded >= 1.0 else
                     f"context: {pct(note.grounded)} read off the capture, "
                     f"the rest earlier fills\n", style="grey42")
        if note.run >= LONG_RUN:
            t.append(f"filled inside a {note.run:,}-byte unread run, where "
                     f"fills measured {LONG_RUN_PRECISION} right\n",
                     style="yellow")


def _byte(v: int) -> str:
    return repr(chr(v)) if 0x20 <= v < 0x7F else f"0x{v:02x}"


def _wrap(s: str, width: int) -> str:
    return "\n".join(s[i:i + width] for i in range(0, len(s), width)) or "-"
