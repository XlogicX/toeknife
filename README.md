# ToeKnife

Carves partial DEFLATE data, and shows you the bits while it does it.

Give it a fragment of a DEFLATE stream with no headers and no Huffman table, and
it will find where the bitstream starts, decode what it can, and tell you
exactly which unknown bytes are standing between you and the rest of the data.

The user manual is in [docs/manual/](docs/manual/README.md): install, the
concepts, nine workflows on real damage, models and sessions.

## Install

Needs Python 3.10+ and [Textual](https://textual.textualize.io/). The byte model
has no dependencies at all.

```
uv venv
uv pip install -e .
```

or with plain pip:

```
python3 -m venv .venv
.venv/bin/pip install -e .
```

Then either activate the venv, or just call the binary directly:

```
.venv/bin/toeknife samples/fragment.bin
```

There is nothing to run inside `toeknife/`. The modules use package-relative
imports, so `python3 toeknife/tui/app.py` will not work. Entry points are the
`toeknife` command or `python -m toeknife`.

Sample data to try:

```
.venv/bin/toeknife samples/fragment.bin                 # carve a fragment
.venv/bin/toeknife samples/complete-dynamic.deflate     # browse a whole stream
.venv/bin/toeknife samples/body-fragment.bin --header samples/partialhuff.bin
```

## The idea

A DEFLATE stream is literals mixed with back-references. When you carve a
fragment out of the middle of one, its back-references point behind the start of
what you have, into data you do not.

The old version of this tool filled those in with `?` and let you supply guesses
in an indexed side file. That threw away the most useful thing about a
back-reference: it is not a request for bytes you might not have, it is an
assertion that two output positions hold the same byte.

So ToeKnife indexes output bytes as integers, with the fragment's first decoded
byte at position 0 and the unknown pre-fragment window at negative positions,
and records every back-reference as a union of two positions. Bytes that follow
from a literal resolve immediately. What is left is a set of equality classes,
each one a single unknown byte that may appear in several places at once.

On the sample fragment below that turns 17 unknown output positions into **7
unknown bytes**. Fill one and it lands everywhere it belongs.

## The interactive view

```
toeknife fragment.bin
```

```
 blocks              tape                                    bits
┌──────────────────┬───────────────────────────────────────┬──────────────────┐
│ ▼ input          │        ···◀5,6▶··················◀6,35▶│ match 6,35       │
│   49 bytes       │  -31 ░░░░ ur.░░░ur is the only of thos│ bit 280..296     │
│   mode: fragment │                                       │ out 38..43       │
│   start: bit 3   │        ····◀6,35▶····◀4─▶·····        │ reads 3..8       │
│   holes: 7 / 17  │   44 e that has░░░ur lett░░░░in it     │                  │
│ ▼ block #0       │                                       │ litlen code      │
│   literals: 42   │                                       │   0000100        │
│   matches: 3     │                                       │   sym 260 len 6  │
│   ▶ literal/len  │                                       │ dist code        │
│   ▶ distance     │                                       │   01010          │
│                  │                                       │   sym 10 → 33    │
└──────────────────┴───────────────────────────────────────┴──────────────────┘
 out 38   57 bytes   7 holes / 17 slots   eob
```

Two rows per line. The top row is what the compressor wrote, with each
back-reference drawn exactly as wide as the bytes it produced. The bottom row is
the output, one column per byte, so the two line up cell for cell.

Because the span width *is* the match length, the label can afford to shrink:

```
◀───6,35───▶   room to spare
─6,35─         narrower, so the arrows go rather than crowd the digits
←4521          too narrow for the pair, so keep the distance: the length
               is already on screen as the width of the span
───            too narrow for anything, select it to read the bits
```

A comma always means `length,distance`. A number without one is a distance. On
a real 420 KB stream that labels 69% of matches with the full pair and 25% with
the distance, leaving under 6% unlabelled.

Colour says how much you know:

| | |
|---|---|
| green | literal, read straight from the bitstream |
| cyan | copied from a byte that is already known |
| yellow | your guess, or a byte one of your guesses implies |
| magenta | a model fill, or a byte one of them implies |
| red `░` | unknown, and fillable |
| grey `░` | window space no back-reference reaches, nothing to solve |

Green and cyan are facts. Yellow is something you asserted. Magenta is something
a model suggested. Red is a question. Four different kinds of claim, and they
never share a colour.

Non-printable bytes render as Unicode Control Pictures (`␀␁␂…`, `␡`) so every
byte occupies exactly one cell and still says what it is.

### Keys

Every printable character is reserved for filling holes, since typing the byte
you think belongs somewhere is the whole point. Commands are on ctrl and
function keys, so nothing you type can trigger one by accident.

The footer shows only the keys for where the recovery is: no model yet,
filling, or done. `ctrl+k` opens a searchable menu of every command with its
key, and `f7` says what is worth doing next.

| key | |
|---|---|
| `ctrl+k` | menu: every command, searchable, with its key; includes "fill from a suspected copy", which lays a candidate file over the capture and lets the checksum prove or refute it (`--fill-from FILE` on the command line) |
| `f7` | what next: the steps worth taking now, then every route to a lost table, why the rest cannot run, and enter to run one |
| `← → ↑ ↓` | move the byte cursor |
| `tab` / `shift+tab` | next / previous hole |
| any character | fill the hole under the cursor, and every position equal to it — or replace a model fill or a guess already there |
| `ctrl+x` | the same, entered as a hex value |
| `backspace` | clear the guess under the cursor |
| `ctrl+z` / `ctrl+r` | undo / redo |
| `ctrl+n` / `ctrl+p` | next / previous back-reference |
| `enter` | follow the selected match to the bytes it reads |
| `f2` | alignment picker |
| `f3` | resync: block boundaries found by scanning, each with what it recovers |
| `f4` | proposals: what the model would fill, with confidence and context |
| `f5` | model: choose a corpus (this machine, a catalogue, or a model you built), save the loaded model, train a byte model, switch the fill mode |
| `ctrl+f` | fill surest class first, descending to the `f4` floor |
| `ctrl+e` | take back the model's own fills, keep yours, score again |
| `ctrl+g` | go to an output position |
| `ctrl+w` | write recovered bytes |
| `ctrl+s` | save session |
| `ctrl+t` | table statistics: what the block's Huffman codes claim, beside what it emitted |
| `f6` | show / hide the blocks pane |
| `f8` | checksum: what the container recorded against what came out |
| `f1` | help |

Selecting a back-reference highlights the source bytes it copies from, which is
the terminal version of hovering over it.

Expanding a block in the left pane lists every code in its tables, which answers
"what is the code for `e`" and nothing else. The `table statistics` row on that
block — or `ctrl+t` for whichever block the cursor is in — reads the same tables
as evidence instead. A canonical Huffman code spends a 3-bit code on something
the encoder expected one time in eight, so the code lengths **are** a frequency
claim about the data that produced them, recoverable without decoding a byte of
the body. The screen puts that claim beside what the block actually emitted:

```
118 of 279 symbols carry a code, 3-13 bits.  Kraft 1.0000 (complete)
159 of the 256 byte values never appear as a literal here

code length per byte value, in bits         cheapest codes first
     0 1 2 3 4 5 6 7 8 9 a b c d e f         symbol      bits  declared  measured
 20  6 · b c · 9 · 9 9 9 7 c 7 7 7 9         length 4      3b    12.50%    10.01%
 30  7 7 7 7 7 7 8 8 7 7 8 a · b · c         length 5      3b    12.50%     9.68%
 60  7 6 8 7 7 6 7 8 8 6 a 8 6 7 6 6         0x65 'e'      6b     1.56%     1.87%
 80  b · · · · · d · c c · · · · · ·         0x20 ' '      6b     1.56%     1.82%
```

The two columns are different kinds of fact — one is arithmetic on the header,
the other a count off the wire, taken from the alphabet entry the decoder read
rather than inferred back from the value it meant. The gap between them is the
useful part: a table that does not fit its own block is what a foreign or a
wrongly-recovered table looks like from the inside. `tab` moves between the
literal/length, distance and code-length alphabets, and heat is absolute rather
than scaled per table, so a 6-bit code reads the same in every block.

On a code set recovered without its assignment there is no byte grid and no byte
names: the codes and their lengths are real, which byte each one means is not
known, and the screen says so rather than printing a value it would be inventing.

With the cursor on a hole, the bits pane shows what the model would put there
and how sure it is. `f4` lists every suggestion with its runners-up and the text
around it, so a byte can be judged in place rather than as a bare number;
nothing is applied until you accept it. Either route undoes with a single
`ctrl+z`, however many classes it filled.

### The route board runs things

`f7` lists every route to a lost table, judged against the capture in front of
you, and `enter` starts the one you are on. It used to be read-only, which was
the wrong half of the idea: it would tell you a route was available and leave you
to work out which function key it hid behind, and for two of them there was no key
at all — the related-capture transplant and the known-plaintext judge existed only
on the command line.

Three kinds of row, and the difference is the point:

- `▸ run` — applies here and can be started now.
- `▸ ask` — applies, but needs evidence the tool does not have. Selecting it
  prompts for the path. "Needs a second capture" is not a dead end.
- listed only — either it does not apply to this capture, or it was measured
  during development and is not in this release (`unshipped`). Each carries its
  cost and the reason, printed on the row rather than tucked behind a cursor that
  cannot reach it.

That last detail matters more than it sounds. `kp-climb` is the case that proves
it: what an operator needs to know is not "needs plaintext" but that searching for
a table against plaintext **was measured and does not pay** — the only score you
can compute correlates +0.37 with being *further* from the truth. A
board that hid that behind a selection would be worse than one that never
mentioned the route.

**Bit rot is wired in now.** Every bit of the header is present and some are
inverted, which makes it a correction problem rather than a completion one — the
search is the flips over the header, not the tables over the alphabet. The
rates were measured by classifying each candidate against the true output, which
is exactly what an operator does not have; `core/bitrot.py` is the same
enumeration with the answer key removed, and a
candidate survives by decoding cleanly to the block's end.

It reports every survivor and picks none. Measured over all 250 single-bit flips
of a real header: the true repair survives **every** time, it is the only survivor
129 times, and it is first in the list 78% of the time. The other 22% is why
nothing is applied without being chosen. Applying one re-decodes the capture and
says so in the status line — a repair that parses is a candidate, not the
original.

Not in this release: the two dictionary sweeps (they need a 100 MB table
census), the mid-header fill, and the
alphabet gate — which is a filter with nothing to filter until a sweep is wired,
so an action for it would have nothing to do.

### Typing over the model

A model fill and a byte you typed are both hypotheses, so typing over either one
replaces it. A byte read from the bitstream, or copied by a back-reference, is a
fact, and typing over that is refused with a message saying which of the two it
is — for a copy, that the byte to correct is the one it copies from. Overruling a
fill re-colours it from the model's magenta to your yellow, because the point of
the colours is who asserted the byte; `ctrl+z` puts back what was displaced, with
the colour it had.

The loop this makes possible is the reason it matters. Measured on a 2 KB gap in
a 60 KB stream, corpus of Austen against a payload of markdown, threshold 0.90:
a first `ctrl+f` pass filled 72 classes at 57.9% precision, and a **second pass
found nothing at all** — the solver had said everything it could. After 40
corrected bytes were typed in, the same second pass found **17 more classes at
71.3% precision**. Nothing was retrained to get that; the proposals read the tape
live, so a byte you fix is context for its neighbours immediately. Where a
self-trained component is in play (`--self-train`, or no corpus at all) the
solver is also rebuilt on your next `f4` so that component sees the correction,
which took the same comparison from 5 classes at 33% to 21 at 53%.

One trial on one specimen, so treat the sizes as indicative rather than as a
measured rate — the direction is the durable part.

Which makes *alternating* with the model worth more than taking turns in bulk.
With a simulated perfect reader filling the most readable classes at a 0.995
floor, 200 typed classes spent 20 at a time with a `ctrl+f` between each got the
model to 334 correct fills at 78.6% precision; the same 200 typed all at once
before a single `ctrl+f` got 122 at 70.1%. 2.7x, from ordering alone — each fill
is context for your next read, and once the model moves, different classes become
the legible ones. So the loop is: read what you can, let it take the sure ones,
read again, and only lower the floor when you run out of things you can read.

Nothing discounts a byte you typed: scoring never looks at provenance, so yours
carries the weight of one DEFLATE emitted. That is what makes the loop work, and
it means a mistyped byte does as much damage as a bad model fill while wearing
your colour, so nothing will question it later. `ctrl+z`, and the ledger, are the
recourse.

The loop has one gap the model cannot close by itself. Filling resolves a class,
so the solver never scores it again — a byte it got wrong early is permanent, and
the context that would refute it arrives too late. `ctrl+e` takes back every byte
*the model* filled, keeps every byte you typed, and scores the lot again. In the
same simulation, after 200 typed classes: filling again on top of its own
mistakes left 754 wrong bytes, and retracting first left **76**. It costs
coverage, because a reopened class that no longer clears the floor stays unknown,
which is the direction this tool prefers.

### The checksum is in the view, not only in the pipeline

The status bar carries the container's verdict as one word — `crc ok`,
`crc MISMATCH`, `adler ?` — because a decode that is quietly wrong looks exactly
like a good one everywhere else on screen: every byte green or cyan, no holes,
status `complete`. It is recomputed when the tape changes rather than on every
keystroke.

`f8` opens the whole of it: what the wrapper recorded, what this decode
produced, the verdict in the same words `--verify` uses, and what the checksum
can still do about whatever is left. From there `enter` pins unknown bytes with
it — CRC32 is affine over GF(2), so four or fewer classes are determined
outright and past that it removes 32 bits of freedom for the model to choose
inside — and `ctrl+l` alternates the model and the checksum until neither moves.
Both are ordinary edits: one `ctrl+z` undoes a run, and every outcome including
the refusals goes into the ledger.

A stream carrying only Adler-32 says so and says why it is weaker: it is two
sums modulo 65521 rather than a linear system, so it rules candidates out
instead of solving, and surviving combinations are counted rather than trusted.

### Choosing a corpus

The corpus is the biggest lever on a fill, and `f5` is a picker rather than a
prompt, because "give me a path" is no guidance at all when the question is what
KIND of data would help. Three sections, all of them discovered rather than
recited:

1. **On this machine** — prose you already have, first: documentation
   directories in the project you are working in and beside the capture itself,
   with their file and byte counts. Then one row per data type with the roots
   that actually exist here, a row beneath each for sweeping the whole
   filesystem instead, and a freeform path. A directory is only offered when it
   is mostly prose, because committing a row trains on everything under it and
   a capture's own directory usually holds the captures.
2. **From outside** — specific files that can be fetched, with size and licence,
   plus any URL you name. Nothing leaves the machine until you confirm; the
   fetch is cached under `~/.toeknife/corpora` and recorded in the ledger.
3. **Models you built earlier** — every `.tkm`, `.model` or `.pkl` found in
   `~/.toeknife/models`, `$TOEKNIFE_MODELS`, the working directory and the
   capture's own directory, listed with its order, size and what it was trained
   on.

Selecting a row shows what committing would do — which directories, how many
files, how many bytes, which URL and how large, a model's provenance — and a
second `enter` commits. Whatever is picked is then measured against this stream,
with the same warning the headless path gives when the score is worse than
knowing nothing. Deciding the corpus after seeing the data beats deciding it on
the command line before you have.

`samples/fragment.bin` is the worked example. At bit 3 it decodes to seven
unknown classes over seventeen positions, and the answer is

```
ers <window> four. four is the only of those that has four letters in it
```

Nothing on this machine solves it outright, and the way it fails is the useful
part. The repo's own prose is the best corpus for it — 108 KB, better fitting
than a 985 KB word list, because the kind matches:

```
toeknife samples/fragment.bin --corpus README.md docs --bit 3 --solve \
         --solve-confidence 0.9 --dump
```

```
er?????????????????????????? ?our. ?our is the only of those that has ?our letter??in it
```

Four of the seven classes, none of them wrong. What is left needs meaning
rather than statistics: `lett` → `er` is settled by English, but `letter`
versus `letters` is not, and `?our` could be four, your, sour or pour — only
the sentence's self-reference picks it. The model fills what the language
determines and stops where the puzzle starts, which is the behaviour to want
from it.

## Alignment

DEFLATE is not byte aligned, so a carved fragment starts at an unknown bit
offset, usually part way through a symbol. The old workflow walked offsets one
at a time asking "does this look right?" up to 26 times. ToeKnife ranks them all
and shows what each one recovers:

```
$ toeknife --data '2f2dd203930a...' --list
 bit   score  lit  mch    out  holes  status       recovered
   3    8.86   42    3     57      7  eob          ur.░░░ur is the only of those that has░░░ur lett░░░░in i
  11    8.85   41    3     56      8  eob ⊂2       r.░░░░r is the only of those that has░░░░r lett░░░░in it
   2    8.83   42    3     57      7  eob          ·r.░░░·r is the only of those that has░░░·r lett░░░░in i
  19    8.83   40    3     55      9  eob ⊂1       .░░░░░ is the only of those that has░░░░░ lett░░░░in it
```

`⊂N` means that decode is a tail of offset N and recovers strictly less. Huffman
codes are self-synchronising, so several offsets land on the same symbol stream
with a symbol or two shaved off the front. That test is exact: offset B is a tail
of offset A when B is one of the bit positions at which A started a symbol.

**The ranking is advisory and the previews are the point.** Scoring combines
hard structural evidence (invalid codes, distances past the window, whether the
decode ends cleanly at end-of-block, how much of the fragment it explains) with
a plausibility estimate. When several offsets are all individually valid and the
disputed evidence is a handful of bytes, no scorer separates them reliably. A
person reading the previews does it instantly. Scoring exists to sort the list,
not to make the decision.

Pointing `--corpus` at data resembling what you are carving trains a byte model
and sharpens the ranking considerably:

```
toeknife fragment.bin --corpus /var/log/ --profile text
```

## Containers

| | |
|---|---|
| ZIP | and everything built on it: jar, apk, docx, xlsx, epub |
| gzip | multi-member, with CRC32 and ISIZE |
| zlib | including preset dictionaries |
| PNG | pixel data across IDAT chunks, plus zTXt/iTXt/iCCP text |
| PDF | `/FlateDecode` object streams |
| SWF | compressed `CWS` bodies |
| WOFF | per-table compression |
| TIFF | Deflate strips and tiles, either byte order |
| Anvil | `.mca`/`.mcr` region chunks, gzip or zlib per chunk |

```
toeknife archive.zip                    # list the entries
toeknife archive.zip --entry 0          # open one, by index
toeknife archive.zip --entry evidence   # or by name
toeknife archive.zip --entry 0 --verify # check it against the recorded CRC
```

```
archive.zip: zip container, 3 entries
  EOCD at 493: 3 entries, central directory 172 B at 321

  #      offset       comp     uncomp     crc32 method    fl name
  0          39         77       2640  1abdf0d1 deflate   -D hello.txt
  1         156         30         30  b0849683 stored    -- stored.bin
  2         231         90       5280  ad455b65 deflate   -D nested/deep.txt
```

Locating the bytes is the boring half. The half worth having is that the wrapper
records what the data looked like *before* compression:

```
$ toeknife archive.zip --entry hello --verify
hello.txt: size 2640/2640 ok  crc 1abdf0d1/1abdf0d1 ok
verified: the recovered bytes are the original bytes.
```

That turns "does this look right?" into arithmetic. It matters more than it
sounds, because **a corrupted DEFLATE stream usually still decodes.** Corrupt
200 bytes in the middle of a 250 KB entry and you get 254,242 bytes out instead
of 254,394, full of plausible-looking text. Nothing about the decode complains.
Only the checksum notices:

```
evidence.txt: size 254242/254394 MISMATCH  crc 240fe3a5/7c2a9899 MISMATCH
FAILED: the decode does not match what the container recorded.
```

A partial recovery reports the size and leaves the CRC alone, rather than
reporting a failure that means nothing: a checksum over output that still has
unknown bytes is not evidence of anything.

### A ZIP describes itself twice

Once per entry in a local header, once centrally at the end. Ordinary readers
trust the central directory. A forensic reader cannot, because the tail is
exactly what truncation takes. So both are parsed and reconciled:

- **No end-of-central-directory** — falls back to scanning for `PK\x03\x04`
  local headers, keeping only hits whose name and extra-field lengths are
  self-consistent. Entries still verify against their CRCs afterward.
- **Prepended data** — a self-extracting stub or a carve that lost the front
  shifts every recorded offset by a constant. The EOCD says how big the
  directory is and where the file ends, so the shift is measurable and applied.
- **Local and central disagree** — reported rather than silently resolved. That
  is damage or hand-editing, and it is worth saying out loud.
- **Streamed entries** — written to a pipe, sizes are zero in the local header
  and live in a trailing data descriptor, with or without its optional
  signature.
- **ZIP64** — `0xFFFFFFFF` sentinels resolved from the extended information
  extra field.
- **`PK\x05\x06` inside a comment** — a candidate EOCD is accepted only when its
  declared comment length accounts for exactly the bytes after it.
- **Encrypted, or not DEFLATE** — flagged and refused with a reason, not
  attempted.

### gzip and zlib

Neither records how long its compressed data is, so the end of a stream can only
be found by decoding to it. In gzip that matters: a member which will not decode
takes the position of every member after it, so a damaged file falls back to
scanning for the `1f 8b` magic and says that the lengths it reports are unknown.

gzip's trailer is the useful part. CRC32 of the original and its length mod 2³²,
which is a checksum and a known output length. zlib carries an Adler-32 instead.

```
$ toeknife evidence.gz --verify
evidence.txt: size 19200/19200 ok  crc 3f6649d8/3f6649d8 ok
verified: the recovered bytes are the original bytes.
```

A zlib stream with **FDICT** was compressed against a preset dictionary that is
not in the file. Other tools refuse those. ToeKnife decodes them and lets the
references reaching into the missing dictionary become holes, which is the same
problem as a reference behind the start of a fragment and the tape already
models it. On a 19 KB test stream that recovers 7,800 bytes outright and reduces
the rest to 35 unknown bytes, one per byte of the dictionary.

Recovering that dictionary from context is a different matter, and mostly does
not work: 7 of 36 characters at best in testing, because the positions
referencing it are surrounded by other unknowns and there is little to condition
on. The tool reports the situation accurately rather than guessing at it.

zlib's signature is weak. Exhaustively, 66 of the 65536 possible header byte
pairs pass the method, window and checksum-multiple constraints, so random data
offers a false header roughly every 1100 bytes. Detection always confirms by
decoding, which takes that to about every 15000.

### The rest

A stream is not always one contiguous run of bytes. **PNG** pixel data is a
single zlib stream cut across however many IDAT chunks the encoder felt like
emitting, with a length, a type and a CRC wedged between each pair, so the
payloads are gathered rather than sliced. The spec requires that run to be
consecutive; when it is not, ToeKnife says so instead of quietly joining the
pieces, because a stream reassembled from a broken run is missing bytes in the
middle and decodes to nonsense somewhere without ever failing outright. Chunk
CRCs cover the compressed bytes rather than the output, so they localise damage
but cannot verify a recovery.

**PDF** streams are found by scanning rather than by reading the cross-reference
table, which is correct for a valid file and useless here: the xref is at the
end and the end is what damage takes. Compressed bytes spell anything given
enough of them, so the `stream` keyword only counts when it directly follows the
dictionary describing it, and keywords inside an already-identified body are
skipped.

**Anvil** has no magic at all. It is recognised by the shape of its location
table: a whole number of 4 KiB sectors, every entry pointing inside the file and
past the two header sectors. Compression is per chunk, so one region file can
mix gzip and zlib.

**TIFF** accepts both 8 (Adobe's registration of Deflate) and 32946 (the
original extension), in either byte order. **WOFF** compresses per table, and
its directory checksums are over the *uncompressed* table, so unlike a PNG chunk
CRC they are a genuine constraint on a recovery.

## Two fragments with something missing between them

The case this exists for is a capture that lost packets mid-transfer. You have
the head of a stream, so you have its Huffman tables, and you have a later piece,
and between them is a hole.

```
toeknife head.bin --gap tail.bin --corpus /var/log/
toeknife head.bin --gap tail.bin --total-size 57482    # exact, no search
```

Two unknowns, different in kind. The *compressed* gap is the ordinary alignment
problem. The *output* gap is the hard one: the missing bytes produced some
number of output bytes, and until that number is right, every back-reference in
the later fragment that reaches across the gap lands on the wrong byte. Getting
it wrong does not fail. It silently reads the wrong text.

Where the container records the decompressed length, the gap is arithmetic
rather than search: the recorded total minus what the two fragments produce.
gzip's ISIZE and a ZIP entry's uncompressed size both give it outright, and
ToeKnife takes it automatically when it is there.

Without one it has to be searched for, and the signal is that a wrong offset
makes the cross-gap references read gibberish. How well that works depends
entirely on how much still reaches across:

| output gap | bytes read across it | search result |
|---|---|---|
| 1 KB | 1940 | found exactly |
| 2 KB | 1462 | found exactly |
| 4 KB | 805 | found exactly |
| 8 KB | 182 | 38th |
| 16 KB | 23 | hopeless |
| 24 KB | ~0 | nothing to score |

The window is 32 KB, so a gap approaching it leaves nothing reaching over and
nothing to judge. Search is reliable to about 4 KB and unusable well before the
window runs out. When it searches, it says how much evidence it had and shows
the runners-up, because nothing here proves an answer.

With the gap right, every byte recovered is correct: on a 57 KB stream missing
2 KB, 26,013 bytes recovered and none of them wrong.

## When the Huffman table is gone

A fragment that starts inside a dynamic block is unreadable. The table defining
its symbols is behind you, and rebuilding it from the body is not practical:
code lengths 0–15 over up to 286 symbols. Ralf Brown, who wrote two DFRWS papers
on recovering corrupt DEFLATE, put it plainly in ziprec's changelog — the search
was "intractible in the general case, but usable when the Huffman trees are
known".

The way out is to stop trying to rebuild the table and skip to the next one.
Streams emit block headers periodically and each dynamic header carries a fresh
table, so everything from the next boundary onward reads without outside help.

```
toeknife fragment.bin --scan       # find candidate boundaries
toeknife fragment.bin --resync     # decode from the best one
```

Four signatures, strongest first:

| | |
|---|---|
| `sync-flush` | a `00 00 FF FF` marker. `Z_SYNC_FLUSH` emits an empty stored block, so a fresh header starts byte-aligned right after it. Cheap and near-certain |
| `stored` | a stored header, where LEN and NLEN must be complements. Sixteen bits of redundancy |
| `dynamic` | a dynamic header whose code-length code and both tables build cleanly |
| `fixed` | three bits. Meaningless alone, off by default |

Candidates are decode-validated, because a dynamic header parses out of pure
noise about once every 4000 bits. The second stage is what makes the scan usable.

**How much this recovers depends entirely on how the stream was written.**
Measured on 250 KB of varied text with the first third of the compressed stream
discarded:

| written with | boundary found | output recovered |
|---|---|---|
| `Z_SYNC_FLUSH` every 8 KB (streaming, HTTP, pcap) | flush marker, 1.9 KB in | 201 KB, 26% of bytes known outright |
| no flushing (a static `.gz`) | next dynamic header, ~31 KB in | everything before that header is unaccounted for |

Block spacing is the whole story: about 1.3 KB between blocks when a compressor
flushes, against roughly 54 KB when it does not. So this works best in exactly
the case where packets go missing mid-transfer, and worst on a small chunk
carved out of a static archive.

The bytes it does not recover are not lost so much as deferred. On that same
250 KB run, 167,000 unknown output positions collapse to **20,000 unknown
bytes**, an 8:1 reduction, because a back-reference is an equality and the
positions share classes. Everything reported as known is checked: the test suite
asserts that zero known bytes ever disagree with the original.

### When the shape comes back but the meanings do not

A recovered code set is not a table. Reconstructing the shape — how many
literals sit at each code length, the end-of-block length, the 29 length and 30
distance code lengths — is enough to **parse**, because canonical Huffman hands
out code words in (length, symbol) order and every literal symbol sits below
end-of-block, which sits below every length code. The counts alone fix which
code words are literals and which are matches. What they never say is *which
byte* a literal stands for.

That is a real state to be in, and the tool decodes it directly rather than
pretending the assignment is known:

```
toeknife body.bin --codeset cs.json                    # opens the interactive view
toeknife body.bin --codeset cs.json --assignment       # the ledger, headless
toeknife body.bin --codeset cs.json --assignment --guess 0=t 3=' '
```

`--codeset` builds an ordinary session, so `--dump`, `--export` and the
interactive view all work on it; `--assignment` is the output mode that prints
what each literal code has turned out to mean.

In the tape view an unassigned literal gets its own **glyph** — `▒` against the
`░` of any other unknown byte — and deliberately not its own colour. Colour says
where a byte came from, and this one has not come from anywhere yet; it is as
unknown as any hole. What the shape tells you is what filling it buys, because
the two kinds of hole are not the same size: a copy chain is usually a handful of
positions, while a common literal code can be hundreds. The status bar carries
`codes known / total` beside the hole count.

The file is four keys — `a`, `e`, `L`, `D` — plus an optional `bit` saying where
the block body starts, and any provenance you care to keep alongside. That
optional key earns its place: decoding from the **wrong** offset with a valid
code set still reaches a clean end-of-block, so a missing offset does not fail,
it quietly produces a shorter wrong answer that looks like a recovery. Whoever
recovered the table knew the offset, so the file carries it; when it is absent
the tool says the result is unanchored instead of letting it read as a success.

Every literal becomes an unknown byte tied to its code, and every occurrence of
one code joins one equality class — the same union-find a back-reference uses,
because "these two positions came from the same literal code" is the same kind of
assertion as "these two positions hold the same byte". So identifying a single
occurrence settles that code everywhere it fired, and everything chained onto it.

On a 3,440-byte block with 28 literal codes, identifying the largest class first:

| occurrences identified | codes known | bytes known |
|---|---|---|
| 0 | 0 of 28 | 0 (0%) |
| 1 | 1 | 680 (19.8%) |
| 3 | 3 | 1,160 (33.7%) |
| 10 | 10 | 2,000 (58.1%) |
| 28 | 28 | 3,440 (100%) |

The leverage is that a code's class is far larger than the number of times the
code fired: the commonest one above fired **10 times and its class covers 680
positions**, because back-references chain onto it. One byte entered by hand
resolves a fifth of the block.

Provenance does not change and deliberately gains no new colour. A byte that
follows from a guess reads as guess-propagated whether it arrived through a
back-reference or through a shared code, because both are equalities read off the
bitstream and inventing a distinction would be false precision. What the tool
does add is the ledger — which codes have a meaning, how it was arrived at, and
how many positions each unassigned one still covers.

The representation also self-checks for free. Two literal codes in one table are
two different byte values, so if the constraints ever force two codes to the same
byte, the code set being decoded with is the wrong one, and the tool says so
instead of producing output.

## Letting a model fill the holes

```
toeknife fragment.bin --solve-propose --corpus /var/log/    # what it would do
toeknife fragment.bin --solve --corpus /var/log/            # do it
```

On the worked example below, this fills all seven classes correctly and
recovers `ers ` and ` fo`, which are exactly the guesses the old tool made you
write into a `guesses.txt` by hand.

The unknowns are classes, not positions, and that is the whole advantage. A
class appearing at three places in the output gets three independent readings of
what belongs there, from three different surrounding contexts, all voting on one
byte. Ordinary infilling gets one context per unknown. Scoring also runs both
ways: a candidate is charged for how well it predicts the bytes that *follow*
it, since a causal model reading only leftward ignores the right-hand context,
and after a hole that context is usually known and usually decisive.

With no corpus to hand, `--self-train` builds the model from the stream's own
recovered bytes. The readable part of a stream is often the best available
sample of the part that is missing.

**Confidence ranks well and is not a probability.** Measured on a carved
licence-text stream with a held-out order-4 model, against ground truth:

| `--solve-confidence` | classes filled | precision of new fills |
|---|---|---|
| 0.90 | 163 | 60.7% |
| 0.99 | 163 | 90.8% |
| 0.995 | 100 | **99.0%** |
| 0.999 | 44 | 97.7% |
| 0.9995 | 25 | 100.0% |

So it is a dial, and not even a monotonic one: 0.995 beats 0.999 here because
the tail is small enough to be noisy. The default is 0.995, picked by measuring
rather than guessing. It started at 0.99, which turned out to give 90.8% where
0.995 gives 99.0% for a third fewer fills.

How much it buys depends mostly on how much was already known: a hole surrounded
by other holes has no context to judge it by, and no threshold rescues that.

**That table is one stream and one model, and the figure travels only within
its regime.** The same 0.995 floor, measured against the real text
(re-measured 2026-10-28), with the
held-out phrack article p49-14 and an order-6 model trained on the rest of the
archive:

| case | 0.995 fills | precision |
|---|---|---|
| article with its front chopped off (`stack.bin`) | 21 classes | 100% |
| same article, one 536 B packet lost | 167 classes | 29.3% |

On short holes in decoded text, confidence ranks well: classes scored 0.90-0.99
were right 95% of the time and those under 0.5, 19%. (An earlier measurement put
the carve at 65%; that came from a few bytes misread at the fragment's start,
fixed since, not from the model.) Inside a run of 256+ unread bytes nothing
ranks: fills were right about 19% whatever the floor, because each one becomes
context for the next. `^f` and `--solve` stop before filling inside such runs
and say so (`--fill-long-runs`, or "continue anyway", overrides). Models trained
by ToeKnife are now calibrated on held-out text, so their floor is a measured
precision; the side pane shows each fill's figure.

### Layering passes

Repeating a pass with the same model gains nothing: the solver already loops
internally, filling, propagating, and re-scoring only the classes whose context
changed. Alternating two models gains nothing either, for the same reason: a
second opinion re-reads the same evidence.

This section used to claim that a descending threshold ladder "measures as
*exactly* equal to a single pass at its lowest rung". That was measured on a
10-class specimen where it is true and everything clears in the first round
either way, and it is false wherever the tool is actually useful: on a 513-class
phrack carve at a 0.70 floor, walking eight rungs by hand gave 80.3% precision
against 72.3% for one pass at the bottom — 402 more correct bytes and 760 fewer
wrong ones. Order matters because a round applies every class above the floor
using scores computed *before* any of them landed, so the doubtful fills become
context for the sure ones instead of the other way round.

`ctrl+f` now descends by itself, filling the 32 surest classes at a time and
re-scoring what they touched, which is why the threshold is described as a floor
throughout. Doing it by hand is no longer worth anything: one press reaches 80.7%
in 44 seconds where the eight-press ladder took 94. Filling strictly one at a
time, the obvious next step, buys 0.1 points for 4.7x the time.

Alternating the model with the container's **checksum** is different, because a
checksum is not an opinion. `--solve-loop` runs both until neither makes
progress. Each class the model fills is one fewer unknown in the checksum's
linear system, and four or fewer are solvable outright; each class the checksum
pins is new context for the model.

Its most valuable outcome is not filling more. It is this:

```
8 of 10 classes filled over 2 round(s), 2 left; THE CHECKSUM CONTRADICTS WHAT IS
ALREADY FILLED IN, so at least one of these bytes is wrong
```

The model was confident and wrong; the checksum proved it without being able to
say which byte. Nothing else in the tool can do that. Where it does succeed it
says so with the same precision: *"the checksum confirms every recovered byte"*,
and that phrase is only used when a checksum was actually computed and compared,
never for a size match on output that still has holes.

### Predictors

The built-in order-N byte model is the default and needs nothing installed.
Both directions are trained, as ziprec does — its `mklang` builds a forward and
a reverse model by default and notes that forward-only *"degrades
reconstruction"*. The reason is sharp here: the holes that matter sit at the
edge of the pre-fragment window, where everything to the left is permanently
missing and everything to the right is known. Measured:

| model | filled | precision |
|---|---|---|
| forward only (`--forward-only`) | 100 | 99.0% |
| forward + reverse (default) | **109** | **99.1%** |

**Order scales with corpus size, and does it itself.** `--order` defaults to
`auto` and is resolved per corpus, so swapping the corpus in `f5` re-decides it:
3 under 32 KB, 4 under 128 KB, 5 under 8 MB, 6 above. A number you pass always
wins.

The tiers are measured, and the measurement's own verdict is
that this is a small dial. The two ways of judging it disagree: held-out bits per
byte prefers order 2 below 200 KB and order 3 up to 1 MB, and never prefers 4 or
more; filling holes on a stream whose answers are known prefers 5 from 128 KB up,
because the solver only acts where it clears 99.5% and what matters there is the
sharpest predictions rather than the average byte. On a real carve with a corpus
of the same kind but not the same text, order moved precision by at most two
points across orders 2 to 6, while corpus size moved it from 17% to 34%.

Above 8 MB of corpus the longest context wins outright — precision on the carve
track is monotonic in order at both 8 MB and 15 MB, and 15 MB at order 6 recovers
52 correct bytes against order 4's 50. **ziprec's default of 6 is right**, for a
corpus 200 times past where that crossover starts.

What steps the order back down is memory, judged from what the data *is* rather
than how much of it there is, because a model's size is its number of distinct
contexts and prose reuses its 6-grams where machine code does not. Measured peak
per byte of corpus, for the forward/reverse pair:

| order | mixed text | binary |
|---|---|---|
| 3 | 20 | 52 |
| 4 | 78 | 98 |
| 5 | 143 | 199 |
| 6 | 271 | 325 |

The text column is from the whole 30 MB Phrack corpus; plain English prose is
about a third of it at order 6, so the estimate runs high on a novel. That is
deliberate — its only job is to warn before the memory is spent.

**Nothing caps how much corpus is read.** It used to: the readers stopped at 8 MB
and additionally skipped any single file over 8 MB, both silently, so pointing the
tool at a 30 MB corpus trained a model on the first 8 MB in directory order and
said nothing, while the largest file in the corpus was the one guaranteed to be
dropped. A byte count was the wrong instrument anyway, since 8 MB of prose and
8 MB of machine code differ by 2x in what they cost.

What protects memory instead is the order, and when it moves the tool says so:

```
corpus: 969 file(s), 30,208,986 bytes; context order 5 (lowered from 6 to stay
inside 3.0 GB — --order 6 overrides, TOEKNIFE_MEMORY_BUDGET moves it); roughly
4.0 GB of memory, give or take a factor of two
```

In the TUI that line appears in the picker before the second `enter`, so backing
out is one keystroke, and an estimate over budget is a warning rather than a
refusal. The budget is half of free RAM with a 3 GB floor, so a workstation is not
held to a figure chosen for a small VM. It does mean the same corpus can earn a
different order on a different machine; the order is reported before training,
recorded in the model's provenance, and `--order` pins it when a run has to
reproduce.

Most of that memory turned out to be unreadable anyway. The lookup backs off past
any context seen fewer than `min_obs` times, so storing one is memory spent on
something that can never be consulted — and on high-entropy data that is nearly
all of them. Training now skips them outright, using ziprec's bound from
`mklang`: a context occurs no more often than any trigram inside it, so a context
whose leading trigram is too rare cannot itself be frequent enough to read, and
every longer context at that position is refused with it. **10.9 MB of ELF at
order 6 went from 7.8 GB to 3.4 GB with the model's predictions bit-identical**,
and on random bytes it stores 6% of what it used to. `prune()` does the same
thing after the fact for models built the old way, which takes a third off a
saved model's size.

Train once and reuse:

```
toeknife --save-model english.tkm --corpus /usr/share/doc
toeknife fragment.bin --model english.tkm --solve
```

A saved model records the order it was built at, along with what it was trained
on, and the `f5` picker shows both.

Inside the TUI, a finished train offers to keep the model there and then, and
the save row on `f5` writes whatever is loaded at any later point. Models go to
`~/.toeknife/models` by default, which is the first place the picker looks, so
saving is what makes one appear under "models you built earlier" next time. The
path is editable and `$TOEKNIFE_MODELS` adds directories to the search. Worth
doing because training is the expensive half: **17.9 MB of Gutenberg prose
resolves to order 6 and takes 130 s and 1.36 GB to train, against a 71.6 MB file
that loads back in 1.1 s**. On prose the file runs 4-6x the corpus size.

The offer arrives on its own because a save key alone did not work as a design: the
model is only worth saving if you remember to before the process ends, and the
one moment the question answers itself is when the cost has just been paid.

Everything slow in the TUI — training, the filesystem sweep, the boundary scan,
scoring every class for `f4` and `ctrl+f` — runs on a thread with a bar, a note
saying which pass is running, and a clock. `esc` calls it off. Before this, each
of those ran inside the key handler that started it, so the status line it set
never painted and the app ignored the keyboard until it finished; a two-minute
train was indistinguishable from a crash. Nothing is committed until a run
finishes, so a cancelled train leaves the model you already had, and a cancelled
`ctrl+f` leaves what it filled one `ctrl+z` from gone.

With no corpus it trains on the stream's own recovered bytes, which is often the
best available sample of the part that is missing — but only when there are
enough of them. Below 256 recovered bytes that model is a rumour rather than
evidence and is dropped, and a block's own Huffman table is used instead where
it says anything (a fixed-Huffman block's implied frequencies are flat, so it
says nothing). A small fragment with fixed tables and no corpus therefore has
nothing to predict from, and the tool says so rather than inventing a
predictor:

```
predictor: emptyx1
note: nothing to predict from — too few recovered bytes to self-train and no
usable table prior. Pass --corpus data of the same kind
```

`--self-train` *adds* the stream's own bytes to a corpus model rather than
replacing it, so a corpus stays bidirectional. The fit score is reported for the
corpus model only: a self-trained model is measured on the very bytes it was
trained on, where a good number means it memorised them, not that it can predict
anything.

It trains on decoded bytes and on what you typed, never on the model's own fills.
That distinction is worth 5 points of precision: training on everything readable
teaches the model its own wrong guesses as ground truth, and each one then votes
for itself wherever it fits. It is also why `--self-train` is off by default
whenever a corpus is loaded — even purified, it measured *worse* at every weight
than not doing it at all, monotonically.

### The corpus matters far more than the model

This is the single most useful thing measured here. Mean log2 probability per
byte on held-out data, where -8.0 is what knowing nothing scores:

| | on English | on ELF |
|---|---|---|
| trained on English | **-3.9** | -17.2 |
| trained on ELF | -5.2 | **-2.7** |
| no model at all | -8.0 | -8.0 |

Two things fall out. Binary is *more* predictable than English once the model
has seen the right kind of it. And **a mismatched corpus is far worse than no
corpus**: -17.2 against -8.0. It does not degrade to ignorance, it degrades past
it, becoming confidently wrong, and confidence is exactly what the solver's
threshold reads.

So ToeKnife scores the model against the bytes it has already recovered and
says so before solving:

```
data looks like: ascii-text (100% of recovered bytes are printable ASCII)
WARNING: this model scores -17.2 bits per byte on the recovered data, worse than
the -8.0 of knowing nothing. It is trained on the wrong kind of data and will be
confidently wrong.
```

Point `--corpus` at data of the same kind and the byte model becomes a model of
that kind. That is worth more than any change of architecture below.

### Language models

Available for text: `pip install 'toeknife[ml]'`, then `--predictor hf:distilgpt2`.
It ranks candidates by the log-probability of the whole sequence, since a
byte-level BPE merges across the boundary and a candidate changes how the bytes
before it tokenise. Given a context it cannot read it returns nothing rather
than numbers.

It is the better predictor in the abstract. On held-out licence text it needs
**1.2 bits per byte where the order-4 n-gram needs 2.5** — about twice as good,
and no corpus you assemble will beat it at English.

**It still loses at this job, for a structural reason.** On the sample
fragment's seven unknown classes, with ground truth known:

| predictor | correct top choices |
|---|---|
| order-4 n-gram | 4 of 7, the wrong ones below the threshold |
| distilgpt2 | 0 of 7 |

Most holes worth solving sit in the *pre-fragment window*, where there is little
or no left context. A causal model conditioned on nothing is near-uniform, and
its low confidence there is honest rather than broken. The n-gram wins because
its backoff degrades gracefully on short context and because it scores a
candidate against the bytes that *follow* it, which is where the evidence
actually is when the left side is missing.

An infilling model that reads both directions is the right shape. ByT5 is
exactly that and the off-the-shelf checkpoint does not work (below). That gap is
open.

**Those two models are not the question, though, and the answer changes with a
bigger one.** Measured against a frontier model on 40 classes from a real carve,
answers committed before scoring:
**72% against the n-gram's 42%**, and the whole gap is context — the n-gram is
flat at ~42% however much surrounding text it gets, while the language model goes
from 50% to 82%. So the structural argument above is true of small causal models
and not of the approach.

It still does not make one shippable: one query per class against 4,041 classes,
no calibrated confidence to threshold on, and it cannot use the equality structure
that is this tool's actual advantage. What that measurement *did* pay for was a
bug it exposed in the n-gram's own scoring, worth 10 to 16 points of precision at
matched coverage.

Two integration bugs were fixed on the way, both of which made the LM look worse
than it is: candidates the model declined to score were defaulting to zero,
which beats every real (negative) log-probability and split the confidence
across a pile of unscorable bytes; and whole-sequence scores were compared
against per-byte ones without normalising. It now runs in seconds instead of
twenty minutes.

**ByT5 looked like the right tool and was not, and that now holds to 4x the
size.** Retested at `base` (582M) and `large` (1.2B) on the 40-class benchmark:
**5, 2 and 6 correct of 40** for small, base and large, against the n-gram's 17.
Flat and noisy, so the conclusion below was not a size artefact. The reason looks
distributional -- ByT5 denoises ~20-byte spans in otherwise clean text, while
these are 1-byte spans in text where 40% of the bytes are already gone.

 It is byte-level, so one
forward pass gives all 256 probabilities instead of one pass per candidate, and
span infilling is its pretraining objective. But `byt5-small` off the shelf
echoes the context rather than filling the mask (`brown <mask> jumps over` →
`" jumps over"`, through transformers' own `generate`), and it ranked the truth
96th where distilgpt2 ranked it 2nd. The raw checkpoint needs fine-tuning to be
useful. A fine-tuned ByT5, or a byte model trained on the file types you carve,
is the direction with something left in it.

**Seven models later, the n-gram still wins.** Llama 3.1 8B, Gemma 3 27B and
EvaByte 6.5B all score exactly 13/39 on left-context prediction — three
architectures, two tokenisation schemes, a 3x parameter range, one number — where
the n-gram gets 17/40. Only a frontier reasoning model beats it, at 29/40, and
only because the masking convention can be explained to it in a sentence.

**Mixing predictors did not work, and the code says so.** Combining the corpus
model with a self-trained one, or with the frequencies implied by a dynamic
Huffman table, raises precision only by filling less. Compared at matched
coverage rather than matched threshold, the plain corpus model wins on both
axes: 100 fills at 99.0% against the mixture's 67 at 98.5%. The machinery is
there because the language-model tier needs it, but it is not the default and it
is not an improvement.

**The table-alphabet constraint made no difference either** on the data tested:
identical fills, identical precision. It only has anything to say about a
dynamic block whose table omits byte values, and where the model was already
proposing bytes the table allows, eliminating the rest changes nothing. UTF-8
constraints are the ones with teeth, and only on UTF-8.

Model fills get their own provenance and their own colour, distinct from both
decoded bytes and your own guesses, and a whole run undoes as one step.

### Solving against the container's checksum

```
toeknife FILE --checksum-solve --corpus /var/log/
```

A checksum is a constraint on the unknown bytes, not just a verdict at the end.
CRC32 is affine over GF(2), so unknown bytes enter it *linearly* and recovering
them is a linear system rather than a search. Because the unknowns are classes
rather than positions, a stream with eleven thousand unknown positions across
thirty-five classes is a system in two hundred and eighty unknowns, and
thirty-two equations bite hard on that.

Four unknown classes are generically determined outright. More than that, the
checksum removes 32 bits of freedom and the model chooses among what survives:
five unknown bytes are 2⁴⁰ possibilities before the checksum and 2⁸ after it.

On a zlib stream compressed against a preset dictionary the file does not
contain, with both a CRC32 and an Adler-32 available:

```
checksum: exactly one of 1048576 combinations satisfies crc32 and adler32
recovered b'ensic'
size 19200/19200 ok  crc 3f6649d8/3f6649d8 ok  adler 1967a306/1967a306 ok
```

**Adler-32 on its own is much weaker than its width suggests** and will accept
wrong answers. It is two sums modulo 65521, and across five unknown bytes the
reachable sums number in the low thousands. Asked to recover five bytes from
Adler-32 alone, the solver found `etesa` where the truth was `ensic`: both have
the same byte sum and the same weighted sum, so both satisfy it. So every
surviving candidate is counted and the count is reported. One survivor is a
result; three survivors is the model guessing, and it says so.

Adding a real CRC32 turns that wrong answer into an honest "no combination
fits", which is the outcome a forensic tool should prefer.

## Command line

Everything works headless so it can go in a pipeline.

```
toeknife FILE                      open the interactive view
toeknife --data HEX                take the fragment as hex
toeknife FILE --list               rank starting offsets, with previews
toeknife FILE --dump               symbol stream, infgen style
toeknife FILE --dump -v            ... annotated with each symbol's bits
toeknife FILE --tables             the Huffman tables in full
toeknife FILE --export out.bin     write recovered bytes
toeknife FILE --guess -3=' ' -2=f  fill holes non-interactively
toeknife FILE --header hdr.bin     take Huffman tables from a separate fragment
toeknife FILE --scan               find block boundaries in a table-less fragment
toeknife FILE --codeset cs.json    decode with a recovered code set (meanings
                                   unknown); add --assignment for the ledger
toeknife FILE --resync             decode from the best boundary found
toeknife FILE --solve-propose      what the model would fill, and how sure it is
toeknife FILE --solve              fill the holes it is confident about
toeknife FILE --checksum-solve     pin unknown bytes with the container checksum
toeknife FILE --gap TAIL           assemble two fragments across a missing piece
toeknife FILE --entry N --verify   check a container entry against its CRC
```

`--header` replaces the old `--dynamic`: when the block header survived in one
place and the body in another, point at the header fragment and its tables get
used to decode the body.

Full streams work too, wrapped in zlib or gzip or raw. ToeKnife detects the
container, and the blocks pane becomes a full header and Huffman table browser
with every field still pointing at its bits.

## Worked example

The fragment is the tail of a fixed-Huffman compression of

> Some numbers are one, two, three, and four. four is the only of those that has four letters in it

```
$ toeknife --data '2f2dd203930a99c50a2519a90af97939950af9690a2519f9c5a90a251989250a1989c5102539a9252520e332f314324b00' --dump
! toeknife
! 49 bytes, fragment, start bit 3
!
literal 'ur.
match 5 6   ! unresolved
literal ' is the only of those that has
match 6 35   ! unresolved
literal 'lett
match 4 79   ! unresolved
literal 'in it
end
! 7 unknown byte classes covering 17 output positions
!   hole at -31: also at 48
!   hole at -30: also at 49
!   hole at -29: also at 50
!   hole at -28: also at 51
!   hole at -3: also at 3, 38
!   hole at -2: also at 4, 39
!   hole at -1: also at 5, 40
```

Seven bytes to work out, and it tells you where each one is echoed. `lett` at
position 44 followed by a four-byte hole reads like `letters `, and position 3
sits after `ur.` so it wants a space then `fo`:

```
$ toeknife --data '...' --guess -3=' ' -2=f -1=o -31=e -30=r -29=s -28=' ' --export out.bin
$ cat out.bin
ers ???????????????????????? four. four is the only of those that has four letters in it
```

In the interactive view that is seven keystrokes with `tab` between them.

## Correctness

The decoder is differentially tested against `zlib` for stored, fixed and
dynamic blocks, multi-block streams, overlapping copies, empty distance trees
and RLE-only data, and its symbol stream is compared line for line against Mark
Adler's [infgen](https://github.com/madler/infgen). They agree exactly, 732
lines deep, on real streams.

```
pytest
```

## Fixed since v1

Found while rewriting, all confirmed against zlib:

- **Distance code 29 was undecodable.** The distance table was built as
  `range(0,29)`, giving codes 0 through 28. Code 29 covers distances 24577 to
  32768, so any long-range back-reference failed. Real files hit this.
- **Dynamic code lengths were read as two independent runs.** RFC 1951 encodes
  the literal/length and distance code lengths as one concatenated sequence, so
  a repeat code may straddle the boundary. Reading them separately corrupts the
  distance table whenever that happens.
- **The `--table` distance filter did nothing.** It compared a binary code
  string against `'2'`, which is never equal.
- **Symbol decoding was a linear scan** over every code in the table, for every
  symbol. Fine on a 49-byte fragment, unusable on a real file. Now a flat lookup
  table.
- **HLIT and HDIST were unbounded.** Both fields are five bits, so they can
  encode 288 literal/length codes and 32 distance codes, but RFC 1951 allows at
  most 286 and 30. zlib rejects the excess with "too many length or distance
  symbols"; we accepted it. Beyond being wrong, it was a measurable share of the
  false positives when scanning damaged data for block headers.

The overlapping-copy logic, for the record, was correct. I checked before
changing it.

## Justification

In section 1.1 of RFC-1951:

*The data format defined by this specification does not attempt to:*
*Allow random access to compressed data;*

This doesn't mean that it's impossible, just that the specification isn't
written to for this purpose. ToeKnife helps to assist with some random access.
