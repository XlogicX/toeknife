# Samples

`streaming-fragment.bin` and `fixed-fragment.bin` are built by
`build/makesample.py`; the rest at this level are the original
proof-of-concept fixtures and predate most of the tool.

`damaged/` is a separate set, built by `build/makedamaged.py`: containers
rather than fragments, one file per way a file goes wrong — a ZIP whose entry
is off by a single flipped bit, a gzip whose second member takes the rest of
the file down with it, a PNG with a damaged chunk. See `damaged/README.md`.

Both made samples are raw DEFLATE with **the first third discarded**, which is
the case the tool exists for: a capture that begins mid-stream. What separates
them is how the stream was *written*, and that turns out to decide almost
everything about how much comes back.

## `fixed-fragment.bin` — decodes completely, no holes

95,555 bytes. Recovers **163,840 bytes, every one of them known, zero holes**,
byte-exact against the original: 62.5% of the source text, and the only thing
lost is what was in the discarded third.

```
toeknife samples/fixed-fragment.bin              # open it, then press f3
toeknife samples/fixed-fragment.bin --resync     # or go straight there
toeknife samples/fixed-fragment.bin --scan       # or just list the boundaries
```

**`f3` does this from inside the view**, which matters because opening a
mid-stream capture with no flags takes the fragment path and says nothing about
the better one: 2,211 known bytes and 2,449 holes here, against 163,840 known
and none.
`f3` scans, shows every boundary with the output and hole count it was measured
to give, and re-opens the stream at the one you pick. `f2` is a different
question — it chooses a bit offset for a headerless fragment, not a block
boundary.

Two properties of the writer do that, and neither is luck:

- **`Z_FIXED`** forces fixed-Huffman blocks, whose tables are defined by RFC 1951
  instead of carried in a header. Losing the head of the stream therefore loses
  no table; all that is needed is the right bit alignment, which the scan finds.
- **`Z_FULL_FLUSH`** resets the LZ77 window at each flush, so no back-reference
  after a marker can point before it. That is what removes the holes. The usual
  reason a fragment has unknowns is copies reaching into the window that went
  missing with the head — and after a full flush there is nothing back there to
  reach for.

21 blocks: 11 fixed, and 10 empty stored blocks, which is what a flush marker
*is*.

## `streaming-fragment.bin` — the realistic case, recovered in part

68,030 bytes, written the way an HTTP server or chunked response writes one:
`Z_SYNC_FLUSH` every 8 KB, dynamic Huffman. Sync flush ends the block but
**keeps** the window, so back-references still reach across the damage.

| | |
|---|---|
| boundary found | sync marker at byte 1,855, resync at bit 14,872 |
| output recovered | **172,032 bytes** across 43 blocks |
| known outright | **42,861 bytes**, and **0 of them disagree** with the original |
| still unknown | 137,445 positions, in only **8,274 distinct unknown bytes**: a 17:1 collapse |

That last row is the point of the representation. A back-reference asserts that
two positions hold the same byte, so unknowns are equality classes rather than
independent holes: the largest single class is **456 positions**, so one
keystroke fills 456 cells. `tab` walks 8,274 decisions to cover 137,445 positions.

### Filling the holes with a model

With no corpus the model trains on the 42,861 bytes it can read, and at the
default 99.5% floor `--solve` fills one class: 51 positions, all right.
Dropping the floor to 50% fills 397 classes, and only 26.3% of their positions
are right.

A corpus does better. `f5` downloads two English books, "Project Gutenberg:
Pride and Prejudice" and "Project Gutenberg: Moby-Dick", 2 MB together. Train on
them and fill:

```
toeknife samples/streaming-fragment.bin --resync --solve --solve-confidence 0.7 \
    --corpus ~/.toeknife/corpora/www.gutenberg.org-25e43630-pg1342.txt \
             ~/.toeknife/corpora/www.gutenberg.org-05cc7d1d-pg2701.txt
```

Scored against the original text, position by position:

| model | floor | classes filled | positions filled | right |
|---|---|---|---|---|
| no corpus | 99.5% | 1 | 51 | 51 (100%) |
| no corpus | 50% | 397 | 7,884 | 2,076 (26.3%) |
| the two books | 90% | 56 | 392 | 130 (33.2%) |
| the two books | 70% | 1,039 | 49,066 | 45,607 (93.0%) |
| the two books | 50% | 2,036 | 77,451 | 71,736 (92.6%) |

With the books the floor is a measured precision, taken on held-out book text,
and the best class the model reaches is 97.9%, so a 99.5% floor fills nothing.
The 90% row is the odd one: 56 classes, mostly wrong. Every run with the books
also prints a warning that the model scores -7.5 bits per byte on the
recovered data, close to the -8.0 of knowing nothing: the tool's own measure
that the books are a poor match for this sample. Read that warning before you
trust a floor.

`f5` trains on a corpus without leaving the tool, and `f4` then shows what it
would fill with confidence and context before anything is committed.

## What they decode to

Both carry `damaged/payload.txt`: 262,144 bytes of The Adventures of Sherlock
Holmes, the same pinned text as the damaged set, so the recovered tape reads as
prose and `build/makesample.py` rebuilds both files byte for byte.
