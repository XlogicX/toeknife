# Damaged samples

The samples one level up are about carving: a fragment that begins mid-stream
with its Huffman table behind the damage. These are about the other half of the
tool — containers, and the fact that **a corrupted DEFLATE stream usually still
decodes**. Each file is built to make one behaviour visible, and every number
below was measured against these files, not estimated.

Rebuild them with

```
.venv/bin/python samples/build/makedamaged.py
.venv/bin/python samples/build/makepcap.py
```

The payload is `payload.txt` in this directory: the first 256 KB of *The
Adventures of Sherlock Holmes*, public domain, pinned so a clean checkout
rebuilds byte-identical files. It is prose, so recovered bytes read as text, and
it is in none of the corpora the manual trains on, so a model never holds the
answer. Nothing here is
random that is not seeded, and no header carries a timestamp.

| file | what it shows |
|---|---|
| `evidence.zip` | one flipped bit that only the CRC notices; 200 bytes of burst damage that the size notices; a stored entry and a genuinely encrypted one, both refused with a reason |
| `evidence-noeocd.zip` | the central directory is gone, so entries are found by scanning for local headers — and still verify |
| `evidence-sfx.zip` | 4 KB prepended, so every recorded offset is wrong by a constant the tool measures |
| `capture.gz` | a member that will not decode takes the position of every member after it |
| `preset-dict.zlib` | FDICT, dictionary absent: references into it become holes rather than a refusal |
| `capture-head.bin` + `capture-tail.bin` | two fragments with 2 KB of output missing between them |
| `carved-midblock.bin` | the bad case: no flushing, table gone, `--scan` and `--routes` |
| `lost-table.bin` | raw deflate, no container: the first block's table is destroyed and 95.7% of the payload still comes back byte-exact |
| `bitrot-header.gz` | one flipped bit in a dynamic header: nothing decodes until `f7` undoes it, and the gzip CRC says whether it undid the right one |
| `plot-damaged.png` | chunk CRC localises damage; the zlib Adler-32 is what actually fails |
| `plot-splitrun.png` | an IDAT run the spec forbids, reported rather than silently joined |
| `report-noxref.pdf` | streams found by scanning, because the cross-reference table is what damage takes |

Start with `evidence.zip`. It is the one that makes the case for the whole
container layer.

**`guide.html` in this directory is the step-by-step version**: per sample, which
keys to press in the interactive view and in what order, with what each one puts
on screen. Open it in a browser. This README is the command-line half; the guide
is the half you read next to the terminal.

## `evidence.zip` — the checksum earns its keep

```
$ toeknife samples/damaged/evidence.zip
samples/damaged/evidence.zip: zip container, 5 entries
  EOCD at 101848: 5 entries, central directory 279 B at 101569

  #      offset       comp     uncomp     crc32 method    fl name
  0          39      10502      24000  92aa9c69 deflate   -D notes.txt
  1       10580      37637      96000  8d7c5e32 deflate   -D rotted.md
  2       48258      50943     130000  2dc67a69 deflate   -D shredded.md
  3       99241        256        256  8ac51697 stored    -- stored.bin
  4       99537       2032       4000  54bfcbdb deflate   E- secret.txt
```

Five entries, four kinds of answer.

```
$ toeknife samples/damaged/evidence.zip --entry notes --verify
notes.txt: size 24000/24000 ok  crc 92aa9c69/92aa9c69 ok
verified: the recovered bytes are the original bytes.

$ toeknife samples/damaged/evidence.zip --entry rotted --verify
rotted.md: size 96000/96000 ok  crc 6da73ded/8d7c5e32 MISMATCH
FAILED: the decode does not match what the container recorded.

$ toeknife samples/damaged/evidence.zip --entry shredded --verify
shredded.md: size 130019/130000 MISMATCH  crc ae6697a7/2dc67a69 MISMATCH
FAILED: the decode does not match what the container recorded.

$ toeknife samples/damaged/evidence.zip --entry stored --verify
toeknife: stored.bin: stored, not compressed

$ toeknife samples/damaged/evidence.zip --entry secret --verify
toeknife: secret.txt: entry is encrypted
```

`rotted.md` is **one flipped bit** — byte 12545, bit 0 of the compressed data —
chosen by searching for a flip that leaves every Huffman code legal. The stream
inflates to exactly the right length. 64 of the 96,000 output bytes are wrong.
Nothing in the decode has anything to say about it, and 32 bits of arithmetic at
the end of the entry catch it.

`shredded.md` is the loud version: 200 bytes of the compressed data replaced
with garbage. It still produces 130,019 bytes where 130,000 were recorded, 96.9%
of them printable ASCII, the same share as the original. The first 64,637 bytes
are correct and 61,426 differ from the original, which reads like this at the
seam:

```
sticking up a station from time to time, or stopping the wagL anon byra one
elsenly fastweenhave ave ave atake me oyour “Td the tl yournl-holda tall you my
cortall ter whte
```

That is what damage looks like when nothing checks: prose for two thirds of the
file, then something that is almost prose, and no error anywhere.

`secret.txt` is really encrypted, with the original PKWARE stream cipher and the
password `forensics`, rather than an entry with the flag bit set as a prop. The
point is the refusal, not the cipher.

### The same archive, two ways it can arrive

`evidence-noeocd.zip` is that file with the central directory and the
end-of-central-directory record cut off, which is what truncation takes first:

```
$ toeknife samples/damaged/evidence-noeocd.zip
samples/damaged/evidence-noeocd.zip: zip container, 5 entries
  ! no end-of-central-directory record: the archive is truncated, or this is only a piece of one
  ! recovered 5 entries by scanning for local headers; sizes may be missing for streamed entries
```

All five come back, and `--entry rotted --verify` still reports the same CRC
mismatch — the local header carries a copy of the checksum.

`evidence-sfx.zip` is the same archive with 4096 bytes glued to the front, so
every offset the directory records is short by 4096:

```
$ toeknife samples/damaged/evidence-sfx.zip
  EOCD at 105944: 5 entries, central directory 279 B at 101569
  offsets are shifted by +4096 bytes (prepended data, or a truncated front)
```

The shift is measured, not guessed: the EOCD says how big the directory is and
where the file ends, and the difference is arithmetic. Entries verify afterwards.

## `capture.gz` — one bad member costs you the rest

gzip records no compressed length, so the end of a member is found by decoding to
it. Member 1 here is damaged 40 bytes into its deflate data:

```
$ toeknife samples/damaged/capture.gz
  member 0 at 0: OS Unix, mtime 0
  member 1 at 16672: OS Unix, mtime 0, deflate data did not decode (bad-table)
  member 2 at 36959: OS Unix, mtime 0
  ! member 1 did not decode; the walk picked up again at member 2 (offset 36959), the next gzip header found by scanning whose data decodes to its end
```

Member 0 verifies against its trailer. Member 2 is found by scanning for `1f 8b`
and taken only because its data decodes to its own end; member 1's CRC32 and
length are read from the 8 bytes in front of it.

## `preset-dict.zlib` — a dictionary that is not in the file

FDICT is set and the 40-byte dictionary was never shipped. Other tools refuse
this. Here the references reaching into it become holes, which is the same
problem as a reference behind the start of a fragment:

```
$ toeknife samples/damaged/preset-dict.zlib --solve --verify
toeknife: zlib stream: compressed against preset dictionary 3fb40f8b, which is
not in this file: references reaching into it become holes
data looks like: utf-8 (100% of sampled chunks decode as UTF-8)
solver: filled 0 of 28 classes (0 output positions) in 0 rounds; stopped
because nothing above 99.5% confidence. 28 classes still open.
inconclusive: 28 unknown byte classes remain (1608 positions). Fill them and check again.
```

19,200 bytes of output, and the whole missing dictionary reduces to 28 unknown
bytes. With no corpus the model trains on the stream's own bytes, and none of
the 28 clears the floor: nothing filled, and nothing claimed. `--dict` finds the
real dictionary when a file holding it is to hand.

## `capture-head.bin` + `capture-tail.bin` — a hole in the middle

The case a lossy capture produces. 2048 output bytes are missing between the two
fragments. With the size recorded somewhere, the gap is arithmetic:

```
$ toeknife samples/damaged/capture-head.bin --gap samples/damaged/capture-tail.bin --total-size 60000
gap: the tail after the loss starts with a block header of its own, which decodes cleanly to the end; it is read from there
gap: output gap of 2,048 bytes, from the recorded decompressed size
```

The tail was cut at a flush, so it opens on a block header and carries its own
table. 48,665 of the 60,000 bytes come back known, every one of them right; the
rest are the gap and the copies later blocks make out of it.

Without the size, the gap has to be searched for, and the evidence is how much
still reaches across it:

```
$ toeknife samples/damaged/capture-head.bin --gap samples/damaged/capture-tail.bin --dump
gap: gap of 709 bytes, the best of 8,193 tried (6097 bytes read across it); a search result, which nothing here proves
gap: it scores 1.09, against a median of 1.07 over 200 gaps sampled; that barely stands out, so the gap may be past the 8,192 searched (--gap-max)
```

It gets this one wrong: 709 against the true 2,048. With no corpus the score
rests on the stream's own bytes, the best gap barely beats the median, and the
tool says so. The recorded size is the reliable route; the search is a lead.

## `carved-midblock.bin` — the bad case, on purpose

60 KB carved out of the middle of a 100 KB stream written with no flushing at
all. It begins inside a dynamic block, so the table is gone, and the next header
is a long way off.

```
$ toeknife samples/damaged/carved-midblock.bin --scan
position                     kind          output       blocks  holes  status     note
bit 234216 (byte 29277)      dynamic        80813 B out    2 blk   8787 holes  truncated  HLIT=275 HDIST=30 HCLEN=17
bit 477631 (byte 59703.7)    dynamic          553 B out    1 blk    480 holes  truncated  HLIT=272 HDIST=30 HCLEN=15
bit 446007 (byte 55750.7)    dynamic           28 B out    1 blk      0 holes  bad-code   HLIT=266 HDIST=8 HCLEN=9
bit 347089 (byte 43386.1)    dynamic           12 B out    1 blk      0 holes  bad-code   HLIT=260 HDIST=12 HCLEN=11
...
```

The list below the first two entries is the reason the scan validates by
decoding: a dynamic header parses out of noise roughly every 4000 bits, and
every one of those candidates built a legal-looking table before its body fell
over. Stored-block headers fire on noise too, once in 65,536 positions, and a
stored block "decodes" whatever follows it; one here produced 30,513 bytes of
compressed data, so a stored candidate is kept only when the stream carries on
correctly after it.

```
$ toeknife samples/damaged/carved-midblock.bin --resync --export out.bin
resynced at bit 234216 (dynamic), recovering 80813 bytes
wrote 111248 bytes to out.bin (13910 known, 97338 filled with '?')
```

**29 KB of the fragment is unreachable and most of the rest is holes**, because
nothing was ever flushed: back-references reach into a window that went missing
with the front of the stream. Compare `samples/streaming-fragment.bin`, written
with a sync flush every 8 KB, which gives 42,861 known bytes from a similar
carve. The difference is entirely the writer, and this file is here to make that
comparison possible. Every one of the 13,910 known bytes matches the original.

`--routes` is the other reason this file exists. It prints what could be tried
against a lost table, what each costs, and why the ones that do not apply do
not:

```
$ toeknife samples/damaged/carved-midblock.bin --routes
  [borrow]
    no  Take the table from a related capture
        needs a second capture | 1-2 s
        needs a second capture of the same or similar content; supply one and its token overlap decides the odds
    ... Sweep the 1,828-table dictionary
        needs the table census | 1.3 s
        applies here, but this route is not in this release
  [search]
    no  Judge this table against known plaintext
        needs operator-supplied plaintext | instant
        needs known plaintext to score candidates against; without it there is nothing to tell a right table from a wrong one
```

Hand it something you know is in the output and the answer changes:

```
$ tail -c +100001 samples/damaged/payload.txt | head -c 400 > known.txt
$ toeknife samples/damaged/carved-midblock.bin --plaintext known.txt --routes
  [search]
    OK  Judge this table against known plaintext
        needs operator-supplied plaintext | instant
```

## `lost-table.bin` — a destroyed table that costs almost nothing

The interesting question about a lost Huffman table is not whether it can be
recovered — it usually cannot — but how much of the file it takes with it. That
depends entirely on something the operator does not choose: whether the encoder
flushed.

A 60 KB article compressed in one go is a **single dynamic block**. Its header is
the only one in the stream, and destroying it is fatal: no parse, nothing to
resync to, and no route that recovers it. Approximate tables make this worse
rather than better — one built from a near-identical text will parse tens of
thousands of bytes to a clean end-of-block and produce *zero* correct ones.

This specimen is the same damage in a stream that survives it. 48,000 bytes,
raw deflate with **no container** — no CRC32, no recorded length, nothing outside
the bitstream to confirm anything — flushed every 2,048 bytes, so it is 49 blocks
of which 24 carry their own dynamic table. The first block's table is destroyed:
every bit between BTYPE and its first symbol is randomised, so the block still
announces itself as dynamic and the decode dies as `bad-table`.

```
$ toeknife samples/damaged/lost-table.bin
  67 bytes out, 32 known           <- effectively nothing
```

Press `f3`. The flush markers are `00 00 ff ff`, which is the highest-confidence
signature the scan looks for, so the boundaries are found immediately:

```
  bit 8560  sync-flush  ->  45,952 bytes, 0 holes
```

Take it, and the tape holds **45,952 of 48,000 bytes — 95.7% — byte-exact,
with no holes at all**. What is lost is the destroyed block's own 2,048 bytes and
nothing else.

`Z_FULL_FLUSH` is why, and it is worth understanding because the same file built
with `Z_SYNC_FLUSH` behaves quite differently. A full flush resets the window, so
no later block can reference the destroyed one and every byte after the damage
decodes exactly. A sync flush keeps the window, back-references reach into the
lost 2 KB, and the unknowns propagate forward through the match chains: the same
45,952 bytes are covered, but only 17,094 of them are *known* and the rest become
1,364 hole classes for the model and the operator to fill.

Neither number is available from the file itself. There is no checksum here, so
"95.7% recovered" is a statement about a specimen whose payload we happen to
have. On a real capture the tool can tell you the parse landed and the tape has
no holes; it cannot tell you the bytes are right.

## `bitrot-header.gz` — the one where the repair route earns its keep

Every other sample here is missing something. This one is missing nothing: all
48,000 bytes of the original are present, and one bit of the dynamic header is
inverted. That difference is the whole reason `f7`'s repair route exists —
recovery is a correction problem, so the search is the flips over the header
rather than the tables over the alphabet, and the difference is about a hundred
orders of magnitude.

It opens as a total loss:

```
$ toeknife samples/damaged/bitrot-header.gz
  out 0   0 bytes   size MISMATCH   bad-table
```

Zero bytes. The table that decodes the table is wrong, so nothing downstream of
it can be read. Press `f7`, choose **Repair a header whose bits are wrong**, and
enter:

```
680 of 680 1-bit flips over 680 header bits: 1 parse, and 1 matches the recorded size and checksum. That one is the original.
  1. bit 3 → 48,000 B, 48,000 known  checksum ok
```

One survivor out of 680 candidates. Enter applies it, and the status line
becomes:

```
  out 0   block #0 dynamic   48000 bytes   complete   crc ok   end
```

`crc ok` is the part that matters. The route itself can only say a candidate
parses; it has no way to know whether it parses *correctly*, which is why it
reports every survivor and picks none. Here the container settles it — the gzip
trailer records a CRC32 of the original, it matches, and the repair is a
recovery rather than a plausible-looking guess.

Getting that CRC took a fix. A gzip member's end is found by decoding it, so
damage that stops the decode used to stop the walk before it read the trailer —
discarding the one field that could confirm a repair of the very damage that
stopped it. For the first member the trailer's position is not a guess: a gzip
ends with it. It is read from there now, and labelled as inferred rather than
walked to, because that reading is wrong the moment a second member follows.

Two things this sample cannot show, both worth knowing. The repair is unique here
and usually is not: measured over all 250 single-bit flips of a real header, the
true undo survived every time but was the only survivor 129 times, and was top of
the list 78% of the time. And a two-bit flip is not found by a one-bit search — it
says so rather than offering a candidate.

## The two PNGs

`plot-damaged.png` has a byte flipped inside the second IDAT chunk, after its
CRC was computed:

```
$ toeknife samples/damaged/plot-damaged.png
  ! chunk CRC mismatch at 13666 (a chunk CRC covers the compressed bytes, so it
    localises damage but cannot verify a recovery)

$ toeknife samples/damaged/plot-damaged.png --entry 0 --verify
IDAT pixel data: adler de50f8de/0abdf8e8 MISMATCH
```

Two different checksums saying two different things. The chunk CRC says *where*
the damage is and can never say whether a repair worked, because it covers the
compressed bytes. The zlib Adler-32 at the end of the pixel stream covers the
output, so it is the one that judges a recovery.

`plot-splitrun.png` is undamaged but structurally illegal: a `tEXt` chunk sits
between the two IDATs.

```
  ! the IDAT run is not consecutive, which the format forbids; the gathered
    pixel stream is missing bytes from the middle
```

The pixels still verify here, because this file only interleaves rather than
losing anything. The warning matters anyway: a stream reassembled out of a
genuinely broken run is missing bytes in the middle and decodes to nonsense
somewhere without ever failing outright.

## `report-noxref.pdf`

Two `/FlateDecode` objects and no cross-reference table, which is the normal
state of a damaged PDF: the xref lives at the end and the end is what goes.

```
$ toeknife samples/damaged/report-noxref.pdf
  PDF 1.5
  2 /FlateDecode stream(s) found by scanning

  #      offset       comp     uncomp     crc32 method    fl name
  0          66       3839          ?         - deflate   -D obj 1
  1        3984       5422          ?         - deflate   -D obj 2

$ toeknife samples/damaged/report-noxref.pdf --entry 0 --verify
obj 1: adler 0fe52e4b/0fe52e4b ok
verified: the recovered bytes are the original bytes.
```

The uncompressed sizes are `?` because a PDF does not record them. The zlib
Adler-32 still does the verifying.
