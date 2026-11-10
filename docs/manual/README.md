# ToeKnife manual

ToeKnife recovers data from damaged DEFLATE: carves missing their start,
archives with bad bytes, captures with lost packets. It decodes what the bits
still say, marks every byte it can't recover, and helps you fill those with a
model, with the container's checksum, or by hand. Facts and guesses are drawn
in different colours throughout.

Find your situation below, read [Concepts](concepts.md) once, then follow the
workflow.

| You have | Workflow |
|---|---|
| a fragment of a stream whose start is gone (a carve) | [A. Carved fragment](a-carved-fragment.md) |
| a packet capture with a packet missing | [B. Lost packet](b-lost-packet.md) |
| a zip that is damaged, truncated, or wrapped in something | [C. Damaged zip](c-damaged-zip.md) |
| a file with a flipped bit in a block header | [D. Bit rot](d-bit-rot.md) |
| a block whose Huffman table is destroyed | [E. Lost table](e-lost-table.md) |
| two fragments of one stream with a hole between them | [F. Two fragments](f-two-fragments.md) |
| a multi-member gzip with one bad member | [G. Bad gzip member](g-bad-gzip-member.md) |
| a zlib stream that needs a preset dictionary you don't have | [H. Missing dictionary](h-missing-dictionary.md) |
| DEFLATE inside a PNG or a PDF | [I. Inside other formats](i-inside-other-formats.md) |

Two chapters the workflows refer to:

- [Models](models.md): training, calibrating, loading, saving and removing
  the models that fill holes.
- [Sessions and evidence](sessions.md): saving your work, writing the
  output, and tracing where any byte came from.

New here? [Install](install.md) first.

## The screenshots

`docs/manual/shoot.py` makes every screenshot: it runs the real command,
presses the keys the text describes, and saves the screen as SVG. Run
`.venv/bin/python docs/manual/shoot.py` after changing the tool and every
picture is regenerated; a step that can no longer be scripted fails there.
Every specimen but Workflow A's `stack.bin` is built from one Phrack
article by `python3 docs/manual/specimens.py p49-14.txt`, byte for byte;
each page says how to make the same damage from your own files.

`docs/manual/preview.sh` renders the manual to HTML with working links and
screenshots, for reading locally.
