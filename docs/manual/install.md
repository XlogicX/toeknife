# Install

ToeKnife needs Python 3.10 or newer. The only required dependency is Textual,
for the interface.

```
git clone https://github.com/XlogicX/ToeKnife
cd ToeKnife
python3 -m venv .venv
.venv/bin/pip install -e .
```

That puts the `toeknife` command in `.venv/bin/` and nowhere else. Link it
onto your PATH so you can run it from anywhere:

```
ln -s "$PWD/.venv/bin/toeknife" ~/.local/bin/toeknife
```

`~/.local/bin` is on the PATH of most Linux setups (`echo $PATH` to check).
The link points into the venv, so pulling new code needs no reinstall. The
rest of this manual assumes `toeknife` runs as a bare command.

Keep your own files out of `.venv/`. Rebuilding the venv deletes it.

## The byte model (optional)

Byte models (`.tks` files) need PyTorch:

```
.venv/bin/pip install -e '.[ml]'
```

The n-gram models, which most workflows use, need nothing extra.

## Where things live

| Path | Holds |
|---|---|
| `~/.toeknife/models/` | saved models; `f5` lists them, `$TOEKNIFE_MODELS` adds directories |
| `~/.toeknife/trash/` | models removed from `f5`, until you delete them |
| `~/.toeknife/corpora/` | downloaded training text |
| `samples/`, `samples/damaged/` | the repo's specimens |

## Try it

```
toeknife samples/fixed-fragment.bin
```

Three panes, decoded text in the middle. `ctrl+q` quits.
