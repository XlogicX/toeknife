"""Every screen the app imports lazily exists.

The app imports most screens inside the method that opens them, so a screen
removed by mistake fails only when its key is pressed. DictionaryStop went
that way during a rewrite of its neighbour, and nothing else noticed.
"""
import re
from pathlib import Path

import toeknife.tui.modals as modals


def test_every_screen_the_app_imports_exists():
    src = (Path(modals.__file__).parent / "app.py").read_text()
    names = set()
    for m in re.finditer(r"from \.modals import (\([^)]*\)|[^\n]+)", src):
        names.update(n for n in re.split(r"[,\s()]+", m.group(1)) if n)
    assert names
    assert [n for n in sorted(names) if not hasattr(modals, n)] == []
