"""Keep the tests off the operator's own machine.

Two directories here are real and outside the repo: `~/.toeknife/models`, which
the TUI now writes to, and `~/.toeknife/corpora`, which the fetch caches into. A
test that reaches either one reads whatever the machine happens to have and, in
the models case, leaves a file behind -- one turned up as a 16 MB model in a real
home directory, written by a test run pressing enter on the save offer.

So every test gets its own empty pair, and anything that wants the real search
path has to say so by patching these back.
"""
import pytest

from toeknife.predict import corpora


@pytest.fixture(autouse=True)
def _isolate_toeknife_home(tmp_path, monkeypatch):
    models = tmp_path / "toeknife-models"
    monkeypatch.setattr(corpora, "MODEL_DIRS", (str(models),))
    monkeypatch.setattr(corpora, "CACHE", str(tmp_path / "toeknife-corpora"))
    # The env var is another way into the search path, and inheriting the
    # operator's would defeat the point.
    monkeypatch.delenv("TOEKNIFE_MODELS", raising=False)
    # Training records this machine's rate there; a test's tiny corpus would
    # leave a figure that makes every real estimate absurd.
    monkeypatch.setenv("TOEKNIFE_RATES", str(tmp_path / "rates.json"))
    return models
