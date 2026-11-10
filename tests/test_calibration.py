"""Confidence, measured on held-out text, so the floor means precision."""
import glob

import pytest

from toeknife.predict import calibration as C
from toeknife.predict.solver import LONG_RUN, Proposal


def test_the_fit_never_rewards_being_less_sure():
    # Confidence and correctness out of order on purpose.
    rows = ([(0.2, 0, True)] * 30 + [(0.5, 0, False)] * 30
            + [(0.9, 0, True)] * 40 + [(0.99, 0, False)] * 2)
    cal = C.fit(rows)
    precisions = [p for _lo, p, _n in cal.short]
    assert precisions == sorted(precisions), "a surer class must not measure worse"


def test_a_handful_all_right_is_not_a_certainty():
    """Five right out of five must not clear a 99.5% floor."""
    rows = [(0.1, 0, False)] * 60 + [(0.999, 0, True)] * 5
    cal = C.fit(rows)
    assert cal.precision(0.999) < 0.9


def test_long_runs_get_their_own_figure():
    rows = [(0.9, 0, True)] * 60 + [(0.9, LONG_RUN, False)] * 40
    cal = C.fit(rows)
    assert cal.precision(0.9, 0) > 0.9
    assert cal.precision(0.9, LONG_RUN) < 0.1


def test_too_little_to_measure_is_not_a_calibration():
    assert C.fit([(0.9, 0, True)] * 10) is None


def test_a_calibration_survives_the_round_trip():
    cal = C.fit([(c / 100, 0, c % 3 != 0) for c in range(100)])
    back = C.Calibration.from_dict(cal.to_dict())
    assert back.short == cal.short and back.precision(0.5) == cal.precision(0.5)


def test_whole_files_are_held_out_deterministically():
    paths = [f"f{i:03}" for i in range(100)]
    train, _docs = C.holdout(paths)
    assert len(train) == 100 - 5
    assert C.holdout(paths)[0] == train


def test_training_calibrates_and_the_model_keeps_it(tmp_path):
    """Step 5: every model trained here arrives calibrated, and saving it
    keeps the calibration with it."""
    from toeknife.predict.context import ContextModel
    paths = sorted(glob.glob("docs/manual/*.md"))
    if len(paths) < 5:
        pytest.skip("needs the manual's pages as a corpus")
    model, cal, train = C.train_files(paths, order=4)
    assert cal is not None and cal.classes >= C.MIN_CLASSES
    assert model.calibration is cal
    out = tmp_path / "m.tkm"
    model.save(out)
    back = ContextModel.load(out)
    assert back.calibration is not None
    assert back.calibration.short == cal.short


def test_the_floor_reads_measured_precision_when_there_is_one():
    p = Proposal(0, (0,), 65, -1.0, 0.999)
    assert p.gate == 0.999
    p.precision = 0.71
    assert p.gate == 0.71


def test_classes_next_to_a_gap_get_their_own_curve():
    """A carve's 90% band delivered 88% while next to a
    lost packet the same band delivered 53%. Pooled, the carve's figure would
    carry fills near a gap."""
    rows = ([(0.95, 0, False, True)] * 60 + [(0.95, 0, True, False)] * 40
            + [(0.95, 0, True, True)] * 10)
    cal = C.fit(rows)
    assert cal.precision(0.95, 0, near=False) > 0.9
    assert cal.precision(0.95, 0, near=True) < 0.3


def test_a_calibration_without_a_gap_curve_falls_back_to_the_long_run_figure():
    """Models calibrated before the gap regime existed must not lend the
    carve curve to a class beside a lost packet."""
    rows = [(0.9, 0, True)] * 60 + [(0.9, LONG_RUN, False)] * 40
    cal = C.fit(rows)
    assert not cal.near
    assert cal.precision(0.9, 0, near=True) == cal.long[0]


def test_a_byte_model_calibration_survives_the_round_trip():
    cal = C.fit_filler([(c / 200, 0, False, c % 3 != 0) for c in range(200)])
    d = cal.to_dict()
    back = C.Calibration.from_dict(d, kind="slm")
    assert back.kind == "slm" and back.short == cal.short
    assert C.Calibration.from_dict(d) is None, (
        "an n-gram must never load a byte model's curve as its own")
    ngram = C.fit([(c / 100, 0, True) for c in range(100)]).to_dict()
    assert C.Calibration.from_dict(ngram, kind="slm") is None


def test_a_tie_is_one_block_however_its_answers_sort():
    """Sorted, the wrong answers at a tie come first. Pooled on arrival they
    sank into the block below and left the right ones a step of their own,
    so 75% right at a saturated confidence read as 99.99%."""
    rows = ([(0.5, 0, False)] * 50 + [(0.5, 0, True)] * 50
            + [(1.0, 0, False)] * 25 + [(1.0, 0, True)] * 75)
    cal = C.fit(rows)
    assert cal.precision(1.0) == pytest.approx(76 / 102, abs=1e-3)
    assert cal.precision(1.0) < 0.9



def test_held_out_files_spread_across_a_big_corpus():
    """The first 12 every-20th files of phrack were its early issues only, and
    their curve let a 50% floor near a gap deliver 29%."""
    paths = [f"f{i:04}" for i in range(2000)]
    train, _docs = C.holdout(paths)
    held = sorted(set(paths) - set(train))
    assert len(held) == C.CAP
    assert held[0] < "f0100" and held[-1] > "f1900", "spread end to end"


# ---- a byte model calibrated by the tool, not by a research script ---------

def test_a_small_corpus_holds_back_text_past_what_the_trainer_reads(tmp_path):
    """Under 4 whole files to spare, the held-out text is each file past
    --limit: the trainer never reads it, and every file still trains."""
    paths = []
    for i in range(5):
        p = tmp_path / f"a{i}.txt"
        p.write_bytes(bytes([65 + i]) * 60_000)
        paths.append(p)
    (tmp_path / "short.txt").write_bytes(b"x" * 9000)
    paths.append(tmp_path / "short.txt")
    train, docs = C.filler_holdout(paths, limit=48_000)
    assert len(train) == 6, "nothing whole is held back"
    assert [n.rsplit("/", 1)[1] for n, _ in docs] == [
        f"a{i}.txt@48000" for i in range(5)], "too short to have a tail: kept out"
    assert docs[2][1] == b"C" * 12_000


def test_a_big_corpus_holds_back_whole_files_for_a_byte_model(tmp_path):
    paths = []
    for i in range(200):
        p = tmp_path / f"f{i:04}"
        p.write_bytes(b"x" * 100)
        paths.append(p)
    train, docs = C.filler_holdout(paths, limit=48_000)
    assert len(train) == 190 and len(docs) == 10
    assert not {n for n, _ in docs} & set(train), "held out means not trained"


def test_held_out_text_reads_from_an_offset(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_bytes(bytes(range(256)) * 200)
    assert C.read_held_out(f"{p}@256")[1][:2] == b"\x00\x01"
    assert C.read_held_out(f"{p}@10")[1][0] == 10
    assert len(C.read_held_out(str(p))[1]) == C.DOC_BYTES
    odd = tmp_path / "x@y.txt"
    odd.write_bytes(b"abc")
    assert C.read_held_out(str(odd))[1] == b"abc", "an @ in a name is a name"


def _tiny_byte_model(path):
    torch = pytest.importorskip("torch")
    from toeknife.predict.bytemodel import ByteFiller
    torch.manual_seed(0)
    m = ByteFiller(d_model=32, layers=1, heads=2, window=33)
    torch.save({"model": m.state_dict(), "cfg": m.cfg,
                "held_out": ["doc.txt"]}, path)
    return path


def _prose(n):
    words = (b"the quick brown fox jumps over a lazy dog while seven "
             b"packets of compressed text go missing in transit ").split()
    import random
    r = random.Random(1)
    out = bytearray()
    while len(out) < n:
        out += r.choice(words) + b" "
    return bytes(out[:n])


def test_calibrating_a_byte_model_writes_a_copy_with_its_curve(tmp_path,
                                                               monkeypatch):
    """`--calibrate-slm` from the run's own record of what it held out:
    the original stays as it was, the copy loads calibrated."""
    from toeknife import cli
    from toeknife.predict.bytemodel import ClassFiller
    src = _tiny_byte_model(tmp_path / "m.tks")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "doc.txt").write_bytes(_prose(12_000))
    before = src.read_bytes()
    assert cli.main(["--calibrate-slm", str(src),
                     "--corpus", str(corpus)]) == 0
    assert src.read_bytes() == before, "the measured model is left alone"
    cal = ClassFiller.load(tmp_path / "m-cal.tks").calibration
    assert cal is not None and cal.kind == "slm"
    assert cal.held_out == [str(corpus / "doc.txt")]
    assert cli.main(["--calibrate-slm", str(src),
                     "--corpus", str(corpus)]) == 1, "never overwrites"


def test_calibrating_needs_to_know_what_the_model_never_saw(tmp_path):
    from toeknife import cli
    src = _tiny_byte_model(tmp_path / "m.tks")
    assert cli.main(["--calibrate-slm", str(src)]) == 1, (
        "held-out names recorded but nowhere to find them")
    assert not (tmp_path / "m-cal.tks").exists()


def test_held_out_files_that_moved_are_found_by_name(tmp_path):
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.txt").write_bytes(b"a")
    found, missing = C.find_held_out(
        ["/gone/a.txt@48000", "/gone/b.txt"], [tmp_path / "corpus"])
    assert found == [f"{tmp_path}/corpus/a.txt@48000"]
    assert missing == ["/gone/b.txt"]


def test_a_held_out_document_loses_a_packet_at_five_places():
    """Near a gap the n-gram is rarely sure, so its curve's top steps rested
    on a handful of classes; five packet positions give them 2-5x more."""
    import random
    r = random.Random(3)
    # Varied enough to compress to ~20 KB, so even the 3/4 packet leaves the
    # 1 KB after it that a session needs.
    doc = bytes(r.choice(b"etaoinshrdlu cmfwyp") for _ in range(32_768))
    assert len(C.PACKET_AT) == 5
    sessions = list(C._sessions(doc, None))
    assert len(sessions) == 1 + len(C.CUTS) + 5, (
        "a carve, the front cuts, and one session per packet position")


def test_a_floor_says_what_it_rests_on():
    cal = C.Calibration(short=[(0.5, 0.6, 400), (0.9, 0.92, 50)],
                        near=[(0.5, 0.55, 300), (0.99, 0.91, 9)])
    assert cal.resting(0.9) == 50
    assert cal.resting(0.9, near=True) == 9
    assert cal.resting(0.5) == 450
    assert cal.resting(0.95) == 0


def test_one_kind_of_damage_can_be_scored_alone():
    import random
    r = random.Random(3)
    doc = bytes(r.choice(b"etaoinshrdlu cmfwyp") for _ in range(32_768))
    only = list(C._sessions(doc, None, cuts=(), packets=((1, 4), (2, 3)),
                            carved=False))
    assert len(only) == 2


SAMPLE = str(__import__("pathlib").Path(__file__).resolve().parent.parent
             / "samples" / "fragment.bin")


def test_the_cli_fills_with_a_byte_model(tmp_path, capsys):
    """`.tks` loaded in the TUI only; the CLI could calibrate one and never
    fill with it (wrap-up #4)."""
    from toeknife import cli
    src = _tiny_byte_model(tmp_path / "m.tks")
    assert cli.main([SAMPLE, "--solve-propose", "--byte-model", str(src),
                     "--fill-mode", "slm"]) == 0
    err = capsys.readouterr().err
    assert "bits/byte" in err, "it says how the model reads this capture"
    assert "raw confidence" in err, (
        "an uncalibrated byte model alone has no curve to read the floor on")


def test_a_byte_model_fill_mode_needs_a_byte_model(capsys):
    from toeknife import cli
    with pytest.raises(SystemExit) as e:
        cli.main([SAMPLE, "--solve", "--fill-mode", "slm"])
    assert e.value.code == 1
    assert "needs --byte-model" in capsys.readouterr().err
