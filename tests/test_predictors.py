"""Predictor tiers: constraints, mixing, profiles, persistence."""
import binascii
import zlib

import pytest

from toeknife.core.session import Session
from toeknife.core.tape import Tape
from toeknife.predict.base import UNIFORM, Uniform
from toeknife.predict.constraints import (Alphabet, CONTINUATION, Fixed, Utf8,
                                          combine)
from toeknife.predict.context import ContextModel
from toeknife.predict.mixture import Mixture, TablePrior, Weighted
from toeknife.predict.profiles import build, detect

FRAGMENT = binascii.unhexlify(
    "2f2dd203930a99c50a2519a90af97939950af9690a2519f9c5a90a251989250a"
    "1989c5102539a9252520e332f314324b00")


def fragment_session():
    return Session(FRAGMENT, mode="fragment", profile="text")


# ---- constraints ------------------------------------------------------------

def test_utf8_demands_a_continuation_byte_mid_character():
    u = Utf8()
    allowed = u.allowed("é".encode()[:1])          # a two-byte lead, one to go
    assert allowed == set(CONTINUATION)
    assert 0x41 not in allowed


def test_utf8_forbids_a_continuation_byte_on_a_boundary():
    u = Utf8()
    allowed = u.allowed(b"plain ascii")
    assert allowed is not None
    assert not (allowed & CONTINUATION), "a character cannot start with 10xxxxxx"
    assert 0xC0 not in allowed and 0xF5 not in allowed, "never valid anywhere"
    assert ord("x") in allowed


def test_utf8_declines_when_the_context_is_not_utf8_at_all():
    """A constraint that cannot tell has to say so.  Ruling bytes out on the
    strength of a misread context is worse than ruling nothing out."""
    assert Utf8().allowed(b"\xc0\xc1\xf5") is None
    assert Utf8().allowed(b"\xff\xfe\xff\xfe") is None


def test_utf8_uses_the_bytes_after_the_hole_too():
    u = Utf8(strict=True)
    following_continuation = u.allowed(b"ok ", b"\x81\x82")
    assert following_continuation is not None
    assert all(b >= 0xC2 for b in following_continuation), \
        "only a lead byte can precede a continuation"
    following_ascii = u.allowed(b"ok ", b"abc")
    assert all(b < 0x80 for b in following_ascii)


def test_the_table_alphabet_only_speaks_when_it_knows_something():
    """A fixed-Huffman block gives every byte a code, so it excludes nothing and
    must report no opinion rather than a set of all 256."""
    session = fragment_session()
    assert Alphabet.from_trace(session.trace) is None       # fixed tables

    full = binascii.unhexlify(
        "8dcbd10980400c04d156b600b10b0b0961258133caada0767fda811fef676096"
        "dbb6a311629d564eac7bffb0940e2b6b8f5213ae480fa4de04fe7ae601")
    dynamic = Session(full)
    alpha = Alphabet.from_trace(dynamic.trace)
    assert alpha is not None
    allowed = alpha.allowed(b"")
    assert 0 < len(allowed) < 256
    assert ord("E") in allowed, "the text starts with 'Example'"


def test_combining_constraints_intersects_and_refuses_the_empty_set():
    a = Fixed({1, 2, 3}, "a")
    b = Fixed({3, 4, 5}, "b")
    assert combine([a, b], b"") == {3}
    # contradictory constraints say something about themselves, not the data
    assert combine([Fixed({1}), Fixed({2})], b"") is None
    assert combine([], b"") is None


# ---- mixing -----------------------------------------------------------------

def test_a_mixture_of_one_is_the_predictor_itself():
    m = ContextModel(order=3).train(b"forensic analysis " * 200)
    mixed = Mixture([Weighted(m, 1.0)])
    for byte in (ord("a"), ord("z"), 0):
        assert mixed.logprob(byte, b"foren") == pytest.approx(
            m.logprob(byte, b"foren"))


def test_a_mixture_sits_between_its_parts():
    a = ContextModel(order=3).train(b"aaaaaaaa" * 100)
    b = ContextModel(order=3).train(b"bbbbbbbb" * 100)
    mixed = Mixture([Weighted(a, 1.0), Weighted(b, 1.0)])
    for byte in (ord("a"), ord("b")):
        lo = min(a.logprob(byte, b"aa"), b.logprob(byte, b"aa"))
        hi = max(a.logprob(byte, b"aa"), b.logprob(byte, b"aa"))
        assert lo <= mixed.logprob(byte, b"aa") <= hi


def test_a_mixture_applies_its_constraints_when_ranking():
    m = ContextModel(order=3).train(b"forensic analysis " * 200)
    mixed = Mixture([Weighted(m, 1.0)], [Fixed({ord("a"), ord("b")}, "only ab")])
    ranked = mixed.rank(b"foren")
    assert {v for v, _ in ranked} == {ord("a"), ord("b")}


def test_a_mixture_ranks_the_candidates_it_was_given_not_the_right_context():
    """rank() is (left, candidates, right), as HFPredictor and ByT5Predictor
    implement it and as HoleSolver._score_batched calls it.  A mixture with the
    old (left, right, values) order took the candidate pool as its right-hand
    context and the right-hand context as its candidates, so it ranked the bytes
    *following* a hole as if they were the answers -- at confidence 1.0."""
    m = ContextModel(order=3).train(b"forensic analysis " * 200)
    mixed = Mixture([Weighted(m, 1.0)])
    ranked = mixed.rank(b"analy", [ord("s"), ord("z")], b"is of data")
    assert {v for v, _ in ranked} == {ord("s"), ord("z")}
    assert ranked[0][0] == ord("s")


def test_mixing_does_not_throw_away_the_reverse_direction():
    """The holes that matter sit where there is no left context at all, so a
    mixture has to pass the right-hand context to any part that reads
    backwards rather than collapsing to the forward half."""
    from toeknife.predict.context import Bidirectional
    bi = Bidirectional.train_on(b"forensic analysis of data. " * 200, order=4)
    mixed = Mixture([Weighted(bi, 1.0)])
    assert hasattr(mixed, "logprob_both")
    # nothing on the left, "nalysis" on the right: only the reverse half can say
    both = mixed.logprob_both(ord("a"), b"", b"nalysis")
    assert both == pytest.approx(bi.logprob_both(ord("a"), b"", b"nalysis"))
    assert both > mixed.logprob(ord("a"), b"")


def test_the_table_prior_prefers_bytes_with_shorter_codes():
    full = binascii.unhexlify(
        "8dcbd10980400c04d156b600b10b0b0961258133caada0767fda811fef676096"
        "dbb6a311629d564eac7bffb0940e2b6b8f5213ae480fa4de04fe7ae601")
    prior = TablePrior.from_trace(Session(full).trace)
    assert prior is not None
    # 'e' and ' ' are frequent in "Example sentance for forensic analysis"
    assert prior.logprob(ord(" "), b"") > prior.logprob(0xFE, b"")
    assert prior.order == 0, "it knows frequencies, not order"


def test_uniform_predictor_is_flat():
    u = Uniform()
    assert u.logprob(0, b"") == UNIFORM == u.logprob(255, b"anything")


# ---- profiles ---------------------------------------------------------------

def test_ascii_text_is_detected():
    session = fragment_session()
    profile = detect(session.tape, session.trace)
    assert profile.name == "ascii-text"
    assert profile.text_like


def test_utf8_is_detected_and_brings_its_constraint():
    tape = Tape()
    for b in ("café naïve résumé " * 60).encode():
        tape.emit_literal(b)
    profile = detect(tape)
    assert profile.name == "utf-8"
    assert any(c.name == "utf-8" for c in profile.constraints)


def test_binary_is_not_mistaken_for_text():
    import random
    rng = random.Random(1)
    tape = Tape()
    for _ in range(4000):
        tape.emit_literal(rng.randrange(256))
    profile = detect(tape)
    assert not profile.text_like
    assert not any(c.name == "utf-8" for c in profile.constraints), \
        "a hard constraint on the wrong data is worse than none"


def test_an_elf_is_recognised():
    tape = Tape()
    for b in b"\x7fELF\x02\x01\x01" + bytes(200):
        tape.emit_literal(b)
    for b in b"forensic" * 10:
        tape.emit_literal(b)
    assert detect(tape).name == "elf"


def test_detection_declines_on_too_little_evidence():
    tape = Tape()
    for b in b"short":
        tape.emit_literal(b)
    assert detect(tape).name == "unknown"


def test_build_assembles_something_usable_with_no_corpus_at_all():
    """The fragment has 47 known bytes, too few to self-train on, and fixed
    Huffman tables, which carry no usable frequencies.  There is genuinely
    nothing to build from, and saying so beats inventing a predictor."""
    session = fragment_session()
    mixture, profile = build(session)
    assert profile.name == "ascii-text"
    assert mixture.logprob(ord("e"), b"th") <= 0.0
    assert "empty" in mixture.describe(), mixture.describe()


def test_build_uses_what_is_available_when_there_is_enough_of_it():
    corpus = ContextModel(order=4).train(b"forensic analysis of data. " * 400)
    session = fragment_session()
    mixture, _ = build(session, corpus)
    assert "corpus" in mixture.describe()
    assert mixture.logprob(ord("s"), b"analysi") > mixture.logprob(0xFE, b"analysi")


def test_a_fixed_huffman_table_contributes_no_prior():
    """Every byte gets a code of nearly the same length, so the implied
    frequencies are flat and mixing them in only dilutes what else is there."""
    assert TablePrior.from_trace(fragment_session().trace) is None


# ---- persistence ------------------------------------------------------------

def test_a_model_survives_a_round_trip(tmp_path):
    corpus = b"forensic analysis of compressed data, repeatedly. " * 200
    model = ContextModel(order=4).train(corpus)
    path = tmp_path / "m.tkm"
    model.save(path)
    back = ContextModel.load(path)
    assert back.order == model.order
    assert back.trained_bytes == model.trained_bytes
    for ctx in (b"foren", b"anal", b""):
        for byte in (ord("s"), ord("y"), 0):
            assert back.logprob(byte, ctx) == pytest.approx(
                model.logprob(byte, ctx))


# ---- reporting how far along a slow train is --------------------------------

# Bigger than context.STEP, so several reports land in each pass rather than one.
PROGRESS_CORPUS = b"forensic analysis of compressed evidence. " * 4000


def test_training_reports_one_bar_across_all_of_its_passes():
    """A bidirectional order-6 train is five passes over the corpus in two
    halves. The caller draws one bar, so `total` must not move and `done` must
    not go backwards when a pass or a half ends."""
    from toeknife.predict.context import Bidirectional
    seen = []
    Bidirectional.train_on(PROGRESS_CORPUS, order=4,
                           progress=lambda d, t, n: seen.append((d, t, n)))
    assert len(seen) > 4, "more than one report per half"
    dones = [d for d, _, _ in seen]
    assert dones == sorted(dones), "a bar that goes backwards is worse than none"
    totals = {t for _, t, _ in seen}
    assert len(totals) == 1, f"total moved: {totals}"
    assert dones[-1] <= totals.pop()
    notes = [n for _, _, n in seen]
    assert any("forwards" in n for n in notes)
    assert any("backwards" in n for n in notes)
    assert any("counting trigrams" in n for n in notes)
    assert any("training" in n for n in notes)


def test_the_bar_is_sized_for_the_passes_that_will_actually_run():
    """min_obs=1 skips the trigram pre-count and the prune, so promising three
    passes would leave the bar stuck at a third."""
    from toeknife.predict.context import ContextModel
    seen = []
    m = ContextModel(order=3, min_obs=1)
    assert m.passes == 1
    m.train_corpus(PROGRESS_CORPUS,
                   progress=lambda d, t, n: seen.append((d, t, n)))
    assert {t for _, t, _ in seen} == {len(PROGRESS_CORPUS)}
    assert ContextModel(order=3).passes == 3


def test_raising_from_the_progress_callback_stops_the_training():
    """This is how the interactive path cancels: the worker learns about it at
    its next report and unwinds, and nothing has been assigned yet."""
    from toeknife.predict.context import Bidirectional

    class Stop(Exception):
        pass

    def cb(done, total, note):
        if done:
            raise Stop(note)

    with pytest.raises(Stop):
        Bidirectional.train_on(PROGRESS_CORPUS, order=4, progress=cb)


def test_progress_does_not_change_what_gets_trained():
    from toeknife.predict.context import ContextModel
    quiet = ContextModel(order=4).train_corpus(PROGRESS_CORPUS)
    watched = ContextModel(order=4).train_corpus(
        PROGRESS_CORPUS, progress=lambda *a: None)
    assert watched.trained_bytes == quiet.trained_bytes
    for o in range(5):
        assert watched._counts[o] == quiet._counts[o]
        assert watched._totals[o] == quiet._totals[o]


def test_scoring_classes_reports_and_can_be_stopped():
    """f4 scores every unfilled class before it can show anything, which is
    seconds of work on a real fragment."""
    import binascii

    from toeknife.core.session import Session
    from toeknife.predict.context import ContextModel
    from toeknife.predict.solver import HoleSolver
    from toeknife.predict.profiles import build

    frag = binascii.unhexlify(
        "2f2dd203930a99c50a2519a90af97939950af9690a2519f9c5a90a251989250a"
        "1989c5102539a9252520e332f314324b00")
    s = Session(frag, mode="fragment", profile="text")
    s.model = ContextModel(order=4).train(b"the quick brown fox. " * 200)
    mix, _prof = build(s, corpus_model=s.model, self_train=False)
    solver = HoleSolver(s, mix)

    seen = []
    solver.propose(progress=lambda d, t, n: seen.append((d, t, n)))
    assert seen and seen[0][1] == len(s.tape.holes())
    assert [d for d, _, _ in seen] == sorted(d for d, _, _ in seen)
    assert "scoring class" in seen[0][2]

    class Stop(Exception):
        pass

    def cb(done, total, note):
        raise Stop(note)

    with pytest.raises(Stop):
        solver.propose(progress=cb)


# ---- the optional language-model tier ---------------------------------------

hf = pytest.importorskip("toeknife.predict.hf", reason="needs torch+transformers")
pytestmark_hf = pytest.mark.skipif(not hf.available(),
                                   reason="torch and transformers not installed")


@pytest.mark.skipif(not hf.available(), reason="needs torch and transformers")
def test_the_language_model_ranks_text_it_should_know():
    """It earns its place on exactly the case the byte model got wrong: the
    substitution experiment produced 'Iicensor' for 'Licensor'."""
    p = hf.HFPredictor(order=120).load()
    cands = [ord(c) for c in "abcdefghijklmnopqrstuvwxyz .,"]
    ranked = p.rank(b'"Licens', cands, b'or" shall mean the copyright owner')
    assert ranked
    assert chr(ranked[0][0]) == "e"


@pytest.mark.skipif(not hf.available(), reason="needs torch and transformers")
def test_the_language_model_declines_on_bytes_it_cannot_read():
    """Asked about data it cannot read it would still return numbers, and those
    numbers are noise dressed as evidence."""
    p = hf.HFPredictor().load()
    assert p.rank(b"\x89PNG\r\n\x1a\n\x00\x00\xff\xfe\xfd", [65, 66, 67]) == []
    assert p.logprob(65, b"\xff\xfe\xfd\xc0\xc1") == UNIFORM


def test_the_backend_is_optional_and_says_so_when_missing():
    """Everything else works without it, and the failure has to be actionable."""
    assert isinstance(hf.available(), bool)
    with pytest.raises(hf.MissingBackend, match="toeknife\\[ml\\]"):
        raise hf.MissingBackend("this predictor needs torch and transformers: "
                                "pip install 'toeknife[ml]'")


# ---- guarding against the wrong corpus --------------------------------------

def test_a_mismatched_corpus_is_worse_than_no_model_at_all():
    """The finding behind sanity_check.  A model trained on the wrong kind of
    data does not degrade to ignorance, it degrades past it: it is confidently
    wrong, and confidence is what the solver's threshold reads."""
    import glob, pathlib
    english = b"".join(pathlib.Path(p).read_bytes()
                       for p in sorted(glob.glob("/usr/share/common-licenses/*"))[:8])
    binary = b"".join(pathlib.Path(p).read_bytes()[:120000]
                      for p in sorted(glob.glob("/usr/bin/*"))[:10])
    if len(english) < 50000 or len(binary) < 50000:
        pytest.skip("not enough system files to build both corpora")

    text_model = ContextModel(order=4).train(english[:200000])
    held_binary = binary[-60000:]
    score = text_model.score(held_binary)
    assert score < -8.0, (
        f"an English model on binary scored {score:.1f}; the point of the check "
        f"is that this goes below the -8.0 of knowing nothing")


def test_sanity_check_flags_a_model_that_will_hurt():
    from toeknife.predict.profiles import sanity_check
    import glob, pathlib
    english = b"".join(pathlib.Path(p).read_bytes()
                       for p in sorted(glob.glob("/usr/share/common-licenses/*"))[:8])
    binary = b"".join(pathlib.Path(p).read_bytes()[:120000]
                      for p in sorted(glob.glob("/usr/bin/*"))[:10])
    if len(english) < 50000 or len(binary) < 50000:
        pytest.skip("not enough system files to build both corpora")

    tape = Tape()
    for b in binary[:40000]:
        tape.emit_literal(b)
    wrong = ContextModel(order=4).train(english[:200000])
    score, warning = sanity_check(wrong, tape)
    assert score < -8.0
    assert "worse than" in warning and "wrong kind of data" in warning

    right = ContextModel(order=4).train(binary[-400000:] or binary)
    score, warning = sanity_check(right, tape)
    assert score > -8.0
    assert warning == "", warning


def test_sanity_check_declines_on_a_sample_too_small_to_judge():
    from toeknife.predict.profiles import sanity_check
    tape = Tape()
    for b in b"only a few bytes":
        tape.emit_literal(b)
    _, warning = sanity_check(ContextModel(order=4).train(b"anything" * 100), tape)
    assert "too few" in warning


class TestStorageTimeFiltering:
    """`train_corpus` skips contexts the lookup could never consult, using
    ziprec's bound: a context occurs no more often than any trigram inside it."""

    def corpora(self):
        import random
        rng = random.Random(11)
        prose = (b"the quick brown fox jumps over the lazy dog. forensic "
                 b"analysis of compressed evidence proceeds slowly. " * 200)
        noise = bytes(rng.randrange(256) for _ in range(40_000))
        return {"prose": prose, "noise": noise,
                "mixed": prose[:20_000] + noise[:20_000]}

    @pytest.mark.parametrize("order", [3, 4, 6])
    def test_it_stores_exactly_what_the_slow_path_keeps(self, order):
        from toeknife.predict.context import ContextModel
        for name, blob in self.corpora().items():
            slow = ContextModel(order=order).train(blob).prune()
            fast = ContextModel(order=order).train_corpus(blob)
            for o in range(order + 1):
                assert slow._totals[o] == fast._totals[o], (name, o)
                assert slow._counts[o] == fast._counts[o], (name, o)

    @pytest.mark.parametrize("order", [4, 6])
    def test_predictions_are_identical_not_merely_close(self, order):
        import random
        from toeknife.predict.context import ContextModel
        blob = self.corpora()["mixed"]
        slow = ContextModel(order=order).train(blob).prune()
        fast = ContextModel(order=order).train_corpus(blob)
        rng = random.Random(3)
        for _ in range(2000):
            i = rng.randrange(len(blob))
            ctx = blob[max(0, i - order):i]
            for byte in (blob[i], rng.randrange(256)):
                assert slow.logprob(byte, ctx) == fast.logprob(byte, ctx)

    def test_it_actually_skips_most_of_a_high_entropy_corpus(self):
        """The point of the exercise: 10 MB of ELF at order 6 went from 7.8 GB
        to 2.15 GB, because nearly every long context there is a singleton."""
        from toeknife.predict.context import ContextModel
        noise = self.corpora()["noise"]
        full = ContextModel(order=6).train(noise)
        kept = ContextModel(order=6).train_corpus(noise)
        before = sum(len(t) for t in full._totals)
        after = sum(len(t) for t in kept._totals)
        assert after < before / 4

    def test_a_model_that_keeps_singletons_is_left_alone(self):
        """min_obs=1 means the lookup consults everything, so nothing may be
        filtered out from under it."""
        from toeknife.predict.context import ContextModel
        blob = self.corpora()["noise"]
        a = ContextModel(order=5, min_obs=1).train(blob)
        b = ContextModel(order=5, min_obs=1).train_corpus(blob)
        assert sum(len(t) for t in a._totals) == sum(len(t) for t in b._totals)

    def test_a_context_split_across_files_is_not_treated_as_rare(self, tmp_path):
        """The hazard the whole-corpus rule exists for: a trigram that occurs
        once per file is rare in every file and common in the corpus. Filtering
        per file would throw it away."""
        from toeknife.predict.context import ContextModel
        files = []
        for i in range(3):
            p = tmp_path / f"{i}.txt"
            p.write_bytes(b"ordinary padding text. " * 20 + b"QZQVWX"
                          + b" more ordinary padding. " * 20)
            files.append(p)
        model = ContextModel(order=5).train_files(files)
        assert b"QZQVW" in model._totals[5]
        assert model._totals[5][b"QZQVW"] == 3


class TestIsThisByteModelAnyUseHere:
    """The n-gram's rule, measured the byte model's way.

    A corpus model on the wrong kind of data scores -17.2 bits per byte against
    the -8.0 of knowing nothing. A byte model on the wrong kind reaches -28,
    measured on an ELF binary with a phrack-trained model -- so it is the more
    dangerous mistake, not the lesser one.
    """

    def _filler(self, favourite=None):
        """A stand-in with a controllable distribution, so the verdict can be
        tested without torch weights."""
        torch = pytest.importorskip("torch")
        from toeknife.predict.bytemodel import ClassFiller

        class _Model:
            def __init__(self, fav):
                self.fav = fav

            def __call__(self, ids):
                out = torch.full((ids.shape[0], ids.shape[1], 256), -20.0)
                if self.fav is not None:
                    out[:, :, self.fav] = 0.0
                return out

        return ClassFiller(_Model(favourite), window=9, batch=8)

    def _tape(self, data=b"the quick brown fox jumps over the lazy dog " * 4):
        from toeknife.core.tape import Tape
        tape = Tape()
        for b in data:
            tape.emit_literal(b)
        return tape

    def test_a_model_that_is_confidently_wrong_is_flagged(self):
        from toeknife.predict.bytemodel import sanity_check
        tape = self._tape()
        # Names a byte that is never the answer, with high confidence.
        bits, top1, warning = sanity_check(self._filler(favourite=0), tape,
                                           sample=40)
        assert top1 == 0.0
        assert bits < -8.0, "worse than knowing nothing"
        assert "wrong kind of data" in warning

    def test_a_model_that_is_right_is_not_flagged(self):
        from toeknife.predict.bytemodel import sanity_check
        data = b"a" * 400
        bits, top1, warning = sanity_check(self._filler(favourite=ord("a")),
                                           self._tape(data), sample=40)
        assert top1 == 1.0 and warning == ""
        assert bits > -1.0

    def test_too_little_recovered_to_judge_says_so(self):
        from toeknife.predict.bytemodel import sanity_check
        bits, top1, warning = sanity_check(self._filler(), self._tape(b"abc"))
        assert "too few recovered bytes" in warning

    def test_the_answer_is_hidden_before_the_model_is_asked(self):
        """The check would be worthless if the model could read the byte it is
        being asked to predict: every model would score perfectly."""
        torch = pytest.importorskip("torch")
        from toeknife.predict.bytemodel import UNKNOWN, sanity_check
        seen = []

        class _Peeking:
            def __call__(self, ids):
                seen.append(ids.clone())
                return torch.full((ids.shape[0], ids.shape[1], 256), -8.0)

        from toeknife.predict.bytemodel import ClassFiller
        sanity_check(ClassFiller(_Peeking(), window=9, batch=8), self._tape(),
                     sample=8)
        assert seen, "the model was never called"
        for ids in seen:
            assert (ids[:, 9 // 2] == UNKNOWN).all(), "the answer leaked in"


class TestWhatAFillWillCost:
    """~40x slower per hole than the n-gram is acceptable for an opt-in thorough
    pass only if the operator sees the number before starting, not while
    waiting."""

    def _filler(self):
        torch = pytest.importorskip("torch")
        from toeknife.predict.bytemodel import ClassFiller

        class _Model:
            def __call__(self, ids):
                return torch.full((ids.shape[0], ids.shape[1], 256), -8.0)

        return ClassFiller(_Model(), window=9, batch=8)

    def test_a_model_that_has_done_no_work_admits_it_has_no_rate(self):
        assert self._filler().rate() is None

    def test_the_rate_comes_from_work_actually_done(self):
        f = self._filler()
        f.note_work(1000, 2.0)
        assert f.rate() == pytest.approx(500.0)

    def test_too_little_work_is_not_a_rate(self):
        """A handful of windows is noise, and an estimate built on it would be
        worse than saying nothing."""
        f = self._filler()
        f.note_work(8, 0.001)
        assert f.rate() is None

    def test_the_cost_is_counted_the_way_the_solver_spends_it(self):
        """One pass per occurrence, capped at the solver's own limit -- so it is
        not simply the number of unknown positions."""
        from toeknife.predict.bytemodel import fill_cost

        class _Hole:
            def __init__(self, n):
                self.positions = tuple(range(n))

        class _Tape:
            def holes(self):
                return [_Hole(3), _Hole(100), _Hole(24)]

        f = self._filler()
        f.note_work(1000, 10.0)                      # 100 windows/s
        windows, seconds = fill_cost(f, _Tape(), max_occurrences=24)
        assert windows == 3 + 24 + 24
        assert seconds == pytest.approx(0.51)

    def test_no_rate_means_no_time_claimed(self):
        from toeknife.predict.bytemodel import fill_cost

        class _Tape:
            def holes(self):
                class _H:
                    positions = (1, 2)
                return [_H()]

        windows, seconds = fill_cost(self._filler(), _Tape())
        assert windows == 2 and seconds == 0.0
