"""Choosing a corpus: the types, the scan, and the trap in already-compressed
file formats."""
import os
import zlib

import pytest

from toeknife.predict import corpora
from toeknife.predict.context import Bidirectional, ContextModel


class TestRegistry:
    def test_every_type_says_what_the_stream_actually_holds(self):
        for t in corpora.TYPES:
            assert t.contains, f"{t.key} does not say what it contains"

    def test_every_type_names_an_external_source(self):
        for t in corpora.TYPES:
            assert t.sources, f"{t.key} has nowhere to get data from"
            for s in t.sources:
                assert s.url.startswith("http") and s.licence

    def test_categories_cover_every_type(self):
        assert {t.category for t in corpora.TYPES} <= set(corpora.CATEGORIES)

    def test_png_is_marked_as_needing_inflation(self):
        """The trap: a PNG's stream holds scanlines, not the file's bytes."""
        assert corpora.BY_KEY["png"].needs_inflating

    def test_text_and_code_are_not(self):
        for key in ("text-en", "markup", "elf"):
            assert not corpora.BY_KEY[key].needs_inflating

    def test_sources_can_be_listed_by_category(self):
        text = corpora.sources_for("text")
        assert text and all(t.category == "text" for t, _s in text)


class TestScan:
    def test_a_plan_is_proposed_before_anything_is_read(self, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"hello " * 100)
        (tmp_path / "b.png").write_bytes(b"not a png")
        plan = corpora.plan("text-en", [tmp_path])
        assert [p.name for p in plan.files] == ["a.txt"]
        assert plan.total_bytes == 600

    def test_the_plan_explains_itself(self, tmp_path):
        (tmp_path / "a.txt").write_bytes(b"x" * 10)
        text = corpora.plan("text-en", [tmp_path]).render()
        assert "the stream holds" in text and "looking in" in text

    def test_an_empty_plan_says_what_to_do_about_it(self, tmp_path):
        text = corpora.plan("elf", [tmp_path]).render()
        assert "nothing found" in text

    def test_whole_system_is_recorded_as_such(self, tmp_path):
        plan = corpora.plan("text-en", [tmp_path], whole_system=True)
        assert plan.whole_system
        assert "whole filesystem" in plan.render()

    def test_inflating_types_warn_in_the_plan(self, tmp_path):
        (tmp_path / "x.png").write_bytes(b"junk")
        assert "INFLATED" in corpora.plan("png", [tmp_path]).render()


class TestTrainingBytes:
    def test_plain_types_train_on_the_file_itself(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_bytes(b"the quick brown fox")
        got = corpora.training_bytes(corpora.BY_KEY["text-en"], p)
        assert got == b"the quick brown fox"

    def test_a_png_trains_on_its_scanlines_not_its_bytes(self, tmp_path):
        """Training on the file's own bytes here means training on compressed
        noise, which is the -17.2 case reached while looking correct."""
        png = _tiny_png(b"\x00" + b"\x40\x50\x60" * 8)
        p = tmp_path / "x.png"
        p.write_bytes(png)
        got = corpora.training_bytes(corpora.BY_KEY["png"], p)
        assert got and got != png
        assert b"\x40\x50\x60" in got

    def test_a_broken_compressed_file_yields_nothing_not_garbage(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"rubbish" * 10)
        assert corpora.training_bytes(corpora.BY_KEY["png"], p) == b""


class TestProvenance:
    @pytest.mark.parametrize("cls", (ContextModel, Bidirectional))
    def test_a_saved_model_carries_what_it_was_trained_on(self, tmp_path, cls):
        m = (ContextModel(order=2).train(b"hello world " * 20)
             if cls is ContextModel
             else Bidirectional.train_on(b"hello world " * 20, order=2))
        p = tmp_path / "m.model"
        m.save(p, provenance={"type": "text-en", "files": 3})
        back = cls.load(p)
        assert back.provenance == {"type": "text-en", "files": 3}

    def test_a_model_saved_without_provenance_reads_as_unknown(self, tmp_path):
        """Empty means unknown, which is the honest answer for models saved
        before this existed -- not a claim that it was trained on nothing."""
        m = ContextModel(order=2).train(b"abc" * 50)
        p = tmp_path / "m.model"
        m.save(p)
        assert ContextModel.load(p).provenance == {}

    def test_saved_models_are_listed_with_their_lineage(self, tmp_path):
        m = ContextModel(order=2).train(b"abc" * 50)
        m.save(tmp_path / "one.model", provenance={"type": "elf"})
        found = corpora.saved_models(tmp_path)
        assert len(found) == 1
        assert found[0]["provenance"] == {"type": "elf"}
        assert found[0]["trained_bytes"] == 150

    def test_listing_a_missing_directory_is_empty_not_an_error(self, tmp_path):
        assert corpora.saved_models(tmp_path / "nope") == []

    def test_the_canonical_directory_is_the_first_place_looked(self, tmp_path,
                                                              monkeypatch):
        """Where a model is written has to be where it is then found, or the
        picker stays empty however many models get saved."""
        monkeypatch.setattr(corpora, "MODEL_DIRS", (str(tmp_path),))
        monkeypatch.delenv("TOEKNIFE_MODELS", raising=False)
        assert corpora.model_dir() == tmp_path
        assert corpora.model_search_paths()[0] == tmp_path

    def test_the_suggested_name_says_what_the_model_read(self):
        """The operator mostly presses enter on this, so it has to carry the two
        things that decide whether a saved model is the one to reuse."""
        assert corpora.suggest_model_name(
            {"download": "gutenberg-en-moby-dick", "order": 6}
        ) == "gutenberg-en-moby-dick-o6.tkm"
        assert corpora.suggest_model_name({"type": "text-en", "order": 4}) \
            == "text-en-o4.tkm"
        assert corpora.suggest_model_name(
            {"url": "https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
             "order": 5}) == "gutenberg.org-pg2701-o5.tkm"
        assert corpora.suggest_model_name(
            {"path": "/home/me/corpora/moby-dick.txt", "order": 5}) \
            == "moby-dick-o5.tkm"
        assert corpora.suggest_model_name(
            {"path": "/usr/share/doc", "order": 4}) == "doc-o4.tkm"
        # No lineage at all still produces a legal filename rather than ".tkm"
        assert corpora.suggest_model_name({}) == "corpus.tkm"
        assert corpora.suggest_model_name({"type": "../../etc/passwd"}) \
            == "etc-passwd.tkm"

    def test_the_suggested_path_never_proposes_an_overwrite(self, tmp_path,
                                                           monkeypatch):
        """Two models off the same corpus at the same order differ by what was
        truncated or swept, and the name cannot say which -- so the second one
        gets a suffix instead of the first one's bytes."""
        monkeypatch.setattr(corpora, "MODEL_DIRS", (str(tmp_path),))
        prov = {"type": "text-en", "order": 4}
        first = corpora.suggest_model_path(prov)
        assert first == tmp_path / "text-en-o4.tkm"
        first.write_bytes(b"not really a model")
        assert corpora.suggest_model_path(prov) == tmp_path / "text-en-o4-2.tkm"

    def test_the_suggested_path_survives_a_directory_that_does_not_exist(
            self, tmp_path, monkeypatch):
        """It is proposed before anything is created, which is the normal case on
        a machine that has never saved one."""
        missing = tmp_path / "nope" / "models"
        monkeypatch.setattr(corpora, "MODEL_DIRS", (str(missing),))
        assert corpora.suggest_model_path({"order": 3}) \
            == missing / "corpus-o3.tkm"


def _tiny_png(scanlines: bytes) -> bytes:
    """A minimal but real PNG, so the inflation path is exercised for real."""
    import struct

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", 8, 1, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(scanlines))
            + chunk(b"IEND", b""))


# ---- getting it from outside -------------------------------------------------

class TestDownloads:
    def test_every_download_names_a_type_that_exists(self):
        for d in corpora.DOWNLOADS:
            assert d.type_key in corpora.BY_KEY, d.key
            assert d.url.startswith("https://"), d.key
            assert d.approx > 0 and d.licence, d.key

    def test_download_keys_are_unique(self):
        keys = [d.key for d in corpora.DOWNLOADS]
        assert len(keys) == len(set(keys))

    def test_every_text_type_has_something_to_fetch(self):
        """A type nobody can get data for is a row that wastes the operator's
        time; the machine-local roots are not always there."""
        for key in ("text-en", "text-es", "text-de", "markup"):
            assert corpora.downloads_for(key), key

    def test_a_url_that_is_not_http_is_refused_before_anything_opens(self):
        with pytest.raises(corpora.FetchError):
            corpora.fetch("file:///etc/passwd")

    def test_cache_paths_are_stable_and_distinct(self):
        a = corpora.cache_path("https://example.com/a/corpus.txt")
        b = corpora.cache_path("https://example.com/b/corpus.txt")
        assert a == corpora.cache_path("https://example.com/a/corpus.txt")
        assert a != b
        assert "example.com" in a.name and a.name.endswith("corpus.txt")

    def test_a_fetched_container_is_inflated_and_text_is_left_alone(self, tmp_path):
        """The trap this module exists for, on the fetch path: training on a
        .gz as bytes trains on noise."""
        prose = b"forensic analysis of compressed evidence. " * 40
        gz = tmp_path / "corpus.gz"
        gz.write_bytes(zlib.compress(prose))          # zlib wrapper, detected
        blob, how = corpora.corpus_bytes(gz)
        assert how == "inflated" and blob == prose

        txt = tmp_path / "corpus.txt"
        txt.write_bytes(prose)
        blob, how = corpora.corpus_bytes(txt)
        assert how == "as-is" and blob == prose


class TestModelDiscovery:
    def test_a_model_is_found_by_its_extension_wherever_it_landed(self, tmp_path,
                                                                  monkeypatch):
        """--save-model writes wherever it is pointed, and the README's own
        example writes english.tkm into the working directory."""
        model = ContextModel(order=3).train(b"forensic bytes " * 50)
        model.save(tmp_path / "english.tkm", {"corpus": "docs"})
        monkeypatch.setenv("TOEKNIFE_MODELS", str(tmp_path))
        # The canonical directory is a real place with real models in it now
        # that f9 writes there, so a test that counts what was found has to say
        # which directories it means.
        monkeypatch.setattr(corpora, "MODEL_DIRS", ())
        found = corpora.find_models()
        assert [m["name"] for m in found] == ["english.tkm"]
        got = found[0]
        assert got["order"] == 3 and got["trained_bytes"] > 0
        assert got["provenance"] == {"corpus": "docs"}
        assert got["direction"] == "forward only"

    def test_a_bidirectional_model_reports_both_halves(self, tmp_path, monkeypatch):
        """The default --save-model writes one of these, and reading its order
        means looking one level down."""
        model = Bidirectional.train_on(b"forensic bytes " * 50, order=4)
        model.save(tmp_path / "both.model")
        monkeypatch.setenv("TOEKNIFE_MODELS", str(tmp_path))
        got = corpora.find_models()[0]
        assert got["direction"] == "forward+reverse" and got["order"] == 4

    def test_something_that_is_not_a_model_is_reported_not_crashed_on(
            self, tmp_path, monkeypatch):
        (tmp_path / "junk.pkl").write_bytes(b"not a pickle at all")
        monkeypatch.setenv("TOEKNIFE_MODELS", str(tmp_path))
        got = corpora.find_models()[0]
        assert got["error"] and "order" not in got

    def test_the_search_path_is_ordered_and_deduplicated(self, tmp_path,
                                                         monkeypatch):
        monkeypatch.setenv("TOEKNIFE_MODELS",
                           f"{tmp_path}{os.pathsep}{tmp_path}")
        paths = corpora.model_search_paths()
        assert paths[0] == tmp_path
        assert len(paths) == len(set(paths))


class TestNearbyCorpora:
    """What is already to hand, which is usually the best-fitting corpus and
    was the one thing the picker made you type from memory."""

    def make_docs(self, root, n=5, ext=".md"):
        d = root / "docs"
        d.mkdir()
        for i in range(n):
            (d / f"note{i}{ext}").write_bytes(b"prose about compressed data. " * 40)
        return d

    def test_a_docs_directory_beside_the_capture_is_offered(self, tmp_path):
        self.make_docs(tmp_path)
        found = {str(p) for p, _, _ in corpora.nearby_corpora([tmp_path])}
        assert str(tmp_path / "docs") in found

    def test_the_working_directory_is_searched_too(self, tmp_path, monkeypatch):
        self.make_docs(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert any(p.name == "docs" for p, _, _ in corpora.nearby_corpora())

    def test_the_working_directory_itself_is_not_offered_wholesale(
            self, tmp_path, monkeypatch):
        """On a checkout that means the virtualenv, and a corpus of compiled
        wheels is the -17.2 bits-per-byte case."""
        self.make_docs(tmp_path)
        monkeypatch.chdir(tmp_path)
        assert all(p.resolve() != tmp_path.resolve()
                   for p, _, _ in corpora.nearby_corpora())

    def elsewhere(self, tmp_path, monkeypatch):
        """Somewhere with nothing in it, so the repo's own docs/ (found via the
        working directory) does not answer for the fixture."""
        work = tmp_path / "work"
        work.mkdir()
        monkeypatch.chdir(work)

    def test_a_directory_of_compressed_samples_is_not_offered(self, tmp_path,
                                                              monkeypatch):
        """A capture's own directory usually holds the captures, and training
        on those trains on noise while looking like the obvious choice."""
        self.elsewhere(tmp_path, monkeypatch)
        d = tmp_path / "docs"
        d.mkdir()
        for i in range(3):
            (d / f"read{i}.md").write_bytes(b"a little prose " * 10)
        for i in range(4):
            (d / f"sample{i}.gz").write_bytes(zlib.compress(b"x" * 200_000))
        assert not corpora.nearby_corpora([tmp_path])

    def test_counts_are_the_prose_ones(self, tmp_path, monkeypatch):
        self.elsewhere(tmp_path, monkeypatch)
        d = self.make_docs(tmp_path, n=4)
        (d / "notes.bin").write_bytes(b"\x00" * 10)     # a stray, below the 20%
        rows = {p: (f, t) for p, f, t in corpora.nearby_corpora([tmp_path])}
        assert d in rows
        files, total = rows[d]
        assert files == 4
        assert total == sum(f.stat().st_size for f in d.glob("*.md"))

    def test_a_directory_with_too_little_in_it_is_not_offered(self, tmp_path,
                                                               monkeypatch):
        self.elsewhere(tmp_path, monkeypatch)
        d = tmp_path / "docs"
        d.mkdir()
        (d / "one.md").write_bytes(b"not much of a corpus")
        assert not corpora.nearby_corpora([tmp_path])


class TestOrderScaling:
    """The order the corpus earns.

    Every test here pins the memory budget. It is derived from free RAM by
    default, so without pinning these would assert one thing on a workstation and
    another on a CI box -- and the point of the tier table is that a given corpus
    earns a given order for stated reasons.
    """

    @pytest.fixture(autouse=True)
    def _fixed_budget(self, monkeypatch):
        monkeypatch.setenv("TOEKNIFE_MEMORY_BUDGET", "3G")

    """The tiers as measured."""

    def test_the_tiers_rise_with_the_corpus_and_then_stop(self):
        rising = [corpora.auto_order(n) for n in
                  (1, 10_000, 32_768, 100_000, 131_072, 1 << 20)]
        assert rising == sorted(rising)
        assert max(rising) <= corpora.ORDER_MAX

    def test_a_big_corpus_gets_the_long_context_it_earned(self):
        """At 8 MB and 15 MB the carve track is monotonic in order and 6 wins,
        which is ziprec's default. An earlier version capped this at 4."""
        assert corpora.auto_order(8 << 20) == 6
        assert corpora.auto_order(64 << 20) == 6

    def test_the_order_steps_down_by_what_the_data_costs_not_its_size(self):
        """The check follows what the data is: 11 MB of ELF at order 6 costs
        3.4 GB and steps down a rung, while 11 MB of prose costs 1.8 GB and does
        not. Binary is about twice text at every order, measured."""
        prose = b"the quick brown fox jumps over the lazy dog. " * 400
        binary = bytes(range(256)) * 200
        assert corpora.auto_order(11 << 20, prose) == 6
        assert corpora.auto_order(11 << 20, binary) == 5
        assert corpora.auto_order(64 << 20, binary) == 3

    def test_a_large_text_corpus_steps_down_too_now_that_it_is_all_read(self):
        """This changed on purpose. Text used to skip the memory check entirely,
        which was survivable only because the readers silently stopped at 8 MB --
        so a text corpus never got big enough to matter. They no longer stop.

        It is a step down in context, never a truncation of the corpus, and
        `order_note` states it with the override beside it."""
        prose = b"the quick brown fox jumps over the lazy dog. " * 400
        # 64 MB of mixed text at order 6 is ~17 GB, at 5 ~9 GB, at 4 ~5 GB.
        assert corpora.auto_order(64 << 20, prose) == 3
        order, note = corpora.order_note(64 << 20, prose)
        assert order == 3
        assert "lowered from 6" in note and "--order 6" in note

    def test_the_budget_moves_with_the_environment(self, monkeypatch):
        prose = b"the quick brown fox jumps over the lazy dog. " * 400
        monkeypatch.setenv("TOEKNIFE_MEMORY_BUDGET", "16G")
        assert corpora.memory_budget() == 16 << 30
        assert corpora.auto_order(30 << 20, prose) == 6
        monkeypatch.setenv("TOEKNIFE_MEMORY_BUDGET", "512M")
        assert corpora.memory_budget() == 512 << 20
        monkeypatch.setenv("TOEKNIFE_MEMORY_BUDGET", "nonsense")
        assert corpora.memory_budget() >= corpora.MEMORY_BUDGET

    def test_the_budget_fits_the_machine_rather_than_a_fixed_guess(self,
                                                                  monkeypatch):
        """A fixed 3 GB shortens the context on any mixed-text corpus past about
        11 MB, which is the size range the measurements say a long context pays
        in. It is a floor now, not a ceiling."""
        monkeypatch.delenv("TOEKNIFE_MEMORY_BUDGET", raising=False)
        assert corpora.memory_budget() >= corpora.MEMORY_BUDGET
        monkeypatch.setattr(corpora, "available_memory", lambda: 64 << 30)
        assert corpora.memory_budget() == 32 << 30
        monkeypatch.setattr(corpora, "available_memory", lambda: 1 << 30)
        assert corpora.memory_budget() == corpora.MEMORY_BUDGET

    def test_the_cost_estimate_uses_the_table_for_the_kind_of_data(self):
        prose = b"the quick brown fox jumps over the lazy dog. " * 400
        binary = bytes(range(256)) * 200
        for order in (3, 4, 5, 6):
            assert corpora.estimate_memory(1 << 20, order, prose) == \
                (1 << 20) * corpora.TEXT_COST[order]
            assert corpora.estimate_memory(1 << 20, order, binary) == \
                (1 << 20) * corpora.BINARY_COST[order]
            # No sample means assume the expensive kind, not the cheap one.
            assert corpora.estimate_memory(1 << 20, order) == \
                (1 << 20) * corpora.BINARY_COST[order]
            assert corpora.TEXT_COST[order] < corpora.BINARY_COST[order]

    def test_the_table_is_anchored_to_a_real_measurement(self):
        """The figures come from training the 30 MB Phrack corpus at each order
        and reading peak RSS: 0.57, 2.20, 4.32 and 7.62 GB. If this drifts, it
        drifted because someone recalibrated, which should be deliberate."""
        prose = b"the quick brown fox jumps over the lazy dog. " * 400
        got = corpora.estimate_memory(30_208_986, 6, prose) / 2 ** 30
        assert 7.0 < got < 8.2, f"order 6 on 30 MB of mixed text: {got:.2f} GB"

    def test_the_note_says_what_it_will_cost_before_anything_is_spent(self):
        prose = b"the quick brown fox jumps over the lazy dog. " * 400
        order, note = corpora.order_note(2 << 20, prose)
        assert order == 5 and "from its size" in note
        assert "memory" in note and "over the" not in note
        # An order the operator pinned is reported as theirs, cost included.
        order, note = corpora.order_note(2 << 20, prose, 6)
        assert order == 6 and "you set it" in note
        # And a pinned order that blows the budget warns instead of refusing.
        order, note = corpora.order_note(200 << 20, prose, 6)
        assert order == 6 and "over the" in note

    def test_a_small_binary_corpus_is_not_penalised(self):
        binary = bytes(range(256)) * 200
        assert corpora.auto_order(300_000, binary) == corpora.auto_order(300_000)

    def test_what_counts_as_textual(self):
        assert corpora.looks_textual(b"plain english prose, mostly.\n" * 10)
        assert not corpora.looks_textual(bytes(range(256)))
        assert not corpora.looks_textual(b"")

    def test_a_sample_can_come_from_bytes_or_from_files(self, tmp_path):
        assert corpora.sample_of(b"abcdef", limit=3) == b"abc"
        a, b = tmp_path / "a.txt", tmp_path / "b.txt"
        a.write_bytes(b"hello ")
        b.write_bytes(b"world")
        assert corpora.sample_of([a, b], limit=32) == b"hello world"
        assert corpora.sample_of([tmp_path / "gone.txt"]) == b""

    @pytest.mark.parametrize("size,order", [
        (5_000, 3), (31_000, 3), (32_768, 4), (100_000, 4),
        (131_072, 5), (322_617, 5), (2 << 20, 5), (8 << 20, 6), (64 << 20, 6),
    ])
    def test_the_measured_table(self, size, order):
        assert corpora.auto_order(size) == order

    def test_an_order_the_operator_asked_for_always_wins(self):
        assert corpora.resolve_order(6, 1_000) == 6
        assert corpora.resolve_order("3", 1 << 30) == 3

    def test_auto_and_none_both_mean_measure_it(self):
        assert corpora.resolve_order("auto", 1_000) == corpora.auto_order(1_000)
        assert corpora.resolve_order(None, 1 << 20) == corpora.auto_order(1 << 20)

    def test_a_tiny_corpus_does_not_get_a_long_context(self):
        """An order-6 context on 5 KB is seen once and fires overconfidently,
        which is the failure the threshold cannot catch."""
        assert corpora.auto_order(5_000) <= 3


class TestNothingIsSilentlyTruncated:
    """The complaint that caused this: a 30 MB corpus trained as 8 MB, quietly.

    Reading is capped nowhere now. What protects memory is the order, which is
    chosen from what the data costs and which says out loud when it moves.
    """

    def test_training_reads_every_file_it_is_given(self, tmp_path):
        from toeknife.predict.context import Bidirectional, ContextModel
        # Comfortably past the 8 MB the readers used to stop at.
        chunk = b"forensic analysis of compressed evidence. " * 25_000  # ~1 MB
        files = []
        for i in range(10):
            p = tmp_path / f"part{i:02d}.txt"
            p.write_bytes(chunk)
            files.append(p)
        total = sum(f.stat().st_size for f in files)
        assert total > (8 << 20), "the fixture has to exceed the old cap"

        m = ContextModel(order=2).train_files(files)
        assert m.trained_bytes == total, "read a slice instead of the corpus"
        b = Bidirectional.train_files(files, order=2)
        assert b.trained_bytes == total

    def test_a_cap_is_still_available_to_anyone_who_wants_one(self, tmp_path):
        from toeknife.predict.context import ContextModel
        p = tmp_path / "big.txt"
        p.write_bytes(b"x" * 100_000)
        assert ContextModel(order=2).train_files([p],
                                                max_bytes=1000).trained_bytes == 1000

    def test_a_file_larger_than_the_old_cap_is_no_longer_skipped(self, tmp_path):
        """It skipped any single file over 8 MB, so the largest file in a corpus
        -- usually the most substantial -- was the one guaranteed to be lost."""
        big = tmp_path / "big.txt"
        big.write_bytes(b"prose. " * 1_300_000)          # ~9 MB
        small = tmp_path / "small.txt"
        small.write_bytes(b"prose. " * 10)
        found = corpora.find(corpora.BY_KEY["text-en"], [tmp_path])
        assert {f.name for f in found} == {"big.txt", "small.txt"}

    def test_a_scan_that_hits_its_backstop_says_so(self, tmp_path):
        for i in range(12):
            (tmp_path / f"f{i:02d}.txt").write_bytes(b"prose. " * 20)
        full = corpora.plan("text-en", [tmp_path])
        assert len(full.files) == 12 and not full.truncated
        assert "sample of what is there" not in full.render()
        clipped = corpora.plan("text-en", [tmp_path], limit=5)
        assert len(clipped.files) == 5 and clipped.truncated
        assert "sample of what is there" in clipped.render()


class TestByteModelsInTheRegistry:
    """A byte model and an n-gram share a directory and are not interchangeable.

    The registry has to tell them apart without loading either, because the
    picker draws a list before the operator has chosen anything, and reading a
    checkpoint means importing torch and unpickling tens of megabytes.
    """

    def _fake_slm(self, tmp_path, name="m.tks", provenance=None):
        torch = pytest.importorskip("torch")
        blob = {"cfg": {"d_model": 512, "layers": 6, "heads": 8, "window": 129},
                "model": {}, "steps": 144048, "val_acc": 0.554,
                "provenance": provenance or {"corpus": "phrack", "result": "14"}}
        torch.save(blob, tmp_path / name)
        return tmp_path / name

    def test_a_checkpoint_is_read_as_a_byte_model_not_an_ngram(self, tmp_path):
        self._fake_slm(tmp_path)
        found = corpora.saved_models(tmp_path)
        assert len(found) == 1
        m = found[0]
        assert m["kind"] == corpora.SLM
        assert m["window"] == 129 and m["d_model"] == 512
        assert m["provenance"]["result"] == "14"
        assert "error" not in m

    def test_an_ngram_still_reads_as_an_ngram(self, tmp_path):
        Bidirectional.train_on(b"the quick brown fox " * 200, order=3).save(
            tmp_path / "n.tkm", provenance={"corpus": "test"})
        m = corpora.saved_models(tmp_path)[0]
        assert m["kind"] == corpora.NGRAM
        assert m["order"] == 3

    def test_listing_a_byte_model_twice_does_not_re_read_the_checkpoint(
            self, tmp_path, monkeypatch):
        """The sidecar is a cache and the second listing must use it.

        Proved by making torch.load fail rather than by damaging the file: the
        cache is invalidated by size and mtime, so damaging the checkpoint would
        correctly invalidate it and would test the opposite of this.
        """
        torch = pytest.importorskip("torch")
        p = self._fake_slm(tmp_path)
        first = corpora.saved_models(tmp_path)[0]
        assert (tmp_path / (p.name + ".json")).is_file()

        def refuse(*a, **k):
            raise AssertionError("the checkpoint was read a second time")

        monkeypatch.setattr(torch, "load", refuse)
        second = corpora.saved_models(tmp_path)[0]
        assert second["window"] == first["window"] == 129
        assert "error" not in second

    def test_a_changed_checkpoint_invalidates_its_cache(self, tmp_path):
        """Staleness is the risk a cache adds, and here it would misreport the
        window a solve is about to be run with."""
        p = self._fake_slm(tmp_path)
        assert corpora.saved_models(tmp_path)[0]["window"] == 129
        torch = pytest.importorskip("torch")
        torch.save({"cfg": {"d_model": 256, "layers": 6, "heads": 4,
                            "window": 257}, "model": {},
                    "provenance": {"corpus": "other"}}, p)
        m = corpora.saved_models(tmp_path)[0]
        assert m["window"] == 257, "a stale cache reported the old window"

    def test_a_corrupt_cache_falls_back_to_the_checkpoint(self, tmp_path):
        p = self._fake_slm(tmp_path)
        corpora.saved_models(tmp_path)
        (tmp_path / (p.name + ".json")).write_text("{ not json")
        m = corpora.saved_models(tmp_path)[0]
        assert m["window"] == 129 and "error" not in m

    def test_registering_records_lineage_and_refuses_to_overwrite(
            self, tmp_path):
        src = self._fake_slm(tmp_path, name="src.tks", provenance={})
        dest_dir = tmp_path / "models"
        out = corpora.register_slm(src, "named", {"corpus": "phrack",
                                                  "p38_5": "52.8% / 67.4%"},
                                   directory=dest_dir)
        assert out.name == "named.tks"
        m = corpora.saved_models(dest_dir)[0]
        assert m["provenance"]["p38_5"] == "52.8% / 67.4%"
        with pytest.raises(FileExistsError):
            corpora.register_slm(src, "named", {}, directory=dest_dir)

    def test_an_ngram_is_cached_too_and_the_cache_notices_a_change(
            self, tmp_path):
        """Reading an n-gram's order means unpickling all of it, so it caches.

        The risk a cache adds is staleness, and staleness here would report the
        wrong order for a model the operator is about to train a solve on. Size
        and mtime invalidate it, checked by writing a genuinely different model
        over the first.
        """
        path = tmp_path / "n.tkm"
        Bidirectional.train_on(b"the quick brown fox " * 200, order=3).save(
            path, provenance={"corpus": "first"})
        first = corpora.saved_models(tmp_path)[0]
        assert first["order"] == 3
        assert (tmp_path / "n.tkm.json").is_file()

        Bidirectional.train_on(b"something else entirely " * 300, order=5).save(
            path, provenance={"corpus": "second"})
        second = corpora.saved_models(tmp_path)[0]
        assert second["order"] == 5, "a stale cache reported the old order"
        assert second["provenance"]["corpus"] == "second"

    def test_something_that_is_not_a_byte_model_says_so(self, tmp_path):
        torch = pytest.importorskip("torch")
        torch.save({"nothing": "useful"}, tmp_path / "x.tks")
        m = corpora.saved_models(tmp_path)[0]
        assert m["kind"] == corpora.SLM
        assert m["error"] == "not a ToeKnife byte model"


class TestWhetherACorpusCanTrainAByteModel:
    """`carves.feasible` answers "can training happen", before four days rather
    than after. It never answers "would the result be good" -- that is
    bytemodel.sanity_check's job, against a real capture."""

    def test_files_too_small_to_carve_are_refused_outright(self, tmp_path):
        from toeknife.predict import carves
        for i in range(4):
            (tmp_path / f"t{i}.txt").write_bytes(b"hello world\n" * 20)
        usable, tried, positions, verdict = carves.feasible(
            sorted(tmp_path.glob("*")))
        assert usable == 0 and positions == 0
        assert "cannot train a byte model" in verdict

    def test_incompressible_data_carves_and_teaches_nothing(self, tmp_path):
        """The trap this exists for. Random bytes carve perfectly -- every file
        succeeds -- and yield zero maskable positions, because incompressible
        data has no back-references and therefore no equality classes. It looks
        healthy and would train on nothing."""
        import random
        from toeknife.predict import carves
        rng = random.Random(2)
        for i in range(4):
            (tmp_path / f"r{i}.bin").write_bytes(
                bytes(rng.randrange(256) for _ in range(30000)))
        usable, tried, positions, verdict = carves.feasible(
            sorted(tmp_path.glob("*")))
        assert usable == tried, "random data does carve; that is the point"
        assert positions == 0
        assert "thinly" in verdict

    def test_nothing_at_all_says_so(self):
        from toeknife.predict import carves
        assert carves.feasible([])[3] == "nothing to train on"


class TestFindingTheCaptureInsideTheCorpus:
    """Training on the file being recovered makes every number afterwards
    fiction, and an operator pointing at a corpus cannot know their capture came
    out of it."""

    def _tape(self, data):
        from toeknife.core.tape import Tape
        tape = Tape()
        for b in data:
            tape.emit_literal(b)
        return tape

    @staticmethod
    def _names(hits):
        from pathlib import Path
        return [Path(p).name for _f, p in hits]

    BODY = (b"the quick brown fox jumps over the lazy dog while nobody is "
            b"watching it at all, and then it does the same thing again "
            b"somewhere else entirely for reasons of its own choosing. ")

    def test_the_file_the_capture_came_from_is_found(self, tmp_path):
        from toeknife.predict import carves
        (tmp_path / "source.txt").write_bytes(b"header\n" + self.BODY * 6)
        (tmp_path / "other.txt").write_bytes(b"header\n" + b"nothing alike " * 60)
        hits = carves.contamination(self._tape(self.BODY * 6),
                                    sorted(tmp_path.glob("*")))
        assert self._names(hits) == ["source.txt"]

    def test_shared_boilerplate_alone_does_not_convict(self, tmp_path):
        """The naive version of this matched 270 of phrack's 969 files, because
        mastheads and separator bars are not unique. Scoring by the fraction of
        the capture's long runs a file holds is what separates them."""
        from toeknife.predict import carves
        boiler = b"=" * 78 + b"\nPhrack Inc. Volume One\n" + b"=" * 78 + b"\n"
        for i in range(4):
            (tmp_path / f"f{i}.txt").write_bytes(
                boiler + f"article {i} ".encode() * 400)
        tape = self._tape(boiler + self.BODY * 6)
        assert carves.contamination(tape, sorted(tmp_path.glob("*"))) == []

    def test_a_tape_too_holed_to_search_says_so_rather_than_clearing_it(self):
        """"Nothing found" about a capture that could not be looked for would be
        a lie, and it is the answer an operator would act on."""
        from toeknife.predict import carves
        assert carves.enough_evidence(self._tape(b"short")) is False
        # A tape recovered whole is ONE run, and is the best evidence there is.
        assert len(carves.clean_runs_of(self._tape(self.BODY * 6), 64)) == 1
        assert carves.enough_evidence(self._tape(self.BODY * 6)) is True


class TestTrainingTimeIsEstimated:
    """Training printed its memory and then ran for minutes unannounced."""

    def test_a_machine_with_no_runs_gets_the_reference_rate_and_says_so(self):
        note = corpora.training_time_note(17_926_231, 6)
        assert "about 9 min" in note and "another machine" in note

    def test_a_run_records_its_rate_and_the_next_estimate_uses_it(self):
        corpora.record_training_rate(5, 10_000_000, 100.0)
        assert "about 2 min" in corpora.training_time_note(12_000_000, 5)
        assert "this machine's measured" in corpora.training_time_note(
            12_000_000, 5)
        near = corpora.training_time_note(12_000_000, 6)
        assert "measured at order 5" in near

    def test_the_note_carries_it(self):
        _order, note = corpora.order_note(2 << 20, b"plain prose " * 100)
        assert "about" in note
