"""Supervising a training run that outlives the process that started it.

The thing under test is not the trainer -- it is the claim that the app can be
closed, or crash, or the machine rebooted, and the run is still findable and
still resumable afterwards. So most of these work on a run directory written by
hand, because that is exactly the state a later session arrives to: a spec, a
log, and no memory of anything.
"""
import json
import os
import time
from pathlib import Path

import pytest

from toeknife.predict import training


@pytest.fixture(autouse=True)
def _runs_here(tmp_path, monkeypatch):
    monkeypatch.setenv("TOEKNIFE_TRAINING", str(tmp_path / "training"))


REPORT = ("  102:01    11,166,321       2.537     2.211     42.2%   "
          "lr 2.99e-04   9.6%")
LATER = (" 1630:10   251,905,178       1.596     1.784     53.0%   "
         "lr 1.82e-05  93.4%")
FINISHED = ("done (budget reached): 144,048 of 144,048 steps "
            "(302,298,429 predictions), val loss 1.847, val top-1 55.1%")


def _run(name="r", log="", pid=None, **spec):
    d = training.runs_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    body = dict(name=name, command=["/bin/true"], started=time.time(),
                marker=name)
    body.update(spec)
    if pid is not None:
        body["pid"] = pid
    (d / training.SPEC).write_text(json.dumps(body))
    (d / training.LOG).write_text(log)
    return training.load(d)


class TestReadingALogNobodyWasWatching:
    def test_progress_comes_from_the_last_report(self):
        r = _run(log="\n".join(["noise", REPORT, "best 51.8% -> x.pt", LATER]))
        p = r.progress()
        assert p.predictions == 251_905_178
        assert p.fraction == pytest.approx(0.934)
        assert p.val_top1 == pytest.approx(53.0)

    def test_a_run_with_no_reports_yet_says_so(self):
        assert _run(log="building 778 carves...\n").progress() is None

    def test_the_completion_line_is_the_only_claim_a_run_ended_on_purpose(self):
        assert _run(log=REPORT).finished() is None
        assert _run(log=REPORT + "\n" + FINISHED).finished() == "budget reached"

    def test_eta_extrapolates_from_the_run_s_own_rate(self):
        r = _run(log=REPORT)          # 102:01 elapsed at 9.6%
        eta = r.eta_seconds()
        assert eta == pytest.approx((102 * 60 + 1) * (1 - 0.096) / 0.096, rel=1e-3)

    def test_a_finished_run_has_no_eta(self):
        assert _run(log=LATER.replace("93.4%", "100.0%")).eta_seconds() is None


class TestWhetherItIsActuallyRunning:
    """A pid is not liveness. It can be recycled, wedged, or long gone."""

    def test_a_pid_that_is_not_ours_does_not_count_as_alive(self):
        # This process exists and is emphatically not the run's trainer.
        r = _run(pid=os.getpid(), marker="a-marker-no-cmdline-contains")
        assert r.process_alive() is False
        assert r.state() == "stopped"

    def test_a_pid_that_is_ours_counts(self):
        r = _run(pid=os.getpid(), marker="python")
        assert r.process_alive() is True

    def test_a_missing_pid_is_stopped_not_crashed(self):
        assert _run(log=REPORT).state() == "stopped"

    def test_a_finished_run_reads_as_finished_even_though_nothing_runs(self):
        assert _run(log=REPORT + "\n" + FINISHED).state() == "finished"

    def test_alive_but_silent_is_its_own_state(self, tmp_path):
        """Alive by pid and saying nothing is not the same as healthy, and it is
        the shape a wedged run takes."""
        r = _run(pid=os.getpid(), marker="python", log=REPORT)
        old = time.time() - training.QUIET_SECONDS - 60
        os.utime(r.log, (old, old))
        assert r.state() == "quiet"


class TestStartingOne:
    def test_a_launched_run_is_findable_from_nothing_but_the_directory(self):
        r = training.launch("x", ["/bin/sh", "-c", "sleep 30"],
                            spec={"marker": "sleep 30"})
        assert (r.path / training.SPEC).is_file()
        found = training.runs()
        assert [f.name for f in found] == ["x"]
        assert found[0].process_alive()
        found[0].stop()

    def test_a_relative_interpreter_survives_the_cwd_change(self, tmp_path):
        """The run's directory becomes its cwd, so a relative program name would
        resolve against the wrong place and die as a bare FileNotFoundError."""
        script = tmp_path / "prog.sh"
        script.write_text("#!/bin/sh\necho ran\n")
        script.chmod(0o755)
        here = os.getcwd()
        os.chdir(tmp_path)
        try:
            r = training.launch("rel", ["./prog.sh"], spec={"marker": "prog"})
        finally:
            os.chdir(here)
        assert Path(r.spec["command"][0]).is_absolute()
        for _ in range(50):
            if "ran" in r.log.read_text():
                break
            time.sleep(0.1)
        assert "ran" in r.log.read_text()

    def test_an_interpreter_symlink_is_not_followed(self, tmp_path):
        """A virtualenv's bin/python is a symlink to the system interpreter.
        Resolving it launches the system python with none of the venv's
        packages, which surfaces as ModuleNotFoundError: torch."""
        real = tmp_path / "real-python"
        real.write_text("#!/bin/sh\ntrue\n")
        real.chmod(0o755)
        link = tmp_path / "venv-python"
        link.symlink_to(real)
        here = os.getcwd()
        os.chdir(tmp_path)
        try:
            r = training.launch("venv", ["./venv-python"], spec={"marker": "v"})
        finally:
            os.chdir(here)
        assert r.spec["command"][0].endswith("venv-python"), \
            "the symlink was resolved away"

    def test_resuming_asks_the_trainer_to_continue(self):
        r = _run(command=["/bin/true", "--steps", "10"])
        assert r.resume_command()[-1] == "--resume"


class TestLearningThisMachineSSpeed:
    """The training screen's costs were probed on one machine and are wrong on
    any other. A run that has actually happened here measured the real figure for
    its own shape, so the screen can stop quoting somebody else's hardware."""

    STEPS = 144_048

    def _run_at(self, name, fraction, active_min, **cfg):
        line = (f"  600:00    50,000,000       1.600     1.900     50.0%   "
                f"lr 1.0e-04  {fraction * 100:.1f}%  active   {active_min}:00")
        return _run(name, log=line,
                    config=dict(d_model=512, layers=6, window=129,
                                steps=self.STEPS, **cfg))

    def test_a_run_measures_its_own_rate(self):
        # A quarter of 144,048 steps in 600 minutes of running.
        r = self._run_at("a", 0.25, 600)
        assert r.steps_per_second() == pytest.approx(
            self.STEPS * 0.25 / (600 * 60))

    def test_the_rate_uses_the_clock_that_stops_when_the_machine_sleeps(self):
        """Dividing by wall time would report a run that hibernated as a slower
        machine -- and this figure is used to size the NEXT run."""
        slept = self._run_at("b", 0.25, 300)      # 600 wall, 300 running
        assert slept.progress().elapsed_s == 600 * 60
        assert slept.steps_per_second() == pytest.approx(
            self.STEPS * 0.25 / (300 * 60))

    def test_a_prediction_budgeted_run_cannot_report_steps(self):
        """Its fraction is a fraction of predictions, and steps are not
        recoverable from it, so it declines rather than inventing one."""
        r = _run("c", log=("  600:00  50,000,000  1.6  1.9  50.0%  "
                           "lr 1.0e-04  25.0%  active   300:00"),
                 config=dict(d_model=512, layers=6, window=129))
        assert r.steps_per_second() is None

    def test_barely_started_is_not_a_rate(self):
        assert self._run_at("d", 0.0001, 5).steps_per_second() is None

    def test_rates_are_keyed_by_what_changes_the_cost(self, tmp_path):
        self._run_at("wide", 0.25, 600, )
        rates = training.measured_rates()
        assert (512, 6, 129) in rates

    def test_the_newest_run_wins(self):
        """A machine that got faster or slower should be believed about now."""
        import time
        old = self._run_at("old", 0.25, 600)
        old.spec["started"] = time.time() - 10_000
        (old.path / training.SPEC).write_text(json.dumps(old.spec))
        new = self._run_at("new", 0.50, 600)       # twice the steps, same time
        new.spec["started"] = time.time()
        (new.path / training.SPEC).write_text(json.dumps(new.spec))
        rate = training.measured_rates()[(512, 6, 129)]
        assert rate == pytest.approx(self.STEPS * 0.50 / (600 * 60))


class TestSavingAFinishedRun:
    """A finished run's model reaches f5 only through the models directory,
    as a .tks carrying where it came from."""

    def _finished(self, calibrated=False):
        torch = pytest.importorskip("torch")
        from toeknife.predict import calibration
        from toeknife.predict.bytemodel import ByteFiller
        r = _run(name="slm-20261107-120000", log=FINISHED,
                 corpus="/data/phrackarchive",
                 config=dict(d_model=32, layers=1, heads=2, window=33,
                             steps=30))
        m = ByteFiller(d_model=32, layers=1, heads=2, window=33)
        blob = dict(model=m.state_dict(), cfg=m.cfg, val_acc=0.551,
                    held_out=["/data/phrackarchive/a.txt",
                              "/data/phrackarchive/b.txt"])
        if calibrated:
            blob["calibration"] = calibration.fit_filler(
                [(c / 200, 0, False, c % 3 != 0) for c in range(200)]
            ).to_dict()
        torch.save(blob, r.path / "bytemodel-best.pt")
        torch.save(dict(blob, val_acc=0.4), r.path / "bytemodel.pt")
        return r

    def test_the_best_checkpoint_is_the_one_saved(self):
        r = self._finished()
        assert training.checkpoint(r).name == "bytemodel-best.pt"

    def test_saved_it_is_listed_as_a_byte_model_with_its_lineage(self):
        from toeknife.predict import corpora
        r = self._finished(calibrated=True)
        dest = training.suggest_name(r)
        assert dest.name.startswith("phrackarchive-slm-") and \
            dest.suffix == ".tks"
        training.save_run(r, dest)
        [m] = corpora.find_models()
        assert m["kind"] == corpora.SLM and m["path"] == str(dest)
        prov = m["provenance"]
        assert prov["corpus"] == "phrackarchive"
        assert prov["run"] == r.name and prov["steps"] == 30
        assert prov["val_top1"] == "55.1%" and prov["held_out"] == 2
        assert m["calibrated"], "the run's curve travels with it"

    def test_a_second_save_is_offered_a_new_name_and_never_overwrites(self):
        r = self._finished()
        first = training.save_run(r, training.suggest_name(r))
        second = training.suggest_name(r)
        assert second != first and second.stem.endswith("-2")
        with pytest.raises(FileExistsError):
            training.save_run(r, first)
