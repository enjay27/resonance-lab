from types import SimpleNamespace

import pytest

import mlflow_compare

T0 = 1_790_963_363_000


class FakeClient:
    def __init__(self, runs, experiment="1"):
        self._runs, self._experiment = runs, experiment

    def get_experiment_by_name(self, name):
        return SimpleNamespace(experiment_id=self._experiment) if self._experiment else None

    def search_runs(self, experiment_ids, filter_string="", max_results=1000, order_by=None):
        return self._runs


def mlflow_run(name, status="FINISHED", start=T0, params=None, metrics=None):
    return SimpleNamespace(info=SimpleNamespace(run_name=name, status=status, start_time=start),
                           data=SimpleNamespace(params=params or {}, metrics=metrics or {}, tags={}))


def test_the_script_prints_the_table(capsys):
    client = FakeClient([mlflow_run("hy-fast-1", params={"profile": "hy-mt2-1.8b-fast", "learning_rate": "0.0002"}, metrics={"eval.chrf": 59.3})])

    mlflow_compare.main([], client=client)

    out = capsys.readouterr().out
    assert "hy-fast-1" in out and "59.3" in out and "0.0002" in out


def test_markdown_output_for_the_memory_notes(capsys):
    mlflow_compare.main(["--markdown"], client=FakeClient([mlflow_run("r", params={"profile": "p"})]))

    assert capsys.readouterr().out.startswith("| run |")


def test_options_reach_the_table(capsys):
    runs = [mlflow_run("a", params={"profile": "p"}, metrics={"eval.chrf": 50.0}), mlflow_run("b", params={"profile": "p"}, metrics={"eval.chrf": 60.0}),
            mlflow_run("k", status="KILLED", params={"profile": "p"})]

    mlflow_compare.main(["--sort", "chrf", "--limit", "1"], client=FakeClient(runs))
    first = capsys.readouterr().out
    mlflow_compare.main(["--all"], client=FakeClient(runs))
    everything = capsys.readouterr().out

    first_column = [line.split()[0] for line in first.splitlines()[1:]]
    assert first_column == ["b"] and "KILLED" not in first  # sorted by chrF, one row, the killed run hidden
    assert "KILLED" in everything


def test_an_unknown_experiment_exits_1_with_the_reason(capsys):
    with pytest.raises(SystemExit) as exc:
        mlflow_compare.main(["--experiment", "nope"], client=FakeClient([], experiment=None))

    assert exc.value.code == 1 and "nope" in capsys.readouterr().out


def test_without_tracking_settings_it_says_how_to_set_them_up(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mlflow_compare, "MLFLOW_ENV_FILE", str(tmp_path / "none.env"))
    for key in ("MLFLOW_TRACKING_URI", "RESONANCE_MLFLOW"):
        monkeypatch.delenv(key, raising=False)

    with pytest.raises(SystemExit) as exc:
        mlflow_compare.main([])

    assert exc.value.code == 1 and ".env.mlflow" in capsys.readouterr().out
