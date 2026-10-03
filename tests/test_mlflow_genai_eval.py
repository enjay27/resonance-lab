import json

import pytest

import genai_eval
import mlflow_compare
import mlflow_genai_eval

ROWS = [
    {"original": "こんにちは", "reference": "안녕하세요", "prediction": "안녕하세요", "raw_output": "안녕하세요", "category": "chat"},
    {"original": "草", "reference": "ㅋㅋ", "prediction": "고블린", "raw_output": "고블린", "category": "slang"},
]


class FakeClient:
    def __init__(self):
        self.tags = {}

    def set_tag(self, run_id, key, value):
        self.tags[(run_id, key)] = value


class Evaluation:
    """Stands in for the MLflow side of the script: records what it was asked to evaluate."""

    def __init__(self, run_id="run-1"):
        self.run_id, self.calls = run_id, []

    def __call__(self, data, scorers, experiment):
        self.calls.append({"data": data, "scorers": scorers, "experiment": experiment})
        return self.run_id


@pytest.fixture
def predictions(tmp_path):
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS), encoding="utf-8")
    return str(path)


@pytest.fixture
def no_merge_record(monkeypatch):
    monkeypatch.setattr(mlflow_genai_eval, "read_merge_record", lambda merged_dir: None)


def run_main(predictions, *extra, client=None, evaluation=None):
    client, evaluation = client or FakeClient(), evaluation or Evaluation()
    mlflow_genai_eval.main(["--model", "hy-mt2-1.8b", "--predictions", predictions, *extra], client=client,
                           run_evaluation=evaluation, scorers=lambda: ["scorer"])
    return client, evaluation


def test_every_row_goes_to_the_evaluation_in_the_eval_experiment(predictions, no_merge_record):
    _, evaluation = run_main(predictions)
    [call] = evaluation.calls
    assert call["data"] == genai_eval.evaluation_data(ROWS)
    assert call["scorers"] == ["scorer"]
    assert call["experiment"] == mlflow_genai_eval.MLFLOW_EVAL_EXPERIMENT


def test_the_run_is_tagged_with_the_model_the_prompt_and_the_sample_count(predictions, no_merge_record):
    client, _ = run_main(predictions, "--prompt", "training")
    assert client.tags[("run-1", "profile")] == "hy-mt2-1.8b"
    assert client.tags[("run-1", "eval.prompt")] == "training"
    assert client.tags[("run-1", "eval.n")] == "2"
    assert ("run-1", "training_run") not in client.tags  # no merge record: a model from before runs existed


def test_the_run_names_the_training_run_the_merged_model_came_from(predictions, monkeypatch):
    monkeypatch.setattr(mlflow_genai_eval, "read_merge_record",
                        lambda merged_dir: {"run": "20261002-1030", "profile": "hy-mt2-1.8b"})
    client, _ = run_main(predictions)
    assert client.tags[("run-1", "training_run")] == "hy-mt2-1-8b-20261002-1030"


def test_the_fast_profile_is_a_model_of_its_own(predictions, no_merge_record):
    client, _ = run_main(predictions, "--fast")
    assert client.tags[("run-1", "profile")] == "hy-mt2-1.8b-fast"


def test_a_failing_tag_does_not_hide_the_evaluation(predictions, no_merge_record, capsys):
    class Broken(FakeClient):
        def set_tag(self, run_id, key, value):
            raise RuntimeError("server went away")

    run_main(predictions, client=Broken())
    assert "could not tag" in capsys.readouterr().out


def test_without_predictions_it_says_to_run_eval_first(tmp_path, capsys, no_merge_record):
    with pytest.raises(SystemExit) as stop:
        mlflow_genai_eval.main(["--model", "hy-mt2-1.8b", "--predictions", str(tmp_path / "none.jsonl")],
                               client=FakeClient(), run_evaluation=Evaluation())
    assert stop.value.code == 1
    assert "eval.py" in capsys.readouterr().out


def test_a_server_that_does_not_answer_is_a_message_not_a_traceback(predictions, no_merge_record, capsys):
    def failing(data, scorers, experiment):
        raise ConnectionError("no route to NAS")

    with pytest.raises(SystemExit) as stop:
        mlflow_genai_eval.main(["--model", "hy-mt2-1.8b", "--predictions", predictions], client=FakeClient(),
                               run_evaluation=failing, scorers=lambda: [])
    assert stop.value.code == 1
    assert "no route to NAS" in capsys.readouterr().out


def test_a_missing_pandas_says_which_requirements_file_has_it(predictions, no_merge_record, capsys):
    def no_pandas(data, scorers, experiment):
        raise ModuleNotFoundError("No module named 'pandas'", name="pandas")

    with pytest.raises(SystemExit) as stop:
        mlflow_genai_eval.main(["--model", "hy-mt2-1.8b", "--predictions", predictions], client=FakeClient(),
                               run_evaluation=no_pandas, scorers=lambda: [])
    assert stop.value.code == 1
    assert "pandas" in capsys.readouterr().out


def test_without_tracking_settings_it_says_how_to_set_them_up(monkeypatch, tmp_path, capsys, predictions, no_merge_record):
    monkeypatch.setattr(mlflow_compare, "MLFLOW_ENV_FILE", str(tmp_path / "missing.env"))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("RESONANCE_MLFLOW", raising=False)
    with pytest.raises(SystemExit) as stop:
        mlflow_genai_eval.main(["--model", "hy-mt2-1.8b", "--predictions", predictions])
    assert stop.value.code == 1
    assert ".env.mlflow" in capsys.readouterr().out


def test_the_llamafactory_requirements_name_pandas_for_the_skinny_client():
    """mlflow-skinny 3.16.1's genai.evaluate fails with ModuleNotFoundError: pandas when it is not installed (checked)."""
    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "requirements-llamafactory.txt"), encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    assert any(re.match(r"pandas\b", line) for line in lines)
