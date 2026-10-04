import json
from types import SimpleNamespace

import pytest

import compare_runs
from compare_runs import CompareError

T0 = 1_790_963_363_000  # 2026-10-02 17:49:23 UTC


def run(name="hy-mt2-1.8b-fast-20261002-180510", status="FINISHED", start=T0, params=None, metrics=None, tags=None):
    return {"name": name, "status": status, "start_ms": start, "params": params or {}, "metrics": metrics or {}, "tags": tags or {}}


FULL = run(
    params={"profile": "hy-mt2-1.8b-fast", "learning_rate": "0.0002", "num_train_epochs": "3.0", "effective_batch": "32", "data.rows_passed": "4210"},
    metrics={"eval.best_loss": 0.6912, "eval.best_loss_step": 60.0, "train.loss": 1.2381, "eval.chrf": 59.34, "eval.term_accuracy": 0.048,
             "eval.jp_leak_rate": 0.039, "eval.n": 51.0},
    tags={"eval.prompt": "training", "dataset.revision": "b37c268fd5d0143e2b6ff0fd36e9c436a8289b25", "train.best_checkpoint": "checkpoint-60"},
)


# --- one run as one row ----------------------------------------------------------------------------------------------


def test_a_run_becomes_a_row_of_the_things_a_sweep_is_compared_on():
    row = compare_runs.summarize(FULL)

    assert row["profile"] == "hy-mt2-1.8b-fast" and row["lr"] == "0.0002" and row["epochs"] == "3" and row["batch"] == "32"
    assert row["eval_loss"] == "0.6912" and row["at_step"] == "60" and row["train_loss"] == "1.2381"
    assert row["chrf"] == "59.3" and row["term_acc"] == "4.8%" and row["jp_leak"] == "3.9%" and row["prompt"] == "training"
    assert row["rows"] == "4210" and row["data"] == "b37c268f" and row["when"] == "2026-10-02 17:49"
    assert row["run"] == "hy-mt2-1.8b-fast-20261002-180510" and row["status"] == "FINISHED"


def test_what_a_run_never_logged_is_a_dash_not_an_error():
    row = compare_runs.summarize(run(params={"profile": "hy-mt2-1.8b-fast"}))  # trained, no eval stage yet

    assert row["chrf"] == row["term_acc"] == row["eval_loss"] == row["lr"] == row["data"] == "-"


def test_a_learning_rate_is_shown_the_short_way():
    assert compare_runs.summarize(run(params={"learning_rate": "1e-05"}))["lr"] == "1e-05"
    assert compare_runs.summarize(run(params={"learning_rate": "0.0001"}))["lr"] == "0.0001"
    assert compare_runs.summarize(run(params={"learning_rate": "not a number"}))["lr"] == "not a number"


# --- choosing and ordering the runs -------------------------------------------------------------------------------------


RUNS = [
    run("a", start=T0, params={"profile": "hy-mt2-1.8b-fast"}, metrics={"eval.best_loss": 0.70, "eval.chrf": 50.0}),
    run("b", start=T0 + 1000, params={"profile": "hy-mt2-1.8b-fast"}, metrics={"eval.best_loss": 0.65, "eval.chrf": 58.0}),
    run("c", start=T0 + 2000, params={"profile": "translategemma-4b"}, metrics={"eval.best_loss": 0.88}),
    run("d", start=T0 + 3000, status="KILLED", params={"profile": "hy-mt2-1.8b-fast"}),
    run("e", start=T0 + 4000, params={"profile": "hy-mt2-1.8b-fast"}),  # no eval stage
]


def names(rows):
    return [r["run"] for r in rows]


def test_by_default_only_finished_runs_and_the_newest_first():
    assert names(compare_runs.build_rows(RUNS)) == ["e", "c", "b", "a"]  # the killed run d is hidden
    assert names(compare_runs.build_rows(RUNS, include_unfinished=True)) == ["e", "d", "c", "b", "a"]


def test_the_profile_filter_matches_the_profile_or_the_run_name():
    assert names(compare_runs.build_rows(RUNS, profile="translategemma")) == ["c"]
    assert names(compare_runs.build_rows(RUNS, profile="hy-mt2-1.8b-fast")) == ["e", "b", "a"]
    sweep = [run("sweep-lr1e-4-0001", params={"profile": "x"}), run("sweep-lr2e-4-0002", params={"profile": "x"})]
    assert names(compare_runs.build_rows(sweep, profile="lr2e-4")) == ["sweep-lr2e-4-0002"]


def test_sorting_puts_the_best_first_and_the_missing_last():
    assert names(compare_runs.build_rows(RUNS, sort="eval-loss")) == ["b", "a", "c", "e"]  # lowest loss first, no loss last
    assert names(compare_runs.build_rows(RUNS, sort="chrf")) == ["b", "a", "e", "c"]  # highest first, missing last (newest first among them)
    with pytest.raises(CompareError, match="eval-loss"):
        compare_runs.build_rows(RUNS, sort="nonsense")


def test_the_limit_keeps_the_first_rows_after_sorting():
    assert names(compare_runs.build_rows(RUNS, sort="eval-loss", limit=2)) == ["b", "a"]


# --- the table --------------------------------------------------------------------------------------------------------


def test_the_text_table_has_aligned_columns_and_a_header():
    rows = compare_runs.build_rows([FULL, run("short", params={"profile": "x"})])

    lines = compare_runs.render_text(rows).splitlines()

    assert lines[0].split()[:3] == ["run", "status", "profile"] and len(lines) == 3
    assert len({len(line) for line in lines}) == 1  # every cell padded to its column: the columns line up
    assert lines[1].split()[0] in {"short", "hy-mt2-1.8b-fast-20261002-180510"}


def test_the_markdown_table_has_a_separator_row_and_no_pipes_inside_cells():
    rows = compare_runs.build_rows([run("x|y", params={"profile": "p"})])

    lines = compare_runs.render_markdown(rows).splitlines()

    assert lines[0].startswith("| run |") and set(lines[1].replace("|", "").replace(" ", "")) <= {"-", ":"}
    assert "x\\|y" in lines[2]


def test_no_rows_say_so():
    assert "no runs" in compare_runs.render_text([]).lower()


# --- reading the runs from MLflow (a stand-in client) -------------------------------------------------------------------


class FakeClient:
    def __init__(self, runs, experiment="1"):
        self._runs, self._experiment, self.calls = runs, experiment, []

    def get_experiment_by_name(self, name):
        self.calls.append(("get_experiment_by_name", name))
        return SimpleNamespace(experiment_id=self._experiment) if self._experiment else None

    def search_runs(self, experiment_ids, filter_string="", max_results=1000, order_by=None):
        self.calls.append(("search_runs", list(experiment_ids), max_results, order_by))
        return self._runs


def mlflow_run(name, status="FINISHED", start=T0, params=None, metrics=None, tags=None):
    return SimpleNamespace(info=SimpleNamespace(run_name=name, status=status, start_time=start),
                           data=SimpleNamespace(params=params or {}, metrics=metrics or {}, tags=tags or {}))


def test_fetch_runs_turns_mlflow_runs_into_plain_dicts():
    client = FakeClient([mlflow_run("r1", params={"profile": "p"}, metrics={"eval.chrf": 55.0}, tags={"eval.prompt": "training"})])

    runs = compare_runs.fetch_runs(client, "resonance-lab")

    assert runs == [{"name": "r1", "status": "FINISHED", "start_ms": T0, "params": {"profile": "p"}, "metrics": {"eval.chrf": 55.0},
                     "tags": {"eval.prompt": "training"}}]
    assert client.calls[0] == ("get_experiment_by_name", "resonance-lab") and client.calls[1][1] == ["1"]


def test_an_unknown_experiment_is_a_clear_error():
    with pytest.raises(CompareError, match="resonance-lab"):
        compare_runs.fetch_runs(FakeClient([], experiment=None), "resonance-lab")


def test_the_whole_thing_is_json_free_of_surprises():
    assert json.dumps(compare_runs.summarize(FULL))  # every cell is plain text


# --- the dataset recipe ------------------------------------------------------------------------------------------------


def test_a_run_trained_on_a_recipe_shows_its_name_and_a_plain_run_a_dash():
    with_recipe = compare_runs.summarize(run(tags={"dataset.recipe": "balanced"}))

    assert with_recipe["recipe"] == "balanced"
    assert compare_runs.summarize(FULL)["recipe"] == "-"


def test_the_recipe_is_a_column_of_both_tables_next_to_the_rows_it_chose():
    rows = compare_runs.build_rows([run(params={"profile": "p"}, tags={"dataset.recipe": "balanced"})])
    headers = [header for _, header in compare_runs.COLUMNS]

    assert headers.index("recipe") == headers.index("rows") + 1
    assert "recipe" in compare_runs.render_text(rows).splitlines()[0].split()
    assert "| balanced |" in compare_runs.render_markdown(rows)
