import hashlib
import json
import math
import os
import subprocess

import pytest

import tracking
from tracking import Settings


# --- settings: the URL and credentials live in the gitignored .env.mlflow -----------------------------------------


def _env_file(tmp_path, text):
    path = tmp_path / ".env.mlflow"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_the_env_file_is_parsed_without_comments_blank_lines_and_quotes(tmp_path):
    path = _env_file(tmp_path, '# the NAS\n\nMLFLOW_TRACKING_URI=http://192.168.0.10:5050\nMLFLOW_TRACKING_PASSWORD="p=w d"\n  KEY = v  \nbroken line\n')

    assert tracking.load_env_file(path) == {"MLFLOW_TRACKING_URI": "http://192.168.0.10:5050", "MLFLOW_TRACKING_PASSWORD": "p=w d", "KEY": "v"}


def test_a_missing_env_file_is_empty(tmp_path):
    assert tracking.load_env_file(str(tmp_path / "nope")) == {}


def test_settings_come_from_the_env_file(tmp_path):
    path = _env_file(tmp_path, "MLFLOW_TRACKING_URI=http://nas:5050\nMLFLOW_TRACKING_USERNAME=admin\nMLFLOW_TRACKING_PASSWORD=secret\n")

    assert tracking.tracking_settings(environ={}, env_file=path) == Settings("http://nas:5050", "admin", "secret")


def test_the_environment_wins_over_the_file(tmp_path):
    path = _env_file(tmp_path, "MLFLOW_TRACKING_URI=http://nas:5050\nMLFLOW_TRACKING_PASSWORD=file\n")

    settings = tracking.tracking_settings(environ={"MLFLOW_TRACKING_PASSWORD": "env"}, env_file=path)

    assert settings.password == "env" and settings.uri == "http://nas:5050"


def test_without_a_url_tracking_is_off(tmp_path):
    assert tracking.tracking_settings(environ={}, env_file=str(tmp_path / "nope")) is None


@pytest.mark.parametrize("off", ["0", "false", "off", "no", "FALSE"])
def test_resonance_mlflow_switches_tracking_off(tmp_path, off):
    path = _env_file(tmp_path, "MLFLOW_TRACKING_URI=http://nas:5050\n")

    assert tracking.tracking_settings(environ={"RESONANCE_MLFLOW": off}, env_file=path) is None


def test_only_a_server_url_is_accepted_never_a_local_store(tmp_path):
    # a path or file: URL would silently create a local mlruns/ store: the point is the NAS server
    for uri in ("mlruns", "file:///tmp/x", "sqlite:///x.db"):
        path = _env_file(tmp_path, f"MLFLOW_TRACKING_URI={uri}\n")
        with pytest.raises(ValueError, match="http"):
            tracking.tracking_settings(environ={}, env_file=path)


def test_the_client_environment_has_short_timeouts_and_no_telemetry():
    env = tracking.client_environment(Settings("http://nas:5050", "admin", "secret"))

    assert env["MLFLOW_TRACKING_URI"] == "http://nas:5050"
    assert env["MLFLOW_TRACKING_USERNAME"] == "admin" and env["MLFLOW_TRACKING_PASSWORD"] == "secret"
    assert env["MLFLOW_DISABLE_TELEMETRY"] == "true" and env["DO_NOT_TRACK"] == "true"
    assert int(env["MLFLOW_HTTP_REQUEST_TIMEOUT"]) <= 10 and int(env["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]) <= 1  # a dead NAS must not stall training


def test_the_description_never_shows_the_password():
    text = tracking.describe(Settings("http://nas:5050", "admin", "secret"))

    assert "secret" not in text and "http://nas:5050" in text and "admin" in text


# --- parameters ---------------------------------------------------------------------------------------------------


def test_flatten_params_makes_dotted_string_values():
    flat = tracking.flatten_params({"lr": 1e-05, "lora": {"rank": 32, "target": ["q_proj", "v_proj"]}, "bf16": True, "x": None})

    assert flat == {"lr": "1e-05", "lora.rank": "32", "lora.target": '["q_proj", "v_proj"]', "bf16": "true", "x": "null"}


def test_keys_and_values_are_cut_to_what_mlflow_accepts():
    flat = tracking.flatten_params({"a b#c": 1, "k" * 300: "v", "long": "x" * 7000})

    assert "a b_c" in flat  # `#` is not allowed in a name
    assert all(len(k) <= 250 for k in flat) and len(flat["long"]) == 6000 and flat["long"].endswith("...")


def test_train_params_adds_the_profile_and_the_effective_batch():
    cfg = {"per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "learning_rate": 2e-4}

    params = tracking.train_params("hy-mt2-1.8b", "tencent/Hy-MT2-1.8B", "hy_dense_1_8b", cfg)

    assert params["profile"] == "hy-mt2-1.8b" and params["base_model"] == "tencent/Hy-MT2-1.8B" and params["template"] == "hy_dense_1_8b"
    assert params["effective_batch"] == "8" and params["learning_rate"] == "0.0002"


def test_the_effective_batch_is_left_out_when_the_yaml_does_not_say():
    assert "effective_batch" not in tracking.train_params("p", "m", "t", {"learning_rate": 1})


# --- what the data was --------------------------------------------------------------------------------------------


STATE = {"repo": "someone/bp-chat", "revision": "a" * 40, "include": "dataset_*.jsonl", "raw_sha256": "r" * 64, "rows": 4300,
         "fetched": "2026-10-01T10:00:00+00:00"}
MANIFEST = {"format": "pair", "style": "hy", "reverse": True, "raw_sha256": "r" * 64, "data_sha256": "d" * 64, "val_sha256": "v" * 64,
            "val_rows": 210, "val_fraction": 0.05, "eval_set": "bp-eval-dataset.jsonl", "eval_lines_excluded_from": 51,
            "counts": {"total": 9000, "passed": 8000, "validation": 400, "hallucination": 3, "duplicate": 1000, "eval overlap": 12}}


def test_the_dataset_tags_name_the_revision_and_link_to_it():
    tags = tracking.dataset_tags(STATE, MANIFEST)

    assert tags["dataset.repo"] == "someone/bp-chat" and tags["dataset.revision"] == "a" * 40
    assert tags["dataset.url"] == "https://huggingface.co/datasets/someone/bp-chat/tree/" + "a" * 40
    assert tags["data.train_sha256"] == "d" * 64 and tags["data.val_sha256"] == "v" * 64 and tags["data.raw_sha256"] == "r" * 64
    assert tags["data.eval_set"] == "bp-eval-dataset.jsonl" and tags["data.eval_overlap"] == "12"


def test_a_local_raw_log_is_said_so_instead_of_inventing_a_dataset():
    tags = tracking.dataset_tags(None, MANIFEST)

    assert tags["dataset.source"] == "local raw log" and "dataset.repo" not in tags


def test_without_a_manifest_only_the_fetch_state_is_known():
    tags = tracking.dataset_tags(STATE, None)

    assert tags["dataset.revision"] == "a" * 40 and "data.train_sha256" not in tags


def test_data_params_carry_the_counts_and_the_drop_reasons():
    params = tracking.data_params(MANIFEST)

    assert params["data.rows_total"] == "9000" and params["data.rows_passed"] == "8000" and params["data.rows_validation"] == "400"
    assert params["data.drop.hallucination"] == "3" and params["data.drop.eval overlap"] == "12"
    assert params["data.format"] == "pair" and params["data.reverse"] == "true" and params["data.val_fraction"] == "0.05"
    assert "data.drop.total" not in params


def test_the_prompt_fingerprint_changes_with_the_instruction_text(monkeypatch):
    import prompts

    first = tracking.prompt_fingerprint("hy", True)
    assert first["prompt.style"] == "hy" and first["prompt.reverse"] == "true" and len(first["prompt.sha256"]) == 64
    assert first == tracking.prompt_fingerprint("hy", True)  # stable

    monkeypatch.setitem(prompts.STYLES, "hy", {**prompts.STYLES["hy"], "ja-ko": "changed"})
    assert tracking.prompt_fingerprint("hy", True)["prompt.sha256"] != first["prompt.sha256"]


def test_no_style_is_recorded_as_the_raw_line():
    assert tracking.prompt_fingerprint(None, False) == {"prompt.style": "none", "prompt.reverse": "false", "prompt.sha256": "none"}


# --- results ------------------------------------------------------------------------------------------------------


REPORT = {
    "n": 50, "jp_leakage": 1, "think_leakage": 0, "term_hits": 16, "term_total": 21, "discord_violations": 2, "exact_match": 14,
    "standard": {"bleu": 64.5, "chrf": 66.7, "ter": 44.2},
    "categories": {"party chat": {"total": 20, "jp_leak": 1, "term_miss": 2, "discord_viol": 0}},
}


def test_eval_metrics_are_numbers_with_rates_and_per_category_counts():
    metrics = tracking.eval_metrics(REPORT, comet=0.8123)

    assert metrics["eval.chrf"] == 66.7 and metrics["eval.bleu"] == 64.5 and metrics["eval.ter"] == 44.2 and metrics["eval.comet"] == 0.8123
    assert metrics["eval.n"] == 50 and metrics["eval.jp_leak_rate"] == pytest.approx(0.02) and metrics["eval.exact_match_rate"] == pytest.approx(0.28)
    assert metrics["eval.term_accuracy"] == pytest.approx(16 / 21) and metrics["eval.term_total"] == 21
    assert metrics["eval.discord_violations"] == 2 and metrics["eval.think_leak_rate"] == 0
    assert metrics["eval.cat.party chat.jp_leak"] == 1 and metrics["eval.cat.party chat.total"] == 20


def test_eval_metrics_leave_out_what_was_not_measured():
    metrics = tracking.eval_metrics({**REPORT, "term_total": 0, "term_hits": 0})

    assert "eval.comet" not in metrics and "eval.term_accuracy" not in metrics


def test_train_result_metrics_read_the_trainers_two_files():
    results = {"epoch": 3.0, "total_flos": 1.5e16, "train_loss": 0.505, "train_runtime": 1508.2, "train_samples_per_second": 8.5, "train_steps_per_second": 1.064}
    state = {"global_step": 1605, "best_model_checkpoint": "outputs/x/run1/checkpoint-1000", "log_history": [
        {"loss": 0.9, "step": 10}, {"eval_loss": 0.8972, "step": 100}, {"eval_loss": 0.6656, "step": 1000}, {"eval_loss": 0.69, "step": 1600}]}

    metrics = tracking.train_result_metrics(results, state)

    assert metrics["train.runtime_s"] == 1508.2 and metrics["train.samples_per_s"] == 8.5 and metrics["train.steps_per_s"] == 1.064
    assert metrics["train.loss"] == 0.505
    assert metrics["eval.best_loss"] == 0.6656 and metrics["eval.best_loss_step"] == 1000
    # the same number for every run of a profile is no chart: it is a tag
    assert not {"train.epochs", "train.total_flos", "train.global_step"} & set(metrics)
    assert tracking.trainer_tags(state, results) == {"train.best_checkpoint": "checkpoint-1000", "train.epochs": "3",
                                                     "train.global_step": "1605", "train.total_flos": "15000000000000000"}


def test_train_result_metrics_cope_with_missing_files_and_no_eval_rows():
    assert tracking.train_result_metrics(None, None) == {}
    assert "eval.best_loss" not in tracking.train_result_metrics({"train_loss": 1.0}, {"log_history": [{"loss": 1.0, "step": 1}]})
    assert tracking.trainer_tags(None) == {} and tracking.trainer_tags({"best_model_checkpoint": None}) == {}
    assert tracking.trainer_tags(None, {"epoch": 2.5}) == {"train.epochs": "2.5"}


def _log(*rows):
    return [json.dumps(r) for r in rows] + ["", "{broken"]


def test_step_metrics_become_points_in_mlflows_order_from_the_trainer_log():
    lines = _log(
        {"current_steps": 1, "total_steps": 1605, "loss": 2.5, "lr": 1e-05, "epoch": 0.002, "elapsed_time": "0:00:01"},
        {"current_steps": 100, "total_steps": 1605, "eval_loss": 0.9, "epoch": 0.19, "elapsed_time": "0:01:35"},
    )

    points = tracking.step_metrics(lines, start_ms=1_000_000)

    # (name, value, timestamp_ms, step): MLflow's own order
    assert ("loss", 2.5, 1_001_000, 1) in points and ("learning_rate", 1e-05, 1_001_000, 1) in points and ("epoch", 0.002, 1_001_000, 1) in points
    assert ("eval_loss", 0.9, 1_095_000, 100) in points


def test_step_metrics_skip_non_numbers_and_unreadable_lines():
    lines = _log({"current_steps": 5, "loss": float("nan"), "lr": "x", "elapsed_time": "bad"}, {"current_steps": 6, "loss": 1.0})

    points = tracking.step_metrics(lines, start_ms=5)

    assert points == [("loss", 1.0, 5, 6)]  # the run's start as the time (no elapsed_time), step 6
    assert not any(math.isnan(p[1]) for p in points)


def test_chunks_respect_mlflows_batch_limits():
    assert [len(c) for c in tracking.chunks(list(range(2500)), 1000)] == [1000, 1000, 500]
    assert list(tracking.chunks([], 10)) == []


# --- where and what ran -----------------------------------------------------------------------------------------


def _git(cwd, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=cwd, check=True, capture_output=True)


def test_git_info_names_the_commit_the_branch_and_whether_the_tree_was_dirty(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "claude/demo")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "x")

    clean = tracking.git_info(str(tmp_path))
    (tmp_path / "a.txt").write_text("changed", encoding="utf-8")
    dirty = tracking.git_info(str(tmp_path))

    assert len(clean["git.commit"]) == 40 and clean["git.branch"] == "claude/demo" and clean["git.dirty"] == "false"
    assert dirty["git.dirty"] == "true"


def test_git_info_outside_a_repo_is_empty_not_an_error(tmp_path):
    assert tracking.git_info(str(tmp_path)) == {}


def test_package_versions_list_only_what_is_installed():
    versions = tracking.package_versions(("pytest", "no-such-package-xyz"))

    assert set(versions) == {"pkg.pytest"} and versions["pkg.pytest"][0].isdigit()


def test_gguf_info_has_the_size_and_the_sha(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"abc" * 1000)

    assert tracking.gguf_info(str(path)) == {"gguf.size_bytes": "3000", "gguf.sha256": hashlib.sha256(b"abc" * 1000).hexdigest()}
    assert tracking.gguf_info(str(tmp_path / "nope.gguf")) == {}


def test_the_default_env_file_is_in_the_repo_root_and_ignored():
    import config

    assert config.MLFLOW_ENV_FILE == os.path.join(config.BASE_DIR, ".env.mlflow")


# --- the model size, from the trainer's own log -----------------------------------------------------------------------


def test_the_trainable_parameter_counts_are_read_from_the_trainers_log():
    log = "[INFO|2026] noise\ntrainable params: 13,434,880 || all params: 3,893,000,000 || trainable%: 0.3451\nstep 1\n"

    assert tracking.model_size_tags(log) == {"train.trainable_params": "13434880", "train.all_params": "3893000000"}


def test_a_log_without_the_line_gives_no_tags():
    assert tracking.model_size_tags("") == {} and tracking.model_size_tags("Traceback ...") == {}


# --- the curves: the trainer's log plus the gradient norm only its state file has ----------------------------------------


def test_the_gradient_norm_from_the_trainer_state_joins_the_log_rows_by_step():
    log = _log({"current_steps": 1, "loss": 5.3, "lr": 3e-07, "elapsed_time": "0:00:01"},
               {"current_steps": 2, "loss": 5.1, "lr": 6e-07, "elapsed_time": "0:00:02"})
    state = {"log_history": [{"step": 1, "loss": 5.3, "grad_norm": 18.06}, {"step": 2, "loss": 5.1, "grad_norm": 24.1}]}

    rows = [json.loads(line) for line in tracking.curve_lines(log, state)]

    assert [(r["current_steps"], r["grad_norm"], r["elapsed_time"]) for r in rows] == [(1, 18.06, "0:00:01"), (2, 24.1, "0:00:02")]
    points = tracking.step_metrics(tracking.curve_lines(log, state), start_ms=0)
    assert ("grad_norm", 18.06, 1000, 1) in points  # the existing point builder already knows the key


def test_curves_keep_rows_without_a_matching_state_row_and_skip_junk():
    log = _log({"current_steps": 1, "loss": 5.3, "elapsed_time": "0:00:01"}, {"current_steps": 7, "eval_loss": 0.9, "elapsed_time": "0:00:07"})
    state = {"log_history": [{"step": 1, "grad_norm": 2.0}, {"step": 3, "grad_norm": 9.0}, {"eval_loss": 0.9}, "junk", {"step": 1, "grad_norm": float("nan")}]}

    rows = [json.loads(line) for line in tracking.curve_lines(log, state)]

    assert [r["current_steps"] for r in rows] == [1, 7] and rows[0]["grad_norm"] == 2.0 and "grad_norm" not in rows[1]


def test_without_a_state_file_the_curves_are_the_log_itself():
    log = _log({"current_steps": 1, "loss": 5.3, "elapsed_time": "0:00:01"})

    assert [json.loads(line) for line in tracking.curve_lines(log, None)] == [json.loads(log[0])]


# --- the dataset recipe -------------------------------------------------------------------------------------------


RECIPE = {"name": "balanced", "sha256": "s" * 64, "seed": 42, "keep": ["recruitment spam"], "weights": {"chat": "3", "recruitment/party": "1"},
          "categories_file": "categories.jsonl", "categories_sha256": "c" * 64, "total": 120, "requested": None, "limited_by": "chat",
          "available": {"chat": 90, "recruitment/party": 400}, "selected": {"chat": 90, "recruitment/party": 30}}


def test_the_dataset_tags_name_the_recipe_and_hash_it_and_its_categories_file():
    tags = tracking.dataset_tags(STATE, {**MANIFEST, "recipe": RECIPE})

    assert tags["dataset.recipe"] == "balanced"
    assert tags["dataset.recipe_sha256"] == "s" * 64 and tags["data.categories_sha256"] == "c" * 64


def test_without_a_recipe_there_are_no_recipe_tags_or_params():
    tags = tracking.dataset_tags(STATE, MANIFEST)
    params = tracking.data_params(MANIFEST)

    assert not [key for key in tags if "recipe" in key or "categories" in key]
    assert not [key for key in params if key.startswith(("data.recipe", "data.cat."))]


def test_data_params_say_how_many_lines_each_category_gave_and_what_limited_the_recipe():
    params = tracking.data_params({**MANIFEST, "recipe": RECIPE})

    assert params["data.cat.chat"] == "90" and params["data.cat.recruitment/party"] == "30"
    assert params["data.recipe.seed"] == "42" and params["data.recipe.keep"] == '["recruitment spam"]'
    assert params["data.recipe.limited_by"] == "chat"


def test_a_recipe_that_nothing_limited_has_no_limited_by_param():
    params = tracking.data_params({**MANIFEST, "recipe": {**RECIPE, "limited_by": None}})

    assert "data.recipe.limited_by" not in params
