import json

import runs
import track_records
import tracking
from run_queue import RunQueue

MANIFEST = {
    "format": "pair", "style": "translategemma", "reverse": True, "raw_sha256": "r" * 64, "data_sha256": "t" * 64,
    "val_sha256": "v" * 64, "counts": {"total": 4216, "passed": 4210, "validation": 199, "eval overlap": 5, "duplicate": 1},
    "val_fraction": 0.05, "eval_set": "bp-eval-dataset.jsonl",
}
FETCH = {"repo": "someone/bp-chat", "revision": "a" * 40, "include": "dataset_*.jsonl", "raw_sha256": "r" * 64, "rows": 4216}
TRAIN_CFG = {"model_name_or_path": "google/translategemma-4b-it", "learning_rate": 1e-4, "per_device_train_batch_size": 4,
             "gradient_accumulation_steps": 8}


# --- which run the tracker records into ---------------------------------------------------------------------------


def test_the_tracker_run_is_named_after_the_profile_and_the_training_run():
    local_id, name = track_records.run_identity("translategemma-4b", "20261001-185200")

    assert name == "translategemma-4b-20261001-185200"
    assert local_id == "translategemma-4b-20261001-185200"


def test_the_local_id_is_always_accepted_by_the_queue(tmp_path):
    local_id, _ = track_records.run_identity("hy mt2.1/8b", "20261001-185200")

    assert RunQueue(str(tmp_path / "q.json"), str(tmp_path / "files")).start_run("resonance-lab", "x", local_run_id=local_id) == local_id


def test_a_pre_runs_adapter_has_no_tracker_run():
    assert track_records.stage_run_id({"run": None, "profile": "translategemma-4b"}) is None
    assert track_records.stage_run_id(None) is None
    assert track_records.stage_run_id({"run": "20261001-185200", "profile": "translategemma-4b"}) == "translategemma-4b-20261001-185200"


def test_a_merge_record_names_the_run_for_the_later_stages(tmp_path):
    adapter = tmp_path / "adapter" / "20261001-185200"
    adapter.mkdir(parents=True)
    runs._write_json(str(adapter / runs.RUN_FILE), {"run": "20261001-185200", "profile": "p", "status": "complete"})
    merged = tmp_path / "merged"
    merged.mkdir()
    runs.write_merge_record(str(merged), "p", str(adapter))

    record = runs.read_merge_record(str(merged))

    assert record["run"] == "20261001-185200" and record["profile"] == "p"
    assert runs.read_merge_record(str(tmp_path / "nowhere")) is None


# --- what a training records --------------------------------------------------------------------------------------


def _train(**over):
    args = dict(profile_name="translategemma-4b", base_model="google/translategemma-4b-it", template="gemma3",
                train_cfg=TRAIN_CFG, fetch_state=FETCH, manifest=MANIFEST, run_id="20261001-185200",
                git={"git.commit": "c" * 40, "git.branch": "main", "git.dirty": "false"}, packages={"pkg.torch": "2.10.0"})
    args.update(over)
    return track_records.train_records(**args)


def test_a_training_records_its_recipe_and_how_the_data_was_made():
    params, _ = _train()

    assert params["profile"] == "translategemma-4b" and params["effective_batch"] == "32" and params["learning_rate"] == "0.0001"
    assert params["data.rows_total"] == "4216" and params["data.format"] == "pair" and params["data.drop.eval overlap"] == "5"


def test_a_training_tags_the_data_revision_the_prompt_the_code_and_the_packages():
    _, tags = _train()

    assert tags["stage"] == "train" and tags["run.id"] == "20261001-185200"
    assert tags["dataset.revision"] == "a" * 40 and tags["data.train_sha256"] == "t" * 64
    assert tags["prompt.style"] == "translategemma" and tags["prompt.reverse"] == "true" and len(tags["prompt.sha256"]) == 64
    assert tags["git.commit"] == "c" * 40 and tags["pkg.torch"] == "2.10.0"


def test_a_training_on_a_local_raw_log_says_so():
    _, tags = _train(fetch_state=None)

    assert tags["dataset.source"] == "local raw log" and "dataset.revision" not in tags


def test_without_a_manifest_the_data_and_prompt_are_left_out_not_invented():
    params, tags = _train(manifest=None)

    assert not any(key.startswith("data.") for key in params)
    assert not any(key.startswith("prompt.") for key in tags)


def test_every_record_value_is_a_string_the_tracker_accepts():
    params, tags = _train()

    assert all(isinstance(v, str) for v in params.values()) and all(isinstance(v, str) for v in tags.values())
    json.dumps([params, tags])


def test_the_finished_training_gives_its_metrics_and_the_best_checkpoint():
    results = {"train_runtime": 1500.5, "train_loss": 0.4}
    state = {"global_step": 90, "best_model_checkpoint": "outputs/x/checkpoint-80",
             "log_history": [{"eval_loss": 0.9, "step": 40}, {"eval_loss": 0.7, "step": 80}]}

    metrics, tags = track_records.train_result_records(results, state)

    assert metrics["train.runtime_s"] == 1500.5 and metrics["eval.best_loss"] == 0.7 and "train.global_step" not in metrics
    assert tags == {"train.best_checkpoint": "checkpoint-80", "train.global_step": "90"}


def test_a_training_that_left_no_result_files_records_nothing_and_does_not_fail():
    assert track_records.train_result_records(None, None) == ({}, {})


# --- merge, gguf, eval ---------------------------------------------------------------------------------------------


def test_a_merge_tags_the_run_it_came_from():
    tags = track_records.merge_tags("/x/outputs/p/20261001-185200")

    assert tags == {"stage.merge": "done", "merge.adapter": "20261001-185200"}


def test_a_gguf_records_its_size_and_hash(tmp_path):
    path = tmp_path / "m.gguf"
    path.write_bytes(b"abc")

    tags = track_records.gguf_tags(str(path))

    assert tags["stage.gguf"] == "done" and tags["gguf.size_bytes"] == "3" and len(tags["gguf.sha256"]) == 64
    assert track_records.gguf_tags(str(tmp_path / "missing.gguf")) == {}


REPORT = {
    "n": 10, "standard": {"chrf": 66.7, "bleu": 30.0, "ter": 50.0}, "jp_leakage": 1, "think_leakage": 0, "exact_match": 2,
    "discord_violations": 0, "term_total": 4, "term_hits": 3,
    "categories": {"chat": {"total": 10, "jp_leak": 1, "term_miss": 1, "discord_viol": 0}},
}


def test_an_eval_records_the_scores_and_how_the_text_was_generated():
    metrics, tags = track_records.eval_records(REPORT, comet=0.8, prompt_mode="training")

    assert metrics["eval.chrf"] == 66.7 and metrics["eval.comet"] == 0.8 and metrics["eval.jp_leak_rate"] == 0.1
    assert tags["stage"] == "eval" and tags["eval.prompt"] == "training"
    assert tags["eval.max_new_tokens"] == "256" and tags["eval.decoding"] == "greedy" and tags["eval.batch_size"] == "1"


def test_eval_scores_match_what_the_tracking_helper_makes():
    metrics, _ = track_records.eval_records(REPORT, comet=None, prompt_mode="chat-template")

    assert metrics == tracking.eval_metrics(REPORT)


# --- reading the files the stages leave ---------------------------------------------------------------------------


def test_a_missing_or_broken_json_file_reads_as_none(tmp_path):
    good = tmp_path / "a.json"
    good.write_text('{"x": 1}', encoding="utf-8")
    bad = tmp_path / "b.json"
    bad.write_text("{broken", encoding="utf-8")

    assert track_records.read_json(str(good)) == {"x": 1}
    assert track_records.read_json(str(bad)) is None and track_records.read_json(str(tmp_path / "nope.json")) is None
