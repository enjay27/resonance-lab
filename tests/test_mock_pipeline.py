"""The llamafactory pipeline end to end with only the GPU parts faked (tests/mock_gpu/harness.py).

Proves the orchestration and the file contracts between the stages: that each stage finds what the one before
left, in the places and shapes the real tools use. It does NOT prove that training, merge or quantization work
on a GPU -- those stay `NOT VERIFIED: model part -- no GPU in this session`. Linux / macOS only: the fakes are scripts.
"""

import json
import os
import re
import subprocess
import sys

import pytest

import pipelines
from mock_gpu import harness

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="the fake llamafactory-cli / llama-quantize are scripts, not .exe")

STAGE_LINE = re.compile(r"\] (.+?)\s+\| .*?(SUCCESS|FAILED|MISSING)")


def stage_results(stdout):
    """[(stage name, status)] in the order run_pipeline.py logged them (colour codes stripped)."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
    return [(m.group(1).strip(), m.group(2)) for m in STAGE_LINE.finditer(clean)]


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    work = str(tmp_path_factory.mktemp("mock-pipeline") / "repo")
    return work, harness.run_pipeline(work)


def test_every_stage_runs_in_order_and_the_pipeline_completes(completed):
    work, result = completed
    expected = [(stage.name, "SUCCESS") for stage in pipelines.stages("llamafactory")]
    assert stage_results(result.stdout) == expected, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PIPELINE COMPLETE" in result.stdout


def exists(work, *parts):
    return os.path.exists(os.path.join(work, *parts))


def test_each_stage_leaves_what_the_next_one_reads(completed):
    work, _ = completed
    assert exists(work, "data", "processed", "lora_train_data.jsonl")
    assert exists(work, "data", "dataset_info.json")
    runs = os.path.join(work, "outputs", "hy-mt2-1.8b_lora")
    run_dirs = [d for d in os.listdir(runs) if os.path.isdir(os.path.join(runs, d))]
    assert len(run_dirs) == 1
    with open(os.path.join(runs, run_dirs[0], "run.json"), encoding="utf-8") as f:
        assert json.load(f)["status"] == "complete"
    with open(os.path.join(work, "model_hy-mt2-1.8b_merged", "resonance_run.json"), encoding="utf-8") as f:
        assert json.load(f)["run"] == run_dirs[0]
    assert exists(work, "model_gguf", "bp-hy-mt2-1.8b-q4_k_m.gguf")
    assert exists(work, "outputs", "eval", "hy-mt2-1.8b-chat-template.txt")
    assert exists(work, "outputs", "eval", "hy-mt2-1.8b-chat-template.jsonl")


def test_a_dataset_name_the_stages_did_not_write_stops_the_fine_tuning(tmp_path):
    """The profile names a dataset that Update Dataset never put in dataset_info.json: LLaMA-Factory would fail, so must the mock."""
    def rename_dataset(work):
        path = os.path.join(work, "configs", "llamafactory", harness.PROFILE, "train.yaml")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        assert "\ndataset: bp_translation\n" in text
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text.replace("\ndataset: bp_translation\n", "\ndataset: bp_translations\n"))

    work = str(tmp_path / "repo")
    result = harness.run_pipeline(work, mutate=rename_dataset)
    assert result.returncode == 1
    done = stage_results(result.stdout)
    assert done[-1] == ("Fine-Tuning", "FAILED"), result.stdout
    assert ("Merge LoRA", "SUCCESS") not in done
    assert "[MOCK CONTRACT] llamafactory-cli" in result.stdout and "bp_translations" in result.stdout


def test_the_fake_export_refuses_a_directory_that_is_not_a_trained_adapter(tmp_path):
    """The fakes can fail: an export of a folder without adapter files exits 2 and says why."""
    (tmp_path / "adapter").mkdir()
    (tmp_path / "merge.yaml").write_text(
        "model_name_or_path: x\nadapter_name_or_path: adapter\ntemplate: t\nfinetuning_type: lora\nexport_dir: merged\n", encoding="utf-8")
    env = harness.mock_env(os.path.join(harness.HERE, "bin"))
    result = subprocess.run([os.path.join(harness.HERE, "bin", "llamafactory-cli"), "export", "merge.yaml"], cwd=tmp_path, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "lacks adapter_config.json, adapter_model.safetensors" in result.stderr
