import os

import pytest

import pipelines
from config import BASE_DIR


def stage_names(name):
    return [stage.name for stage in pipelines.stages(name)]


def test_unsloth_runs_the_stages_run_pipeline_always_ran():
    assert stage_names("unsloth") == ["Fetch Data", "Validate", "Preprocessing", "Dataset Split", "Fine-Tuning", "Metadata Fix", "Evaluation"]


def test_every_stage_script_exists():
    for name in pipelines.PIPELINES:
        for stage in pipelines.stages(name):
            assert os.path.isfile(stage.path), f"{name}/{stage.name}: {stage.path} is missing"


def test_stage_paths_are_absolute_and_inside_the_repo():
    for stage in pipelines.stages("unsloth"):
        assert os.path.isabs(stage.path)
        assert stage.path.startswith(BASE_DIR)


def test_a_stage_without_arguments_has_an_empty_tuple():
    assert all(stage.args == () for stage in pipelines.stages("unsloth"))
    assert pipelines.Stage("x", "x.py").args == ()


def test_default_pipeline_is_registered():
    assert pipelines.DEFAULT in pipelines.PIPELINES


def test_unknown_pipeline_lists_the_choices():
    with pytest.raises(ValueError, match="unsloth"):
        pipelines.stages("nope")


def test_llamafactory_runs_merge_before_gguf_export_and_evaluates_last():
    assert stage_names("llamafactory") == [
        "Fetch Data",
        "Validate",
        "Preprocessing",
        "Update Dataset",
        "Fine-Tuning",
        "Merge LoRA",
        "Export GGUF",
        "Evaluation",
    ]


def test_llamafactory_preprocesses_into_the_pair_layout():
    preprocessing = next(s for s in pipelines.stages("llamafactory") if s.name == "Preprocessing")
    assert preprocessing.args == ("--format", "pair", "--prompt", "auto")


def test_unsloth_keeps_the_default_instruction_layout():
    preprocessing = next(s for s in pipelines.stages("unsloth") if s.name == "Preprocessing")
    assert preprocessing.args == ()


def test_llamafactory_is_the_default_pipeline():
    assert pipelines.DEFAULT == "llamafactory"


def test_pipelines_share_validate_and_preprocess():
    shared = {"Validate", "Preprocessing"}
    paths = {name: {s.name: s.path for s in pipelines.stages(name) if s.name in shared} for name in pipelines.PIPELINES}
    assert paths["unsloth"] == paths["llamafactory"]


def test_every_pipeline_ends_with_the_evaluation_stage():
    # The eval only reads the merged model and prints a report; a missing eval set must not
    # keep the GGUF from being exported, so it comes last.
    for name in pipelines.PIPELINES:
        assert stage_names(name)[-1] == "Evaluation", name


def test_eval_dataset_path_is_configured():
    import config

    assert config.EVAL_DATASET_PATH.endswith(os.path.join("data", "eval", "bp-eval-dataset.jsonl"))


def test_llamafactory_requirements_do_not_pull_liger_kernel_by_name():
    # liger-kernel depends on `triton`, which does not exist on Windows (only `triton-windows`), so a bare
    # requirement line breaks `pip install -r` there. It is installed by hand with --no-deps (see the file).
    with open(os.path.join(BASE_DIR, "requirements-llamafactory.txt"), encoding="utf-8") as f:
        requirements = [line.split("#")[0].strip() for line in f]

    assert not any(line.startswith("liger-kernel") for line in requirements)


def test_both_pipelines_fetch_the_data_first():
    for name in pipelines.PIPELINES:
        assert stage_names(name)[0] == "Fetch Data"


# --- running part of a pipeline: --from / --only --------------------------------------------------------------


def _stages():
    return pipelines.stages("llamafactory")


def test_without_a_selection_every_stage_runs():
    assert pipelines.select_stages(_stages()) == _stages()


def test_from_starts_at_the_named_stage_and_runs_the_rest():
    names = [s.name for s in pipelines.select_stages(_stages(), start="Merge LoRA")]

    assert names == ["Merge LoRA", "Export GGUF", "Evaluation"]


def test_only_runs_just_the_named_stage():
    assert [s.name for s in pipelines.select_stages(_stages(), only="Fine-Tuning")] == ["Fine-Tuning"]


@pytest.mark.parametrize("given", ["merge lora", "MERGE-LORA", "merge_lora", " Merge LoRA "])
def test_stage_names_match_case_and_separator_insensitively(given):
    assert [s.name for s in pipelines.select_stages(_stages(), only=given)] == ["Merge LoRA"]


def test_a_unique_prefix_is_enough():
    assert [s.name for s in pipelines.select_stages(_stages(), only="merge")] == ["Merge LoRA"]


def test_an_ambiguous_prefix_names_the_candidates():
    with pytest.raises(ValueError, match="Export GGUF"):
        pipelines.select_stages(_stages(), only="e")  # Export GGUF and Evaluation


def test_an_unknown_stage_lists_the_stages_of_the_pipeline():
    with pytest.raises(ValueError, match="Fine-Tuning"):
        pipelines.select_stages(_stages(), start="Nope")


def test_from_and_only_together_are_refused():
    with pytest.raises(ValueError, match="either"):
        pipelines.select_stages(_stages(), start="Merge LoRA", only="Evaluation")
