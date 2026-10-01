import os

import pytest

import pipelines
from config import BASE_DIR


def stage_names(name):
    return [stage.name for stage in pipelines.stages(name)]


def test_unsloth_runs_the_stages_run_pipeline_always_ran():
    assert stage_names("unsloth") == ["Validate", "Preprocessing", "Dataset Split", "Fine-Tuning", "Metadata Fix", "Evaluation"]


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


def test_default_pipeline_is_registered():
    assert pipelines.DEFAULT in pipelines.PIPELINES


def test_unknown_pipeline_lists_the_choices():
    with pytest.raises(ValueError, match="unsloth"):
        pipelines.stages("nope")
