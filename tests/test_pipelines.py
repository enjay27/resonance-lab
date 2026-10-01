import os

import pytest

import pipelines
from config import BASE_DIR


def stage_names(name):
    return [stage_name for stage_name, _ in pipelines.stages(name)]


def test_unsloth_runs_the_stages_run_pipeline_always_ran():
    assert stage_names("unsloth") == ["Validate", "Preprocessing", "Dataset Split", "Fine-Tuning", "Metadata Fix", "Evaluation"]


def test_every_stage_script_exists():
    for name in pipelines.PIPELINES:
        for stage_name, path in pipelines.stages(name):
            assert os.path.isfile(path), f"{name}/{stage_name}: {path} is missing"


def test_stage_paths_are_absolute_and_inside_the_repo():
    for _, path in pipelines.stages("unsloth"):
        assert os.path.isabs(path)
        assert path.startswith(BASE_DIR)


def test_default_pipeline_is_registered():
    assert pipelines.DEFAULT in pipelines.PIPELINES


def test_unknown_pipeline_lists_the_choices():
    with pytest.raises(ValueError, match="unsloth"):
        pipelines.stages("nope")
