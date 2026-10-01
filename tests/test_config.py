import os
import subprocess
import sys

import config


def raw_logs_with_env(value):
    """config.RAW_LOGS in a fresh interpreter, so the module-level read of the environment is real."""
    env = {k: v for k, v in os.environ.items() if k != "RESONANCE_RAW_LOGS"}
    if value is not None:
        env["RESONANCE_RAW_LOGS"] = value
    result = subprocess.run(
        [sys.executable, "-c", "import config; print(config.RAW_LOGS)"],
        cwd=config.BASE_DIR, env=env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    return result.stdout.strip()


def test_raw_logs_default_is_unchanged():
    assert raw_logs_with_env(None) == os.path.join(config.BASE_DIR, "data", "raw", "raw_translated_logs.jsonl")


def test_raw_logs_can_point_at_a_hand_curated_file(tmp_path):
    curated = str(tmp_path / "bp-training-dataset-final.jsonl")
    assert raw_logs_with_env(curated) == curated


def test_a_relative_raw_logs_path_is_relative_to_the_repo_root():
    assert raw_logs_with_env(os.path.join("data", "raw", "bp-training-dataset-final.jsonl")) == os.path.join(
        config.BASE_DIR, "data", "raw", "bp-training-dataset-final.jsonl"
    )


def test_validate_and_preprocess_read_the_same_file():
    # Both stages import RAW_LOGS from config, so one setting moves the whole pipeline.
    import preprocess
    import validate

    assert preprocess.RAW_LOGS == validate.RAW_LOGS == config.RAW_LOGS
