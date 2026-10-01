import pytest

import preprocess
from config import INSTRUCTION
from conftest import read_jsonl


def test_maps_raw_logs_to_instruction_rows(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "おやすみ！", "translated": "잘 자!"}])
    out = tmp_path / "processed" / "train.jsonl"

    preprocess.transform_for_lora(src, str(out))

    assert read_jsonl(out) == [{"instruction": INSTRUCTION, "input": "おやすみ！", "output": "잘 자!"}]


def test_exits_when_raw_file_is_missing(tmp_path):
    with pytest.raises(SystemExit) as exc:
        preprocess.transform_for_lora(str(tmp_path / "missing.jsonl"), str(tmp_path / "out.jsonl"))
    assert exc.value.code == 1


def test_exits_when_raw_file_is_empty(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [])

    with pytest.raises(SystemExit) as exc:
        preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"))
    assert exc.value.code == 1
