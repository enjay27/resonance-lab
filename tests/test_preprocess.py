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


def test_skips_rows_the_app_never_translated(write_jsonl, tmp_path, capsys):
    # resonance-stream writes `translated: null` for lines it never translated.
    src = write_jsonl(
        "raw.jsonl",
        [
            {"pid": 1, "original": "おやすみ！", "translated": None, "timestamp": 1},
            {"pid": 2, "original": "遺跡1F", "translated": "유적 1F", "timestamp": 2},
            {"pid": 3, "original": "スカイ", "translated": "  ", "timestamp": 3},
            {"pid": 4, "original": None, "translated": "번역", "timestamp": 4},
        ],
    )
    out = tmp_path / "out.jsonl"

    preprocess.transform_for_lora(src, str(out))

    assert read_jsonl(out) == [{"instruction": INSTRUCTION, "input": "遺跡1F", "output": "유적 1F"}]
    assert "Skipped 3 lines without an original or a translation." in capsys.readouterr().out


def test_exits_when_no_row_was_translated(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "おやすみ！", "translated": None}])

    with pytest.raises(SystemExit) as exc:
        preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"))
    assert exc.value.code == 1
