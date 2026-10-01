import json
import re

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
    out = capsys.readouterr().out
    assert re.search(r"Skipped\s*: 3", out)
    assert re.search(r"empty field\s*: 3", out)


def test_exits_when_no_row_was_translated(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "おやすみ！", "translated": None}])

    with pytest.raises(SystemExit) as exc:
        preprocess.transform_for_lora(src, str(tmp_path / "out.jsonl"))
    assert exc.value.code == 1


@pytest.mark.parametrize(
    "original, translated, reason",
    [
        ("おやすみ！", "잘 자!", None),
        ("", "잘 자!", "empty field"),
        ("おやすみ！", "   ", "empty field"),
        ("ムクボ3돌 완료", "무크보 3돌 완료", "hangeul in original"),
        ("おやすみ！", "おやすみ!", "JP residual in translation"),
        ("遺跡1F", "遺跡 1F", "JP residual in translation"),
        ("あ", "가" * 11, "hallucination"),
        ("あ", "가" * 10, None),  # exactly 10x is allowed
        ("ID:1 " + "あ" * 150 + " ID:2", "모집", "recruitment spam"),
        ("ID:1 " + "あ" * 150, "모집", None),  # one ID is a normal recruitment line
        ("あ" * 150 + " ID:1 ID:2", "모집", "recruitment spam"),
        ("ID:1 ID:2", "모집", None),  # short lines are not spam
    ],
)
def test_clean_reason(original, translated, reason):
    assert preprocess.clean_reason(original, translated) == reason


def test_filters_keep_clean_rows_in_order_and_report_each_reason(write_jsonl, tmp_path, capsys):
    src = write_jsonl(
        "raw.jsonl",
        [
            {"original": "遺跡1F", "translated": "유적 1F"},
            {"original": "ムクボ3돌", "translated": "무크보 3돌"},  # hangeul in original
            {"original": "おやすみ", "translated": "おやすみ"},  # JP residual
            {"original": "遺跡1F", "translated": "다른 번역"},  # duplicate of line 1
            "{broken json",
            {"original": "スカイ", "translated": "스카이"},
            {"original": "あ", "translated": "가" * 50},  # hallucination
        ],
    )
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out))

    assert [r["input"] for r in read_jsonl(out)] == ["遺跡1F", "スカイ"]
    assert read_jsonl(out)[0]["output"] == "유적 1F"  # the first occurrence wins
    assert counts == {
        "total": 7,
        "passed": 2,
        "empty field": 0,
        "hangeul in original": 1,
        "JP residual in translation": 1,
        "hallucination": 1,
        "recruitment spam": 0,
        "duplicate": 1,
        "json error": 1,
    }
    report = capsys.readouterr().out
    for label, n in (("Total input", 7), ("Passed", 2), ("Skipped", 5), ("duplicate", 1), ("json error", 1)):
        assert re.search(rf"{label}\s*: {n}\b", report), label


def test_cleaned_text_is_stripped(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "  おやすみ！\n", "translated": " 잘 자! "}])
    out = tmp_path / "out.jsonl"

    preprocess.transform_for_lora(src, str(out))

    assert read_jsonl(out)[0]["input"] == "おやすみ！"
    assert read_jsonl(out)[0]["output"] == "잘 자!"


def test_pair_format_is_what_llamafactory_reads(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"pid": 1, "original": "遺跡1F", "translated": "유적 1F", "timestamp": 5}])
    out = tmp_path / "out.jsonl"

    preprocess.transform_for_lora(src, str(out), fmt="pair")

    assert read_jsonl(out) == [{"original": "遺跡1F", "translated": "유적 1F"}]


def test_unknown_format_is_refused_before_anything_is_written(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "遺跡1F", "translated": "유적 1F"}])
    out = tmp_path / "out.jsonl"

    with pytest.raises(ValueError, match="instruction"):
        preprocess.transform_for_lora(src, str(out), fmt="nope")
    assert not out.exists()


def test_main_takes_the_format_from_the_command_line(write_jsonl, tmp_path, monkeypatch):
    src = write_jsonl("raw.jsonl", [{"original": "遺跡1F", "translated": "유적 1F"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))

    preprocess.main(["--format", "pair"])
    assert list(read_jsonl(out)[0]) == ["original", "translated"]

    preprocess.main([])  # the default stays the unsloth format
    assert list(read_jsonl(out)[0]) == ["instruction", "input", "output"]


# --- prompt styles and the reverse direction (--format pair) -----------------------------------


def test_pair_rows_wrap_the_line_in_the_models_instruction():
    rows = preprocess.pair_rows("杖@2募集", "법사@2 모집", style="hy")

    assert rows == [{
        "original": "Translate the following text into Korean. Note that you should only output the translated "
                    "result without any additional explanation:\n\n杖@2募集",
        "translated": "법사@2 모집",
    }]


def test_pair_rows_without_a_style_are_the_raw_pair():
    assert preprocess.pair_rows("杖@2募集", "법사@2 모집") == [{"original": "杖@2募集", "translated": "법사@2 모집"}]


def test_reverse_adds_the_ko_to_ja_row_with_its_own_instruction():
    forward, backward = preprocess.pair_rows("杖@2募集", "법사@2 모집", style="translategemma", reverse=True)

    assert forward["original"].endswith("Please translate the following Japanese text into Korean:\n杖@2募集")
    assert forward["translated"] == "법사@2 모집"
    assert backward["original"].endswith("Please translate the following Korean text into Japanese:\n법사@2 모집")
    assert backward["translated"] == "杖@2募集"


def test_transform_writes_forward_and_reverse_rows_for_clean_rows_only(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [
        {"original": "杖@2募集", "translated": "법사@2 모집"},
        {"original": "이미 한글", "translated": "x"},  # Hangeul in the source: dropped, and not reversed either
    ])
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), fmt="pair", style="hy", reverse=True)

    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and counts["passed"] == 1
    assert rows[1]["translated"] == "杖@2募集"


def test_style_and_reverse_need_the_pair_format(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "杖", "translated": "법사"}])

    with pytest.raises(ValueError, match="pair"):
        preprocess.transform_for_lora(src, str(tmp_path / "o.jsonl"), fmt="instruction", style="hy")


def test_main_resolves_the_auto_style_from_the_model_parameter(write_jsonl, tmp_path, monkeypatch):
    src = write_jsonl("raw.jsonl", [{"original": "杖@2募集", "translated": "법사@2 모집"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))

    preprocess.main(["--format", "pair", "--prompt", "auto", "--model", "translategemma-4b"])

    assert json.loads(out.read_text(encoding="utf-8"))["original"].startswith("You are a professional Japanese (ja)")


def test_main_without_a_prompt_keeps_the_raw_line(write_jsonl, tmp_path, monkeypatch):
    src = write_jsonl("raw.jsonl", [{"original": "杖@2募集", "translated": "법사@2 모집"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))

    preprocess.main(["--format", "pair"])

    assert json.loads(out.read_text(encoding="utf-8"))["original"] == "杖@2募集"
