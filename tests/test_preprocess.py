import json
import os
import re

import pytest

import config
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
        "JP in original (ko_ja)": 0,
        "Hangeul residual in translation (ko_ja)": 0,
        "hallucination": 1,
        "recruitment spam": 0,
        "eval overlap": 0,
        "eval overlap (near)": 0,
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
    assert len(rows) == 2 and counts["passed"] == 2  # one forward and one reverse row
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


# --- the reverse direction's own filters (experiment/translategemma aab6b66, "process bidirectual training") ---


@pytest.mark.parametrize("japanese,korean,reason", [
    ("杖@2募集", "법사@2 모집", None),
    ("杖@2募集", "법사 杖@2", "JP in original (ko_ja)"),  # kana/kanji left in the Korean that would be the input
    ("杖@2募集", "", "empty field"),
])
def test_reverse_clean_reason_filters_the_korean_side_as_the_source(japanese, korean, reason):
    assert preprocess.clean_reason(korean, japanese, direction="ko-ja") == reason


def test_reverse_clean_reason_flags_hangeul_left_in_the_japanese_answer():
    assert preprocess.clean_reason("법사@2 모집", "杖@2 모집 모집", direction="ko-ja") == "Hangeul residual in translation (ko_ja)"


def test_reverse_clean_reason_keeps_the_ten_times_and_spam_rules():
    assert preprocess.clean_reason("가", "あ" * 11, direction="ko-ja") == "hallucination"
    spam = "ID: 1 ID: 2 " + "가" * 150
    assert preprocess.clean_reason(spam, "あ", direction="ko-ja") == "recruitment spam"


def test_forward_filters_are_unchanged_by_the_direction_argument():
    assert preprocess.clean_reason("이미 한글", "x") == "hangeul in original"
    assert preprocess.clean_reason("杖", "법사") is None


def test_reverse_is_tried_only_for_forward_rows_and_counted_on_its_own(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [
        {"original": "杖@2募集", "translated": "법사@2 모집"},          # forward ok, reverse ok
        {"original": "杖募集" * 4, "translated": "법"},                # forward ok; reverse: Japanese 12x the Korean
        {"original": "이미 한글", "translated": "x"},                  # forward dropped: no reverse attempted
    ])
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), fmt="pair", reverse=True)

    assert counts["total"] == 3 + 2  # three forward attempts, two reverse attempts
    assert counts["passed"] == 3     # 2 forward + 1 reverse
    assert counts["hallucination"] == 1  # the reverse one
    assert counts["hangeul in original"] == 1
    assert len(out.read_text(encoding="utf-8").splitlines()) == 3


def test_reverse_input_that_is_already_a_seen_source_is_a_duplicate(write_jsonl, tmp_path):
    # The second line's Korean answer equals the first line's input: as a reverse source it would repeat a seen input.
    src = write_jsonl("raw.jsonl", [
        {"original": "AAA", "translated": "BBB"},
        {"original": "CCC", "translated": "AAA"},
    ])
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), fmt="pair", reverse=True)

    assert counts["duplicate"] >= 1


def test_the_translategemma_profile_trains_with_room_for_the_instruction():
    import yaml

    for name in ("translategemma-4b", "translategemma-4b-fast", "hy-mt2-1.8b", "hy-mt2-7b"):
        path = f"{config.BASE_DIR}/configs/llamafactory/{name}/train.yaml"
        with open(path, encoding="utf-8") as f:
            assert yaml.safe_load(f)["cutoff_len"] >= 256, name


# --- the eval set stays out of the training data --------------------------------------------------


EVAL_LINE = "今日のレイドは21時から始めます、参加できる人は教えてください"


def test_rows_whose_original_is_in_the_eval_set_are_dropped_and_counted(write_jsonl, tmp_path, capsys):
    src = write_jsonl("raw.jsonl", [
        {"original": "ウルト溜まった", "translated": "궁 찼다!"},  # the eval line
        {"original": "ウルト溜まった！", "translated": "궁 찼다"},  # same line after normalising
        {"original": EVAL_LINE + "ね", "translated": "오늘 레이드는 21시부터예요"},  # near-duplicate
        {"original": "スカイ", "translated": "스카이"},
    ])
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out), eval_originals=["ウルト溜まった", EVAL_LINE])

    assert [r["input"] for r in read_jsonl(out)] == ["スカイ"]
    assert counts["eval overlap"] == 2 and counts["eval overlap (near)"] == 1
    assert counts["passed"] == 1
    assert re.search(r"eval overlap\s*: 2", capsys.readouterr().out)


def test_the_reverse_row_of_an_eval_line_is_not_written_either(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [
        {"original": "ウルト溜まった", "translated": "궁 찼다!"},
        {"original": "スカイ", "translated": "스카이"},
    ])
    out = tmp_path / "out.jsonl"

    preprocess.transform_for_lora(src, str(out), fmt="pair", reverse=True, eval_originals=["ウルト溜まった"])

    assert {r["original"] for r in read_jsonl(out)} == {"スカイ", "스카이"}  # both directions of スカイ only, none of the eval line


def test_without_an_eval_set_nothing_is_excluded(write_jsonl, tmp_path):
    src = write_jsonl("raw.jsonl", [{"original": "ウルト溜まった", "translated": "궁 찼다!"}])
    out = tmp_path / "out.jsonl"

    counts = preprocess.transform_for_lora(src, str(out))

    assert counts["passed"] == 1 and counts["eval overlap"] == 0


def _eval_file(tmp_path, rows):
    path = tmp_path / "eval.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_main_excludes_the_eval_set_by_default_when_the_file_exists(write_jsonl, tmp_path, monkeypatch, capsys):
    src = write_jsonl("raw.jsonl", [
        {"original": "ウルト溜まった", "translated": "궁 찼다!"},
        {"original": "スカイ", "translated": "스카이"},
    ])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", _eval_file(tmp_path, [{"original": "ウルト溜まった", "translated": "궁 찼다!"}]))

    preprocess.main(["--format", "pair"])

    assert [r["original"] for r in read_jsonl(out)] == ["スカイ"]
    assert "Excluding eval lines" in capsys.readouterr().out


def test_main_keep_eval_turns_the_exclusion_off(write_jsonl, tmp_path, monkeypatch):
    src = write_jsonl("raw.jsonl", [{"original": "ウルト溜まった", "translated": "궁 찼다!"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", _eval_file(tmp_path, [{"original": "ウルト溜まった", "translated": "x"}]))

    preprocess.main(["--format", "pair", "--keep-eval"])

    assert len(read_jsonl(out)) == 1


def test_main_says_so_when_there_is_no_eval_set(write_jsonl, tmp_path, monkeypatch, capsys):
    src = write_jsonl("raw.jsonl", [{"original": "スカイ", "translated": "스카이"}])
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(tmp_path / "out.jsonl"))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(tmp_path / "missing.jsonl"))

    preprocess.main(["--format", "pair"])

    assert "no eval set" in capsys.readouterr().out


def test_main_stops_on_an_unreadable_eval_set(write_jsonl, tmp_path, monkeypatch):
    src = write_jsonl("raw.jsonl", [{"original": "スカイ", "translated": "스카이"}])
    bad = tmp_path / "eval.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(tmp_path / "out.jsonl"))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(bad))

    with pytest.raises(SystemExit) as exc:
        preprocess.main(["--format", "pair"])
    assert exc.value.code == 1


# --- the sidecar manifest says how the training file was made ---------------------------------------


def test_main_writes_the_manifest_next_to_the_output(write_jsonl, tmp_path, monkeypatch):
    import manifest

    src = write_jsonl("raw.jsonl", [
        {"original": "ウルト溜まった", "translated": "궁 찼다!"},
        {"original": "スカイ", "translated": "스카이"},
    ])
    out = tmp_path / "out.jsonl"
    eval_path = _eval_file(tmp_path, [{"original": "ウルト溜まった", "translated": "x"}])
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", eval_path)

    preprocess.main(["--format", "pair", "--prompt", "hy", "--reverse"])

    meta = manifest.read_manifest(manifest.manifest_path(str(out)))
    assert (meta["format"], meta["style"], meta["reverse"]) == ("pair", "hy", True)
    assert meta["data_sha256"] == manifest.file_sha256(str(out))
    assert meta["raw_sha256"] == manifest.file_sha256(src)
    assert meta["counts"]["eval overlap"] == 1 and meta["counts"]["passed"] == 2  # スカイ forward + reverse
    assert meta["eval_lines_excluded_from"] == 1


def test_main_without_a_prompt_records_no_style(write_jsonl, tmp_path, monkeypatch):
    import manifest

    src = write_jsonl("raw.jsonl", [{"original": "スカイ", "translated": "스카이"}])
    out = tmp_path / "out.jsonl"
    monkeypatch.setattr(preprocess, "RAW_LOGS", src)
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(tmp_path / "missing.jsonl"))

    preprocess.main([])

    meta = manifest.read_manifest(manifest.manifest_path(str(out)))
    assert (meta["format"], meta["style"], meta["reverse"], meta["eval_set"]) == ("instruction", None, False, None)


def test_a_failed_preprocess_leaves_no_stale_manifest(write_jsonl, tmp_path, monkeypatch):
    import manifest

    out = tmp_path / "out.jsonl"
    stale = manifest.manifest_path(str(out))
    with open(stale, "w", encoding="utf-8") as f:
        f.write("{}")
    monkeypatch.setattr(preprocess, "RAW_LOGS", write_jsonl("raw.jsonl", [{"original": "x", "translated": None}]))
    monkeypatch.setattr(preprocess, "PROCESSED_LOGS", str(out))
    monkeypatch.setattr(preprocess, "EVAL_DATASET_PATH", str(tmp_path / "missing.jsonl"))

    with pytest.raises(SystemExit):
        preprocess.main([])

    assert not os.path.exists(stale)
