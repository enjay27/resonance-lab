import pytest

import validate


def test_passes_when_no_original_contains_hangeul(write_jsonl, capsys):
    path = write_jsonl("raw.jsonl", [{"original": "おやすみ！", "translated": "잘 자!"}])

    validate.check_hangeul_in_original(path)

    assert "No Hangeul detected" in capsys.readouterr().out


def test_raises_local_validation_error_when_original_contains_hangeul(write_jsonl):
    path = write_jsonl(
        "raw.jsonl",
        [
            {"original": "おやすみ！", "translated": "잘 자!"},
            {"original": "ムクボ3돌 완료!", "translated": "무크보 3돌 완료!"},
        ],
    )

    with pytest.raises(validate.ValidationError, match="Total 1 lines"):
        validate.check_hangeul_in_original(path)


def test_skips_invalid_json_lines(write_jsonl, capsys):
    path = write_jsonl("raw.jsonl", ["{not json", {"original": "遺跡1F", "translated": "유적 1F"}])

    validate.check_hangeul_in_original(path)

    out = capsys.readouterr().out
    assert "[Line 1] Invalid JSON format." in out
    assert "No Hangeul detected" in out
