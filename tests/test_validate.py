import pytest

import validate


def _clean(n):
    return [{"original": f"おやすみ{i}", "translated": f"잘 자{i}"} for i in range(n)]


def test_a_clean_file_passes_and_says_so(write_jsonl, capsys):
    path = write_jsonl("raw.jsonl", _clean(10))

    stats = validate.validate_raw_logs(path)

    assert stats["rows"] == 10 and stats["hangeul_in_original"] == 0 and stats["structure_errors"] == 0
    assert "OK" in capsys.readouterr().out


def test_a_few_stray_hangeul_lines_are_reported_but_do_not_stop_the_pipeline(write_jsonl, capsys):
    # preprocess.py drops these rows anyway: one line of Korean chat must not halt the whole run.
    rows = _clean(99) + [{"original": "ムクボ3돌 완료!", "translated": "무크보 3돌 완료!"}]
    path = write_jsonl("raw.jsonl", rows)

    stats = validate.validate_raw_logs(path)

    assert stats["hangeul_in_original"] == 1
    out = capsys.readouterr().out
    assert "[Line 100] Hangeul in original: ムクボ3돌 완료!" in out
    assert "1/100" in out


def test_many_hangeul_originals_mean_the_wrong_file_and_stop_the_pipeline(write_jsonl):
    rows = _clean(8) + [{"original": f"한글만 {i}", "translated": "x"} for i in range(2)]  # 20%
    path = write_jsonl("raw.jsonl", rows)

    with pytest.raises(validate.ValidationError, match="Hangeul"):
        validate.validate_raw_logs(path)


def test_only_the_first_few_hangeul_lines_are_listed(write_jsonl, capsys):
    rows = _clean(300) + [{"original": f"한글 {i}", "translated": "x"} for i in range(20)]  # 6%: passes
    path = write_jsonl("raw.jsonl", rows)

    validate.validate_raw_logs(path)

    assert capsys.readouterr().out.count("] Hangeul in original:") == validate.LINES_SHOWN


def test_invalid_json_is_counted_not_ignored(write_jsonl, capsys):
    path = write_jsonl("raw.jsonl", _clean(199) + ["{not json"])  # 0.5%: under the limit

    stats = validate.validate_raw_logs(path)

    assert stats["structure_errors"] == 1
    assert "[Line 200] Invalid JSON format." in capsys.readouterr().out


def test_structural_damage_above_the_limit_stops_the_pipeline(write_jsonl):
    path = write_jsonl("raw.jsonl", _clean(100) + ["{not json"] * 5)  # ~4.8%

    with pytest.raises(validate.ValidationError, match="structure"):
        validate.validate_raw_logs(path)


@pytest.mark.parametrize("row", [
    {"original": "あ"},                       # no translated key
    {"translated": "아"},                     # no original key
    {"original": 5, "translated": "아"},      # original is not text
    [1, 2],                                   # not an object
])
def test_rows_without_the_fields_the_app_writes_are_structural_errors(write_jsonl, row):
    path = write_jsonl("raw.jsonl", [row])

    with pytest.raises(validate.ValidationError, match="structure"):
        validate.validate_raw_logs(path)


def test_an_untranslated_line_is_fine(write_jsonl):
    # resonance-stream writes `translated: null` for lines it never translated; preprocess skips them.
    path = write_jsonl("raw.jsonl", _clean(5) + [{"pid": 1, "original": "おやすみ", "translated": None, "timestamp": 1}])

    assert validate.validate_raw_logs(path)["structure_errors"] == 0


def test_blank_lines_are_not_rows(write_jsonl):
    path = write_jsonl("raw.jsonl", ["", *_clean(3), "   "])

    assert validate.validate_raw_logs(path)["rows"] == 3


def test_an_empty_file_stops_the_pipeline(write_jsonl):
    with pytest.raises(validate.ValidationError, match="no rows"):
        validate.validate_raw_logs(write_jsonl("raw.jsonl", []))


def test_a_missing_file_is_a_validation_error_not_a_traceback(tmp_path):
    with pytest.raises(validate.ValidationError, match="not found"):
        validate.validate_raw_logs(str(tmp_path / "missing.jsonl"))


def test_main_exits_one_with_the_reason_when_validation_fails(write_jsonl, monkeypatch, capsys):
    monkeypatch.setattr(validate, "RAW_LOGS", write_jsonl("raw.jsonl", []))

    with pytest.raises(SystemExit) as exc:
        validate.main()

    assert exc.value.code == 1
    assert "no rows" in capsys.readouterr().out


def test_the_limits_come_from_config():
    import config

    assert validate.MAX_STRUCTURE_ERRORS == config.VALIDATE_MAX_STRUCTURE_ERRORS
    assert validate.MAX_HANGEUL_IN_ORIGINAL == config.VALIDATE_MAX_HANGEUL_IN_ORIGINAL
