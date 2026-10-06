import json
import re

import pytest

import glossary
import translation_check as check

GLOSSARY = glossary.Glossary(
    season="T", doc_version="1.0.0",
    required=(glossary.Required(re.compile("墓"), None, "저주받은 무덤"),
              glossary.Required(re.compile("継"), re.compile("継続"), "계속")),
    banned=(glossary.Banned("와이프", None, "wife"), glossary.Banned("이지만", re.compile("イージー"), "although")),
    fixes=())


def inputs(*pairs):
    return {i: {"i": i, "ja": ja} for i, ja in pairs}


def row(i, ko, terms=(), **extra):
    return {"i": i, "ko": ko, "terms": [list(t) for t in terms], **extra}


def problems(want, got):
    return check.check_output(want, got, GLOSSARY)


# --- a clean output ----------------------------------------------------------------------------------------------------------


def test_a_complete_clean_output_has_no_problems():
    want = inputs((1, "墓M6 5周 ＠D2"), (2, "おつかれさまでした"))

    got = [row(1, "저주받은 무덤 M6 5회 @D2", [("墓", "저주받은 무덤")]), row(2, "수고하셨습니다")]

    assert problems(want, got) == []


# --- completeness ----------------------------------------------------------------------------------------------------------


def test_a_missing_a_duplicate_and_an_unknown_id_are_each_reported():
    want = inputs((1, "a"), (2, "b"), (3, "c"))

    found = problems(want, [row(1, "가"), row(1, "나"), row(9, "다")])

    assert any("missing" in p and "2" in p and "3" in p for p in found)
    assert any("i=1" in p and "duplicated" in p for p in found)
    assert any("unknown" in p and "9" in p for p in found)


@pytest.mark.parametrize("bad", [row(1, ""), row(1, "   "), {"i": 1}, {"i": 1, "ko": 5}])
def test_a_row_needs_a_non_empty_text(bad):
    assert any("i=1" in p and "ko" in p for p in problems(inputs((1, "a")), [bad]))


# --- Japanese left ---------------------------------------------------------------------------------------------------------


def test_kana_and_kanji_left_in_the_korean_are_reported():
    found = problems(inputs((1, "あ")), [row(1, "안녕 こんにちは")])

    assert any("Japanese" in p for p in found)
    assert any("Japanese" in p for p in problems(inputs((1, "あ")), [row(1, "개척 開拓")]))


def test_the_katakana_middle_dot_and_the_long_vowel_mark_of_a_kaomoji_are_not_japanese():
    assert problems(inputs((1, "(・ω・)ー")), [row(1, "(・ω・)ー")]) == []


# --- terms ------------------------------------------------------------------------------------------------------------------


def test_terms_are_pairs_and_the_korean_of_each_is_in_the_text():
    want = inputs((1, "墓"))

    assert any("pairs" in p for p in problems(want, [{"i": 1, "ko": "저주받은 무덤", "terms": ["墓"]}]))
    assert any("verbatim" in p for p in problems(want, [row(1, "저주받은 무덤", [("墓", "묘지")])]))
    assert problems(want, [row(1, "저주받은 무덤", [("墓", "저주받은 무덤")])]) == []


# --- numbers ----------------------------------------------------------------------------------------------------------------


def test_a_number_of_the_japanese_missing_from_the_korean_is_reported_unless_flagged():
    want = inputs((1, "５周 20ch"))

    assert any("number" in p for p in problems(want, [row(1, "5회")]))
    assert problems(want, [row(1, "5회", flag="the channel is unclear")]) == []
    assert problems(want, [row(1, "5회 20채널")]) == []


def test_digits_the_korean_adds_for_a_kanji_numeral_are_fine():
    assert problems(inputs((1, "三種")), [row(1, "3종")]) == []


# --- the glossary -----------------------------------------------------------------------------------------------------------


def test_a_required_name_missing_from_the_korean_is_reported_unless_flagged():
    want = inputs((1, "墓M6"))

    assert any("저주받은 무덤" in p for p in problems(want, [row(1, "무덤 M6")]))
    assert problems(want, [row(1, "무덤 M6", flag="unsure")]) == []


def test_an_unless_pattern_exempts_a_longer_term():
    want = inputs((1, "継NM"), (2, "継続します"))

    assert any("계속" in p and "i=1" in p for p in problems(want, [row(1, "NM"), row(2, "이어서 해요")]))
    assert problems(want, [row(1, "계속 NM"), row(2, "이어서 해요")]) == []


def test_a_banned_rendering_is_reported_and_a_when_pattern_limits_it():
    want = inputs((1, "ワイプ"), (2, "イージーのみ"), (3, "それだけ"))

    assert any("와이프" in p for p in problems(want, [row(1, "와이프"), row(2, "이지 난이도만"), row(3, "그것만")]))
    assert any("이지만" in p and "i=2" in p for p in problems(want, [row(1, "전멸"), row(2, "이지만"), row(3, "그것만")]))
    assert problems(want, [row(1, "전멸"), row(2, "이지 난이도만"), row(3, "해보지만 어렵네")]) == []  # 이지만 inside other words, no イージー: allowed


# --- the final lines ---------------------------------------------------------------------------------------------------------


def test_final_lines_get_the_glossary_and_japanese_checks_by_their_own_field_names():
    rows = [{"i": 1, "original": "墓", "translated": "무덤"}, {"i": 2, "original": "ワイプ", "translated": "와이프"},
            {"i": 3, "original": "あ", "translated": "あ"}, {"i": 4, "original": "墓", "translated": "무덤", "flag": "unsure"}]

    found = check.check_lines(rows, GLOSSARY)

    assert [p.split(":")[0] for p in found if "i=1" in p or "i=2" in p or "i=3" in p or "i=4" in p] == ["i=1", "i=2", "i=3"]


# --- reading ---------------------------------------------------------------------------------------------------------------


def test_reading_a_jsonl_skips_blank_lines_and_names_a_damaged_one(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_text('{"i": 1, "ko": "가"}\n\n{not json\n[1, 2]\n', encoding="utf-8")

    rows, found = check.read_jsonl(str(path))

    assert rows == [{"i": 1, "ko": "가"}]
    assert any("line 3" in p for p in found) and any("line 4" in p for p in found)


def test_reading_a_missing_file_is_a_problem_not_a_crash(tmp_path):
    rows, found = check.read_jsonl(str(tmp_path / "none.jsonl"))

    assert rows == [] and "none.jsonl" in found[0]


def test_the_reader_decodes_utf8_whatever_the_platform(tmp_path):
    path = tmp_path / "out.jsonl"
    path.write_bytes((json.dumps({"i": 1, "ko": "한국어"}, ensure_ascii=False) + "\n").encode("utf-8"))

    assert check.read_jsonl(str(path))[0][0]["ko"] == "한국어"
