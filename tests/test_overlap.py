import pytest

from overlap import EvalOverlap, match_key


@pytest.mark.parametrize(
    "a, b",
    [
        ("おやすみ！", "おやすみ"),  # trailing punctuation
        ("ＩＤ：1234", "id:1234"),  # full-width, case
        ("遺跡 1F", "遺跡1F"),  # whitespace
        ("  スカイ\n", "スカイ"),
        ("杖@2募集!!", "杖@2募集"),
    ],
)
def test_match_key_ignores_width_case_space_and_punctuation(a, b):
    assert match_key(a) == match_key(b)


def test_match_key_keeps_the_letters_that_decide_meaning():
    assert match_key("杖@2募集") != match_key("弓@2募集")
    assert match_key("ー") == "ー"  # the long-vowel mark is a letter, not punctuation


def test_exact_match_after_normalising():
    overlap = EvalOverlap(["おやすみ！", "遺跡1F"])

    assert overlap.check("おやすみ") == "exact"
    assert overlap.check("遺跡 1F") == "exact"
    assert overlap.check("スカイ") is None


def test_near_match_needs_a_long_enough_line_and_a_high_ratio():
    eval_line = "今日のレイドは21時から始めます、参加できる人は教えてください"
    overlap = EvalOverlap([eval_line])

    assert overlap.check(eval_line + "ね") == "near"  # one character added
    assert overlap.check("今日のレイドは21時から始めます") is None  # clearly shorter: a different line
    assert overlap.check("明日は何時から集合ですか、みなさん教えてください") is None


def test_short_lines_only_match_exactly():
    # "ありがとう" vs "ありがとうございます": similar strings, different lines; short lines must not be fuzzy-matched.
    overlap = EvalOverlap(["ありがとう"])

    assert overlap.check("ありがとうございます") is None
    assert overlap.check("ありがとう!") == "exact"


def test_empty_and_punctuation_only_lines_never_match():
    overlap = EvalOverlap(["！！！", "", "   "])

    assert overlap.check("???") is None
    assert overlap.check("") is None
    assert len(overlap) == 0
