import pytest

import categorizer
from categorizer import BASELINE, categorize, classify
from taxonomy import load_taxonomy

WALL = "ギルメン募集 " + " ".join(f"ID:{n}" for n in range(20001, 20021))


@pytest.mark.parametrize("text, root", [
    # a long pasted wall with several IDs
    (WALL, "spam"),
    ("ID:1 ID:2 " + "あ" * 150, "spam"),
    # automated notices
    ("【システム】メンテナンスのお知らせ", "bot"),
    ("[システム] 鉱石を入手しました", "bot"),
    ("System: 報酬を獲得しました", "bot"),
    ("〇〇がパーティに参加しました", "bot"),
    # recruitment
    ("遺跡1F 29k↑ @T1", "recruitment"),
    ("杖@2募集", "recruitment"),
    ("ギルメン募集！初心者歓迎", "recruitment"),
    ("ID:1234 タンク募集", "recruitment"),
    ("新規ギルドです、仲間探しています", "recruitment"),
    ("２人で潜れる方募集", "recruitment"),  # full-width digit
    ("火力募集中ですか？", "recruitment"),  # a recruitment word wins over the question mark
    # social formulas
    ("おはようございます", "social"),
    ("こんばんは〜", "social"),
    ("ありがとうございました", "social"),
    ("お疲れ様です！", "social"),
    ("乙です", "social"),
    # coordination
    ("準備OKです", "coordination"),
    ("ちょっと待って", "coordination"),
    ("ボス行きます", "coordination"),
    ("回復お願い！", "coordination"),
    # questions: a question word or mark wins over the topic
    ("この装備はどこで手に入りますか？", "question"),
    ("ボスの攻略法教えてください", "question"),
    ("マケに何を出せばいいですか？", "question"),
    ("これって強いの？", "question"),
    ("週末は何するの？", "question"),
    # statements about the game, the market included
    ("火力足りないからビルド変えようかな", "game"),
    ("メインクエスト詰まってる", "game"),
    ("アプデ後にラグがひどい", "game"),
    ("マーケットの相場が下がってる", "game"),
    ("出品したけど全然売れない", "game"),
    # reactions are chat
    ("草", "chat"),
    ("wwwww", "chat"),
    ("ｗｗｗ", "chat"),
    ("すごい！", "chat"),
    ("えぐｗ", "chat"),
    # nothing readable
    ("…", "other"),
    ("。。。", "other"),
    ("?", "other"),
    ("ｱ", "other"),
    ("👍", "other"),
    ("(´・ω・`)", "other"),
    # no rule fires: the baseline says nothing instead of guessing
    ("お腹すいたなあ", None),
    ("新しいスマホ買った", None),
    ("", None),
    ("   ", None),
])
def test_the_baseline_puts_a_line_in_its_root_or_says_nothing(text, root):
    assert categorize(text) == root


def test_a_greeting_inside_a_long_line_is_not_a_social_formula():
    text = "おはよう、今日は新しい作戦についてみんなで少し話したいんだけど時間ある人いたら教えてほしい"

    assert len(text) > 30 and categorize(text) != "social"


def test_a_long_line_without_ids_is_not_spam():
    assert categorize("あ" * 200) is None


def test_the_match_names_the_rule_that_fired():
    match = classify("杖@2募集")

    assert match.root == "recruitment" and match.rule
    assert classify("お腹すいたなあ") is None
    assert classify(WALL).rule != classify("杖@2募集").rule


def test_every_root_it_returns_is_a_root_of_the_taxonomy():
    roots = set(load_taxonomy().roots)

    for text in ("草", "…", "杖@2募集", "準備OK", "乙です", "メンテ延長だって", "ボス行きます", "【システム】x", WALL, "どこ？"):
        assert categorize(text) in roots


def test_full_width_and_half_width_forms_give_the_same_root():
    assert categorize("ｗｗｗ") == categorize("www") and categorize("準備ＯＫ") == categorize("準備OK")
    assert categorize("どこですか？") == categorize("どこですか?")


def test_the_baseline_has_a_name_for_the_categories_file():
    assert BASELINE == "regex-v1" and categorizer.BASELINE == BASELINE
