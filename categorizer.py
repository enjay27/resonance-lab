"""A rule-based first pass at a chat message's root category -- Gate 1's baseline. Pure, no torch, no network.

Rules over the normalised text (NFKC, lower case: full-width and half-width forms are one) say which root of
configs/category_taxonomy.json a line is in, or nothing: a line no rule recognises is left uncategorised, never guessed.
They run in order and the first that fires wins, so a stronger signal (a recruitment word) beats a weaker one (a question
mark). A model-based gate has to beat these on a hand-labelled sample (gate_eval.py) to be worth a GPU pass.

Known gaps: a flood of the same text needs to see several lines; everyday chat has no rule, so it stays uncategorised;
a human writing "…を入手しました" reads as a bot notice.
"""

import re
import unicodedata
from typing import NamedTuple

BASELINE = "regex-v1"  # recorded as `by` in the categories file


class Match(NamedTuple):
    root: str
    rule: str


# A readable character: ASCII letters and digits, kana, kanji, Hangul. Emoticons, punctuation and emoji have none.
_WORD = re.compile(r"[0-9a-zぁ-んァ-ヶ一-龥가-힣]")
_SYSTEM_TAG = re.compile(r"^[\[【(（<]?\s*(システム|system|通知|お知らせ)\s*[\]】)）>:：]")
_SYSTEM_NOTICE = re.compile(r"(入手|獲得|取得)しました|に(参加|加入|入室|退出)しました|が成立しました")
_RECRUIT = re.compile(
    r"募集|ギルメン|メンバー.{0,6}(探|募)|仲間.{0,4}探|@\s?(?:t|h|d|dps|ヒラ|タンク|火力|\d)|\d+k\s?[↑+]|\bid:\s?\d+|潜れる方|一緒に行ける人"
)
_SOCIAL = re.compile(
    r"^(おはよ|こんにちは|こんばんは|こんちゃ|おつ|お疲れ|ありがと|どうも|ども|よろしく|ごめん|すみません|すいません|おめでと|お先|おやすみ|またね|また明日|乙|ただいま|いってきます|お帰り)"
)
_COORDINATION = re.compile(
    r"準備\s?(ok|おk|完了|できた|おけ)|待って|集合|行きます|いきます|行くよ|いくよ|スタート|開始します|離脱|撤退|次(いきましょう|行きましょう)|回復(お願い|ください|頼む)|蘇生|ヘイト取|タゲ"
)
_QUESTION = re.compile(
    r"\?|ですか|ますか|でしょうか|教えて|どうやって|どこで|どこに|どこが|何を|何が|誰か.{0,8}(いる|わかる)|いくつ|いくら|どの|どれ|どう(すれば|したら|やれば)"
)
_MARKET = re.compile(r"マーケット|マケ|相場|出品|手数料|売れ|売る|売って|買え|価格|値段|高く売|安く買")
_GAME = re.compile(
    r"ビルド|スキル|ダメージ|火力|クラス|ボス|装備|強化|防御|ヘイト|タンク|ヒーラー|ウルト|クールタイム|レベル|クエ|周回|素材|マップ|遺跡|ダンジョン"
    r"|メンテ|アプデ|アップデート|更新|バグ|ラグ|イベント|ログイン(できな|出来な|エラー|障害)|サーバー|ガチャ"
)
_LAUGHTER = re.compile(r"^(?=.*[w草笑])[w草笑!~〜]+$")
_REACTION_WORD = re.compile(r"^(すご|えぐ|ナイス|やば|うま|かわい|つよ|さすが|いいね|いいな|うける|ウケる)")

# (rule name, root, test on the normalised text), in the order they are tried
_RULES = (
    ("wall", "spam", lambda t: len(t) > 150 and t.count("id:") > 1),  # as preprocess.clean_reason's "recruitment spam"
    ("symbols-only", "other", lambda t: not _WORD.search(t)),
    ("system-tag", "bot", _SYSTEM_TAG.search),
    ("system-notice", "bot", _SYSTEM_NOTICE.search),
    ("recruitment-words", "recruitment", _RECRUIT.search),
    ("social-formula", "social", lambda t: len(t) <= 30 and _SOCIAL.search(t)),
    ("call-out", "coordination", lambda t: len(t) <= 24 and _COORDINATION.search(t)),
    ("question-words", "question", _QUESTION.search),
    ("market-words", "game", _MARKET.search),
    ("game-words", "game", _GAME.search),
    ("laughter", "chat", _LAUGHTER.match),
    ("reaction-word", "chat", lambda t: len(t) <= 12 and _REACTION_WORD.match(t)),
    ("lone-character", "other", lambda t: len(t) == 1),
)


def normalise(text):
    return unicodedata.normalize("NFKC", text or "").strip().lower()


def classify(text):
    """The Match (root, rule) of the first rule that fires on `text`, or None."""
    normal = normalise(text)
    if not normal:
        return None
    for name, root, test in _RULES:
        if test(normal):
            return Match(root, name)
    return None


def categorize(text):
    """The root category of `text`, or None when no rule recognises it."""
    match = classify(text)
    return match.root if match else None
