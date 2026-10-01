"""Character rules shared by the data cleaning (preprocess.py) and the eval metrics (eval_metrics.py)."""

import re

# Kana and kanji: Hiragana U+3041-3096, Katakana U+30A1-30FA, CJK ideographs U+4E00-9FFF.
JP_PATTERN = re.compile(r'[ぁ-ゖァ-ヺ一-鿿]')
HANGEUL_PATTERN = re.compile(r'[가-힣]')
