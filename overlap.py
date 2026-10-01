"""Does a training line also occur in the eval set? Pure, no torch.

A fine-tuned model that has seen the eval lines scores far too well (`roadmap/first-training-run-2026-10-01.md`,
"Caution"), so preprocess.py drops them from the training data. Lines are compared after normalising, and long
lines also by similarity: chat repeats itself with small changes (an added `ね`, `!!`, other width).
"""

import unicodedata
from difflib import SequenceMatcher

NEAR_RATIO = 0.9  # SequenceMatcher ratio at or above which two long lines count as the same line
NEAR_MIN_LENGTH = 10  # normalised length both lines need before similarity is used; shorter ones match exactly only


def match_key(text):
    """`text` as compared: NFKC (full/half width), case-folded, without spaces, punctuation and symbols."""
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(ch for ch in folded if unicodedata.category(ch)[0] not in "ZPSC")


class EvalOverlap:
    """The eval originals; `check(line)` says whether a training line is one of them."""

    def __init__(self, eval_originals, near_ratio=NEAR_RATIO, near_min_length=NEAR_MIN_LENGTH):
        self.near_ratio = near_ratio
        self.near_min_length = near_min_length
        self.keys = {key for key in map(match_key, eval_originals) if key}
        self._long = sorted(key for key in self.keys if len(key) >= near_min_length)

    def __len__(self):
        return len(self.keys)

    def check(self, text):
        """"exact" (same after normalising), "near" (a long line that is almost the same) or None."""
        key = match_key(text)
        if not key:
            return None
        if key in self.keys:
            return "exact"
        if len(key) < self.near_min_length:
            return None
        for other in self._long:
            # The ratio cannot reach near_ratio when the lengths differ this much: skip the expensive comparison.
            if 2 * min(len(key), len(other)) / (len(key) + len(other)) < self.near_ratio:
                continue
            if SequenceMatcher(None, key, other, autojunk=False).ratio() >= self.near_ratio:
                return "near"
        return None
