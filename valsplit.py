"""Which training lines are held out for validation. Pure, no torch.

The split is decided per line by a hash of its normalised text (overlap.match_key), not by shuffling rows:
- variants of a line (`おやすみ` / `おやすみ！`) and the two directions of one pair stay on the same side, so validation
  loss is not flattered by near-copies of what the model trained on (LLaMA-Factory's `val_size` splits rows at random);
- a line keeps its side when the dataset grows (the data is refreshed about every 2 months), so validation loss stays
  comparable between runs.
The hash is sha1, not Python's salted `hash()`, so the split is the same in every process and on every machine.
"""

import hashlib

from overlap import match_key


def bucket(text):
    """A number in [0, 1) that depends only on the normalised text."""
    key = match_key(text) or (text or "")
    return int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def is_validation(text, fraction):
    """True when the line belongs to the validation set that holds about `fraction` of all lines."""
    return bucket(text) < fraction
