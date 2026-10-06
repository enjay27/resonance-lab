"""Is what the translation agents wrote complete, clean and in line with the glossary? Pure: rows in, a list of problems out (empty = fine).

Two shapes are checked with the same rules. An agent's OUTPUT row is `{i, ko, terms, flag?}` against its INPUT row `{i, ja, ...}`
(`check_output`: also completeness, the `terms` list and the numbers); a FINAL row is `{i, original, translated, flag?}` (`check_lines`).
A line the agent flagged (`flag` = an English note on its doubt) is exempt from the glossary and number rules: the doubt is the report.
"""

import collections
import json
import re

from glossary import Glossary  # noqa: F401  (the type the checks take)

# Kana and kanji. The katakana middle dot (U+30FB) and the long-vowel mark (U+30FC) are punctuation: kaomoji and bullets copy them from the source.
JAPANESE = re.compile("[ぁ-ゟァ-ヺヽ-ヿㇰ-ㇿ一-鿿]")
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")
SHOWN = 30  # ids listed in one "missing" line


def read_jsonl(path):
    """(rows, problems): the JSON objects of a UTF-8 JSONL file; blank lines are skipped, a line that is not an object is a problem naming it."""
    rows, problems = [], []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError as error:
        return [], [f"{path}: cannot be read ({error})"]
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            problems.append(f"line {number}: not valid JSON")
            continue
        if not isinstance(value, dict):
            problems.append(f"line {number}: not a JSON object")
            continue
        rows.append(value)
    return rows, problems


def japanese_left(text):
    return JAPANESE.search(text) is not None


def _glossary_problems(i, ja, ko, flagged, glossary):
    found = []
    if japanese_left(ko):
        found.append(f"i={i}: Japanese characters left in the Korean: {ko[:70]!r}")
    for rule in glossary.required:
        if rule.ja.search(ja) and not (rule.unless and rule.unless.search(ja)) and rule.ko not in ko and not flagged:
            found.append(f"i={i}: the Japanese has {rule.ja.pattern!r} but the Korean has no '{rule.ko}': {ja[:40]!r} -> {ko[:50]!r}")
    for rule in glossary.banned:
        if rule.ko in ko and (rule.when is None or rule.when.search(ja)):
            found.append(f"i={i}: contains '{rule.ko}' ({rule.why}): {ko[:60]!r}")
    return found


def check_lines(rows, glossary):
    """Problems of final rows `{i, original, translated, flag?}`: Japanese left, required names missing, banned renderings."""
    found = []
    for r in rows:
        found += _glossary_problems(r["i"], r["original"], r["translated"], bool(r.get("flag")), glossary)
    return found


def check_output(inputs, outputs, glossary):
    """Problems of an agent's output rows against `inputs` ({i: input row with `ja`}): completeness first, then each row."""
    found, got = [], {}
    for r in outputs:
        i = r.get("i")
        if i in got:
            found.append(f"i={i}: duplicated")
            continue
        got[i] = r
    missing, unknown = sorted(set(inputs) - set(got)), sorted(i for i in got if i not in inputs)
    if missing:
        found.append(f"missing i: {missing[:SHOWN]}{' ...' if len(missing) > SHOWN else ''} ({len(missing)} total)")
    if unknown:
        found.append(f"unknown i: {unknown[:SHOWN]}")
    for i, r in sorted((i, r) for i, r in got.items() if i in inputs):
        ko = r.get("ko")
        if not isinstance(ko, str) or not ko.strip():
            found.append(f"i={i}: needs a non-empty string ko")
            continue
        ja, flagged = inputs[i]["ja"], bool(r.get("flag"))
        found += _glossary_problems(i, ja, ko, flagged, glossary)
        terms = r.get("terms", [])
        if not isinstance(terms, list) or any(not (isinstance(t, list) and len(t) == 2 and all(isinstance(x, str) for x in t)) for t in terms):
            found.append(f"i={i}: terms must be a list of [ja, ko] pairs")
        else:
            found += [f"i={i}: term {ko_term!r} is not verbatim in ko" for _, ko_term in terms if ko_term not in ko]
        # a digit of the Japanese gone from the Korean (the Korean may add digits for kanji numerals)
        lost = collections.Counter(re.findall(r"\d+", ja.translate(FULLWIDTH_DIGITS))) - collections.Counter(re.findall(r"\d+", ko.translate(FULLWIDTH_DIGITS)))
        if lost and not flagged:
            found.append(f"i={i}: the Korean lost number(s) {sorted(lost.elements())} of the Japanese and has no flag: {ja[:40]!r} -> {ko[:40]!r}")
    return found
