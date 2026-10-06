"""The mechanics of labelling a season's chat lines with agents (or by hand), apart from the agents and from the judge.

The labelling guide (docs/labeling-guide.md) says how a line gets its category; this module is what surrounds that: the distinct lines of the raw
logs, the batches and the brief for the agents, the check of what they wrote, the labels file, and the judge's sample made from it. The labels may
use categories the taxonomy does not have yet ("proposed" in the guide); `configs/label_map.json` says where each one goes for the judge (a taxonomy
path, or out), and a test keeps the map and the guide in step. Pure: no files, no model.
"""

import collections
import json
import re

from glossary import VERSION_RX, SEMVER
from valsplit import bucket
from dataset_recipe import line_key

DEV_FRACTION = 0.7  # the share of the judge's sample that is the dev set (tune the cutoff on it); the rest is the test set
DROPPED_SECTIONS = ("Changelog", "Updating for a new season", "4. What season")  # results and maintenance are not rules


class LabelError(Exception):
    """The labels or the label map cannot be used; the message names the line or the file."""


class LabelMap:
    """doc_version: the version of the labeling guide the map was written for. map: proposed label -> taxonomy path. exclude: labels for lines the judge never sees."""

    def __init__(self, season, doc_version, map, exclude):
        self.season, self.doc_version, self.map, self.exclude = season, doc_version, dict(map), tuple(exclude)


def load_label_map(path, taxonomy):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise LabelError(f"{path}: label map not found") from None
    except (OSError, ValueError) as error:
        raise LabelError(f"{path}: not a readable JSON file ({error})") from None
    if not isinstance(data, dict) or not isinstance(data.get("map"), dict) or not isinstance(data.get("exclude", []), list):
        raise LabelError(f"{path}: needs \"map\": {{label: taxonomy path}} and \"exclude\": [label, ...]")
    doc_version = data.get("doc_version")
    if not isinstance(doc_version, str) or not SEMVER.match(doc_version):
        raise LabelError(f"{path}: doc_version must look like 1.0.0")
    exclude = data.get("exclude", [])
    for label, target in data["map"].items():
        if label in taxonomy.paths:
            raise LabelError(f"{path}: '{label}' is already a category of the taxonomy: remove it from the map")
        if target not in taxonomy.paths:
            raise LabelError(f"{path}: '{label}' maps to '{target}', which is not a category of the taxonomy")
        if label in exclude:
            raise LabelError(f"{path}: '{label}' is both mapped and excluded")
    for label in exclude:
        if label in taxonomy.paths:
            raise LabelError(f"{path}: '{label}' is already a category of the taxonomy: remove it from exclude")
    return LabelMap(str(data.get("season", "")), doc_version, data["map"], exclude)


def known(label, label_map, taxonomy):
    return isinstance(label, str) and (label in taxonomy.paths or label in label_map.map or label in label_map.exclude)


def judge_path(label, label_map, taxonomy):
    """What the judge's sample says for a label: the taxonomy path, or None for a line the judge never sees."""
    if label in taxonomy.paths:
        return label
    if label in label_map.map:
        return label_map.map[label]
    if label in label_map.exclude:
        return None
    raise LabelError(f"'{label}' is neither a category of the taxonomy nor in the label map")


# --- the lines and the agents' batches ---------------------------------------------------------------------------------------


def distinct_lines(rows):
    """[{original, channel, count}] in first-seen order: one entry per line as the dataset compares lines (`line_key`), with how often it was said
    and in which channel most of the time. Rows without a text are skipped."""
    first, counts, channels = {}, collections.Counter(), collections.defaultdict(collections.Counter)
    for row in rows:
        text = row.get("original")
        if not isinstance(text, str) or not text.strip():
            continue
        key = line_key(text)
        first.setdefault(key, text)
        counts[key] += 1
        channels[key][str(row.get("channel") or "?")] += 1
    return [{"original": first[key], "channel": channels[key].most_common(1)[0][0], "count": counts[key]} for key in first]


def input_rows(lines):
    """The rows an agent gets: `i` (1-based), `ch` (W/P/L/B), `n` (how often the line was said) and the `text`."""
    return [{"i": n, "ch": line["channel"][:1].upper(), "n": line["count"], "text": line["original"]} for n, line in enumerate(lines, 1)]


def check_output(inputs, outputs, label_map, taxonomy):
    """Problems of the agents' output rows `{i, cat, unsure?}` against `inputs` ({i: row}); empty = fine."""
    found, got = [], {}
    for r in outputs:
        i = r.get("i")
        if i in got:
            found.append(f"i={i}: duplicated")
            continue
        got[i] = r
    missing, unknown = sorted(set(inputs) - set(got)), sorted(i for i in got if i not in inputs)
    if missing:
        found.append(f"missing i: {missing[:30]}{' ...' if len(missing) > 30 else ''} ({len(missing)} total)")
    if unknown:
        found.append(f"unknown i: {unknown[:30]}")
    for i, r in sorted((i, r) for i, r in got.items() if i in inputs):
        cat = r.get("cat")
        if not isinstance(cat, str):
            found.append(f"i={i}: needs a string `cat`")
        elif not known(cat, label_map, taxonomy):
            found.append(f"i={i}: '{cat}' is not a category of the taxonomy or the label map")
        if "unsure" in r and not isinstance(r["unsure"], bool):
            found.append(f"i={i}: `unsure` must be true or false")
    return found


# --- the labels and the judge's sample ---------------------------------------------------------------------------------------


def assemble(lines, outputs):
    """The labelled lines `{original, category, channel, count, unsure?}`: `lines` (as numbered by `input_rows`) joined to the agents' `{i, cat, unsure?}`."""
    by_i = {r["i"]: r for r in outputs}
    rows, missing = [], []
    for n, line in enumerate(lines, 1):
        out = by_i.get(n)
        if out is None:
            missing.append(n)
            continue
        row = {"original": line["original"], "category": out["cat"], "channel": line["channel"], "count": line["count"]}
        if out.get("unsure"):
            row["unsure"] = True
        rows.append(row)
    if missing:
        raise LabelError(f"not labelled: {missing[:30]}{' ...' if len(missing) > 30 else ''} ({len(missing)} lines)")
    return rows


def judge_sample(rows, label_map, taxonomy, dev_fraction=DEV_FRACTION):
    """(sample rows, {excluded label: count}): what `categorize.py --probe` and `compare_judges.py` read. Labels map to taxonomy paths, excluded labels are
    dropped, and `split` is `dev` or `test` by a hash of the line, so a line keeps its side when the sample grows."""
    sample, excluded = [], collections.Counter()
    for row in rows:
        path = judge_path(row["category"], label_map, taxonomy)
        if path is None:
            excluded[row["category"]] += 1
            continue
        out = {"original": row["original"], "category": path, "channel": row.get("channel", ""), "count": row.get("count", 1),
               "split": "dev" if bucket(row["original"]) < dev_fraction else "test"}
        if row.get("unsure"):
            out["unsure"] = True
        sample.append(out)
    return sample, dict(excluded)


def label_counts(rows):
    """[(label, lines)] sorted by label."""
    return sorted(collections.Counter(r["category"] for r in rows).items())


def format_counts(counts, unsure):
    """The Markdown table for section 4 of the labeling guide."""
    lines = ["| Category | Lines |", "|---|---|"] + [f"| `{label}` | {n:,} |" for label, n in sorted(counts, key=lambda c: (-c[1], c[0]))]
    return "\n".join(lines) + f"\n\n{sum(n for _, n in counts):,} lines, unsure: {unsure}\n"


# --- the brief -----------------------------------------------------------------------------------------------------------------

OUTPUT_FORMAT = """
## Output (this task)

Label every input row. Each row is `{"i": 12, "ch": "W", "n": 3, "text": "..."}`: `ch` is the channel (W world, P party, L local, B beginner),
`n` how often the line was said, `text` the Japanese line. Write one JSON object per input row to your output file (UTF-8, one per line, `ensure_ascii` off):

    {"i": 12, "cat": "recruitment/party", "unsure": true}

- `cat`: one category of section 2 (a taxonomy path, or a proposed label of the table). The most specific path you are sure of, else its root.
- `unsure`: add `true` when you are not sure; leave it out otherwise. Never skip a line, never invent a category.
- When you are done, run the check you were given on your output file and fix every problem it lists.
"""


def build_brief(doc_text):
    """The brief for labelling agents: the guide without its results, its maintenance and its changelog, the version it is, and the output format."""
    version = VERSION_RX.search(doc_text)
    sections = re.split(r"(?m)^(?=## )", doc_text)
    kept = [s for s in sections if not s.startswith("## ") or not any(s[3:].startswith(name) for name in DROPPED_SECTIONS)]
    return "".join(kept).rstrip() + f"\n\n(Labeling guide version {version.group(1) if version else 'unknown'}.)\n" + OUTPUT_FORMAT
