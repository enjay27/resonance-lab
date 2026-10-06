import json
import os

import pytest

import glossary
import labeling_tools as tools
from config import BASE_DIR
from labeling_tools import LabelError
from taxonomy import load_taxonomy
from valsplit import bucket

TAXONOMY = load_taxonomy()
LABEL_MAP_PATH = os.path.join(BASE_DIR, "configs", "label_map.json")
GUIDE = os.path.join(BASE_DIR, "docs", "labeling-guide.md")


def write(tmp_path, **fields):
    data = {"version": 1, "season": "T", "doc_version": "1.0.0", "map": {"coordination/boss_call": "coordination"}, "exclude": ["non_japanese"]}
    data.update(fields)
    path = tmp_path / "map.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def label_map(**fields):
    return tools.LabelMap("T", "1.0.0", fields.get("map", {"coordination/boss_call": "coordination", "recruitment/closed": "recruitment"}),
                          tuple(fields.get("exclude", ("non_japanese",))))


# --- the label map ---------------------------------------------------------------------------------------------------------


def test_a_label_map_loads(tmp_path):
    loaded = tools.load_label_map(write(tmp_path), TAXONOMY)

    assert loaded.map == {"coordination/boss_call": "coordination"} and loaded.exclude == ("non_japanese",) and loaded.doc_version == "1.0.0"


@pytest.mark.parametrize("fields, message", [
    ({"map": {"x/y": "nonsense"}}, "nonsense"),                       # maps to something that is not a taxonomy category
    ({"map": {"chat": "game"}}, "already a category"),                 # a category the taxonomy has no longer needs the map: remove it
    ({"exclude": ["chat"]}, "already a category"),
    ({"map": {"a": "chat"}, "exclude": ["a"]}, "both"),
    ({"doc_version": "1"}, "doc_version"),
    ({"map": []}, "map"),
])
def test_a_malformed_label_map_says_what_is_wrong(tmp_path, fields, message):
    with pytest.raises(LabelError, match=message):
        tools.load_label_map(write(tmp_path, **fields), TAXONOMY)


def test_a_missing_label_map_is_an_error_naming_it(tmp_path):
    with pytest.raises(LabelError, match="nowhere.json"):
        tools.load_label_map(str(tmp_path / "nowhere.json"), TAXONOMY)


def test_the_shipped_label_map_loads_against_the_real_taxonomy_and_matches_the_guide_version():
    shipped = tools.load_label_map(LABEL_MAP_PATH, TAXONOMY)

    assert shipped.doc_version == glossary.doc_version(GUIDE), "docs/labeling-guide.md changed: update configs/label_map.json to match, then set its doc_version"


def test_every_mapped_and_excluded_category_is_in_the_guide_and_the_guide_says_where_it_goes():
    shipped = tools.load_label_map(LABEL_MAP_PATH, TAXONOMY)
    with open(GUIDE, encoding="utf-8") as f:
        section = f.read().split("## 2. The categories", 1)[1].split("## 3.", 1)[0]
    rows = [line for line in section.splitlines() if line.startswith("| `")]

    for proposed, target in shipped.map.items():
        (row,) = [line for line in rows if line.startswith(f"| `{proposed}`")]
        assert row.rstrip().endswith(f"| `{target}` |"), f"the guide's row of {proposed} must end with its judge path `{target}`"
    for name in shipped.exclude:
        assert any(line.startswith(f"| `{name}`") for line in rows)


# --- turning a label into what the judge sees --------------------------------------------------------------------------------


@pytest.mark.parametrize("label, expected", [("chat", "chat"), ("recruitment/guild", "recruitment/guild"),
                                              ("coordination/boss_call", "coordination"), ("recruitment/closed", "recruitment"), ("non_japanese", None)])
def test_the_judge_path_of_a_label(label, expected):
    assert tools.judge_path(label, label_map(), TAXONOMY) == expected


def test_an_unknown_label_is_an_error_naming_it():
    with pytest.raises(LabelError, match="nonsense"):
        tools.judge_path("nonsense", label_map(), TAXONOMY)


# --- the distinct lines ------------------------------------------------------------------------------------------------------


def test_distinct_lines_count_the_repeats_keep_the_first_order_and_the_commonest_channel():
    rows = [{"original": "A", "channel": "WORLD"}, {"original": "B", "channel": "PARTY"}, {"original": "A", "channel": "PARTY"},
            {"original": "A", "channel": "PARTY"}, {"original": " ", "channel": "WORLD"}, {"original": "b ", "channel": "PARTY"}]

    lines = tools.distinct_lines(rows)

    assert [(r["original"], r["channel"], r["count"]) for r in lines] == [("A", "PARTY", 3), ("B", "PARTY", 2)]  # near-copies (case, spaces) are one line


def test_a_row_without_a_text_is_skipped_not_counted():
    assert tools.distinct_lines([{"channel": "W"}, {"original": None}]) == []


# --- the batches for the labelling agents -----------------------------------------------------------------------------------


def test_the_input_rows_for_agents_carry_an_id_the_text_the_channel_and_the_count():
    lines = [{"original": "おはよう", "channel": "PARTY", "count": 3}, {"original": "草", "channel": "WORLD", "count": 1}]

    assert tools.input_rows(lines) == [{"i": 1, "ch": "P", "n": 3, "text": "おはよう"}, {"i": 2, "ch": "W", "n": 1, "text": "草"}]


# --- checking what the agents wrote -----------------------------------------------------------------------------------------


INPUTS = {1: {"i": 1, "ch": "P", "n": 3, "text": "おはよう"}, 2: {"i": 2, "ch": "W", "n": 1, "text": "草"}}


def problems(outputs, inputs=INPUTS):
    return tools.check_output(inputs, outputs, label_map(), TAXONOMY)


def test_a_complete_output_with_known_labels_has_no_problems():
    assert problems([{"i": 1, "cat": "social/greeting"}, {"i": 2, "cat": "chat/reaction", "unsure": True}]) == []


def test_a_proposed_and_an_excluded_label_are_known():
    assert problems([{"i": 1, "cat": "coordination/boss_call"}, {"i": 2, "cat": "non_japanese"}]) == []


def test_missing_duplicate_and_unknown_ids_are_reported():
    found = problems([{"i": 1, "cat": "chat"}, {"i": 1, "cat": "chat"}, {"i": 9, "cat": "chat"}])

    assert any("missing" in p and "2" in p for p in found)
    assert any("i=1" in p and "duplicated" in p for p in found)
    assert any("unknown" in p and "9" in p for p in found)


@pytest.mark.parametrize("bad, message", [({"i": 1, "cat": "nonsense"}, "nonsense"), ({"i": 1}, "cat"), ({"i": 1, "cat": 5}, "cat"),
                                          ({"i": 1, "cat": "chat", "unsure": "yes"}, "unsure")])
def test_a_bad_row_says_what_is_wrong(bad, message):
    found = problems([bad, {"i": 2, "cat": "chat"}])

    assert any("i=1" in p and message in p for p in found)


# --- the labels and the judge's sample -----------------------------------------------------------------------------------------


def labelled():
    return [{"original": "a", "category": "chat", "channel": "PARTY", "count": 4}, {"original": "b", "category": "coordination/boss_call", "channel": "WORLD", "count": 2},
            {"original": "c", "category": "non_japanese", "channel": "WORLD", "count": 1}, {"original": "d", "category": "chat/reaction", "channel": "PARTY", "count": 1, "unsure": True}]


def test_assembling_joins_the_labels_to_the_lines():
    lines = [{"original": "a", "channel": "PARTY", "count": 4}, {"original": "b", "channel": "WORLD", "count": 2}]

    rows = tools.assemble(lines, [{"i": 1, "cat": "chat"}, {"i": 2, "cat": "coordination/boss_call", "unsure": True}])

    assert rows == [{"original": "a", "category": "chat", "channel": "PARTY", "count": 4},
                    {"original": "b", "category": "coordination/boss_call", "channel": "WORLD", "count": 2, "unsure": True}]


def test_assembling_a_line_nobody_labelled_is_an_error():
    with pytest.raises(LabelError, match="not labelled"):
        tools.assemble([{"original": "a", "channel": "P", "count": 1}], [])


def test_the_judge_sample_maps_labels_drops_the_excluded_and_splits_dev_and_test_by_the_line_not_by_position():
    sample, excluded = tools.judge_sample(labelled(), label_map(), TAXONOMY, dev_fraction=0.7)

    assert [r["original"] for r in sample] == ["a", "b", "d"] and excluded == {"non_japanese": 1}
    assert sample[1]["category"] == "coordination" and sample[2]["unsure"] is True and "unsure" not in sample[0]
    assert all(r["split"] == ("dev" if bucket(r["original"]) < 0.7 else "test") for r in sample)
    assert tools.judge_sample(list(reversed(labelled())), label_map(), TAXONOMY)[0][0]["split"] == [r for r in sample if r["original"] == "d"][0]["split"]


def test_the_judge_sample_is_what_the_gate_reads(tmp_path):
    from gate_eval import read_sample

    sample, _ = tools.judge_sample(labelled(), label_map(), TAXONOMY)
    path = tmp_path / "sample.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in sample), encoding="utf-8")

    assert [r["category"] for r in read_sample(str(path))] == ["chat", "coordination", "chat/reaction"]  # read_sample refuses a label that is not a taxonomy path


def test_the_counts_are_sorted_and_the_table_is_markdown_with_the_unsure_total():
    counts = tools.label_counts(labelled())

    assert counts == [("chat", 1), ("chat/reaction", 1), ("coordination/boss_call", 1), ("non_japanese", 1)]
    table = tools.format_counts(counts, unsure=1)
    assert table.startswith("| Category | Lines |") and "| `chat` | 1 |" in table and "unsure: 1" in table


# --- the brief for labelling agents ------------------------------------------------------------------------------------------


GUIDE_TEXT = """# T

**Version:** 1.0.1 · **Season:** S1

## 1. The rules of labelling
One label per line.

## 2. The categories
| `chat` | x |

## 4. What season 1 produced
133 unsure

## Updating for a new season
steps

## Changelog
| 1.0.1 | x |
"""


def test_the_brief_is_the_guide_without_the_results_and_maintenance_sections_plus_the_output_format():
    brief = tools.build_brief(GUIDE_TEXT)

    assert "One label per line." in brief and "`chat`" in brief
    for gone in ("133 unsure", "Updating for a new season", "Changelog"):
        assert gone not in brief
    assert '"i"' in brief and '"cat"' in brief and '"unsure"' in brief and "version 1.0.1" in brief.lower()
