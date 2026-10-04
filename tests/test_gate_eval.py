import json
import os

import pytest

import config
from categorizer import categorize
from gate_eval import SampleError, evaluate_categorizer, format_gate_report, read_sample
from taxonomy import load_taxonomy, root_of

SAMPLE = os.path.join(os.path.dirname(__file__), "fixtures", "gate1_sample.jsonl")


def write(tmp_path, rows):
    path = tmp_path / "sample.jsonl"
    path.write_text("\n".join(row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    return str(path)


# --- the sample file -------------------------------------------------------------------------------------------------


def test_the_made_up_sample_covers_every_root_with_valid_labels():
    rows = read_sample(SAMPLE)
    taxonomy = load_taxonomy()
    per_root = {}
    for row in rows:
        assert row["category"] in taxonomy.paths, row
        per_root[root_of(row["category"])] = per_root.get(root_of(row["category"]), 0) + 1

    assert set(per_root) == set(taxonomy.roots)
    assert min(per_root.values()) >= 5
    assert len({row["original"] for row in rows}) == len(rows)  # no line twice


def test_the_sample_is_made_up_short_lines_only():
    rows = read_sample(SAMPLE)

    assert all(len(row["original"]) < 250 for row in rows)
    assert all(set(row) == {"original", "category"} for row in rows)  # no translations, no player ids, no timestamps


def test_read_sample_checks_every_row(tmp_path):
    ok = {"original": "草", "category": "chat/reaction"}
    for bad, message in [("{oops", "line 2"), ({"original": "x"}, "line 2"), ({"original": "x", "category": "Chat"}, "line 2"),
                         ({"original": "x", "category": "chat/nope"}, "line 2"), ({"original": "  ", "category": "chat"}, "line 2")]:
        with pytest.raises(SampleError, match=message):
            read_sample(write(tmp_path, [ok, bad]))


def test_read_sample_names_a_missing_file(tmp_path):
    with pytest.raises(SampleError, match="nope.jsonl"):
        read_sample(str(tmp_path / "nope.jsonl"))


# --- scoring a categorizer ---------------------------------------------------------------------------------------------

ROWS = [
    {"original": "a", "category": "chat/reaction"},
    {"original": "b", "category": "chat/casual"},
    {"original": "c", "category": "game/combat"},
    {"original": "d", "category": "game/market"},
    {"original": "e", "category": "social/greeting"},
    {"original": "f", "category": "other"},
]
ANSWERS = {"a": "chat", "b": None, "c": "game", "d": "question", "e": "chat", "f": "other"}


def test_a_categorizer_is_scored_at_the_root_and_may_say_nothing():
    report = evaluate_categorizer(ROWS, ANSWERS.get)

    assert report["n"] == 6 and report["covered"] == 5 and report["abstained"] == 1
    assert report["correct"] == 3 and report["wrong"] == 2
    assert report["accuracy"] == pytest.approx(3 / 6) and report["precision"] == pytest.approx(3 / 5) and report["coverage"] == pytest.approx(5 / 6)


def test_per_root_precision_and_recall_and_the_confusions():
    report = evaluate_categorizer(ROWS, ANSWERS.get)

    chat = report["per_root"]["chat"]
    assert (chat["support"], chat["predicted"], chat["correct"]) == (2, 2, 1)  # answered for "a" (right) and "e" (a greeting)
    assert chat["precision"] == pytest.approx(1 / 2) and chat["recall"] == pytest.approx(1 / 2)
    assert report["per_root"]["social"]["recall"] == 0 and report["per_root"]["social"]["precision"] is None  # never predicted
    assert report["confusion"]["game"] == {"game": 1, "question": 1}
    assert report["confusion"]["chat"] == {"chat": 1, "-": 1}  # "-" is "no answer"


def test_an_empty_sample_is_an_error():
    with pytest.raises(SampleError, match="empty"):
        evaluate_categorizer([], ANSWERS.get)


def test_the_report_text_says_accuracy_coverage_and_each_root():
    text = format_gate_report(evaluate_categorizer(ROWS, ANSWERS.get), "regex-v1")

    for expected in ("regex-v1", "6 lines", "accuracy", "coverage", "chat", "game", "social", "no answer"):
        assert expected in text, expected


# --- the baseline on the made-up sample: a regression guard, not a measurement ---------------------------------------


def test_the_baseline_stays_precise_on_the_made_up_sample():
    report = evaluate_categorizer(read_sample(SAMPLE), categorize)

    # Made-up lines are cleaner than real chat: this guards the rules against breaking, it says nothing about real accuracy.
    assert report["precision"] >= 0.9
    assert report["coverage"] >= 0.7
    assert report["per_root"]["spam"]["recall"] > 0  # the wall rule fires
    assert config.GATE1_SAMPLE.endswith(os.path.join("data", "eval", "gate1-sample.jsonl"))
