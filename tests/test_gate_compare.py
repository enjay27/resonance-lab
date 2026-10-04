import json

import pytest

import gate_compare
from dataset_recipe import line_key
from judge_client import ChoiceAnswer

# eight labelled lines; the truth is the first word of the category
ROWS = [{"original": f"line{i}", "category": category} for i, category in enumerate(
    ["social", "social", "game", "game", "chat", "chat", "recruitment", "recruitment"])]


def key(i):
    return line_key(f"line{i}")


def answers(*triples):
    """{line key: answer} from (line number, choice, margin); None = the judge gave no answer."""
    out = {}
    for number, choice, margin in triples:
        out[key(number)] = None if choice is None else {"choice": choice, "margin": margin, "probabilities": {choice: 0.5 + margin / 2}}
    return out


def record(label, triples, seconds=0.5, **extra):
    return gate_compare.probe_record(label, f"systemone:{label}:abcd1234", answers(*triples), seconds_per_line=seconds, use_channel=False, **extra)


# A is right on 0-5 (confident), wrong on 6 (taken for chat), no answer on 7.
A = record("a", [(0, "social", .9), (1, "social", .8), (2, "game", .7), (3, "game", .6), (4, "chat", .5), (5, "chat", .4),
                 (6, "chat", .2), (7, None, 0)])
# B is right on 0-3 and 6-7, wrong on 4, 5 (taken for social), less confident.
B = record("b", [(0, "social", .5), (1, "social", .5), (2, "game", .3), (3, "game", .2), (4, "social", .6), (5, "social", .1),
                 (6, "recruitment", .9), (7, "recruitment", .8)], seconds=1.5)


# --- collecting what a judge answered ----------------------------------------------------------------------------------


class FakeJudge:
    def __init__(self, answers):
        self.answers, self.asked = answers, []

    def answer(self, text, channel=None):
        self.asked.append((text, channel))
        return self.answers.get(text)


def test_collect_answers_keys_them_by_line_and_keeps_the_probabilities():
    judge = FakeJudge({"line0": ChoiceAnswer("social", {"social": 0.7, "chat": 0.2, "game": 0.1}, 0.5)})

    collected = gate_compare.collect_answers(ROWS[:2], judge, use_channel=False)

    assert collected[key(0)] == {"choice": "social", "margin": pytest.approx(0.5), "probabilities": {"social": 0.7, "chat": 0.2, "game": 0.1}}
    assert collected[key(1)] is None  # the judge failed on it: no answer, but the line is in the record


def test_the_channel_is_asked_with_only_when_the_probe_used_it():
    rows = [{"original": "line0", "category": "social", "channel": "PARTY"}]

    plain, with_channel = FakeJudge({}), FakeJudge({})
    gate_compare.collect_answers(rows, plain, use_channel=False)
    gate_compare.collect_answers(rows, with_channel, use_channel=True)

    assert plain.asked == [("line0", None)] and with_channel.asked == [("line0", "PARTY")]


# --- saving and loading ----------------------------------------------------------------------------------------------


def test_a_record_round_trips_as_utf8_json_without_chat_lines(tmp_path):
    path = gate_compare.save_probe(str(tmp_path), A)

    assert path.endswith("a.json")
    text = open(path, encoding="utf-8").read()
    assert "line0" not in text  # answers are keyed by the sha1 of a line, never the line
    (loaded,) = gate_compare.load_probes(str(tmp_path))
    assert loaded == json.loads(json.dumps(A))


def test_a_saved_probe_is_never_replaced_by_another_run_of_the_same_label(tmp_path):
    gate_compare.save_probe(str(tmp_path), A)

    with pytest.raises(FileExistsError, match="another label"):
        gate_compare.save_probe(str(tmp_path), A)


@pytest.mark.parametrize("label", ["", "a b", "../x", "9b/q8", "é"])
def test_a_label_is_a_plain_file_name(label):
    with pytest.raises(ValueError, match="label"):
        gate_compare.probe_record(label, "x", {}, seconds_per_line=0, use_channel=False)


def test_the_probes_load_sorted_by_label_and_a_missing_folder_is_no_probes(tmp_path):
    gate_compare.save_probe(str(tmp_path), B)
    gate_compare.save_probe(str(tmp_path), A)

    assert [r["label"] for r in gate_compare.load_probes(str(tmp_path))] == ["a", "b"]
    assert gate_compare.load_probes(str(tmp_path / "nowhere")) == []


def test_a_damaged_probe_file_is_an_error_naming_the_file(tmp_path):
    (tmp_path / "bad.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(gate_compare.CompareError, match="bad.json"):
        gate_compare.load_probes(str(tmp_path))


# --- the comparison ------------------------------------------------------------------------------------------------------


def test_the_scores_at_the_cutoff_count_no_answer_as_wrong():
    comparison = gate_compare.compare_probes([A, B], ROWS, cutoff=0.3)
    a, b = comparison["rows"]

    # A at 0.3: answers 0-5 (6 at margin .2 is below it, 7 has none): 6 right of 8
    assert (a["covered"], a["correct"], a["accuracy"], a["coverage"]) == (6, 6, 0.75, 0.75) and a["precision"] == 1.0
    # B at 0.3: 0,1,2,4(wrong),6,7 answered (3 at .2 and 5 at .1 are below): 5 right of 6 answered
    assert (b["covered"], b["correct"]) == (6, 5) and b["precision"] == pytest.approx(5 / 6)
    assert a["seconds_per_line"] == 0.5 and b["seconds_per_line"] == 1.5


def test_the_raw_accuracy_ignores_the_cutoff_so_it_is_the_models_not_the_settings():
    a, b = gate_compare.compare_probes([A, B], ROWS, cutoff=0.9)["rows"]

    assert a["raw_accuracy"] == 6 / 8 and b["raw_accuracy"] == 6 / 8  # A: 0-5 right; B: 0-3 and 6-7 right


def test_the_accuracy_comes_with_a_95_percent_interval_that_shows_how_little_eight_lines_say():
    a = gate_compare.compare_probes([A], ROWS, cutoff=0.3)["rows"][0]

    low, high = a["raw_interval"]
    assert low < 0.75 < high and high - low > 0.4


def test_wilson_interval_matches_the_textbook_value():
    low, high = gate_compare.wilson(50, 100)

    assert (round(low, 3), round(high, 3)) == (0.404, 0.596)
    assert gate_compare.wilson(0, 0) is None


def test_the_pairs_say_where_two_judges_differ_and_which_was_right():
    comparison = gate_compare.compare_probes([A, B], ROWS, cutoff=0.3)

    (pair,) = comparison["pairs"]
    assert (pair["a"], pair["b"]) == ("a", "b")
    # they give another answer on 4, 5 (A right), 6 and 7 (B right; A has no answer on 7: "no answer" is an answer that differs)
    assert pair["differ"] == 4 and pair["only_a"] == 2 and pair["only_b"] == 2


def test_coverage_at_the_target_precision_is_the_most_a_judge_answers_while_staying_that_precise():
    comparison = gate_compare.compare_probes([A, B], ROWS, cutoff=0.3, target_precision=0.9)
    a, b = comparison["rows"]

    assert a["coverage_at_target"] == 0.75  # cutoff 0.3: 6 answered, all right (0.2 would add the wrong line 6)
    assert b["coverage_at_target"] is not None and b["coverage_at_target"] <= 0.75


def test_a_target_nobody_reaches_is_none():
    bad = record("bad", [(i, "other", .9) for i in range(8)])

    assert gate_compare.compare_probes([bad], ROWS, cutoff=0.3)["rows"][0]["coverage_at_target"] is None


def test_a_probe_made_on_another_sample_is_flagged_with_how_many_lines_it_lacks():
    short = record("short", [(0, "social", .9)])

    row = gate_compare.compare_probes([short], ROWS, cutoff=0.3)["rows"][0]

    assert row["missing"] == 7
    assert "7 lines" in gate_compare.format_comparison(gate_compare.compare_probes([short], ROWS, cutoff=0.3))


def test_the_text_names_every_judge_the_cutoff_and_the_size_of_the_sample():
    text = gate_compare.format_comparison(gate_compare.compare_probes([A, B], ROWS, cutoff=0.3))

    for needle in ("systemone:a:abcd1234", "systemone:b:abcd1234", "0.3", "8 lines", "75%", "0.50 s", "1.50 s"):
        assert needle in text
    assert "only a right" in text.lower() or "a right, b wrong" in text.lower()


def test_comparing_nothing_or_an_empty_sample_is_an_error():
    with pytest.raises(gate_compare.CompareError, match="no probes"):
        gate_compare.compare_probes([], ROWS, cutoff=0.3)
    with pytest.raises(gate_compare.CompareError, match="sample"):
        gate_compare.compare_probes([A], [], cutoff=0.3)
