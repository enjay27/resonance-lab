import json
from fractions import Fraction

import pytest

import gate_pipeline
from dataset_recipe import Recipe, line_key
from gate_eval import SampleError, read_sample


def lines_of(**by_text):
    """distinct lines as categorize.distinct_lines yields them: (key, original, channel)."""
    return [(line_key(text), text, channel) for text, channel in by_text.items()]


def numbered(prefix, n, channel=None):
    return [(line_key(f"{prefix}{i}"), f"{prefix}{i}", channel) for i in range(n)]


def guess_by_prefix(original, channel):
    return {"s": "social", "r": "recruitment", "q": "question"}.get(original[0])  # anything else: no guess


# --- the draft of the labelled sample ----------------------------------------------------------------------------------


def test_the_draft_takes_the_same_number_from_each_guessed_category():
    lines = numbered("s", 30) + numbered("r", 30) + numbered("q", 30)

    rows = gate_pipeline.draft_sample(lines, 12, seed=1, guess=guess_by_prefix)

    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts == {"social": 4, "recruitment": 4, "question": 4}


def test_a_category_that_runs_out_gives_the_rest_to_the_others():
    lines = numbered("s", 2) + numbered("r", 30) + numbered("q", 30)

    rows = gate_pipeline.draft_sample(lines, 12, seed=1, guess=guess_by_prefix)

    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts["social"] == 2 and len(rows) == 12 and counts["recruitment"] == counts["question"] == 5


def test_the_draft_is_no_longer_than_the_lines_there_are():
    assert len(gate_pipeline.draft_sample(numbered("s", 3), 10, seed=1, guess=guess_by_prefix)) == 3


def test_the_same_seed_gives_the_same_draft_and_another_seed_another():
    lines = numbered("s", 50) + numbered("r", 50)

    first = gate_pipeline.draft_sample(lines, 20, seed=1, guess=guess_by_prefix)

    assert first == gate_pipeline.draft_sample(list(reversed(lines)), 20, seed=1, guess=guess_by_prefix)
    assert first != gate_pipeline.draft_sample(lines, 20, seed=2, guess=guess_by_prefix)


def test_lines_the_judge_has_no_guess_for_are_a_stratum_of_their_own_marked_so():
    rows = gate_pipeline.draft_sample(numbered("s", 5) + numbered("x", 5), 6, seed=1, guess=guess_by_prefix)

    assert {row["category"] for row in rows} == {"social", gate_pipeline.UNGUESSED}


def test_an_unreviewed_guess_cannot_pass_as_a_label(tmp_path):
    """The marker is not a category: read_sample refuses the file until the human has put a real label on the line."""
    rows = gate_pipeline.draft_sample(numbered("x", 2), 2, seed=1, guess=guess_by_prefix)
    path = tmp_path / "gate1-sample.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")

    with pytest.raises(SampleError, match="not a category"):
        read_sample(str(path))


def test_the_channel_of_a_line_is_kept_in_the_row():
    rows = gate_pipeline.draft_sample(lines_of(**{"s1": "PARTY", "s2": None}), 2, seed=1, guess=guess_by_prefix)

    assert {row["original"]: row.get("channel") for row in rows} == {"s1": "PARTY", "s2": None}
    assert all("channel" not in row for row in rows if row["original"] == "s2")


def test_lines_already_in_the_sample_are_not_drawn_again():
    lines = numbered("s", 10)
    done = {key for key, _, _ in lines[:6]}

    rows = gate_pipeline.draft_sample(lines, 10, seed=1, guess=guess_by_prefix, exclude=done)

    assert len(rows) == 4 and not {line_key(row["original"]) for row in rows} & done


def test_the_guess_gets_the_channel_too():
    seen = []
    gate_pipeline.draft_sample(lines_of(**{"s1": "PARTY"}), 1, seed=1, guess=lambda text, channel: seen.append((text, channel)))

    assert seen == [("s1", "PARTY")]


def test_the_draft_file_is_written_utf8_and_never_over_an_edited_one(tmp_path):
    path = tmp_path / "gate1-sample.draft.jsonl"
    rows = [{"original": "おはよう", "category": "social"}]

    gate_pipeline.write_draft(str(path), rows)
    assert json.loads(path.read_text(encoding="utf-8")) == rows[0] and "おはよう" in path.read_text(encoding="utf-8")

    with pytest.raises(FileExistsError, match="overwrite"):
        gate_pipeline.write_draft(str(path), [])
    gate_pipeline.write_draft(str(path), [], overwrite=True)
    assert path.read_text(encoding="utf-8") == ""


def test_the_draft_path_sits_next_to_the_sample():
    assert gate_pipeline.draft_path("data/eval/gate1-sample.jsonl") == "data/eval/gate1-sample.draft.jsonl"


# --- the channel mix ---------------------------------------------------------------------------------------------------


def test_the_channel_report_counts_lines_per_channel():
    report = gate_pipeline.channel_report(lines_of(a="PARTY", b="PARTY", c="WORLD", d=None))

    assert report == {"total": 4, "with_channel": 3, "channels": {"PARTY": 2, "WORLD": 1}}
    text = gate_pipeline.format_channel_report(report)
    assert "PARTY" in text and "2" in text and "3 of 4" in text


def test_without_any_channel_the_report_says_what_that_means_for_the_judge():
    text = gate_pipeline.format_channel_report(gate_pipeline.channel_report(lines_of(a=None, b=None)))

    assert "no line has a channel" in text and "--use-channel" in text


# --- the recipe against the categories ---------------------------------------------------------------------------------


def recipe(**weights):
    return Recipe(seed=1, total=None, keep=(), weights={name: Fraction(weight) for name, weight in weights.items()})


def test_the_preview_says_what_each_category_would_contribute():
    keys = [line_key(f"s{i}") for i in range(20)] + [line_key(f"r{i}") for i in range(4)] + [line_key("unknown")]
    categories = {**{line_key(f"s{i}"): "social" for i in range(20)}, **{line_key(f"r{i}"): "recruitment" for i in range(4)}}

    preview = gate_pipeline.recipe_preview(recipe(social=1, recruitment=1), categories, keys)

    rows = {row["key"]: row for row in preview["rows"]}
    assert rows["social"]["available"] == 20 and rows["recruitment"]["available"] == 4
    assert rows["recruitment"]["target"] == 4 and rows["social"]["target"] == 4  # equal weights: the short category ends it
    assert preview["limited_by"] == "recruitment" and preview["total"] == 8
    assert preview["uncategorized"] == 1 and preview["distinct"] == 25
    assert rows["social"]["natural"] == pytest.approx(20 / 24) and rows["social"]["share"] == pytest.approx(0.5)


def test_children_count_for_their_root_and_a_category_the_recipe_does_not_name_is_outside():
    keys = [line_key("a"), line_key("b"), line_key("c")]
    categories = {line_key("a"): "game/combat", line_key("b"): "game", line_key("c"): "spam"}

    preview = gate_pipeline.recipe_preview(recipe(game=1), categories, keys)

    assert preview["rows"][0]["available"] == 2 and preview["outside"] == 1


def test_the_preview_is_text_with_a_marker_for_the_limiting_category():
    keys = [line_key("a")]
    preview = gate_pipeline.recipe_preview(recipe(social=1, chat=1), {line_key("a"): "social"}, keys)

    text = gate_pipeline.format_recipe_preview(preview, "balanced-v1")

    assert "balanced-v1" in text and "social" in text and "limits" in text


# --- the decision text, the progress -----------------------------------------------------------------------------------


def report(accuracy, coverage=1.0, precision=None, n=200):
    covered = round(n * coverage)
    return {"n": n, "covered": covered, "abstained": n - covered, "correct": round(n * accuracy), "wrong": 0, "accuracy": accuracy,
            "coverage": coverage, "precision": precision if precision is not None else accuracy, "per_root": {}, "confusion": {}}


def test_the_decision_text_names_the_judge_the_cutoff_and_the_scores_against_the_rules():
    text = gate_pipeline.decision_text(by="kev-local:kev-9b@v1.0+bf16:ab12cd34", cutoff=0.3, judge=report(0.80, 0.9, 0.88),
                                       rules=report(0.62, 0.7, 0.85), seconds_per_line=0.2, lines=50000, categorized=41000)

    assert "kev-local:kev-9b@v1.0+bf16:ab12cd34" in text and "0.3" in text
    assert "80%" in text and "62%" in text and "200 lines" in text
    assert "beats the rules" in text
    assert "41,000" in text and "50,000" in text and "0.2 s" in text


def test_the_decision_text_says_when_the_judge_does_not_beat_the_rules():
    text = gate_pipeline.decision_text(by="x", cutoff=0.3, judge=report(0.55), rules=report(0.62))

    assert "does NOT beat the rules" in text


def test_the_decision_text_leaves_out_what_it_was_not_given():
    text = gate_pipeline.decision_text(by="x", cutoff=0.3, judge=report(0.9), rules=report(0.6))

    assert "full pass" not in text and "None" not in text


def test_the_progress_text_has_a_speed_and_an_estimate():
    text = gate_pipeline.eta_text(done=12000, total=48000, seconds=600)

    assert "12,000 / 48,000" in text and "25%" in text and "20.0 lines/s" in text and "30 min" in text


def test_the_progress_text_before_any_line_has_no_speed():
    assert "no speed yet" in gate_pipeline.eta_text(done=0, total=10, seconds=1)


def test_the_estimate_reads_in_hours_when_it_is_long():
    assert "2 h 5 min" in gate_pipeline.eta_text(done=1, total=1001, seconds=7.5)
