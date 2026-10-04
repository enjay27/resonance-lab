import json

import pytest

import gate_judge
from dataset_recipe import line_key, read_categories
from gate_judge import GateJudge, JournalError
from judge_client import SystemOneClient
from judge_stub import StubJudge
from taxonomy import TaxonomyError, load_taxonomy

TAXONOMY = load_taxonomy()


def judge_for(stub, cutoff=0.3):
    options = gate_judge.choice_options(TAXONOMY)
    return GateJudge(SystemOneClient("http://stub", post=stub), options, cutoff)


# --- the question ------------------------------------------------------------------------------------------------------


def test_the_options_of_gate_1_are_the_roots_each_with_its_description():
    options = gate_judge.choice_options(TAXONOMY)

    assert list(options) == list(TAXONOMY.roots) and "other" in options
    assert all(options[root] == TAXONOMY.paths[root] and options[root] for root in options)


def test_the_options_of_a_root_are_its_direct_children_by_path():
    options = gate_judge.choice_options(TAXONOMY, "game")

    assert set(options) == {"game/combat", "game/progress", "game/system", "game/market"}
    assert options["game/market"] == TAXONOMY.paths["game/market"]


def test_a_root_without_children_has_no_options_to_choose_between():
    with pytest.raises(TaxonomyError, match="coordination"):
        gate_judge.choice_options(TAXONOMY, "coordination")
    with pytest.raises(TaxonomyError, match="nope"):
        gate_judge.choice_options(TAXONOMY, "nope")


def test_the_judge_id_names_the_model_and_changes_with_the_question():
    options = gate_judge.choice_options(TAXONOMY)

    first = gate_judge.judge_id("models/Julia-1-Q8_0.gguf", gate_judge.INSTRUCTIONS, options)

    assert first.startswith("systemone:Julia-1-Q8_0:") and len(first.split(":")[2]) == 8
    assert first == gate_judge.judge_id("models/Julia-1-Q8_0.gguf", gate_judge.INSTRUCTIONS, dict(options))
    assert first != gate_judge.judge_id("Kev-4B-Q4_K_M.gguf", gate_judge.INSTRUCTIONS, options)
    assert first != gate_judge.judge_id("models/Julia-1-Q8_0.gguf", "Another question?", options)
    assert first != gate_judge.judge_id("models/Julia-1-Q8_0.gguf", gate_judge.INSTRUCTIONS, {**options, "social": "changed"})


def test_a_line_is_the_state_alone_or_with_its_channel():
    assert gate_judge.state_for("2人募集") == "2人募集"
    assert gate_judge.state_for("2人募集", "PARTY") == {"channel": "PARTY", "message": "2人募集"}
    assert gate_judge.state_for("2人募集", "") == "2人募集"
    assert gate_judge.state_for("2人募集", None) == "2人募集"


# --- the judge: the answer, the cutoff, the failures -------------------------------------------------------------------


def test_a_clear_answer_is_the_root():
    stub = StubJudge({"2人募集": {"recruitment": 0.9, "social": 0.1}})

    judge = judge_for(stub)

    assert judge.predict("2人募集") == "recruitment"
    answer = judge.answer("2人募集")
    assert answer.choice == "recruitment" and answer.margin == pytest.approx(0.8)


def test_a_margin_below_the_cutoff_is_no_answer_and_a_tie_always_is():
    stub = StubJudge({"草": {"social": 0.5, "chat": 0.3, "other": 0.2}, "うーん": {"social": 0.5, "chat": 0.5}})

    assert judge_for(stub, cutoff=0.3).predict("草") is None  # margin 0.2
    assert judge_for(stub, cutoff=0.1).predict("草") == "social"
    assert judge_for(stub, cutoff=0.0).predict("うーん") in ("social", "chat")  # a tie has margin 0: cutoff 0 still answers
    assert judge_for(stub, cutoff=0.01).predict("うーん") is None


def test_a_failing_request_is_no_answer_and_is_counted_not_raised():
    judge = judge_for(StubJudge(fail={"x"}))

    assert judge.predict("x") is None and judge.answer("x") is None
    assert judge.failures == 2


def test_an_answer_is_asked_once_per_line_and_channel():
    stub = StubJudge()
    judge = judge_for(stub)

    judge.answer("a"), judge.answer("a"), judge.answer("a", "PARTY")

    assert stub.states() == ["a", {"channel": "PARTY", "message": "a"}]


def test_the_cutoff_sweep_scores_every_cutoff_from_one_pass():
    rows = [{"original": "a", "category": "social"}, {"original": "b", "category": "recruitment"},
            {"original": "c", "category": "social"}]
    stub = StubJudge({"a": {"social": 0.9, "other": 0.1}, "b": {"social": 0.6, "recruitment": 0.4}, "c": {"social": 0.55, "other": 0.45}})
    judge = judge_for(stub)

    sweep = gate_judge.cutoff_sweep(rows, judge, [0.0, 0.15, 0.5])

    assert [(s["cutoff"], s["covered"], s["correct"]) for s in sweep] == [(0.0, 3, 2), (0.15, 2, 1), (0.5, 1, 1)]
    assert sweep[2]["precision"] == 1.0 and sweep[2]["coverage"] == pytest.approx(1 / 3)
    assert len(stub.states()) == 3  # one request per line, not per cutoff


# --- the journal: one row per judged line, resumable ---------------------------------------------------------------------


def row(key, by="systemone:m:abc12345", choice="social", margin=0.8):
    return {"key": key, "by": by, "choice": choice, "margin": margin, "p": 0.9, "probabilities": {choice: 0.9}}


def test_a_missing_journal_is_empty(tmp_path):
    assert gate_judge.read_journal(str(tmp_path / "none.jsonl"), "systemone:m:abc12345") == {}


def test_the_journal_round_trips_and_keeps_the_first_row_of_a_key(tmp_path):
    path = tmp_path / "pass.jsonl"
    path.write_text(json.dumps(row("a" * 40)) + "\n" + json.dumps(row("b" * 40, choice="chat")) + "\n", encoding="utf-8")

    rows = gate_judge.read_journal(str(path), "systemone:m:abc12345")

    assert {k: r["choice"] for k, r in rows.items()} == {"a" * 40: "social", "b" * 40: "chat"}


def test_a_torn_last_line_from_a_crash_is_dropped_but_a_damaged_middle_line_is_an_error(tmp_path):
    path = tmp_path / "pass.jsonl"
    path.write_text(json.dumps(row("a" * 40)) + "\n" + '{"key": "bbb', encoding="utf-8")
    assert list(gate_judge.read_journal(str(path), "systemone:m:abc12345")) == ["a" * 40]

    path.write_text('{"key": "bbb\n' + json.dumps(row("a" * 40)) + "\n", encoding="utf-8")
    with pytest.raises(JournalError, match="line 1"):
        gate_judge.read_journal(str(path), "systemone:m:abc12345")


def test_rows_by_another_judge_are_refused_so_two_judges_never_mix(tmp_path):
    path = tmp_path / "pass.jsonl"
    path.write_text(json.dumps(row("a" * 40, by="systemone:other:99999999")) + "\n", encoding="utf-8")

    with pytest.raises(JournalError, match="systemone:other:99999999.*--force"):
        gate_judge.read_journal(str(path), "systemone:m:abc12345")


def test_the_categories_are_the_journal_rows_with_a_margin_at_the_cutoff(tmp_path):
    rows = {"a" * 40: row("a" * 40, choice="recruitment", margin=0.9), "b" * 40: row("b" * 40, margin=0.29),
            "c" * 40: row("c" * 40, choice="other", margin=0.3)}

    categories, uncertain = gate_judge.categories_from(rows, 0.3)

    assert [(c["key"], c["category"]) for c in categories] == [("a" * 40, "recruitment"), ("c" * 40, "other")]
    assert categories[0] == {"key": "a" * 40, "category": "recruitment", "by": "systemone:m:abc12345", "p": 0.9, "margin": 0.9}
    assert uncertain == 1


def test_the_derived_file_is_one_a_recipe_reads(tmp_path):
    out = tmp_path / "categories.jsonl"
    categories, _ = gate_judge.categories_from({line_key("おはよう"): row(line_key("おはよう"))}, 0.3)

    gate_judge.write_categories(str(out), categories)

    assert read_categories(str(out)) == {line_key("おはよう"): "social"}


def test_a_torn_tail_is_cut_before_appending_so_the_next_row_does_not_glue_to_it(tmp_path):
    path = tmp_path / "pass.jsonl"
    path.write_text(json.dumps(row("a" * 40)) + "\n" + '{"key": "bbb', encoding="utf-8")

    gate_judge.trim_torn_tail(str(path))

    assert path.read_text(encoding="utf-8") == json.dumps(row("a" * 40)) + "\n"


def test_a_complete_last_row_without_its_newline_is_kept_and_terminated(tmp_path):
    path = tmp_path / "pass.jsonl"
    path.write_text(json.dumps(row("a" * 40)), encoding="utf-8")

    gate_judge.trim_torn_tail(str(path))

    assert path.read_text(encoding="utf-8") == json.dumps(row("a" * 40)) + "\n"
    gate_judge.trim_torn_tail(str(tmp_path / "missing.jsonl"))  # nothing to do, no error
