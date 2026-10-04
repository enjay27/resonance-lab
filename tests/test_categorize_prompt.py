"""`categorize.py --judge-prompt NAME`: the judge is asked another question (judge_prompts.py); everything else is the shared pass."""

import json

import pytest

import categorize
import gate_compare
import gate_judge
import judge_prompts
from conftest import read_jsonl
from judge_client import QUESTION_ID
from judge_stub import StubJudge
from taxonomy import load_taxonomy

ANSWERS = {"おはよう": {"social": 0.9, "chat": 0.1}, "2人募集": {"recruitment": 0.8, "social": 0.2}}


def by_of(name):
    instructions, options = judge_prompts.judge_question(load_taxonomy(), judge_prompts.load_variant(name))
    return gate_judge.judge_id("models/Stub-Q8_0.gguf", instructions, options)


def raw_of(write_jsonl, *originals):
    return write_jsonl("raw.jsonl", [{"original": text, "translated": "x"} for text in originals])


def run(tmp_path, raw, stub, *extra):
    categorize.main(["--raw", raw, "--out", str(tmp_path / "categories.jsonl"), "--journal", str(tmp_path / "judge.jsonl"),
                     "--judge-url", "http://stub", "--cutoff", "0.3", *extra], post=stub)


def asked(stub):
    """The question of the first choice request: (instructions, option names)."""
    question = next(c["questions"][QUESTION_ID] for c in stub.calls if c["questions"][QUESTION_ID]["type"] == "choice")
    return question["instructions"], list(question["criteria"]), question["criteria"]


def test_without_the_flag_the_judge_is_asked_todays_question_and_keeps_todays_id(write_jsonl, tmp_path):
    stub = StubJudge(ANSWERS)

    run(tmp_path, raw_of(write_jsonl, "おはよう"), stub)

    instructions, names, _ = asked(stub)
    assert instructions == gate_judge.INSTRUCTIONS and names == list(load_taxonomy().roots)
    assert {r["by"] for r in read_jsonl(tmp_path / "judge.jsonl")} == {by_of("default")}


def test_a_variant_changes_what_the_judge_is_asked_and_its_id(write_jsonl, tmp_path):
    stub = StubJudge(ANSWERS)

    run(tmp_path, raw_of(write_jsonl, "おはよう"), stub, "--judge-prompt", "v2-no-other")

    _, names, criteria = asked(stub)
    assert "other" not in names and "@T1" in criteria["recruitment"]
    assert {r["by"] for r in read_jsonl(tmp_path / "judge.jsonl")} == {by_of("v2-no-other")} != {by_of("default")}


def test_the_journal_of_another_wording_stops_the_run_until_force(write_jsonl, tmp_path, capsys):
    raw = raw_of(write_jsonl, "おはよう")
    run(tmp_path, raw, StubJudge(ANSWERS))
    before = (tmp_path / "judge.jsonl").read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, raw, StubJudge(ANSWERS), "--judge-prompt", "v2")

    assert stopped.value.code == 1 and "--force" in capsys.readouterr().out
    assert (tmp_path / "judge.jsonl").read_text(encoding="utf-8") == before


def test_an_unknown_variant_names_the_ones_there_are_before_the_model_is_asked(write_jsonl, tmp_path, capsys):
    stub = StubJudge(ANSWERS)

    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, raw_of(write_jsonl, "おはよう"), stub, "--judge-prompt", "nope")

    assert stopped.value.code == 2
    assert stub.calls == []
    assert "v2-no-other" in capsys.readouterr().err


def test_the_flag_needs_a_judge(write_jsonl, tmp_path):
    with pytest.raises(SystemExit) as stopped:
        categorize.main(["--raw", raw_of(write_jsonl, "おはよう"), "--out", str(tmp_path / "c.jsonl"), "--judge-prompt", "v2"])

    assert stopped.value.code == 2


def test_a_saved_probe_says_which_wording_it_was_made_with(tmp_path):
    sample = tmp_path / "sample.jsonl"
    sample.write_text(json.dumps({"original": "おはよう", "category": "social"}, ensure_ascii=False), encoding="utf-8")

    categorize.main(["--probe", str(sample), "--judge-url", "http://stub", "--judge-prompt", "clean", "--probe-dir", str(tmp_path / "p"),
                     "--save-probe", "stub-clean"], post=StubJudge(ANSWERS))

    (record,) = gate_compare.load_probes(str(tmp_path / "p"))
    assert record["by"] == by_of("clean") and "prompt clean" in record["notes"]


def test_a_probe_without_the_flag_notes_the_default_wording(tmp_path):
    sample = tmp_path / "sample.jsonl"
    sample.write_text(json.dumps({"original": "おはよう", "category": "social"}, ensure_ascii=False), encoding="utf-8")

    categorize.main(["--probe", str(sample), "--judge-url", "http://stub", "--probe-dir", str(tmp_path / "p"), "--save-probe", "stub-default"],
                    post=StubJudge(ANSWERS))

    (record,) = gate_compare.load_probes(str(tmp_path / "p"))
    assert "prompt default" in record["notes"]


def test_the_local_judge_takes_the_same_flag(write_jsonl, tmp_path):
    stub = StubJudge(ANSWERS)

    class Engine:
        @staticmethod
        def answer(request):
            status, text = stub("kev-local://stub", request, 0)
            return json.loads(text)

    categorize.main(["--raw", raw_of(write_jsonl, "おはよう"), "--out", str(tmp_path / "c.jsonl"), "--journal", str(tmp_path / "j.jsonl"),
                     "--judge-local", "jaredpalmer/kev-0.8b@v1.0", "--judge-prompt", "clean-no-other"], loader=lambda run, device, dtype: Engine())

    assert "other" not in asked(stub)[1]
