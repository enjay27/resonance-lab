import json
import os

import pytest

import categorize
import gate_judge
from conftest import read_jsonl
from dataset_recipe import line_key, read_categories
from judge_stub import StubJudge
from taxonomy import load_taxonomy

SAMPLE = os.path.join(os.path.dirname(__file__), "fixtures", "gate1_sample.jsonl")
BY = gate_judge.judge_id("models/Stub-Q8_0.gguf", gate_judge.INSTRUCTIONS, gate_judge.choice_options(load_taxonomy()))

ANSWERS = {
    "おはよう": {"social": 0.9, "chat": 0.1},
    "2人募集": {"recruitment": 0.8, "social": 0.2},
    "草": {"chat": 0.5, "social": 0.3, "other": 0.2},  # margin 0.2: below the default cutoff in these tests
}


def run(tmp_path, raw, stub, *extra):
    out, journal = tmp_path / "categories.jsonl", tmp_path / "judge.jsonl"
    categorize.main(["--raw", raw, "--out", str(out), "--journal", str(journal), "--judge-url", "http://stub", "--cutoff", "0.3", *extra],
                    post=stub)
    return out, journal


def raw_of(write_jsonl, *originals, **channels):
    return write_jsonl("raw.jsonl", [{"original": text, "translated": "x", **({"channel": channels[text]} if text in channels else {})}
                                     for text in originals])


def test_the_judge_pass_writes_a_journal_row_per_line_and_the_categories_above_the_cutoff(write_jsonl, tmp_path, capsys):
    raw = raw_of(write_jsonl, "おはよう", "2人募集", "草", "おはよう")
    stub = StubJudge(ANSWERS)

    out, journal = run(tmp_path, raw, stub)

    assert stub.states() == ["おはよう", "2人募集", "草"]  # a repeated line is asked once
    rows = {row["key"]: row for row in read_jsonl(journal)}
    assert set(rows) == {line_key("おはよう"), line_key("2人募集"), line_key("草")}
    assert rows[line_key("2人募集")] == {"key": line_key("2人募集"), "by": BY, "choice": "recruitment", "margin": 0.6, "p": 0.8,
                                         "probabilities": {**dict.fromkeys(load_taxonomy().roots, 0.0), "recruitment": 0.8, "social": 0.2}}
    assert [(r["key"], r["category"], r["by"]) for r in read_jsonl(out)] == [
        (line_key("おはよう"), "social", BY), (line_key("2人募集"), "recruitment", BY)]  # 草 (margin 0.2) is left out
    assert read_categories(str(out))  # the file a recipe reads
    text = capsys.readouterr().out
    assert BY in text and "uncategorized" in text


def test_the_channel_goes_to_the_judge_only_when_asked_for(write_jsonl, tmp_path):
    raw = raw_of(write_jsonl, "おはよう", "2人募集", **{"2人募集": "PARTY"})
    plain, with_channel = StubJudge(ANSWERS), StubJudge(ANSWERS)

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    run(tmp_path / "a", raw, plain)
    run(tmp_path / "b", raw, with_channel, "--use-channel")

    assert plain.states() == ["おはよう", "2人募集"]
    assert with_channel.states() == ["おはよう", {"channel": "PARTY", "message": "2人募集"}]  # a line without a channel stays a string


def test_a_second_run_asks_only_about_the_new_lines(write_jsonl, tmp_path):
    out, journal = run(tmp_path, raw_of(write_jsonl, "おはよう", "2人募集"), StubJudge(ANSWERS))
    second = StubJudge(ANSWERS)

    run(tmp_path, write_jsonl("raw2.jsonl", [{"original": t, "translated": "x"} for t in ("おはよう", "2人募集", "草")]), second)

    assert second.states() == ["草"]
    assert len(read_jsonl(journal)) == 3


def test_a_new_cutoff_needs_no_new_request(write_jsonl, tmp_path):
    raw = raw_of(write_jsonl, "おはよう", "草")
    out, _ = run(tmp_path, raw, StubJudge(ANSWERS))
    assert [r["category"] for r in read_jsonl(out)] == ["social"]
    again = StubJudge(ANSWERS)

    categorize.main(["--raw", raw, "--out", str(out), "--journal", str(tmp_path / "judge.jsonl"), "--judge-url", "http://stub",
                     "--cutoff", "0.1"], post=again)

    assert again.states() == []
    assert [r["category"] for r in read_jsonl(out)] == ["social", "chat"]


def test_categories_only_cover_the_lines_of_this_raw_log(write_jsonl, tmp_path):
    run(tmp_path, raw_of(write_jsonl, "おはよう", "2人募集"), StubJudge(ANSWERS))

    out, _ = run(tmp_path, write_jsonl("raw2.jsonl", [{"original": "2人募集", "translated": "x"}]), StubJudge(ANSWERS))

    assert [r["key"] for r in read_jsonl(out)] == [line_key("2人募集")]


def test_a_journal_of_another_judge_stops_the_run_until_force(write_jsonl, tmp_path, capsys):
    raw = raw_of(write_jsonl, "おはよう")
    _, journal = run(tmp_path, raw, StubJudge(ANSWERS, model="models/Old-Q8_0.gguf"))
    before = journal.read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, raw, StubJudge(ANSWERS))
    assert stopped.value.code == 1 and "--force" in capsys.readouterr().out
    assert journal.read_text(encoding="utf-8") == before

    run(tmp_path, raw, StubJudge(ANSWERS), "--force")
    assert {r["by"] for r in read_jsonl(journal)} == {BY}


def test_a_line_the_server_fails_on_is_left_unjudged_and_asked_again_next_time(write_jsonl, tmp_path, capsys):
    raw = raw_of(write_jsonl, "おはよう", "2人募集")

    _, journal = run(tmp_path, raw, StubJudge(ANSWERS, fail={"2人募集"}))
    assert [r["key"] for r in read_jsonl(journal)] == [line_key("おはよう")]
    assert "1 line" in capsys.readouterr().out

    retry = StubJudge(ANSWERS)
    run(tmp_path, raw, retry)
    assert retry.states() == ["2人募集"]


def test_a_server_that_stops_answering_ends_the_run_with_the_journal_intact(write_jsonl, tmp_path, capsys):
    lines = ["おはよう"] + [f"line {n}" for n in range(categorize.MAX_CONSECUTIVE_FAILURES + 5)]
    raw = raw_of(write_jsonl, *lines)
    stub = StubJudge(ANSWERS, fail=set(lines[1:]))

    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, raw, stub)

    assert stopped.value.code == 1 and "resume" in capsys.readouterr().out
    journal = tmp_path / "judge.jsonl"
    assert [r["key"] for r in read_jsonl(journal)] == [line_key("おはよう")]
    assert len(stub.states()) == 1 + categorize.MAX_CONSECUTIVE_FAILURES  # it stopped asking


def test_an_unreachable_server_falls_back_to_the_rules_and_says_so(write_jsonl, tmp_path, capsys):
    raw = raw_of(write_jsonl, "杖@2募集", "おはようございます")

    out, journal = run(tmp_path, raw, StubJudge(down=True))

    captured = capsys.readouterr()
    assert "judge" in captured.err and "regex-v1" in captured.err
    assert {r["by"] for r in read_jsonl(out)} == {"regex-v1"}
    assert not journal.exists()


def test_the_fallback_never_replaces_the_categories_a_judge_made(write_jsonl, tmp_path, capsys):
    raw = raw_of(write_jsonl, "おはよう")
    out, _ = run(tmp_path, raw, StubJudge(ANSWERS))
    before = out.read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, raw, StubJudge(down=True))

    assert stopped.value.code == 1 and "systemone" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == before


def test_probe_with_a_judge_prints_the_rules_the_judge_and_the_cutoff_sweep(tmp_path, capsys):
    out = tmp_path / "categories.jsonl"

    categorize.main(["--probe", SAMPLE, "--out", str(out), "--judge-url", "http://stub", "--cutoff", "0.3"], post=StubJudge())

    text = capsys.readouterr().out
    assert "regex-v1" in text and BY in text and "98 lines" in text
    assert "cutoff" in text and "0.9" in text  # the sweep
    assert not out.exists()


def test_probe_stops_when_the_judge_cannot_be_reached(capsys):
    with pytest.raises(SystemExit) as stopped:
        categorize.main(["--probe", SAMPLE, "--judge-url", "http://stub"], post=StubJudge(down=True))

    assert stopped.value.code == 1 and "[ERROR]" in capsys.readouterr().out


def test_probe_gives_the_judge_the_channel_of_a_sample_line_when_asked(tmp_path):
    sample = tmp_path / "sample.jsonl"
    sample.write_text(json.dumps({"original": "2人募集", "category": "recruitment", "channel": "PARTY"}, ensure_ascii=False) + "\n", encoding="utf-8")
    plain, with_channel = StubJudge(), StubJudge()

    categorize.main(["--probe", str(sample), "--judge-url", "http://stub"], post=plain)
    categorize.main(["--probe", str(sample), "--judge-url", "http://stub", "--use-channel"], post=with_channel)

    assert plain.states() == ["2人募集"]
    assert with_channel.states() == [{"channel": "PARTY", "message": "2人募集"}]


def test_judge_pass_is_the_cli_pass_as_a_function_and_raises_instead_of_exiting(write_jsonl, tmp_path, capsys):
    """What the notebook calls on the judge it already loaded."""
    raw = raw_of(write_jsonl, "おはよう", "2人募集")
    judge = gate_judge.GateJudge(categorize.SystemOneClient("http://stub", post=StubJudge(ANSWERS)), gate_judge.choice_options(load_taxonomy()), 0.3)
    out, journal = tmp_path / "categories.jsonl", tmp_path / "judge.jsonl"

    categorize.judge_pass(raw, str(journal), str(out), judge, BY, 0.3, False)

    assert [r["category"] for r in read_jsonl(out)] == ["social", "recruitment"]
    assert BY in capsys.readouterr().out
    with pytest.raises(categorize.PassError, match="--force"):
        categorize.judge_pass(raw, str(journal), str(out), judge, "systemone:other:00000000", 0.3, False)
