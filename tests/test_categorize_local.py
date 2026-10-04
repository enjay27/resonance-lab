"""`categorize.py --judge-local`: the same pass as --judge-url (journal, cutoff, resume, probe, fallback), the model in this process."""

import json

import pytest

import categorize
import config
import gate_judge
from conftest import read_jsonl
from dataset_recipe import line_key
from judge_stub import StubJudge
from taxonomy import load_taxonomy

RUN = "jaredpalmer/kev-0.8b@v1.0"
OPTIONS = gate_judge.choice_options(load_taxonomy())
BY = gate_judge.judge_id("jaredpalmer/kev-0.8b@v1.0+bf16", gate_judge.INSTRUCTIONS, OPTIONS, kind="kev-local")
ANSWERS = {"おはよう": {"social": 0.9, "chat": 0.1}, "2人募集": {"recruitment": 0.8, "social": 0.2}}


class StubEngine:
    """`answer(request dict) -> body`: a StubJudge behind the engine interface judge_local expects."""

    def __init__(self, stub):
        self.stub = stub

    def answer(self, request):
        status, text = self.stub("kev-local://stub", request, 0)
        if status != 200:
            raise RuntimeError(text)
        return json.loads(text)


class Loader:
    def __init__(self, stub=None, raises=None):
        self.engine = StubEngine(stub) if stub else None
        self.raises, self.calls = raises, []

    def __call__(self, run, device, dtype):
        self.calls.append((run, device, dtype))
        if self.raises:
            raise self.raises
        return self.engine


def raw_of(write_jsonl, *originals):
    return write_jsonl("raw.jsonl", [{"original": text, "translated": "x"} for text in originals])


def run(tmp_path, raw, loader, *extra):
    out, journal = tmp_path / "categories.jsonl", tmp_path / "judge.jsonl"
    categorize.main(["--raw", raw, "--out", str(out), "--journal", str(journal), "--judge-local", RUN, "--cutoff", "0.3", *extra],
                    loader=loader)
    return out, journal


def test_the_judge_id_has_a_kind_so_a_local_run_never_looks_like_a_server_run():
    local = gate_judge.judge_id("jaredpalmer/kev-9b@v1.0+bf16", gate_judge.INSTRUCTIONS, OPTIONS, kind="kev-local")
    server = gate_judge.judge_id("jaredpalmer/kev-9b@v1.0+bf16", gate_judge.INSTRUCTIONS, OPTIONS)

    assert local.startswith("kev-local:kev-9b@v1.0+bf16:")
    assert server.startswith("systemone:")
    assert local.split(":")[-1] == server.split(":")[-1]  # the same question, hashed the same


def test_the_local_pass_writes_the_journal_and_the_categories_like_the_server_pass(write_jsonl, tmp_path, capsys):
    stub, loader = StubJudge(ANSWERS), None
    loader = Loader(stub)

    out, journal = run(tmp_path, raw_of(write_jsonl, "おはよう", "2人募集", "おはよう"), loader)

    assert stub.states() == ["おはよう", "2人募集"]
    assert {r["by"] for r in read_jsonl(journal)} == {BY}
    assert [(r["key"], r["category"]) for r in read_jsonl(out)] == [(line_key("おはよう"), "social"), (line_key("2人募集"), "recruitment")]
    assert loader.calls == [(RUN, None, "bf16")]
    assert BY in capsys.readouterr().out


def test_dtype_and_device_reach_the_loader(write_jsonl, tmp_path):
    loader = Loader(StubJudge(ANSWERS))

    run(tmp_path, raw_of(write_jsonl, "おはよう"), loader, "--dtype", "fp32", "--device", "cpu")

    assert loader.calls == [(RUN, "cpu", "fp32")]


def test_a_bare_judge_local_means_the_configs_run(write_jsonl, tmp_path):
    loader = Loader(StubJudge(ANSWERS))

    categorize.main(["--raw", raw_of(write_jsonl, "おはよう"), "--out", str(tmp_path / "c.jsonl"), "--journal", str(tmp_path / "j.jsonl"),
                     "--judge-local"], loader=loader)

    assert loader.calls == [(config.JUDGE_LOCAL_RUN, None, "bf16")]


def test_a_second_run_resumes_from_the_journal(write_jsonl, tmp_path):
    run(tmp_path, raw_of(write_jsonl, "おはよう"), Loader(StubJudge(ANSWERS)))
    second = StubJudge(ANSWERS)

    run(tmp_path, write_jsonl("raw2.jsonl", [{"original": t, "translated": "x"} for t in ("おはよう", "2人募集")]), Loader(second))

    assert second.states() == ["2人募集"]


def test_a_server_journal_stops_a_local_run_until_force(write_jsonl, tmp_path, capsys):
    raw = raw_of(write_jsonl, "おはよう")
    journal = tmp_path / "judge.jsonl"
    categorize.main(["--raw", raw, "--out", str(tmp_path / "categories.jsonl"), "--journal", str(journal), "--judge-url", "http://stub",
                     "--cutoff", "0.3"], post=StubJudge(ANSWERS))
    before = journal.read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, raw, Loader(StubJudge(ANSWERS)))
    assert stopped.value.code == 1 and "--force" in capsys.readouterr().out
    assert journal.read_text(encoding="utf-8") == before

    run(tmp_path, raw, Loader(StubJudge(ANSWERS)), "--force")
    assert {r["by"] for r in read_jsonl(journal)} == {BY}


def test_a_model_that_does_not_load_falls_back_to_the_rules_and_says_so(write_jsonl, tmp_path, capsys):
    out, journal = run(tmp_path, raw_of(write_jsonl, "杖@2募集", "おはようございます"), Loader(raises=OSError("no such file")))

    captured = capsys.readouterr()
    assert "WARNING" in captured.err and "no such file" in captured.err and "rule baseline" in captured.err
    assert not journal.exists()
    assert read_jsonl(out)


def test_a_model_that_does_not_load_never_replaces_a_file_a_judge_made(write_jsonl, tmp_path, capsys):
    raw = raw_of(write_jsonl, "おはよう")
    out, _ = run(tmp_path, raw, Loader(StubJudge(ANSWERS)))
    before = out.read_text(encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, raw, Loader(raises=OSError("no such file")))

    assert stopped.value.code == 1
    assert out.read_text(encoding="utf-8") == before
    assert "not replacing it" in capsys.readouterr().err


def test_a_local_model_that_keeps_failing_ends_the_run_without_blaming_a_server(write_jsonl, tmp_path, capsys):
    lines = [f"line {n}" for n in range(categorize.MAX_CONSECUTIVE_FAILURES + 3)]

    with pytest.raises(SystemExit):
        run(tmp_path, raw_of(write_jsonl, *lines), Loader(StubJudge(fail=set(lines))))

    out = capsys.readouterr().out
    assert "resume" in out and "server" not in out


def test_probe_with_a_local_judge_prints_both_reports_and_the_sweep(tmp_path, capsys):
    sample = tmp_path / "sample.jsonl"
    sample.write_text("\n".join(json.dumps({"original": text, "category": cat}, ensure_ascii=False)
                                for text, cat in (("おはよう", "social"), ("2人募集", "recruitment"))), encoding="utf-8")

    categorize.main(["--probe", str(sample), "--judge-local", RUN], loader=Loader(StubJudge(ANSWERS)))

    out = capsys.readouterr().out
    assert BY in out and "cutoff" in out.lower()


@pytest.mark.parametrize("extra", [["--judge-url", "http://stub"], ["--judge-model", "x"]])
def test_judge_local_excludes_the_server_options(write_jsonl, tmp_path, extra):
    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, raw_of(write_jsonl, "おはよう"), Loader(StubJudge(ANSWERS)), *extra)

    assert stopped.value.code == 2
