import json
import os

import pytest

import translate_agents

DOC = "# T\n\n**Version:** 1.0.0 · **Season:** S1\n\n## 2. Style rules\nKeep numbers.\n\n## Changelog\n| 1.0.0 | 2026-10-05 | x |\n"
GLOSSARY = {"version": 1, "season": "S1", "doc_version": "1.0.0", "required": [{"ja": "墓", "ko": "저주받은 무덤"}],
            "banned": [{"ko": "와이프", "why": "wife"}], "fixes": [{"when": "ワイプ", "old": "와이프", "new": "전멸", "why": "wife"}]}
LABELLED = [
    {"original": "墓M6 ＠D2", "category": "recruitment/party", "channel": "WORLD"},
    {"original": "ギルド員募集", "category": "recruitment/guild", "channel": "WORLD"},
    {"original": "ワイプした", "category": "chat", "channel": "PARTY"},
    {"original": "내일 뭐하지", "category": "non_japanese", "channel": "WORLD"},
    {"original": "ありがとう", "category": "social", "channel": "PARTY"},
]


@pytest.fixture
def env(tmp_path):
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in LABELLED), encoding="utf-8")
    doc = tmp_path / "doc.md"
    doc.write_text(DOC, encoding="utf-8")
    glossary = tmp_path / "s1.json"
    glossary.write_text(json.dumps(GLOSSARY, ensure_ascii=False), encoding="utf-8")
    return {"root": tmp_path / "work", "labels": str(labels), "doc": doc, "glossary": str(glossary),
            "base": ["--root", str(tmp_path / "work"), "--glossary", str(glossary), "--doc", str(doc)]}


def run(env, *args):
    return translate_agents.main([args[0], *env["base"], *args[1:]])


def season(env):
    return env["root"] / "S1"


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_out(env, round_no, name, rows):
    folder = season(env) / f"round{round_no}" / "out"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


GOOD = [{"i": 1, "ko": "저주받은 무덤 M6 @D2", "terms": [["墓", "저주받은 무덤"]]}, {"i": 3, "ko": "와이프했다", "terms": []},
        {"i": 5, "ko": "감사합니다", "terms": []}]


# --- prepare ------------------------------------------------------------------------------------------------------------------


def test_prepare_writes_the_batches_the_brief_and_the_run_record_and_skips_guild_and_foreign_lines(env, capsys):
    assert run(env, "prepare", "--season", "S1", "--labels", env["labels"]) == 0

    (batch,) = os.listdir(season(env) / "round1" / "in")
    rows = read(season(env) / "round1" / "in" / batch)
    assert [r["i"] for r in rows] == [1, 3, 5] and rows[0] == {"i": 1, "ch": "W", "cat": "recruitment/party", "ja": "墓M6 ＠D2"}
    assert "Keep numbers." in (season(env) / "round1" / "brief.md").read_text(encoding="utf-8")
    record = json.load(open(season(env) / "run.json", encoding="utf-8"))
    assert record["doc_version"] == "1.0.0" and record["counts"]["lines"] == 3 and record["counts"]["skipped"] == {"recruitment/guild": 1, "non_japanese": 1}
    assert "skipped" in capsys.readouterr().out


def test_guild_lines_can_be_included_and_batches_sized(env):
    run(env, "prepare", "--season", "S1", "--labels", env["labels"], "--include-guild", "--size", "2")

    names = sorted(os.listdir(season(env) / "round1" / "in"))
    assert len(names) == 2 and sum(len(read(season(env) / "round1" / "in" / n)) for n in names) == 4


def test_prepare_never_overwrites_a_round_that_exists(env):
    run(env, "prepare", "--season", "S1", "--labels", env["labels"])

    with pytest.raises(SystemExit) as stopped:
        run(env, "prepare", "--season", "S1", "--labels", env["labels"])

    assert stopped.value.code == 1


@pytest.mark.parametrize("name", ["", "../x", "a b", "a/b"])
def test_a_season_is_a_plain_name(env, name):
    with pytest.raises(SystemExit):
        run(env, "prepare", "--season", name, "--labels", env["labels"])


# --- check -------------------------------------------------------------------------------------------------------------------


def test_check_fails_naming_a_batch_without_output_and_passes_a_clean_one(env, capsys):
    run(env, "prepare", "--season", "S1", "--labels", env["labels"])
    (name,) = [n[:-6] for n in os.listdir(season(env) / "round1" / "in")]

    with pytest.raises(SystemExit):
        run(env, "check", "--season", "S1")
    assert name in capsys.readouterr().out

    write_out(env, 1, name, [GOOD[0], {"i": 3, "ko": "전멸했다", "terms": []}, GOOD[2]])
    assert run(env, "check", "--season", "S1") == 0
    assert "OK" in capsys.readouterr().out


def test_check_reports_the_problems_of_an_output_and_fails(env, capsys):
    run(env, "prepare", "--season", "S1", "--labels", env["labels"])
    (name,) = [n[:-6] for n in os.listdir(season(env) / "round1" / "in")]
    write_out(env, 1, name, [{"i": 1, "ko": "무덤 M6 @D2", "terms": []}, GOOD[1], {"i": 5, "ko": "ありがとう", "terms": []}])

    with pytest.raises(SystemExit) as stopped:
        run(env, "check", "--season", "S1")

    out = capsys.readouterr().out
    assert stopped.value.code == 1 and "저주받은 무덤" in out and "Japanese" in out


# --- assemble ----------------------------------------------------------------------------------------------------------------


def prepared_and_translated(env):
    run(env, "prepare", "--season", "S1", "--labels", env["labels"])
    (name,) = [n[:-6] for n in os.listdir(season(env) / "round1" / "in")]
    write_out(env, 1, name, GOOD)


def test_assemble_writes_the_final_lines_the_term_table_and_a_record_of_what_made_them(env):
    prepared_and_translated(env)

    assert run(env, "assemble", "--season", "S1") == 0

    final = read(season(env) / "final.jsonl")
    assert [(r["i"], r["translated"]) for r in final] == [(1, "저주받은 무덤 M6 @D2"), (3, "전멸했다"), (5, "감사합니다")]  # the glossary's fix ran
    assert "墓\t1\t저주받은 무덤 (1)\tyes" in (season(env) / "terms.tsv").read_text(encoding="utf-8")
    meta = json.load(open(season(env) / "final.meta.json", encoding="utf-8"))
    assert meta["doc_version"] == "1.0.0" and meta["lines"] == 3 and meta["problems"] == 0 and meta["glossary_sha1"] and meta["corrections"] == {"wife": 1}


def test_assemble_refuses_when_the_glossary_document_changed_after_the_batches_were_prepared(env, capsys):
    prepared_and_translated(env)
    env["doc"].write_text(DOC.replace("1.0.0 ·", "1.1.0 ·"), encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        run(env, "assemble", "--season", "S1")

    out = capsys.readouterr().out
    assert stopped.value.code == 1 and "1.0.0" in out and "1.1.0" in out
    assert not (season(env) / "final.jsonl").exists()


def test_assemble_exits_1_but_still_writes_the_file_when_a_line_breaks_the_glossary(env):
    run(env, "prepare", "--season", "S1", "--labels", env["labels"])
    (name,) = [n[:-6] for n in os.listdir(season(env) / "round1" / "in")]
    write_out(env, 1, name, [{"i": 1, "ko": "무덤 M6", "terms": []}, GOOD[1], GOOD[2]])

    with pytest.raises(SystemExit) as stopped:
        run(env, "assemble", "--season", "S1")

    assert stopped.value.code == 1
    assert json.load(open(season(env) / "final.meta.json", encoding="utf-8"))["problems"] == 1


def test_assemble_says_which_line_no_round_translated(env, capsys):
    run(env, "prepare", "--season", "S1", "--labels", env["labels"])
    (name,) = [n[:-6] for n in os.listdir(season(env) / "round1" / "in")]
    write_out(env, 1, name, GOOD[:1])

    with pytest.raises(SystemExit):
        run(env, "assemble", "--season", "S1")

    assert "not translated" in capsys.readouterr().out


# --- revise ------------------------------------------------------------------------------------------------------------------


def test_revise_prepares_a_round_of_the_lines_that_break_the_glossary_with_their_previous_translation(env):
    run(env, "prepare", "--season", "S1", "--labels", env["labels"])
    (name,) = [n[:-6] for n in os.listdir(season(env) / "round1" / "in")]
    write_out(env, 1, name, [{"i": 1, "ko": "무덤 M6", "terms": []}, GOOD[1], GOOD[2]])
    with pytest.raises(SystemExit):
        run(env, "assemble", "--season", "S1")

    assert run(env, "revise", "--season", "S1") == 0

    (rev,) = os.listdir(season(env) / "round2" / "in")
    assert read(season(env) / "round2" / "in" / rev) == [{"i": 1, "ch": "W", "cat": "recruitment/party", "ja": "墓M6 ＠D2", "prev": "무덤 M6"}]
    assert "revision round" in (season(env) / "round2" / "brief.md").read_text(encoding="utf-8")
    write_out(env, 2, rev[:-6], [{"i": 1, "ko": "저주받은 무덤 M6 @D2", "terms": [["墓", "저주받은 무덤"]]}])
    assert run(env, "assemble", "--season", "S1") == 0
    assert read(season(env) / "final.jsonl")[0]["translated"] == "저주받은 무덤 M6 @D2"


def test_check_looks_at_the_latest_round_unless_told_a_round(env):
    run(env, "prepare", "--season", "S1", "--labels", env["labels"])
    (name,) = [n[:-6] for n in os.listdir(season(env) / "round1" / "in")]
    write_out(env, 1, name, [{"i": 1, "ko": "무덤 M6", "terms": []}, GOOD[1], GOOD[2]])
    with pytest.raises(SystemExit):
        run(env, "assemble", "--season", "S1")
    run(env, "revise", "--season", "S1")
    (rev,) = [n[:-6] for n in os.listdir(season(env) / "round2" / "in")]
    write_out(env, 2, rev, [{"i": 1, "ko": "저주받은 무덤 M6 @D2", "terms": [["墓", "저주받은 무덤"]]}])

    assert run(env, "check", "--season", "S1") == 0          # round 2 is clean; round 1's bad line is superseded
    with pytest.raises(SystemExit):
        run(env, "check", "--season", "S1", "--round", "1")  # asked for round 1 itself


def test_revise_with_nothing_to_revise_says_so_and_prepares_nothing(env, capsys):
    prepared_and_translated(env)
    run(env, "assemble", "--season", "S1")

    assert run(env, "revise", "--season", "S1") == 0

    assert "nothing to revise" in capsys.readouterr().out and not (season(env) / "round2").exists()


def test_revise_all_takes_every_line(env):
    prepared_and_translated(env)
    run(env, "assemble", "--season", "S1")

    run(env, "revise", "--season", "S1", "--all")

    (rev,) = os.listdir(season(env) / "round2" / "in")
    assert [r["i"] for r in read(season(env) / "round2" / "in" / rev)] == [1, 3, 5]


# --- report ------------------------------------------------------------------------------------------------------------------


def test_report_lists_the_terms_with_more_than_one_rendering_and_the_flagged_lines(env, capsys):
    run(env, "prepare", "--season", "S1", "--labels", env["labels"])
    (name,) = [n[:-6] for n in os.listdir(season(env) / "round1" / "in")]
    write_out(env, 1, name, [{"i": 1, "ko": "저주받은 무덤 M6 @D2", "terms": [["墓", "저주받은 무덤"]]},
                             {"i": 3, "ko": "전멸", "terms": [["ワイプ", "전멸"]], "flag": "unclear"},
                             {"i": 5, "ko": "감사합니다 저주받은 무덤", "terms": [["墓", "무덤"], ["ワイプ", "저주받은 무덤"]]}])
    assert run(env, "assemble", "--season", "S1") == 0

    run(env, "report", "--season", "S1")

    out = capsys.readouterr().out
    assert "lines: 3" in out and "flagged: 1" in out and "more than one rendering" in out
