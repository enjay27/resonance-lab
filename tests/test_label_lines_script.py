import json
import os

import pytest

import label_lines

GUIDE = "# T\n\n**Version:** 1.0.0 · **Season:** S1\n\n## 1. The rules of labelling\nOne label per line.\n\n## Changelog\n| 1.0.0 | 2026-10-05 | x |\n"
LABEL_MAP = {"version": 1, "season": "S1", "doc_version": "1.0.0", "map": {"coordination/boss_call": "coordination"}, "exclude": ["non_japanese"]}
RAW = [{"original": "おはよう", "channel": "PARTY"}, {"original": "20ch 覇者", "channel": "WORLD"}, {"original": "おはよう", "channel": "PARTY"},
       {"original": "내일 뭐하지", "channel": "WORLD"}, {"original": "草", "channel": "PARTY"}]


@pytest.fixture
def env(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in RAW), encoding="utf-8")
    guide = tmp_path / "guide.md"
    guide.write_text(GUIDE, encoding="utf-8")
    label_map = tmp_path / "map.json"
    label_map.write_text(json.dumps(LABEL_MAP), encoding="utf-8")
    return {"raw": str(raw), "guide": guide, "root": tmp_path / "work",
            "base": ["--root", str(tmp_path / "work"), "--guide", str(guide), "--label-map", str(label_map)]}


def run(env, *args):
    return label_lines.main([args[0], *env["base"], *args[1:]])


def season(env):
    return env["root"] / "S1"


def read(path):
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_out(env, name, rows):
    folder = season(env) / "round1" / "out"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


LABELS = [{"i": 1, "cat": "social/greeting"}, {"i": 2, "cat": "coordination/boss_call"}, {"i": 3, "cat": "non_japanese"}, {"i": 4, "cat": "chat/reaction", "unsure": True}]


def prepared(env):
    run(env, "prepare", "--season", "S1", "--raw", env["raw"])
    return [n[:-6] for n in os.listdir(season(env) / "round1" / "in")][0]


# --- prepare ------------------------------------------------------------------------------------------------------------------


def test_prepare_writes_the_distinct_lines_the_batches_the_brief_and_the_record(env, capsys):
    assert run(env, "prepare", "--season", "S1", "--raw", env["raw"]) == 0

    assert [r["original"] for r in read(season(env) / "lines.jsonl")] == ["おはよう", "20ch 覇者", "내일 뭐하지", "草"]
    (batch,) = os.listdir(season(env) / "round1" / "in")
    assert read(season(env) / "round1" / "in" / batch)[0] == {"i": 1, "ch": "P", "n": 2, "text": "おはよう"}
    assert "One label per line." in (season(env) / "round1" / "brief.md").read_text(encoding="utf-8")
    record = json.load(open(season(env) / "run.json", encoding="utf-8"))
    assert record["doc_version"] == "1.0.0" and record["counts"] == {"raw": 5, "distinct": 4}
    assert "4 distinct lines" in capsys.readouterr().out


def test_prepare_reads_several_logs_and_batches_by_size(env, tmp_path):
    second = tmp_path / "second.jsonl"
    second.write_text(json.dumps({"original": "新しい行", "channel": "LOCAL"}, ensure_ascii=False), encoding="utf-8")

    run(env, "prepare", "--season", "S1", "--raw", env["raw"], str(second), "--size", "2")

    assert len(os.listdir(season(env) / "round1" / "in")) == 3  # 5 distinct lines, 2 at most per batch


def test_prepare_never_overwrites_work_that_exists(env):
    prepared(env)

    with pytest.raises(SystemExit) as stopped:
        run(env, "prepare", "--season", "S1", "--raw", env["raw"])

    assert stopped.value.code == 1


def test_a_log_that_cannot_be_read_stops_prepare(env, tmp_path):
    with pytest.raises(SystemExit):
        run(env, "prepare", "--season", "S1", "--raw", str(tmp_path / "none.jsonl"))


# --- check ---------------------------------------------------------------------------------------------------------------------


def test_check_names_a_batch_without_output_then_passes_a_good_one(env, capsys):
    name = prepared(env)

    with pytest.raises(SystemExit):
        run(env, "check", "--season", "S1")
    assert name in capsys.readouterr().out

    write_out(env, name, LABELS)
    assert run(env, "check", "--season", "S1") == 0


def test_check_reports_an_unknown_category(env, capsys):
    name = prepared(env)
    write_out(env, name, [LABELS[0], {"i": 2, "cat": "nonsense"}, LABELS[2], LABELS[3]])

    with pytest.raises(SystemExit):
        run(env, "check", "--season", "S1")

    assert "nonsense" in capsys.readouterr().out


# --- assemble and export -----------------------------------------------------------------------------------------------------


def test_assemble_writes_the_labels_the_judge_sample_and_a_record(env, capsys):
    write_out(env, prepared(env), LABELS)

    assert run(env, "assemble", "--season", "S1") == 0

    labels = read(season(env) / "labels.jsonl")
    assert [(r["original"], r["category"]) for r in labels] == [("おはよう", "social/greeting"), ("20ch 覇者", "coordination/boss_call"), ("내일 뭐하지", "non_japanese"), ("草", "chat/reaction")]
    assert labels[3]["unsure"] is True and labels[0]["count"] == 2
    sample = read(season(env) / "judge-sample.jsonl")
    assert [r["category"] for r in sample] == ["social/greeting", "coordination", "chat/reaction"] and {r["split"] for r in sample} <= {"dev", "test"}
    meta = json.load(open(season(env) / "labels.meta.json", encoding="utf-8"))
    assert meta["doc_version"] == "1.0.0" and meta["lines"] == 4 and meta["unsure"] == 1 and meta["judge_sample"] == 3 and meta["excluded"] == {"non_japanese": 1}
    assert "| `chat/reaction` | 1 |" in capsys.readouterr().out


def test_assemble_refuses_when_the_guide_changed_after_the_batches_were_prepared(env, capsys):
    write_out(env, prepared(env), LABELS)
    env["guide"].write_text(GUIDE.replace("1.0.0 ·", "1.1.0 ·"), encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        run(env, "assemble", "--season", "S1")

    out = capsys.readouterr().out
    assert stopped.value.code == 1 and "1.0.0" in out and "1.1.0" in out and not (season(env) / "labels.jsonl").exists()


def test_assemble_names_the_lines_a_batch_left_unlabelled(env, capsys):
    write_out(env, prepared(env), LABELS[:2])

    with pytest.raises(SystemExit):
        run(env, "assemble", "--season", "S1")

    assert "missing i: [3, 4]" in capsys.readouterr().out


def test_export_rebuilds_the_judge_sample_after_a_label_was_corrected_by_hand(env):
    write_out(env, prepared(env), LABELS)
    run(env, "assemble", "--season", "S1")
    labels = read(season(env) / "labels.jsonl")
    labels[0]["category"] = "social/thanks"
    (season(env) / "labels.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in labels), encoding="utf-8")

    assert run(env, "export", "--season", "S1") == 0

    assert read(season(env) / "judge-sample.jsonl")[0]["category"] == "social/thanks"


def test_export_refuses_a_label_that_is_in_neither_the_taxonomy_nor_the_map(env, capsys):
    write_out(env, prepared(env), LABELS)
    run(env, "assemble", "--season", "S1")
    labels = read(season(env) / "labels.jsonl")
    labels[0]["category"] = "typo/here"
    (season(env) / "labels.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in labels), encoding="utf-8")

    with pytest.raises(SystemExit):
        run(env, "export", "--season", "S1")

    assert "typo/here" in capsys.readouterr().out


def test_report_prints_the_counts_table_for_the_guide(env, capsys):
    write_out(env, prepared(env), LABELS)
    run(env, "assemble", "--season", "S1")
    capsys.readouterr()

    run(env, "report", "--season", "S1")

    out = capsys.readouterr().out
    assert "| Category | Lines |" in out and "unsure: 1" in out


@pytest.mark.parametrize("name", ["", "../x", "a b"])
def test_a_season_is_a_plain_name(env, name):
    with pytest.raises(SystemExit):
        run(env, "prepare", "--season", name, "--raw", env["raw"])


def test_check_batch_looks_at_one_batch_only_and_an_unknown_batch_is_an_error(env, capsys):
    run(env, "prepare", "--season", "S1", "--raw", env["raw"], "--size", "2")
    names = sorted(n[:-6] for n in os.listdir(season(env) / "round1" / "in"))
    rows = read(season(env) / "round1" / "in" / f"{names[0]}.jsonl")
    write_out(env, names[0], [{"i": r["i"], "cat": "chat"} for r in rows])

    assert run(env, "check", "--season", "S1", "--batch", names[0]) == 0
    with pytest.raises(SystemExit):
        run(env, "check", "--season", "S1")
    with pytest.raises(SystemExit):
        run(env, "check", "--season", "S1", "--batch", "nope")
