import json

import pytest

import compare_judges
import gate_compare
from dataset_recipe import line_key

SAMPLE = [("おはよう", "social"), ("2人募集", "recruitment"), ("草", "chat"), ("杖@2募集", "recruitment")]


def sample_file(tmp_path):
    path = tmp_path / "sample.jsonl"
    path.write_text("\n".join(json.dumps({"original": t, "category": c}, ensure_ascii=False) for t, c in SAMPLE), encoding="utf-8")
    return str(path)


def save(tmp_path, label, choices, seconds=0.2):
    answers = {line_key(text): {"choice": choice, "margin": 0.8, "probabilities": {choice: 0.9}} for (text, _), choice in zip(SAMPLE, choices)}
    gate_compare.save_probe(str(tmp_path / "probes"), gate_compare.probe_record(label, f"systemone:{label}:00000000", answers, seconds, False))


def run(tmp_path, *extra):
    compare_judges.main(["--sample", sample_file(tmp_path), "--dir", str(tmp_path / "probes"), "--cutoff", "0.3", *extra])


def test_it_prints_one_table_for_every_saved_probe(tmp_path, capsys):
    save(tmp_path, "9b-q8", ["social", "recruitment", "chat", "recruitment"])
    save(tmp_path, "4b", ["social", "recruitment", "social", "chat"])

    run(tmp_path)

    out = capsys.readouterr().out
    assert "9b-q8" in out and "4b" in out and "4 lines" in out and "systemone:4b:00000000" in out
    assert "9b-q8 vs 4b" in out or "4b vs 9b-q8" in out


def test_only_limits_the_labels_compared(tmp_path, capsys):
    for label in ("a", "b", "c"):
        save(tmp_path, label, ["social", "recruitment", "chat", "recruitment"])

    run(tmp_path, "--only", "a,c")

    out = capsys.readouterr().out
    assert "a vs c" in out and "b vs" not in out and "vs b" not in out


def test_an_unknown_label_names_the_ones_there_are(tmp_path, capsys):
    save(tmp_path, "a", ["social", "recruitment", "chat", "recruitment"])

    with pytest.raises(SystemExit) as stopped:
        run(tmp_path, "--only", "zzz")

    assert stopped.value.code == 1
    err = capsys.readouterr().err
    assert "zzz" in err and "a" in err


def test_without_any_probe_it_says_how_to_make_one(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        run(tmp_path)

    assert stopped.value.code == 1 and "--save-probe" in capsys.readouterr().err


def test_without_a_sample_it_names_the_file(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        compare_judges.main(["--sample", str(tmp_path / "nowhere.jsonl"), "--dir", str(tmp_path)])

    assert stopped.value.code == 1 and "nowhere.jsonl" in capsys.readouterr().err
