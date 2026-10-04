"""Against a REAL llama-server with a decision model; skipped unless RESONANCE_JUDGE_URL is set (CI and the cloud gate skip it).

    RESONANCE_JUDGE_URL=http://127.0.0.1:8080 python -m pytest tests/test_judge_live.py -v

Shape checks only (the answer is well-formed, the pass writes a file a recipe reads): which model is good enough is for
`scripts/categorize.py --probe`, not for a test.
"""

import json
import os

import pytest

import categorize
import gate_judge
from dataset_recipe import line_key, read_categories
from judge_client import SystemOneClient
from taxonomy import load_taxonomy

URL = os.environ.get("RESONANCE_JUDGE_URL")
pytestmark = pytest.mark.skipif(not URL, reason="needs a llama-server with a decision model: set RESONANCE_JUDGE_URL")

LINES = ["おはようございます", "ID:12345 ダンジョン行きませんか？ 2人募集", "これどうやって倒すの？"]


def test_the_server_names_its_model():
    assert SystemOneClient(URL).check_server()


def test_a_japanese_line_gets_a_root_with_probabilities_that_sum_to_one():
    options = gate_judge.choice_options(load_taxonomy())

    answer = SystemOneClient(URL).choice(LINES[1], gate_judge.INSTRUCTIONS, options)

    assert answer.choice in options and set(answer.probabilities) == set(options)
    assert sum(answer.probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert 0.0 <= answer.margin <= 1.0


def test_a_pass_over_a_few_lines_writes_the_journal_and_a_categories_file_a_recipe_reads(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("".join(json.dumps({"original": line, "translated": "x", "channel": "WORLD"}, ensure_ascii=False) + "\n" for line in LINES),
                   encoding="utf-8")
    out, journal = tmp_path / "categories.jsonl", tmp_path / "judge.jsonl"

    categorize.main(["--raw", str(raw), "--out", str(out), "--journal", str(journal), "--judge-url", URL, "--cutoff", "0", "--use-channel"])

    assert len(journal.read_text(encoding="utf-8").splitlines()) == len(LINES)
    categories = read_categories(str(out))
    assert set(categories) == {line_key(line) for line in LINES}
    assert set(categories.values()) <= set(load_taxonomy().roots)
