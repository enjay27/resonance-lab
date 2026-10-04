import json
import sys

import pytest

import eval_metrics as em
from taxonomy import Taxonomy


def sample(jp, ko, category=None):
    row = {"original": jp, "translated": ko}
    if category:
        row["category"] = category
    return row


# --- the eval dataset ---------------------------------------------------------------------------


def test_load_eval_dataset_reads_rows_and_skips_blank_lines(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps(sample("遺跡1F", "유적 1F", "party"), ensure_ascii=False) + "\n\n" + json.dumps(sample("おやすみ", "잘 자"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    rows = em.load_eval_dataset(str(path))

    assert [r["original"] for r in rows] == ["遺跡1F", "おやすみ"]
    assert rows[0]["category"] == "party"


def test_load_eval_dataset_missing_file_says_where_to_put_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="eval dataset"):
        em.load_eval_dataset(str(tmp_path / "nope.jsonl"))


def test_load_eval_dataset_names_the_bad_line(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text(json.dumps(sample("a", "b")) + "\n" + json.dumps({"original": "only source"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 2"):
        em.load_eval_dataset(str(path))


def test_load_eval_dataset_names_a_broken_json_line(tmp_path):
    path = tmp_path / "eval.jsonl"
    path.write_text("{broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line 1"):
        em.load_eval_dataset(str(path))


# --- text rules ---------------------------------------------------------------------------------


def test_strip_think_removes_an_empty_think_block_only():
    assert em.strip_think("<think>\n</think>\n안녕") == "안녕"
    assert em.strip_think("  안녕  ") == "안녕"
    assert em.strip_think("<think>생각</think>안녕") == "<think>생각</think>안녕"  # real reasoning stays visible


@pytest.mark.parametrize("text, expected", [("안녕하세요", False), ("안녕 こんにちは", True), ("遺跡 1F", True), ("ムクボ", True), ("abc 123", False)])
def test_has_jp(text, expected):
    assert em.has_jp(text) is expected


@pytest.mark.parametrize(
    "raw, expected",
    [("안녕", False), ("<think></think>안녕", False), ("<think>\n \n</think>안녕", False), ("<think>음, 번역하면</think>안녕", True), ("<think>끝나지 않는 생각", True)],
)
def test_think_leaked_only_counts_real_content(raw, expected):
    assert em.think_leaked(raw) is expected


def test_term_results_check_each_term_in_the_source():
    results = em.term_results("消化と完凸", "숙제와 완돌")

    assert results == [("消化", "숙제", True), ("完凸", "풀돌", False)]


def test_term_results_is_empty_without_terms():
    assert em.term_results("おやすみ", "잘 자") == []


@pytest.mark.parametrize(
    "jp, pred, expected",
    [("Discordで話そう", "디스코드에서 얘기해요", True), ("Discordで話そう", "Discord에서 얘기해요", False), ("おやすみ", "디스코드", False)],
)
def test_discord_violation(jp, pred, expected):
    assert em.discord_violation(jp, pred) is expected


# --- standard metrics ----------------------------------------------------------------------------


def test_standard_metrics_score_every_sample_not_only_the_first():
    # Regression: experiment/translategemma's eval.py passed references as [[r1], [r2], ...],
    # which sacrebleu scores against the FIRST line only (100.0 here, and 0.0 if only
    # the first prediction is wrong). The correct input is one reference set: [[r1, r2, ...]].
    refs = ["유적 1F에서 모집합니다", "안녕하세요 여러분", "오늘은 숙제를 합니다", "잘 자요"]
    first_right_rest_garbage = [refs[0], "완전히 다른 문장", "전혀 관계 없음", "엉망진창 출력"]
    first_garbage_rest_right = ["엉망진창", refs[1], refs[2], refs[3]]

    right_first = em.standard_metrics(first_right_rest_garbage, refs)["chrf"]
    wrong_first = em.standard_metrics(first_garbage_rest_right, refs)["chrf"]

    assert right_first < 60  # the garbage is counted
    assert wrong_first > 40  # the good lines are counted


def test_standard_metrics_of_a_perfect_run():
    refs = ["유적 1F 에서 파티 모집 합니다 지금 바로", "오늘은 숙제 를 하고 나서 잘 거예요 안녕"]
    metrics = em.standard_metrics(refs, refs)
    assert metrics["chrf"] == pytest.approx(100.0)
    assert metrics["bleu"] == pytest.approx(100.0)
    assert metrics["ter"] == pytest.approx(0.0)


def test_bleu_is_zero_for_very_short_lines_but_chrf_still_works():
    # No line has 4 tokens, so corpus BLEU has no 4-grams and is 0 even for a perfect run;
    # chrF (character based) is the metric to trust for short chat lines.
    refs = ["유적 1F", "잘 자요"]
    metrics = em.standard_metrics(refs, refs)
    assert metrics["bleu"] == 0.0
    assert metrics["chrf"] == pytest.approx(100.0)


def test_comet_score_is_none_when_comet_is_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "comet", None)  # `import comet` raises ImportError
    assert em.comet_score([sample("a", "b")], ["b"]) is None


# --- evaluate + report ---------------------------------------------------------------------------

SAMPLES = [
    sample("消化する", "숙제한다", "slang"),
    sample("Discordで募集", "디스코드로 모집", "party"),
    sample("おやすみ", "잘 자", "chat"),
    sample("消化する", "숙제한다", "slang"),
]
PREDICTIONS = ["숙제한다", "디스코드로 모집", "잘 자 おやすみ", "소화한다"]
RAW = ["숙제한다", "디스코드로 모집", "잘 자 おやすみ", "<think>생각</think>소화한다"]


def test_evaluate_counts_every_rule():
    report = em.evaluate(SAMPLES, PREDICTIONS, RAW)

    assert report["n"] == 4
    assert report["jp_leakage"] == 1
    assert report["think_leakage"] == 1
    assert (report["term_hits"], report["term_total"]) == (1, 2)
    assert report["term_misses"][0]["expected"] == "숙제"
    assert report["discord_violations"] == 1
    assert report["exact_match"] == 2  # lines 1 and 2
    assert set(report["standard"]) == {"bleu", "chrf", "ter"}


def test_evaluate_breaks_down_by_category():
    report = em.evaluate(SAMPLES, PREDICTIONS, RAW)

    slang = report["categories"]["slang"]
    assert {key: slang[key] for key in ("total", "jp_leak", "term_miss", "discord_viol")} == {"total": 2, "jp_leak": 0, "term_miss": 1, "discord_viol": 0}
    assert report["categories"]["party"]["discord_viol"] == 1
    assert report["categories"]["chat"]["jp_leak"] == 1


def test_samples_without_a_category_are_unknown():
    report = em.evaluate([sample("a", "b")], ["b"])
    assert list(report["categories"]) == ["unknown"]


def test_raw_outputs_default_to_the_predictions():
    assert em.evaluate([sample("a", "b")], ["b"])["think_leakage"] == 0


def test_evaluate_refuses_mismatched_lengths():
    with pytest.raises(ValueError, match="2 samples but 1 predictions"):
        em.evaluate([sample("a", "b"), sample("c", "d")], ["b"])


def test_evaluate_refuses_an_empty_dataset():
    with pytest.raises(ValueError, match="empty"):
        em.evaluate([], [])


def test_format_report_has_every_section_and_flags_bad_lines():
    text = em.format_report(em.evaluate(SAMPLES, PREDICTIONS, RAW), SAMPLES, PREDICTIONS, comet=0.8765)

    for expected in (
        "chrF", "BLEU", "TER", "COMET : 0.8765", "JP Leakage", "1/4 (25.0%)", "Think Leakage", "Term Accuracy : 1/2 (50.0%)",
        "Expected '숙제' for '消化'", "Exact Match", "[slang] (2 samples)", "[party]", "Full Output Log", "⚠ JP", "⚠ discord",
    ):  # fmt: skip
        assert expected in text, expected


def test_format_report_without_comet_says_so():
    report = em.evaluate([sample("a", "b")], ["b"])
    assert "COMET : not available" in em.format_report(report, [sample("a", "b")], ["b"], comet=None)


def test_format_report_without_terms_says_so():
    report = em.evaluate([sample("おやすみ", "잘 자")], ["잘 자"])
    assert "No term-containing samples" in em.format_report(report, [sample("おやすみ", "잘 자")], ["잘 자"])


def test_term_misses_are_capped_in_the_report():
    samples = [sample("消化", "숙제") for _ in range(9)]
    text = em.format_report(em.evaluate(samples, ["소화"] * 9), samples, ["소화"] * 9)
    assert text.count("Expected '숙제' for '消化'") == 5


# --- per category: chrF, term accuracy, roots ---------------------------------------------------------------------------


def test_each_category_has_its_own_chrf_and_term_counts():
    report = em.evaluate(SAMPLES, PREDICTIONS, RAW)

    party, slang, chat = (report["categories"][name] for name in ("party", "slang", "chat"))
    assert party["chrf"] == pytest.approx(100.0)  # its one prediction is its reference
    assert 0 < slang["chrf"] < 100 and 0 < chat["chrf"] < 100
    assert (slang["term_total"], slang["term_miss"]) == (2, 1) and party["term_total"] == 0


def test_a_sub_category_counts_for_its_root_too():
    samples = [sample("消化する", "숙제한다", "game/combat"), sample("消化する", "숙제한다", "game/market"), sample("おやすみ", "잘 자", "chat")]

    report = em.evaluate(samples, ["숙제한다", "소화한다", "잘 자"], taxonomy=TAX)

    assert set(report["categories"]) == {"game/combat", "game/market", "chat"}
    assert set(report["roots"]) == {"game", "chat"}
    assert report["roots"]["game"]["total"] == 2 and report["roots"]["game"]["term_miss"] == 1 and report["roots"]["game"]["term_total"] == 2
    assert report["roots"]["chat"]["chrf"] == pytest.approx(100.0)
    assert report["categories"]["game/combat"]["chrf"] == pytest.approx(100.0) and report["roots"]["game"]["chrf"] < 100


def test_a_report_says_how_many_eval_lines_each_root_has():
    report = em.evaluate([sample("a", "b", "chat"), sample("c", "d")], ["b", "d"], taxonomy=TAX)

    assert report["coverage"] == {"counts": {"chat": 1, "game": 0, "other": 0}, "unknown": {}, "unlabeled": 1}


def test_without_a_readable_taxonomy_the_report_has_no_coverage(monkeypatch):
    monkeypatch.setattr(em, "CATEGORY_TAXONOMY", "/nonexistent/taxonomy.json")

    assert em.evaluate([sample("a", "b")], ["b"])["coverage"] is None


TAX = Taxonomy(roots=("chat", "game", "other"), paths={"chat": "c", "game": "g", "game/combat": "x", "game/market": "m", "other": "o"})


def test_the_report_has_a_score_table_per_root_and_marks_the_ones_with_few_lines():
    samples = [sample("消化する", "숙제한다", "game")] * 12 + [sample("おやすみ", "잘 자", "chat")] * 3
    predictions = ["숙제한다"] * 12 + ["잘 자"] * 3

    text = em.format_report(em.evaluate(samples, predictions, taxonomy=TAX), samples, predictions)

    table = text.split("--- Category Scores ---")[1].split("\n---")[0]
    game_row = next(line for line in table.splitlines() if line.strip().startswith("game"))
    chat_row = next(line for line in table.splitlines() if line.strip().startswith("chat"))
    assert "12" in game_row and "100.0" in game_row and "12/12" in game_row and "small" not in game_row
    assert "3" in chat_row and "small" in chat_row


def test_the_score_columns_line_up_whatever_the_names_are():
    samples = [sample("a", "b", "chat"), sample("c", "d", "game/combat")]

    table = em.format_report(em.evaluate(samples, ["b", "d"], taxonomy=TAX), samples, ["b", "d"]).split("--- Category Scores ---")[1].split("\n\n")[0]
    rows = [line for line in table.splitlines() if line.startswith("  ") and "category" not in line and not line.startswith("  (")]
    header = next(line for line in table.splitlines() if "category" in line)

    assert all(row.index("100.0") + len("100.0") == header.index("chrF") + len("chrF") for row in rows)

    flat = [sample("a", "b", "chat")]
    nested = [sample("a", "b", "game/combat")]

    assert "Sub-categories" not in em.format_report(em.evaluate(flat, ["b"], taxonomy=TAX), flat, ["b"])
    assert "Sub-categories" in em.format_report(em.evaluate(nested, ["b"], taxonomy=TAX), nested, ["b"])


def test_the_coverage_section_shows_empty_roots_unlabeled_and_unknown_lines():
    samples = [sample("a", "b", "chat"), sample("c", "d"), sample("e", "f", "Chat")]

    text = em.format_report(em.evaluate(samples, ["b", "d", "f"], taxonomy=TAX), samples, ["b", "d", "f"])

    section = text.split("--- Eval Set Coverage")[1].split("--- Category Breakdown")[0]
    assert "game" in section and "no lines" in section
    assert "other" in section and section.count("no lines") == 1  # 'other' is the catch-all: it may stay empty
    assert "1 line" in section and "no category" in section
    assert "Chat" in section and "not in the taxonomy" in section


def test_no_coverage_section_without_a_taxonomy(monkeypatch):
    monkeypatch.setattr(em, "CATEGORY_TAXONOMY", "/nonexistent/taxonomy.json")
    samples = [sample("a", "b", "chat")]

    assert "Eval Set Coverage" not in em.format_report(em.evaluate(samples, ["b"]), samples, ["b"])
