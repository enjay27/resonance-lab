"""How well does a categorizer put chat lines in the right root? Pure, no torch, no network.

A hand-labelled sample (`{"original", "category"}` per line, a taxonomy root or child) is the answer key; a categorizer is any
function line -> root or None (None = no answer). The baseline in categorizer.py and, later, the judge-based gate are scored the
same way, at the root level, so they compare. A sample of made-up lines is in tests/fixtures/gate1_sample.jsonl: it shows the
format and guards the rules, it does not measure real accuracy -- label real lines yourself (config.GATE1_SAMPLE, gitignored).
"""

import json

from taxonomy import TaxonomyError, load_taxonomy, root_of

NO_ANSWER = "-"


class SampleError(Exception):
    """The labelled sample cannot be used; the message names the file and line."""


def read_sample(path):
    """The rows of a labelled sample; every label must be a category of the taxonomy."""
    try:
        taxonomy = load_taxonomy()
    except TaxonomyError as error:
        raise SampleError(str(error)) from None
    try:
        f = open(path, encoding="utf-8")
    except OSError as error:
        raise SampleError(f"{path}: cannot be read ({error.strerror or error})") from None
    rows = []
    with f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError:
                raise SampleError(f"{path} line {number}: not valid JSON") from None
            original = row.get("original") if isinstance(row, dict) else None
            category = row.get("category") if isinstance(row, dict) else None
            if not isinstance(original, str) or not original.strip() or not isinstance(category, str):
                raise SampleError(f"{path} line {number}: needs a non-empty \"original\" and a \"category\"")
            if category not in taxonomy.paths:
                raise SampleError(f"{path} line {number}: '{category}' is not a category of configs/category_taxonomy.json")
            rows.append({"original": original, "category": category})
    return rows


def evaluate_categorizer(rows, predict):
    """Score `predict(original) -> root or None` against the labelled rows, at the root level."""
    if not rows:
        raise SampleError("the labelled sample is empty")
    roots = {}
    confusion = {}
    correct = covered = 0
    for row in rows:
        truth, answer = root_of(row["category"]), predict(row["original"])
        covered += answer is not None
        correct += answer == truth
        confusion.setdefault(truth, {})
        confusion[truth][answer or NO_ANSWER] = confusion[truth].get(answer or NO_ANSWER, 0) + 1
        roots.setdefault(truth, {"support": 0, "predicted": 0, "correct": 0})["support"] += 1
        if answer is not None:
            roots.setdefault(answer, {"support": 0, "predicted": 0, "correct": 0})["predicted"] += 1
        if answer == truth:
            roots[truth]["correct"] += 1
    for stats in roots.values():
        stats["precision"] = stats["correct"] / stats["predicted"] if stats["predicted"] else None
        stats["recall"] = stats["correct"] / stats["support"] if stats["support"] else None
    n = len(rows)
    return {
        "n": n, "covered": covered, "abstained": n - covered, "correct": correct, "wrong": covered - correct,
        "accuracy": correct / n, "coverage": covered / n, "precision": correct / covered if covered else None,
        "per_root": dict(sorted(roots.items())), "confusion": confusion,
    }


def _percent(value):
    return "-" if value is None else f"{value:.0%}"


def format_gate_report(report, name):
    """The scores as text: the totals, one row per root, and what each root was mistaken for."""
    lines = [
        f"--- Gate 1: {name} on {report['n']} lines ---",
        f"accuracy  : {_percent(report['accuracy'])}  ({report['correct']} right of {report['n']}; no answer counts as wrong)",
        f"coverage  : {_percent(report['coverage'])}  ({report['covered']} answered, {report['abstained']} no answer)",
        f"precision : {_percent(report['precision'])}  (right, of the lines it answered)",
        "",
        f"  {'root':<14}{'lines':>6}{'answered':>10}{'right':>7}{'precision':>11}{'recall':>8}",
    ]
    for root, stats in report["per_root"].items():
        lines.append(f"  {root:<14}{stats['support']:>6}{stats['predicted']:>10}{stats['correct']:>7}"
                     f"{_percent(stats['precision']):>11}{_percent(stats['recall']):>8}")
    lines += ["", "What each root was taken for (\"-\" = no answer):"]
    for truth, answers in sorted(report["confusion"].items()):
        wrong = {answer: n for answer, n in answers.items() if answer != truth}
        if wrong:
            lines.append(f"  {truth:<14}" + ", ".join(f"{answer} x{n}" for answer, n in sorted(wrong.items(), key=lambda item: -item[1])))
    return "\n".join(lines)
