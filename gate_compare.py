"""Which Gate judge is better? One probe per judge is saved, and one table compares them on the same labelled sample.

A probe (`categorize.py --probe SAMPLE --judge-... --save-probe LABEL`, or the notebook) asks a judge about every line of the
sample once. What it answered is saved as `data/eval/gate1-compare/<label>.json`: per line the choice, the margin and the
probabilities, keyed by the sha1 of the line (never the chat line itself). The comparison is computed from those files and the
CURRENT labelled sample, so correcting a label needs no new model run, a judge run once is never run again, and it does not
matter whether a judge was Kev in this process or a GGUF on llama-server.

Pure: no network, no torch, no model. The honest part is `wilson` and the pair counts: on a sample of ~200 lines a difference of
a few points is noise, and "A right where B is wrong" against "B right where A is wrong" is the evidence.
"""

import datetime
import itertools
import json
import math
import os
import re

from dataset_recipe import line_key
from gate_eval import evaluate_categorizer
from gate_judge import SWEEP_CUTOFFS
from taxonomy import root_of

VERSION = 1
LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class CompareError(Exception):
    """The probes cannot be compared; the message says which file or what to do."""


def check_label(label):
    if not isinstance(label, str) or not LABEL.match(label):
        raise ValueError(f"label {label!r} must be a plain file name: letters, digits, '.', '_', '-' (e.g. 9b-q8)")
    return label


def probe_path(directory, label):
    return os.path.join(directory, check_label(label) + ".json")


# --- recording what a judge answered -----------------------------------------------------------------------------------


def collect_answers(rows, judge, use_channel):
    """{line key: {choice, margin, probabilities} or None} for the sample `rows`, from a GateJudge (answers are cached by it, so
    this costs no request after a probe has run). None = the judge failed on the line. The channel goes to the judge only with
    `use_channel`, as in the probe."""
    collected = {}
    for row in rows:
        answer = judge.answer(row["original"], row.get("channel") if use_channel else None)
        collected[line_key(row["original"])] = None if answer is None else {
            "choice": answer.choice, "margin": round(answer.margin, 4),
            "probabilities": {option: round(p, 4) for option, p in answer.probabilities.items()}}
    return collected


def probe_record(label, by, answers, seconds_per_line, use_channel, notes=""):
    """The record to save: who answered (`by` is the judge id: method, model, precision, question), how fast, and the answers."""
    return {"version": VERSION, "label": check_label(label), "by": by, "notes": notes, "use_channel": bool(use_channel),
            "seconds_per_line": round(seconds_per_line, 4), "n": len(answers), "answers": answers,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def save_probe(directory, record):
    """Writes `<directory>/<label>.json`; never replaces a probe (it may be the only record of a long run)."""
    path = probe_path(directory, record["label"])
    if os.path.exists(path):
        raise FileExistsError(f"{path} exists: use another label, or delete that file to probe again")
    os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return path


def load_probes(directory):
    """The saved probes, sorted by label; a missing folder is no probes."""
    if not os.path.isdir(directory):
        return []
    records = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        try:
            with open(path, encoding="utf-8") as f:
                record = json.load(f)
            if not (isinstance(record, dict) and isinstance(record["answers"], dict) and isinstance(record["label"], str)):
                raise ValueError("not a probe record")
            record["by"], record["seconds_per_line"]
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CompareError(f"{path}: not a readable probe file ({error}): delete it or probe again") from None
        records.append(record)
    return sorted(records, key=lambda record: record["label"])


# --- the comparison ----------------------------------------------------------------------------------------------------


def wilson(correct, n, z=1.96):
    """The 95% Wilson score interval of a share, (low, high); None for n = 0."""
    if not n:
        return None
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def _predictor(record, cutoff):
    """`predict(original) -> root or None` from a saved probe: the choice when its margin reaches the cutoff."""
    answers = record["answers"]

    def predict(text):
        answer = answers.get(line_key(text))
        return root_of(answer["choice"]) if answer is not None and answer["margin"] >= cutoff else None

    return predict


def compare_probes(records, rows, cutoff, target_precision=0.9):
    """Every probe scored on the labelled `rows`: at the cutoff (what a pass would categorise), without a cutoff (the model's own
    argmax, with its 95% interval), the most it can answer while staying `target_precision` precise, its speed, and for each pair
    of probes on how many lines they answer differently and who is right where they are not both right."""
    if not records:
        raise CompareError("no probes to compare: run `categorize.py --probe ... --save-probe LABEL` for each judge first")
    if not rows:
        raise CompareError("the labelled sample is empty")
    keys = [line_key(row["original"]) for row in rows]
    out = []
    for record in records:
        at_cutoff = evaluate_categorizer(rows, _predictor(record, cutoff))
        argmax = evaluate_categorizer(rows, _predictor(record, 0.0))
        reaching = [report["coverage"] for report in (evaluate_categorizer(rows, _predictor(record, c)) for c in SWEEP_CUTOFFS)
                    if report["precision"] is not None and report["precision"] >= target_precision]
        out.append({
            "label": record["label"], "by": record["by"], "notes": record.get("notes", ""),
            "accuracy": at_cutoff["accuracy"], "coverage": at_cutoff["coverage"], "precision": at_cutoff["precision"],
            "covered": at_cutoff["covered"], "correct": at_cutoff["correct"],
            "raw_accuracy": argmax["accuracy"], "raw_interval": wilson(argmax["correct"], argmax["n"]),
            "coverage_at_target": max(reaching) if reaching else None,
            "seconds_per_line": record["seconds_per_line"], "use_channel": record.get("use_channel", False),
            "missing": sum(key not in record["answers"] for key in keys),
        })
    pairs = []
    for a, b in itertools.combinations(records, 2):
        differ = only_a = only_b = 0
        for row, key in zip(rows, keys):
            answer_a, answer_b = a["answers"].get(key), b["answers"].get(key)
            choice_a = root_of(answer_a["choice"]) if answer_a else None
            choice_b = root_of(answer_b["choice"]) if answer_b else None
            truth = root_of(row["category"])
            differ += choice_a != choice_b
            only_a += choice_a == truth and choice_b != truth
            only_b += choice_b == truth and choice_a != truth
        pairs.append({"a": a["label"], "b": b["label"], "differ": differ, "only_a": only_a, "only_b": only_b})
    return {"n": len(rows), "cutoff": cutoff, "target_precision": target_precision, "rows": out, "pairs": pairs}


def _percent(value):
    return "-" if value is None else f"{value:.0%}"


def format_comparison(comparison):
    n, cutoff, target = comparison["n"], comparison["cutoff"], comparison["target_precision"]
    rows = comparison["rows"]
    width = max(5, *(len(row["label"]) for row in rows))
    lines = [f"--- Judges compared on {n} lines (cutoff {cutoff}; \"argmax\" = no cutoff, with its 95% interval) ---",
             f"  {'judge':<{width}}  {'accuracy':>8}{'coverage':>9}{'precision':>10}   {'argmax accuracy':>16}  {'95% interval':<13}{f'cov. at {target:.0%} prec.':>18}{'speed':>10}"]
    for row in rows:
        interval = "-" if row["raw_interval"] is None else f"{row['raw_interval'][0]:.0%}-{row['raw_interval'][1]:.0%}"
        lines.append(f"  {row['label']:<{width}}  {_percent(row['accuracy']):>8}{_percent(row['coverage']):>9}{_percent(row['precision']):>10}"
                     f"   {_percent(row['raw_accuracy']):>16}  {interval:<13}{_percent(row['coverage_at_target']):>18}{row['seconds_per_line']:>7.2f} s")
    lines += ["", "Judges:"]
    for row in rows:
        lines.append(f"  {row['label']}: {row['by']}" + (" (with the chat channel)" if row["use_channel"] else "") + (f" - {row['notes']}" if row["notes"] else ""))
        if row["missing"]:
            lines.append(f"    ! made on another sample: it lacks {row['missing']} lines of this one (counted as no answer): probe it again")
    if comparison["pairs"]:
        lines += ["", "Where two judges disagree (\"no answer\" counts as an answer; right = the label's root; no cutoff):"]
        for pair in comparison["pairs"]:
            a, b = pair["a"], pair["b"]
            lines.append(f"  {a} vs {b}: another answer on {pair['differ']} of {n} lines; {a} right, {b} wrong: {pair['only_a']}; "
                         f"{b} right, {a} wrong: {pair['only_b']}")
    lines += ["", f"On {n} lines a gap of a few points is noise (see the intervals): look at the right/wrong counts of the pairs, "
                  "and at the accuracy where it matters (precision at the cutoff you will use)."]
    return "\n".join(lines)
