"""Eval metrics shared by every pipeline's eval stage, so their numbers compare.

A pipeline's `eval.py` only generates predictions (that part needs its model and a GPU); the scoring
and the printed report live here and are unit-tested. Samples are the rows of the eval dataset:
{"original": Japanese, "translated": reference Korean, "category": optional}.
"""

import json
import re

from text_rules import JP_PATTERN

# Localized game terms the model must use: Japanese term in the source -> Korean term expected.
TERM_DICT = {
    '消化': '숙제',
    '完凸': '풀돌',
    '火力': '딜러',
    'ファスト': '속공',
    '器用': '숙련',
    'リキャスト': '쿨타임',
    '盾': '탱커',
    '杖': '법사',
    '弓': '궁수',
    'ウルト': '궁',
    'イマジン': '이매진',
    'ガシャ': '뽑기',
    'ばんわ': '존밤',
    'ヒグマ': '산적 두목',
    'ムークボス': '무크 두목',
}

MISSES_SHOWN = 5


# --- the eval dataset ----------------------------------------------------------------------------


def load_eval_dataset(path):
    """The eval rows of a JSONL file; every row needs `original` and `translated`."""
    try:
        f = open(path, encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(f"eval dataset not found: {path}  (one JSON per line: original, translated[, category])") from None
    rows = []
    with f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise ValueError(f"{path} line {number}: not valid JSON ({e})") from None
            if not isinstance(row, dict) or "original" not in row or "translated" not in row:
                raise ValueError(f"{path} line {number}: needs both 'original' and 'translated'")
            rows.append(row)
    return rows


# --- text rules ----------------------------------------------------------------------------------


def strip_think(text):
    """Drop an empty <think></think> block some chat templates emit, and surrounding space."""
    return re.sub(r'<think>\s*</think>\s*', '', text).strip()


def has_jp(text):
    return bool(JP_PATTERN.search(text))


def think_leaked(raw_output):
    """True when the model wrote real reasoning inside <think>, which must not reach the user."""
    if '<think>' not in raw_output:
        return False
    return len(raw_output.split('<think>')[1].split('</think>')[0].strip()) > 0


def term_results(jp, prediction, term_dict=TERM_DICT):
    """(Japanese term, expected Korean term, used?) for every term that occurs in the source."""
    return [(jp_term, ko_term, ko_term in prediction) for jp_term, ko_term in term_dict.items() if jp_term in jp]


def discord_violation(jp, prediction):
    """Discord is a product name: it must stay Latin, not become 디스코드."""
    return 'discord' in jp.lower() and '디스코드' in prediction


# --- standard metrics ------------------------------------------------------------------------------


def standard_metrics(predictions, references):
    """BLEU, chrF and TER of the whole run (`references`: one reference string per prediction).

    sacrebleu wants ONE reference set -- a list of all the references -- not one list per sample.
    experiment/translategemma's eval.py passed `[[r1], [r2], ...]`, which scores only the first
    line (see the regression test), so its BLEU/chrF/TER numbers are not comparable with these.
    """
    import sacrebleu

    reference_sets = [list(references)]
    return {
        "bleu": sacrebleu.corpus_bleu(predictions, reference_sets).score,
        "chrf": sacrebleu.corpus_chrf(predictions, reference_sets).score,
        "ter": sacrebleu.corpus_ter(predictions, reference_sets).score,
    }


def comet_score(samples, predictions):
    """System-level COMET (wmt22-comet-da), or None when it is not installed or fails.

    Needs `unbabel-comet` and downloads a model; not unit-tested beyond the missing-package path.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        return None
    try:
        import torch

        model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))
        data = [{"src": s["original"], "mt": p, "ref": s["translated"]} for s, p in zip(samples, predictions)]
        return model.predict(data, batch_size=8, gpus=1 if torch.cuda.is_available() else 0).system_score
    except Exception as e:  # a failed download / OOM must not lose the other metrics
        print(f"COMET failed: {e}")
        return None


# --- evaluate + report ---------------------------------------------------------------------------


def evaluate(samples, predictions, raw_outputs=None):
    """Score a run. `raw_outputs` are the undecoded generations (to catch <think>); default: predictions."""
    raw_outputs = predictions if raw_outputs is None else raw_outputs
    if not samples:
        raise ValueError("the eval dataset is empty")
    if len(samples) != len(predictions):
        raise ValueError(f"{len(samples)} samples but {len(predictions)} predictions")

    report = {
        "n": len(samples),
        "jp_leakage": 0,
        "think_leakage": sum(think_leaked(raw) for raw in raw_outputs),
        "term_hits": 0,
        "term_total": 0,
        "term_misses": [],
        "discord_violations": 0,
        "exact_match": 0,
        "categories": {},
        "standard": standard_metrics(predictions, [s["translated"] for s in samples]),
    }
    for sample, prediction in zip(samples, predictions):
        jp, ref = sample["original"], sample["translated"]
        category = report["categories"].setdefault(
            sample.get("category", "unknown"), {"total": 0, "jp_leak": 0, "term_miss": 0, "discord_viol": 0}
        )
        category["total"] += 1

        if has_jp(prediction):
            report["jp_leakage"] += 1
            category["jp_leak"] += 1
        if discord_violation(jp, prediction):
            report["discord_violations"] += 1
            category["discord_viol"] += 1
        if prediction == ref:
            report["exact_match"] += 1
        for jp_term, ko_term, used in term_results(jp, prediction):
            report["term_total"] += 1
            if used:
                report["term_hits"] += 1
            else:
                category["term_miss"] += 1
                report["term_misses"].append({"jp": jp, "pred": prediction, "ref": ref, "term": jp_term, "expected": ko_term})
    return report


def _share(count, n):
    return f"{count}/{n} ({count / n * 100:.1f}%)"


def format_report(report, samples, predictions, comet=None):
    """The report as text: metrics, per-category counts and the full output log."""
    n = report["n"]
    std = report["standard"]
    lines = [
        "--- Standard Metrics ---",
        f"BLEU  : {std['bleu']:.2f}  (higher is better, 0-100)",
        f"chrF  : {std['chrf']:.2f}  (higher is better, 0-100)",
        f"TER   : {std['ter']:.2f}  (lower is better, 0-100)",
        "",
        "--- COMET Score ---",
        f"COMET : {comet:.4f}  (higher is better, 0-1, >0.85 is good)" if comet is not None else "COMET : not available (pip install unbabel-comet)",
        "",
        "--- Custom Metrics ---",
        f"JP Leakage    : {_share(report['jp_leakage'], n)} -- lower is better",
        f"Think Leakage : {_share(report['think_leakage'], n)} -- lower is better",
    ]
    if report["term_total"]:
        lines.append(f"Term Accuracy : {_share(report['term_hits'], report['term_total'])} -- higher is better")
        if report["term_misses"]:
            lines.append(f"\n  Term Misses (first {MISSES_SHOWN}):")
            for miss in report["term_misses"][:MISSES_SHOWN]:
                lines += [
                    f"    JP  : {miss['jp']}",
                    f"    REF : {miss['ref']}",
                    f"    PRED: {miss['pred']}",
                    f"    Expected '{miss['expected']}' for '{miss['term']}'",
                    "",
                ]
    else:
        lines.append("Term Accuracy : No term-containing samples in eval set")
    lines += [
        f"Discord Kept Latin: {report['discord_violations']} violations (디스코드 where Discord was written)",
        f"Exact Match   : {_share(report['exact_match'], n)}",
        "",
        "--- Category Breakdown ---",
    ]
    for name, stats in report["categories"].items():
        lines += [
            f"\n  [{name}] ({stats['total']} samples)",
            f"    JP Leakage     : {stats['jp_leak']}/{stats['total']}",
            f"    Term Misses    : {stats['term_miss']}/{stats['total']}",
            f"    Discord Viol   : {stats['discord_viol']}/{stats['total']}",
        ]
    lines += ["", "--- Full Output Log ---"]
    for sample, prediction in zip(samples, predictions):
        flags = []
        if has_jp(prediction):
            flags.append("⚠ JP")
        if discord_violation(sample["original"], prediction):
            flags.append("⚠ discord")
        lines += [
            f"[{sample.get('category', '?')}]",
            f"  JP  : {sample['original']}",
            f"  REF : {sample['translated']}",
            f"  PRED: {prediction}  {' '.join(flags) if flags else 'OK'}",
            "",
        ]
    return "\n".join(lines)
