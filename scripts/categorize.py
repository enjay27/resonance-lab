"""Gate 1, rule-based baseline: categorise every raw chat message into a root category.

    python scripts/categorize.py                       raw log -> data/processed/categories.jsonl (what `preprocess.py --recipe` reads)
    python scripts/categorize.py --probe [SAMPLE]      score the baseline on a hand-labelled sample; writes nothing

A line no rule recognises is left out of the file (it stays "uncategorized" for a recipe). The file has one row per distinct
line (variants such as `乙です` / `乙です！` are one key; the first occurrence decides): {"key", "category", "by"}.
Untranslated rows are categorised too: they are the new messages whose Korean is still to be written.
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from categorizer import BASELINE, categorize
from config import CATEGORIES_FILE, GATE1_SAMPLE, RAW_LOGS
from dataset_recipe import line_key
from gate_eval import SampleError, evaluate_categorizer, format_gate_report, read_sample


def categorise_raw(raw_path, out_path):
    """Write the categories file for the raw log; returns (lines per root, uncategorized lines, damaged rows)."""
    seen = set()
    per_root, uncategorized, damaged = {}, 0, 0
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(raw_path, encoding="utf-8") as raw, open(out_path, "w", encoding="utf-8") as out:
        for line in raw:
            try:
                row = json.loads(line)
            except ValueError:
                damaged += 1
                continue
            original = (row.get("original") or "").strip() if isinstance(row, dict) else ""
            if not original:
                if isinstance(row, dict):
                    continue
                damaged += 1
                continue
            key = line_key(original)
            if key in seen:
                continue
            seen.add(key)
            category = categorize(original)
            if category is None:
                uncategorized += 1
                continue
            per_root[category] = per_root.get(category, 0) + 1
            out.write(json.dumps({"key": key, "category": category, "by": BASELINE}, ensure_ascii=False) + "\n")
    return per_root, uncategorized, damaged


def report_distribution(per_root, uncategorized, damaged, out_path):
    total = sum(per_root.values()) + uncategorized
    print(f"\n--- Gate 1 ({BASELINE}): {total} distinct lines -> {out_path} ---")
    width = max([len("uncategorized"), *map(len, per_root)])
    for root, count in sorted(per_root.items(), key=lambda item: -item[1]):
        print(f"  {root:<{width}}  {count:>8}  {count / total:>6.1%}")
    print(f"  {'uncategorized':<{width}}  {uncategorized:>8}  {uncategorized / total if total else 0:>6.1%}   (no rule fired: left out of the file)")
    if damaged:
        print(f"{damaged} damaged row(s) skipped")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Categorise the raw chat log into root categories (rule-based baseline).")
    parser.add_argument("--raw", default=RAW_LOGS, help="raw log to categorise (default: %(default)s)")
    parser.add_argument("--out", default=CATEGORIES_FILE, help="categories file to write (default: %(default)s)")
    parser.add_argument("--probe", nargs="?", const=GATE1_SAMPLE, metavar="SAMPLE",
                        help="score the baseline on a hand-labelled sample instead (default sample: %s)" % GATE1_SAMPLE)
    args = parser.parse_args(argv)

    if args.probe is not None:
        try:
            rows = read_sample(args.probe)
        except SampleError as error:
            print(f"[ERROR] {error}")
            sys.exit(1)
        print(format_gate_report(evaluate_categorizer(rows, categorize), BASELINE))
        return

    if not os.path.exists(args.raw):
        print(f"[ERROR] Raw data not found at: {args.raw}")
        sys.exit(1)
    per_root, uncategorized, damaged = categorise_raw(args.raw, args.out)
    report_distribution(per_root, uncategorized, damaged, args.out)


if __name__ == "__main__":
    main()
