"""Label a season's chat lines with agents: prepare the batches, check what they wrote, assemble the labels and the judge's sample.

    python scripts/label_lines.py prepare  --season S1 --raw data/raw/raw_translated_logs.jsonl [more logs] [--size 300]
    python scripts/label_lines.py check    --season S1 [--batch B]    # every batch against its input and the labeling guide's categories
    python scripts/label_lines.py assemble --season S1                 # labels.jsonl + judge-sample.jsonl + labels.meta.json (the counts table for the guide)
    python scripts/label_lines.py export   --season S1                 # judge-sample.jsonl again from labels.jsonl (after labels were corrected by hand)
    python scripts/label_lines.py report   --season S1                 # the counts table again

The agents are started by whoever runs this (`.claude/skills/season-data/SKILL.md`): one per batch file `round1/in/<batch>.jsonl`, the brief `round1/brief.md` as its
instructions, its answer in `round1/out/<batch>.jsonl`. Everything lives in data/labeling/<season>/ (gitignored: players' chat). `assemble` refuses when
docs/labeling-guide.md is not the version the batches were prepared with. The judge's sample is for `categorize.py --probe` / `compare_judges.py`: copy it to
data/eval/gate1-sample.jsonl yourself (nothing here overwrites the hand-labelled sample).
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glossary as glossary_lib
import labeling_tools as tools
import translation_assemble as assemble
import translation_check as check
from config import BASE_DIR, LABEL_MAP, LABELING_DIR, LABELING_GUIDE_DOC
from taxonomy import TaxonomyError, load_taxonomy

SEASON = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def fail(message):
    print(f"[ERROR] {message}")
    sys.exit(1)


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path, data):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.write("\n")


def git_sha():
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=BASE_DIR, capture_output=True, text=True, timeout=10)
        return done.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Season:
    def __init__(self, args):
        if not SEASON.match(args.season):
            fail(f"season {args.season!r} must be a plain name (letters, digits, '.', '_', '-'), e.g. S1")
        self.name, self.args = args.season, args
        self.dir = os.path.join(args.root, args.season)
        self.run_path = os.path.join(self.dir, "run.json")
        self.lines_path = os.path.join(self.dir, "lines.jsonl")
        self.labels_path = os.path.join(self.dir, "labels.jsonl")
        self.sample_path = os.path.join(self.dir, "judge-sample.jsonl")
        self.in_dir, self.out_dir = os.path.join(self.dir, "round1", "in"), os.path.join(self.dir, "round1", "out")

    def taxonomy(self):
        try:
            return load_taxonomy()
        except TaxonomyError as error:
            fail(error)

    def label_map(self):
        try:
            return tools.load_label_map(self.args.label_map, self.taxonomy())
        except tools.LabelError as error:
            fail(error)

    def doc_version(self):
        try:
            return glossary_lib.doc_version(self.args.guide)
        except glossary_lib.GlossaryError as error:
            fail(error)

    def run_record(self):
        if not os.path.exists(self.run_path):
            fail(f"{self.run_path}: no run yet: `prepare` first")
        with open(self.run_path, encoding="utf-8") as f:
            return json.load(f)

    def read_lines(self):
        lines, problems = check.read_jsonl(self.lines_path)
        if problems:
            fail(f"{self.lines_path}: {problems[0]}")
        return lines


def prepare(args):
    season = Season(args)
    if os.path.exists(season.run_path):
        fail(f"{season.run_path} exists: it holds the agents' work; delete {season.dir} to prepare again")
    label_map, raw = season.label_map(), []
    for path in args.raw:
        rows, problems = check.read_jsonl(path)
        if problems and not rows:
            fail(f"{path}: {problems[0]}")
        raw += rows
    lines = tools.distinct_lines(raw)
    if not lines:
        fail("no lines with an `original` text in the raw logs")
    try:
        batches = assemble.split_batches(tools.input_rows(lines), args.size, "batch")
    except assemble.AssembleError as error:
        fail(error)
    write_jsonl(season.lines_path, lines)
    for name, rows in batches:
        write_jsonl(os.path.join(season.in_dir, name + ".jsonl"), rows)
    with open(args.guide, encoding="utf-8") as f:
        brief = tools.build_brief(f.read())
    with open(os.path.join(season.dir, "round1", "brief.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(brief)
    write_json(season.run_path, {"season": season.name, "doc_version": season.doc_version(), "label_map_sha1": glossary_lib.file_sha1(args.label_map),
                                 "label_map_season": label_map.season, "git_sha": git_sha(), "batch_size": args.size, "counts": {"raw": len(raw), "distinct": len(lines)},
                                 "batches": [name for name, _ in batches]})
    print(f"{len(raw)} raw lines, {len(lines)} distinct lines in {len(batches)} batches in {season.in_dir}")
    print(f"brief: {os.path.join(season.dir, 'round1', 'brief.md')} (labeling guide {season.doc_version()}); outputs go to round1/out/<batch>.jsonl")
    return 0


def read_inputs(season, name):
    rows, problems = check.read_jsonl(os.path.join(season.in_dir, name + ".jsonl"))
    if problems:
        fail(f"batch {name}: {problems[0]}")
    return {r["i"]: r for r in rows}


def check_command(args):
    season = Season(args)
    taxonomy, label_map, record = season.taxonomy(), season.label_map(), season.run_record()
    bad, names = 0, [n for n in record["batches"] if not args.batch or n == args.batch]
    if not names:
        fail(f"no batch named {args.batch!r} (batches: {', '.join(record['batches'])})")
    for name in names:
        outputs, problems = check.read_jsonl(os.path.join(season.out_dir, name + ".jsonl"))
        if problems and not outputs:
            print(f"{name}: no output yet ({problems[0]})")
            bad += 1
            continue
        problems += tools.check_output(read_inputs(season, name), outputs, label_map, taxonomy)
        if problems:
            bad += 1
            print(f"{name}: {len(problems)} problem(s):")
            for problem in problems[:60]:
                print("  ", problem)
        else:
            print(f"{name}: OK ({len(outputs)} lines, {sum(bool(r.get('unsure')) for r in outputs)} unsure)")
    if bad:
        fail(f"{bad} batch(es) with problems")
    return 0


def write_sample(season, rows, label_map, dev_fraction):
    try:
        sample, excluded = tools.judge_sample(rows, label_map, season.taxonomy(), dev_fraction)
    except tools.LabelError as error:
        fail(error)
    write_jsonl(season.sample_path, sample)
    return sample, excluded


def finish(season, rows, label_map, record, dev_fraction):
    sample, excluded = write_sample(season, rows, label_map, dev_fraction)
    unsure = sum(bool(r.get("unsure")) for r in rows)
    write_json(os.path.join(season.dir, "labels.meta.json"), {
        "season": season.name, "doc_version": record["doc_version"], "label_map_sha1": record["label_map_sha1"], "git_sha": record["git_sha"],
        "lines": len(rows), "unsure": unsure, "judge_sample": len(sample), "excluded": excluded,
        "dev": sum(r["split"] == "dev" for r in sample), "test": sum(r["split"] == "test" for r in sample)})
    print(tools.format_counts(tools.label_counts(rows), unsure))
    print(f"judge sample: {len(sample)} lines ({sum(r['split'] == 'dev' for r in sample)} dev / {sum(r['split'] == 'test' for r in sample)} test), excluded: {excluded or 'none'}")
    print(f"written: {season.labels_path}, {season.sample_path}, labels.meta.json")


def assemble_command(args):
    season = Season(args)
    taxonomy, label_map, record = season.taxonomy(), season.label_map(), season.run_record()
    try:
        assemble.require_current(record, season.doc_version())
    except assemble.StaleBatches as error:
        fail(error)
    outputs = []
    for name in record["batches"]:
        rows, problems = check.read_jsonl(os.path.join(season.out_dir, name + ".jsonl"))
        if problems and not rows:
            fail(f"{name}: no output yet: {problems[0]}")
        found = tools.check_output(read_inputs(season, name), rows, label_map, taxonomy)
        if found:
            fail(f"{name}: {found[0]} (run `check`)")
        outputs += rows
    try:
        # `i` is global: input_rows numbered all the distinct lines once, whatever batch a line went into
        labelled = tools.assemble(season.read_lines(), outputs)
    except tools.LabelError as error:
        fail(error)
    write_jsonl(season.labels_path, labelled)
    finish(season, labelled, label_map, record, args.dev_fraction)
    return 0


def export(args):
    season = Season(args)
    label_map, record = season.label_map(), season.run_record()
    rows, problems = check.read_jsonl(season.labels_path)
    if problems:
        fail(f"{season.labels_path}: {problems[0]}")
    finish(season, rows, label_map, record, args.dev_fraction)
    return 0


def report(args):
    season = Season(args)
    rows, problems = check.read_jsonl(season.labels_path)
    if problems:
        fail(f"{season.labels_path}: {problems[0]}")
    print(tools.format_counts(tools.label_counts(rows), sum(bool(r.get("unsure")) for r in rows)))
    return 0


COMMANDS = {"prepare": prepare, "check": check_command, "assemble": assemble_command, "export": export, "report": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--season", required=True, help="a plain name, e.g. S1: the work lives in <root>/<season>/")
    parser.add_argument("--root", default=LABELING_DIR, help="where the seasons live (default: %(default)s)")
    parser.add_argument("--guide", default=LABELING_GUIDE_DOC, help="the labeling guide (default: %(default)s)")
    parser.add_argument("--label-map", default=LABEL_MAP, help="the label map (default: %(default)s)")
    parser.add_argument("--raw", nargs="+", help="prepare: the raw chat logs (JSONL with `original` and optionally `channel`)")
    parser.add_argument("--batch", help="check: only this batch (an agent checks its own output, not the others')")
    parser.add_argument("--size", type=int, default=300, help="prepare: at most this many lines per batch (default: %(default)s)")
    parser.add_argument("--dev-fraction", type=float, default=tools.DEV_FRACTION, help="assemble/export: the share of the judge sample that is dev (default: %(default)s)")
    args = parser.parse_args(argv)
    if args.command == "prepare" and not args.raw:
        parser.error("prepare needs --raw")
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
