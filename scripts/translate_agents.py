"""Translate a season's chat lines with agents: prepare the batches, check what they wrote, revise, assemble.

    python scripts/translate_agents.py prepare  --season S1 --labels data/eval/gate1-labels.jsonl [--size 200] [--include-guild]
    python scripts/translate_agents.py check    --season S1 [--round N] [--batch B]  # every batch of the latest (or the given) round against its input and the glossary
    python scripts/translate_agents.py assemble --season S1                   # final.jsonl, terms.tsv, final.meta.json (exit 1 when a line breaks the glossary)
    python scripts/translate_agents.py revise   --season S1 [--all]           # the next round: the lines that break the glossary, with their `prev`
    python scripts/translate_agents.py report   --season S1                   # counts, flagged lines, terms rendered more than one way

The agents themselves are started by whoever runs this (`.claude/skills/season-data/SKILL.md`): one agent per batch file `round<N>/in/<batch>.jsonl`, the brief
`round<N>/brief.md` as its instructions, its answer in `round<N>/out/<batch>.jsonl`. Everything lives in data/translation/<season>/ (gitignored: players' chat).
`assemble` refuses when docs/translation-glossary.md is not the version the batches were prepared with. The glossary rules are configs/glossary/<season>.json.
"""

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glossary as glossary_lib
import translation_assemble as assemble
import translation_check as check
from config import BASE_DIR, GLOSSARY_DIR, GLOSSARY_DOC, TRANSLATION_DIR

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
    """The paths of one season and what is loaded for it."""

    def __init__(self, args):
        if not SEASON.match(args.season):
            fail(f"season {args.season!r} must be a plain name (letters, digits, '.', '_', '-'), e.g. S1")
        self.name, self.args = args.season, args
        self.dir = os.path.join(args.root, args.season)
        self.run_path = os.path.join(self.dir, "run.json")
        self.final_path = os.path.join(self.dir, "final.jsonl")

    def glossary(self):
        try:
            return glossary_lib.load_glossary(self.args.glossary or os.path.join(GLOSSARY_DIR, self.name.lower() + ".json"))
        except glossary_lib.GlossaryError as error:
            fail(error)

    def doc_version(self):
        try:
            return glossary_lib.doc_version(self.args.doc)
        except glossary_lib.GlossaryError as error:
            fail(error)

    def run_record(self):
        if not os.path.exists(self.run_path):
            fail(f"{self.run_path}: no run yet: `prepare` first")
        with open(self.run_path, encoding="utf-8") as f:
            return json.load(f)

    def round_dir(self, number):
        return os.path.join(self.dir, f"round{number}")

    def rounds(self, record, only=None):
        return sorted(int(n) for n in record["rounds"] if only is None or int(n) == only)

    def batch_paths(self, number, name):
        return os.path.join(self.round_dir(number), "in", name + ".jsonl"), os.path.join(self.round_dir(number), "out", name + ".jsonl")


def write_round(season, number, batches, brief_text):
    for name, rows in batches:
        write_jsonl(season.batch_paths(number, name)[0], rows)
    with open(os.path.join(season.round_dir(number), "brief.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(brief_text)


def brief_for(season, number):
    with open(season.args.doc, encoding="utf-8") as f:
        return assemble.build_brief(f.read(), number)


def prepare(args):
    season = Season(args)
    glossary = season.glossary()
    if os.path.exists(season.round_dir(1)):
        fail(f"{season.round_dir(1)} exists: it holds the agents' work; delete it to prepare again")
    labelled, problems = check.read_jsonl(args.labels)
    if problems:
        fail(f"{args.labels}: {problems[0]}")
    skip = tuple(s for s in assemble.DEFAULT_SKIP if not (args.include_guild and s == "recruitment/guild")) if args.skip is None else tuple(args.skip.split(","))
    try:
        rows, skipped = assemble.select_lines(labelled, skip)
        batches = assemble.split_batches(rows, args.size, args.prefix)
    except assemble.AssembleError as error:
        fail(error)
    write_round(season, 1, batches, brief_for(season, 1))
    record = assemble.run_record(season.name, season.doc_version(), glossary_lib.file_sha1(args.glossary or os.path.join(GLOSSARY_DIR, season.name.lower() + ".json")),
                                 args.size, 1, {"lines": len(rows), "skipped": skipped}, [name for name, _ in batches], git_sha())
    record["glossary_season"] = glossary.season
    write_json(season.run_path, record)
    print(f"{len(rows)} lines in {len(batches)} batches in {season.round_dir(1)}; skipped {sum(skipped.values())} {skipped or ''}".rstrip())
    print(f"brief: {os.path.join(season.round_dir(1), 'brief.md')} (glossary document {record['doc_version']}); outputs go to round1/out/<batch>.jsonl")
    return 0


def read_inputs(season, number, name):
    path = season.batch_paths(number, name)[0]
    rows, problems = check.read_jsonl(path)
    if problems:
        fail(f"{path}: {problems[0]}")
    return {r["i"]: r for r in rows}


def check_command(args):
    season = Season(args)
    glossary, record = season.glossary(), season.run_record()
    bad, checked = 0, 0
    for number in season.rounds(record, args.round) if args.round else [max(season.rounds(record))]:  # default: the latest round (earlier ones are superseded)
        for name in record["rounds"][str(number)]:
            if args.batch and name != args.batch:
                continue
            checked += 1
            outputs, problems = check.read_jsonl(season.batch_paths(number, name)[1])
            if problems and not outputs:
                print(f"round {number} {name}: no output yet ({problems[0]})")
                bad += 1
                continue
            problems += check.check_output(read_inputs(season, number, name), outputs, glossary)
            if problems:
                bad += 1
                print(f"round {number} {name}: {len(problems)} problem(s):")
                for problem in problems[:60]:
                    print("  ", problem)
            else:
                print(f"round {number} {name}: OK ({len(outputs)} lines, {sum(bool(r.get('flag')) for r in outputs)} flagged)")
    if not checked:
        fail(f"no batch named {args.batch!r} in the round (batches: {', '.join(record['rounds'][str(max(season.rounds(record), default=1))])})")
    if bad:
        fail(f"{bad} batch(es) with problems")
    return 0


def assemble_command(args):
    season = Season(args)
    glossary, record = season.glossary(), season.run_record()
    try:
        assemble.require_current(record, season.doc_version())
    except assemble.StaleBatches as error:
        fail(error)
    inputs, rounds = {}, []
    for number in season.rounds(record):
        outputs = []
        for name in record["rounds"][str(number)]:
            batch_inputs = read_inputs(season, number, name)
            inputs.update({i: r for i, r in batch_inputs.items() if number == 1})
            rows, problems = check.read_jsonl(season.batch_paths(number, name)[1])
            if problems and not rows:
                fail(f"round {number} {name}: no output yet: {problems[0]}")
            outputs += rows
        rounds.append(outputs)
    try:
        rows, corrections = assemble.assemble(inputs, rounds, glossary)
    except assemble.AssembleError as error:
        fail(error)
    problems = check.check_lines(rows, glossary)
    table = assemble.term_table(rows)
    write_jsonl(season.final_path, rows)
    with open(os.path.join(season.dir, "terms.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write(assemble.format_terms_tsv(table))
    meta = {"season": season.name, "doc_version": record["doc_version"], "glossary_sha1": record["glossary_sha1"], "git_sha": record["git_sha"],
            "rounds": record["rounds"], "lines": len(rows), "flagged": sum(bool(r.get("flag")) for r in rows), "problems": len(problems),
            "corrections": dict(corrections), "terms": len(table), "terms_with_more_than_one_rendering": len(assemble.multiple_renderings(table))}
    write_json(os.path.join(season.dir, "final.meta.json"), meta)
    print(f"{len(rows)} lines, {meta['flagged']} flagged, {len(table)} terms; corrections: {dict(corrections) or 'none'}")
    print(f"written: {season.final_path}, terms.tsv, final.meta.json (glossary document {record['doc_version']})")
    if problems:
        print(f"{len(problems)} glossary problem(s):")
        for problem in problems[:60]:
            print("  ", problem)
        fail("fix them (`revise` makes the next round) and assemble again")
    return 0


def revise(args):
    season = Season(args)
    glossary, record = season.glossary(), season.run_record()
    if not os.path.exists(season.final_path):
        fail(f"{season.final_path}: `assemble` first")
    final, problems = check.read_jsonl(season.final_path)
    wrong = {int(re.match(r"i=(\d+)", p).group(1)) for p in check.check_lines(final, glossary)} if not args.all else {r["i"] for r in final}
    chosen = [r for r in final if r["i"] in wrong]
    if not chosen:
        print("nothing to revise: every line passes the glossary check")
        return 0
    number = max(season.rounds(record)) + 1
    rows = [{"i": r["i"], "ch": r["channel"], "cat": r["category"], "ja": r["original"], "prev": r["translated"]} for r in chosen]
    batches = assemble.split_batches(rows, args.size or record["batch_size"], "rev")
    try:
        record = assemble.add_round(record, number, [name for name, _ in batches], season.doc_version())
    except assemble.StaleBatches as error:
        fail(error)
    write_round(season, number, batches, brief_for(season, number))
    write_json(season.run_path, record)
    print(f"round {number}: {len(rows)} lines in {len(batches)} batches in {season.round_dir(number)}")
    return 0


def report(args):
    season = Season(args)
    if not os.path.exists(season.final_path):
        fail(f"{season.final_path}: `assemble` first")
    rows, _ = check.read_jsonl(season.final_path)
    table = assemble.term_table(rows)
    print(f"lines: {len(rows)} | flagged: {sum(bool(r.get('flag')) for r in rows)} | distinct game terms: {len(table)}")
    for r in rows:
        if r.get("flag"):
            print(f"  flag i={r['i']}: {r['flag'][:100]}")
    many = assemble.multiple_renderings(table)
    print(f"terms with more than one rendering: {len(many)}")
    for ja in many:
        print(f"  {ja}: " + " | ".join(f"{ko} ({n})" for ko, n in table[ja].items()))
    return 0


COMMANDS = {"prepare": prepare, "check": check_command, "assemble": assemble_command, "revise": revise, "report": report}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--season", required=True, help="a plain name, e.g. S1: the work lives in <root>/<season>/")
    parser.add_argument("--root", default=TRANSLATION_DIR, help="where the seasons live (default: %(default)s)")
    parser.add_argument("--glossary", help="the glossary JSON (default: configs/glossary/<season>.json)")
    parser.add_argument("--doc", default=GLOSSARY_DOC, help="the glossary document (default: %(default)s)")
    parser.add_argument("--labels", help="prepare: the labelled lines (JSONL of {original, category, channel})")
    parser.add_argument("--size", type=int, default=None, help="prepare/revise: at most this many lines per batch (default: 200 / the run's)")
    parser.add_argument("--prefix", default="batch", help="prepare: the batch names (default: %(default)s)")
    parser.add_argument("--include-guild", action="store_true", help="prepare: also translate guild adverts (skipped by default: the guild is Korean)")
    parser.add_argument("--skip", help="prepare: categories not to translate, comma separated (default: guild, non_japanese, other/placeholder)")
    parser.add_argument("--round", type=int, help="check: this round (default: the latest)")
    parser.add_argument("--batch", help="check: only this batch (an agent checks its own output, not the others')")
    parser.add_argument("--all", action="store_true", help="revise: every line, not only the ones that break the glossary")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if not args.labels:
            parser.error("prepare needs --labels")
        args.size = args.size or 200
    return COMMANDS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
