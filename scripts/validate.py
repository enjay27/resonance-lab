import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS, VALIDATE_MAX_HANGEUL_IN_ORIGINAL, VALIDATE_MAX_STRUCTURE_ERRORS
from text_rules import HANGEUL_PATTERN

MAX_STRUCTURE_ERRORS = VALIDATE_MAX_STRUCTURE_ERRORS
MAX_HANGEUL_IN_ORIGINAL = VALIDATE_MAX_HANGEUL_IN_ORIGINAL
LINES_SHOWN = 5  # offending lines printed per problem; the counts cover all of them


class ValidationError(Exception):
    """The raw log is unusable; stops run_pipeline.py at the Validate stage."""


def _is_row(data):
    """A line the app wrote: an object with a text `original` and a `translated` (null while untranslated)."""
    return isinstance(data, dict) and isinstance(data.get("original"), str) and "translated" in data


def validate_raw_logs(file_path, max_structure_errors=MAX_STRUCTURE_ERRORS, max_hangeul=MAX_HANGEUL_IN_ORIGINAL):
    """Check the raw log's shape and return its stats; raise ValidationError when it is not a usable log.

    This is a sanity gate, not the cleaning: preprocess.py drops the bad rows one by one, so a few stray lines
    only get reported. The run stops when the file is empty or missing, when more than `max_structure_errors`
    of its lines are damaged (invalid JSON, missing fields), or when more than `max_hangeul` of its rows have
    Hangeul in the Japanese `original` (the wrong file, or the languages swapped).
    """
    try:
        f = open(file_path, 'r', encoding='utf-8')
    except FileNotFoundError:
        raise ValidationError(f"raw log not found: {file_path}") from None

    print(f"Checking {file_path} ...")
    stats = {"rows": 0, "structure_errors": 0, "hangeul_in_original": 0}
    shown = {"structure": 0, "hangeul": 0}
    with f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            stats["rows"] += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = None
                problem, message = "structure", "Invalid JSON format."
            else:
                problem, message = ("structure", "Missing or malformed fields.") if not _is_row(data) else (None, None)
            if problem is None and HANGEUL_PATTERN.search(data["original"]):
                stats["hangeul_in_original"] += 1
                problem, message = "hangeul", f"Hangeul in original: {data['original']}"
            elif problem == "structure":
                stats["structure_errors"] += 1
            if problem and shown[problem] < LINES_SHOWN:
                shown[problem] += 1
                print(f"[Line {number}] {message}")

    rows = stats["rows"]
    if rows == 0:
        raise ValidationError(f"{file_path} has no rows")
    structure, hangeul = stats["structure_errors"] / rows, stats["hangeul_in_original"] / rows
    print(f"Rows: {rows} | damaged (structure): {stats['structure_errors']}/{rows} ({structure:.1%}, limit {max_structure_errors:.0%})"
          f" | Hangeul in original: {stats['hangeul_in_original']}/{rows} ({hangeul:.1%}, limit {max_hangeul:.0%})")
    if structure > max_structure_errors:
        raise ValidationError(f"{structure:.1%} of the lines are damaged (structure), the limit is {max_structure_errors:.0%}: "
                              "is this the app's dataset_<CHANNEL>.jsonl?")
    if hangeul > max_hangeul:
        raise ValidationError(f"{hangeul:.1%} of the originals contain Hangeul, the limit is {max_hangeul:.0%}: "
                              "wrong file, or source and translation swapped?")
    print("✅ Raw log OK (preprocess.py drops the individual bad rows).")
    return stats


def main():
    try:
        validate_raw_logs(RAW_LOGS)
    except ValidationError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
