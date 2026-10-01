import argparse
import contextlib
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS, PROCESSED_LOGS, INSTRUCTION, EVAL_DATASET_PATH
from eval_metrics import load_eval_dataset
import manifest
from overlap import EvalOverlap
from valsplit import is_validation
from prompts import STYLES, build_prompt, style_for_template
from text_rules import HANGEUL_PATTERN, JP_PATTERN

# --- Filters (from the TranslateGemma pipeline's preprocess) ---

# A row is skipped for exactly one of these, checked in this order.
REASONS = (
    "empty field",
    "hangeul in original",
    "JP residual in translation",
    "JP in original (ko_ja)",
    "Hangeul residual in translation (ko_ja)",
    "hallucination",
    "recruitment spam",
    "eval overlap",
    "eval overlap (near)",
    "duplicate",
    "json error",
)


# Output row layouts, by pipeline: unsloth trains on instruction/input/output rows,
# LLaMA-Factory reads original/translated columns (see update_dataset_info.py).
FORMATS = {
    "instruction": lambda original, translated: {"instruction": INSTRUCTION, "input": original, "output": translated},
    "pair": lambda original, translated: {"original": original, "translated": translated},
}
DEFAULT_FORMAT = "instruction"


def forward_row(original, translated, style=None):
    """The ja->ko `pair` row with the instruction of `style` (None = the raw line)."""
    return {"original": build_prompt(style, "ja-ko", original), "translated": translated}


def reverse_row(original, translated, style=None):
    """The ko->ja row of the same pair: the Korean is the input, the Japanese the answer."""
    return {"original": build_prompt(style, "ko-ja", translated), "translated": original}


def pair_rows(original, translated, style=None, reverse=False):
    """The forward row, and with `reverse` the ko->ja row after it (no filtering here)."""
    rows = [forward_row(original, translated, style)]
    if reverse:
        rows.append(reverse_row(original, translated, style))
    return rows


def clean_reason(original, translated, direction="ja-ko"):
    """Why a row must not be trained on, or None when it is clean.

    Duplicates are not decided here (they need the rows seen so far).
    """
    original = (original or "").strip()
    translated = (translated or "").strip()
    # resonance-stream writes `translated: null` for lines it never translated.
    if not original or not translated:
        return "empty field"
    if direction == "ko-ja":
        # Reverse rows (Korean in, Japanese out): kana/kanji in the Korean input, or Hangeul left in the Japanese answer.
        if JP_PATTERN.search(original):
            return "JP in original (ko_ja)"
        if HANGEUL_PATTERN.search(translated):
            return "Hangeul residual in translation (ko_ja)"
    else:
        # The source is Japanese chat; Hangeul means a wrong row.
        if HANGEUL_PATTERN.search(original):
            return "hangeul in original"
        # Kana/kanji left in the Korean output: the line was not translated.
        if JP_PATTERN.search(translated):
            return "JP residual in translation"
    # Looping output: the translation is more than 10x longer than the input.
    if len(translated) > len(original) * 10:
        return "hallucination"
    # Guild recruitment walls: long lines with several IDs.
    if len(original) > 150 and original.count('ID:') > 1:
        return "recruitment spam"
    return None


def _report(counts):
    skipped = counts["total"] - counts["passed"]
    print("\n--- Preprocessing Report ---")
    width = max(len(reason) for reason in REASONS)
    print(f"{'Total input':<{width + 2}}: {counts['total']}")
    print(f"{'Passed':<{width + 2}}: {counts['passed']}")
    print(f"{'Skipped':<{width + 2}}: {skipped}")
    for reason in REASONS:
        print(f"  {reason:<{width}}: {counts[reason]}")
    if counts["validation"]:
        print(f"{'Validation rows':<{width + 2}}: {counts['validation']} of the passed rows (the rest are training rows)")


def transform_for_lora(input_file, output_file, fmt=DEFAULT_FORMAT, style=None, reverse=False, eval_originals=(),
                       val_file=None, val_fraction=0.0):
    """Clean raw {original, translated} rows into training rows of layout `fmt`.

    Rows whose original is one of `eval_originals` (exactly or nearly: see overlap.py) are dropped, so the
    model is never trained on what it is evaluated on; their reverse rows go with them.
    With `val_file` and `val_fraction`, about that share of the lines is written to `val_file` instead of `output_file`,
    chosen per line by valsplit.py (both directions of a pair go to the same file).
    Returns the counts per outcome (`total`, `passed` -- training and validation rows together --, `validation`
    and one per reason).
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose one of: {', '.join(sorted(FORMATS))}")
    if (style or reverse) and fmt != "pair":
        raise ValueError("a prompt style or --reverse needs the pair format (--format pair)")
    if val_file and fmt != "pair":
        raise ValueError("a validation split needs the pair format (--format pair); the instruction format is split by unsloth/split_dataset.py")
    if not os.path.exists(input_file):
        print(f"[ERROR] Raw data not found at: {input_file}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    counts = {"total": 0, "passed": 0, **{reason: 0 for reason in REASONS}, "validation": 0}
    seen_inputs = set()
    eval_overlap = EvalOverlap(eval_originals)

    with contextlib.ExitStack() as files:
        f_in = files.enter_context(open(input_file, 'r', encoding='utf-8'))
        f_out = files.enter_context(open(output_file, 'w', encoding='utf-8'))
        f_val = files.enter_context(open(val_file, 'w', encoding='utf-8')) if val_file else None
        for line in f_in:
            counts["total"] += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                counts["json error"] += 1
                continue
            original = (data.get("original") or "").strip()
            translated = (data.get("translated") or "").strip()

            reason = clean_reason(original, translated)
            if reason is None:
                found = eval_overlap.check(original)
                if found:
                    reason = "eval overlap" if found == "exact" else "eval overlap (near)"
            if reason is None and original in seen_inputs:
                reason = "duplicate"  # the first occurrence wins
            if reason:
                counts[reason] += 1
                continue

            seen_inputs.add(original)
            counts["passed"] += 1
            # The pair's side is decided by its Japanese line, so the reverse row below follows the forward row.
            sink = f_val if f_val and is_validation(original, val_fraction) else f_out
            counts["validation"] += sink is f_val
            if fmt == "pair":
                sink.write(json.dumps(forward_row(original, translated, style), ensure_ascii=False) + '\n')
            else:
                sink.write(json.dumps(FORMATS[fmt](original, translated), ensure_ascii=False) + '\n')

            # The reverse direction is tried only for rows whose forward direction passed, with its own filters and
            # counted on its own (as experiment/translategemma's bidirectional preprocess did).
            if reverse:
                counts["total"] += 1
                reason = clean_reason(translated, original, direction="ko-ja")
                if reason is None and translated in seen_inputs:
                    reason = "duplicate"
                if reason:
                    counts[reason] += 1
                    continue
                seen_inputs.add(translated)
                counts["passed"] += 1
                counts["validation"] += sink is f_val
                sink.write(json.dumps(reverse_row(original, translated, style), ensure_ascii=False) + '\n')

    _report(counts)
    if counts["passed"] == 0:
        print("[ERROR] Preprocessing produced 0 lines. Check your raw input file.")
        sys.exit(1)
    print(f"Successfully preprocessed {counts['passed']} lines.")
    return counts


def _profile_template(model):
    """The LLaMA-Factory template of the model profile (needs pyyaml; only --prompt auto gets here)."""
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "llamafactory"))
    from lf_tools import load_profile, model_name
    return load_profile(model_name(model)).template


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clean the raw chat log into training rows.")
    parser.add_argument("--format", choices=sorted(FORMATS), default=DEFAULT_FORMAT, dest="fmt",
                        help="row layout (default: %(default)s)")
    parser.add_argument("--prompt", default="none", choices=["none", "auto", *sorted(STYLES)],
                        help="pair format only: instruction put before each line; auto = the style of --model's template "
                             "(default: %(default)s = the raw line)")
    parser.add_argument("--model", help="llamafactory profile for --prompt auto (default: $RESONANCE_LF_PROFILE, else the default)")
    parser.add_argument("--eval-set", default=EVAL_DATASET_PATH,
                        help="eval dataset whose lines are kept out of the training data (default: %(default)s)")
    parser.add_argument("--keep-eval", action="store_true",
                        help="do not drop the eval set's lines from the training data (experiments only: it inflates every score)")
    parser.add_argument("--val-fraction", type=float, default=None,
                        help="share of the lines written to the validation file lora_train_data.val.jsonl "
                             "(default: 0.05 for --format pair, which LLaMA-Factory validates on; 0 for instruction)")
    parser.add_argument("--reverse", action="store_true",
                        help="pair format only: also write every clean row as ko->ja, Korean as the input")
    args = parser.parse_args(argv)
    style = {"none": None, "auto": None}.get(args.prompt, args.prompt)
    if args.prompt == "auto":
        style = style_for_template(_profile_template(args.model))
    val_fraction = (0.05 if args.fmt == "pair" else 0.0) if args.val_fraction is None else args.val_fraction
    if not 0 <= val_fraction < 1:
        parser.error("--val-fraction must be at least 0 and below 1")
    eval_set, eval_originals = _eval_originals(args)
    manifest.remove_manifest(PROCESSED_LOGS)  # a failed run must not leave the old file's manifest or validation rows behind
    val_file = manifest.val_path(PROCESSED_LOGS) if val_fraction > 0 else None
    counts = transform_for_lora(RAW_LOGS, PROCESSED_LOGS, args.fmt, style, args.reverse, eval_originals, val_file, val_fraction)
    if val_file and not counts["validation"]:
        print("[WARNING] No line fell into the validation split (too few lines?): training will refuse this data.")
    path = manifest.write_manifest(PROCESSED_LOGS, args.fmt, style, args.reverse, RAW_LOGS, counts, eval_set, len(eval_originals), val_fraction)
    print(f"Manifest written -> {path}")


def _eval_originals(args):
    """(eval file, eval lines) to keep out of training, per the command line: (None, []) when excluding is off or there is no set."""
    if args.keep_eval:
        print("--keep-eval: the eval set's lines stay in the training data.")
        return None, []
    if not os.path.exists(args.eval_set):
        print(f"There is no eval set at {args.eval_set}: nothing is excluded from the training data.")
        return None, []
    try:
        samples = load_eval_dataset(args.eval_set)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    print(f"Excluding eval lines ({len(samples)}) from {args.eval_set}.")
    return args.eval_set, [sample["original"] for sample in samples]


if __name__ == "__main__":
    main()
