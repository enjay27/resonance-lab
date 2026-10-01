import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS, PROCESSED_LOGS, INSTRUCTION
from prompts import STYLES, build_prompt, style_for_template
from text_rules import HANGEUL_PATTERN, JP_PATTERN

# --- Filters (from the TranslateGemma pipeline's preprocess) ---

# A row is skipped for exactly one of these, checked in this order.
REASONS = (
    "empty field",
    "hangeul in original",
    "JP residual in translation",
    "hallucination",
    "recruitment spam",
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


def pair_rows(original, translated, style=None, reverse=False):
    """The `pair` rows of one clean line: ja->ko with the instruction of `style` (None = the raw line), and, with
    `reverse`, the ko->ja row (the Korean as the input) built the same way."""
    rows = [{"original": build_prompt(style, "ja-ko", original), "translated": translated}]
    if reverse:
        rows.append({"original": build_prompt(style, "ko-ja", translated), "translated": original})
    return rows


def clean_reason(original, translated):
    """Why a row must not be trained on, or None when it is clean.

    Duplicates are not decided here (they need the rows seen so far).
    """
    original = (original or "").strip()
    translated = (translated or "").strip()
    # resonance-stream writes `translated: null` for lines it never translated.
    if not original or not translated:
        return "empty field"
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


def transform_for_lora(input_file, output_file, fmt=DEFAULT_FORMAT, style=None, reverse=False):
    """Clean raw {original, translated} rows into training rows of layout `fmt`.

    Returns the counts per outcome (`total`, `passed` and one per reason).
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose one of: {', '.join(sorted(FORMATS))}")
    if (style or reverse) and fmt != "pair":
        raise ValueError("a prompt style or --reverse needs the pair format (--format pair)")
    make_rows = (lambda o, t: pair_rows(o, t, style, reverse)) if fmt == "pair" else (lambda o, t: [FORMATS[fmt](o, t)])
    if not os.path.exists(input_file):
        print(f"[ERROR] Raw data not found at: {input_file}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    counts = {"total": 0, "passed": 0, **{reason: 0 for reason in REASONS}}
    seen_inputs = set()

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
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
            if reason is None and original in seen_inputs:
                reason = "duplicate"  # the first occurrence wins
            if reason:
                counts[reason] += 1
                continue

            seen_inputs.add(original)
            counts["passed"] += 1
            for row in make_rows(original, translated):
                f_out.write(json.dumps(row, ensure_ascii=False) + '\n')

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
    parser.add_argument("--reverse", action="store_true",
                        help="pair format only: also write every clean row as ko->ja, Korean as the input")
    args = parser.parse_args(argv)
    style = {"none": None, "auto": None}.get(args.prompt, args.prompt)
    if args.prompt == "auto":
        style = style_for_template(_profile_template(args.model))
    transform_for_lora(RAW_LOGS, PROCESSED_LOGS, args.fmt, style, args.reverse)


if __name__ == "__main__":
    main()
