import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS, PROCESSED_LOGS, INSTRUCTION

# --- Filters (from the TranslateGemma pipeline's preprocess) ---
JP_PATTERN = re.compile(r'[ぁ-ゖァ-ヺ一-鿿]')
HANGEUL_PATTERN = re.compile(r'[가-힣]')

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


def transform_for_lora(input_file, output_file):
    """Clean raw {original, translated} rows into LoRA {instruction, input, output} rows.

    Returns the counts per outcome (`total`, `passed` and one per reason).
    """
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
            lora_data = {
                "instruction": INSTRUCTION,
                "input": original,
                "output": translated
            }
            f_out.write(json.dumps(lora_data, ensure_ascii=False) + '\n')

    _report(counts)
    if counts["passed"] == 0:
        print("[ERROR] Preprocessing produced 0 lines. Check your raw input file.")
        sys.exit(1)
    print(f"Successfully preprocessed {counts['passed']} lines.")
    return counts


if __name__ == "__main__":
    transform_for_lora(RAW_LOGS, PROCESSED_LOGS)
