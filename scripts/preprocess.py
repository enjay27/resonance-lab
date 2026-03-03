import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS, PROCESSED_LOGS, BASE_DIR, SYSTEM_PROMPT_PATH

# --- Filters ---
JP_PATTERN     = re.compile(r'[\u3041-\u3096\u30A1-\u30FA\u4e00-\u9fff]')
HANGEUL_PATTERN = re.compile(r'[가-힣]')


def load_system_prompt(path):
    if not os.path.exists(path):
        print(f"[ERROR] system_prompt.txt not found: {path}")
        sys.exit(1)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def is_clean(original, translated, seen_inputs):
    """
    Returns (True, None) if entry passes all checks.
    Returns (False, reason) if entry should be skipped.
    """
    # 1. Empty fields
    if not original or not translated:
        return False, "empty field"

    # 2. Hangeul in original (source should be Japanese only)
    if HANGEUL_PATTERN.search(original):
        return False, "hangeul in original"

    # 3. JP residual in translation (kana/kanji leaked into output)
    if JP_PATTERN.search(translated):
        return False, "JP residual in translation"

    # 4. Hallucination / looping output (translation is 10x longer than input)
    if len(translated) > len(original) * 10:
        return False, f"hallucination (orig={len(original)}, trans={len(translated)})"

    # 5. Spam / guild recruitment walls (long lines with multiple IDs)
    if len(original) > 150 and original.count('ID:') > 1:
        return False, "recruitment spam"

    # 6. Deduplication
    if original in seen_inputs:
        return False, "duplicate"

    return True, None


def transform_for_lora(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"[ERROR] Raw data not found at: {input_file}")
        sys.exit(1)

    system_prompt = load_system_prompt(SYSTEM_PROMPT_PATH)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    seen_inputs = set()
    counts = {
        "total": 0,
        "passed": 0,
        "empty field": 0,
        "hangeul in original": 0,
        "JP residual in translation": 0,
        "hallucination": 0,
        "recruitment spam": 0,
        "duplicate": 0,
        "json error": 0,
    }

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            counts["total"] += 1
            try:
                data = json.loads(line)
                original   = data.get("original", "").strip()
                translated = data.get("translated", "").strip()

                ok, reason = is_clean(original, translated, seen_inputs)

                if not ok:
                    key = "hallucination" if reason and reason.startswith("hallucination") else reason
                    counts[key] = counts.get(key, 0) + 1
                    continue

                seen_inputs.add(original)
                counts["passed"] += 1

                f_out.write(json.dumps({
                    "system":     system_prompt,
                    "original":   original,
                    "translated": translated,
                }, ensure_ascii=False) + '\n')

            except json.JSONDecodeError:
                counts["json error"] += 1

    # --- Report ---
    skipped = counts["total"] - counts["passed"]
    print(f"\n--- Preprocessing Report ---")
    print(f"Total input      : {counts['total']}")
    print(f"Passed           : {counts['passed']}")
    print(f"Skipped          : {skipped}")
    print(f"  empty field    : {counts['empty field']}")
    print(f"  hangeul in src : {counts['hangeul in original']}")
    print(f"  JP residual    : {counts['JP residual in translation']}")
    print(f"  hallucination  : {counts['hallucination']}")
    print(f"  spam           : {counts['recruitment spam']}")
    print(f"  duplicate      : {counts['duplicate']}")
    print(f"  json error     : {counts['json error']}")

    if counts["passed"] == 0:
        print("[ERROR] Preprocessing produced 0 lines.")
        sys.exit(1)

    print(f"\n✅ Saved {counts['passed']} clean entries to {output_file}")


if __name__ == "__main__":
    transform_for_lora(RAW_LOGS, PROCESSED_LOGS)