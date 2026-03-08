import json
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS, PROCESSED_LOGS

# --- Filters ---
JP_PATTERN     = re.compile(r'[\u3041-\u3096\u30A1-\u30FA\u4e00-\u9fff]')
HANGEUL_PATTERN = re.compile(r'[가-힣]')


def is_clean(original, translated, seen_inputs, direction="ja_ko"):
    if not original or not translated:
        return False, "empty field"

    if direction == "ja_ko":
        if HANGEUL_PATTERN.search(original):
            return False, "hangeul in original"
        if JP_PATTERN.search(translated):
            return False, "JP residual in translation"
    elif direction == "ko_ja":
        if JP_PATTERN.search(original):
            return False, "JP in original (ko_ja)"
        if HANGEUL_PATTERN.search(translated):
            return False, "Hangeul residual in translation (ko_ja)"

    if len(translated) > len(original) * 10:
        return False, f"hallucination (orig={len(original)}, trans={len(translated)})"
    if len(original) > 150 and original.count('ID:') > 1:
        return False, "recruitment spam"
    if original in seen_inputs:
        return False, "duplicate"

    return True, None

def make_prompt(src_lang: str, tgt_lang: str, src_lang_full: str, tgt_lang_full: str, text: str) -> str:
    return (
        f"You are a professional {src_lang_full} ({src_lang}) to {tgt_lang_full} ({tgt_lang}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {src_lang_full} text "
        f"while adhering to {tgt_lang_full} grammar, vocabulary, and cultural sensitivities.\n"
        f"Produce only the {tgt_lang_full} translation, without any additional explanations or commentary. "
        f"Please translate the following {src_lang_full} text into {tgt_lang_full}:\n"
        f"{text}"
    )

def transform_for_lora(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"[ERROR] Raw data not found at: {input_file}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    seen_inputs = set()
    counts = {
        "total": 0,
        "passed": 0,
        "empty field": 0,
        "hangeul in original": 0,
        "JP residual in translation": 0,
        "JP in original (ko_ja)": 0,
        "Hangeul residual in translation (ko_ja)": 0,
        "hallucination": 0,
        "recruitment spam": 0,
        "duplicate": 0,
        "json error": 0,
    }

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                original   = data.get("original", "").strip()
                translated = data.get("translated", "").strip()

                # --- Forward: JA → KO ---
                counts["total"] += 1
                ok, reason = is_clean(original, translated, seen_inputs, "ja_ko")
                if not ok:
                    key = "hallucination" if reason and reason.startswith("hallucination") else reason
                    counts[key] = counts.get(key, 0) + 1
                    continue

                seen_inputs.add(original)
                counts["passed"] += 1
                f_out.write(json.dumps({
                    "original":   make_prompt("ja", "ko", "Japanese", "Korean", original),
                    "translated": translated,
                }, ensure_ascii=False) + '\n')

                # --- Inverse: KO → JA (independent, doesn't block forward) ---
                counts["total"] += 1
                ok, reason = is_clean(translated, original, seen_inputs, "ko_ja")
                if ok:
                    seen_inputs.add(translated)
                    counts["passed"] += 1
                    f_out.write(json.dumps({
                        "original":   make_prompt("ko", "ja", "Korean", "Japanese", translated),
                        "translated": original,
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
    print(f"  JP in src      : {counts['JP in original (ko_ja)']}")
    print(f"  KO residual    : {counts['Hangeul residual in translation (ko_ja)']}")
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