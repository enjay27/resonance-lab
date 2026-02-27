import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS, PROCESSED_LOGS, INSTRUCTION


def transform_for_lora(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"[ERROR] Raw data not found at: {input_file}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                lora_data = {
                    "instruction": INSTRUCTION,
                    "input": data["original"],
                    "output": data["translated"]
                }
                f_out.write(json.dumps(lora_data, ensure_ascii=False) + '\n')
                count += 1
            except Exception as e:
                print(e)

    if count == 0:
        print("[ERROR] Preprocessing produced 0 lines. Check your raw input file.")
        sys.exit(1)
    print(f"Successfully preprocessed {count} lines.")


if __name__ == "__main__":
    transform_for_lora(RAW_LOGS, PROCESSED_LOGS)