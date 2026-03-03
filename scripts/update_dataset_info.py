import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SYSTEM_PROMPT_PATH, DATASET_INFO_PATH

def update_dataset_info():
    # 1. Load system prompt
    if not os.path.exists(SYSTEM_PROMPT_PATH):
        print(f"[ERROR] system_prompt.txt not found: {SYSTEM_PROMPT_PATH}")
        sys.exit(1)

    with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
        system_prompt = f.read().strip()

    # 2. Build dataset_info
    dataset_info = {
        "bp_translation": {
            "file_name": "bp-training-dataset-final.jsonl",
            "columns": {
                "prompt": "original",
                "response": "translated"
            },
            "system": system_prompt
        }
    }

    # 3. Write dataset_info.json
    with open(DATASET_INFO_PATH, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=True, indent=2)

    print(f"✅ dataset_info.json updated with system prompt ({len(system_prompt)} chars)")
    print(f"   → {DATASET_INFO_PATH}")

if __name__ == "__main__":
    update_dataset_info()