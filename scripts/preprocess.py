import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS, PROCESSED_LOGS

def transform_for_lora(input_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            lora_data = {
                "instruction": "スタレゾ（Blue Protocol）のチャットログを中立的な韓国語に翻訳...",
                "input": data["original"],
                "output": data["translated"]
            }
            f_out.write(json.dumps(lora_data, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    transform_for_lora(RAW_LOGS, PROCESSED_LOGS)