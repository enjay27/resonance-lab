import json

def transform_for_lora(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:

        for line in f_in:
            data = json.loads(line)

            lora_data = {
                "instruction": "スタレゾ（Blue Protocol）のチャットログを中立的な韓国語に翻訳し、ゲーム用語（NM, EH, クラス名など）を適切に維持してください。",
                "input": data["original"],
                "output": data["translated"]
            }

            f_out.write(json.dumps(lora_data, ensure_ascii=False) + '\n')

transform_for_lora('dataset.jsonl', '../lora_train_data.jsonl')