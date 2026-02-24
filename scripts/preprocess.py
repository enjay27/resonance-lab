import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS, PROCESSED_LOGS


def transform_for_lora(input_file, output_file):
    if not os.path.exists(input_file):
        print(f"[ERROR] Raw data not found at: {input_file}")
        sys.exit(1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    count = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            data = json.loads(line)
            lora_data = {
                "instruction": (
                    "다음 Blue Protocol (스타레조) 채팅 로그를 일본어에서 중립적인 한국어로 번역하세요. "
                    "원본에 명시되지 않은 주어나 목적어(명사)를 임의로 추가하지 마십시오. "
                    "일본어와 마찬가지로 한국어에서도 문맥상 추측 가능한 명사는 생략하여 자연스러운 구어체로 번역하십시오. "
                    "게임 용어, 클래스 약어(T, H, D, 狂, 響) 및 던전/레이드 약어(NM, EH, M16)는 "
                    "일본 서버 컨텍스트에 맞게 적절히 유지하십시오."
                ),
                "input": data["original"],
                "output": data["translated"]
            }
            f_out.write(json.dumps(lora_data, ensure_ascii=False) + '\n')
            count += 1

    if count == 0:
        print("[ERROR] Preprocessing produced 0 lines. Check your raw input file.")
        sys.exit(1)
    print(f"Successfully preprocessed {count} lines.")


if __name__ == "__main__":
    transform_for_lora(RAW_LOGS, PROCESSED_LOGS)