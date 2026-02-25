import json
import os
import re
import sys

from huggingface_hub.errors import ValidationError

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_LOGS # Import the dynamic path

def check_hangeul_in_original(file_path):
    # 한글 범위를 찾는 정규표현식
    hangeul_re = re.compile(r'[가-힣]')
    error_count = 0

    print(f"Checking {file_path} for Hangeul in 'original' field...")

    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                original_text = data.get("original", "")

                # 'original' 필드에 한글이 포함되어 있는지 확인
                if hangeul_re.search(original_text):
                    print(f"[Line {i}] Hangeul detected: {original_text}")
                    error_count += 1
            except json.JSONDecodeError:
                print(f"[Line {i}] Invalid JSON format.")
                continue

    if error_count == 0:
        print("✅ No Hangeul detected in 'original' fields.")
    else:
        print(f"❌ Total {error_count} lines contain Hangeul in 'original' field.")
        raise ValidationError(f"❌ Total {error_count} lines contain Hangeul in 'original' field.")


# 실행
if __name__ == "__main__":
    check_hangeul_in_original(RAW_LOGS)