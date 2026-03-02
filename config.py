import os

# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data Paths ---
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
LORA_DATASET_DIR = os.path.join(BASE_DIR, "lora_dataset")

RAW_LOGS = os.path.join(RAW_DATA_DIR, "raw_translated_logs.jsonl")
PROCESSED_LOGS = os.path.join(PROCESSED_DATA_DIR, "lora_train_data.jsonl")

# --- Model Paths ---
MODEL_NAME = "dnotitia/Qwen3-4B-Instruct-2507"  # Base model from HF
MASTER_MODEL_DIR = os.path.join(BASE_DIR, "model_f16")
CLEAN_MODEL_DIR = os.path.join(BASE_DIR, "model_f16_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# --- Training Config ---
MAX_SEQ_LENGTH = 1024
MAX_STEPS = 1000
LEARNING_RATE = 5e-5

# --- Instruction ---
INSTRUCTION = (
        """
        당신은 '블루 프로토콜: 스타 레조넌스' 일본 서버 전문 번역 엔진입니다. 
        사용자가 입력하는 일본어 채팅 로그를 다음 규칙에 따라 한국어 구어체로 번역하십시오.
        
        1. **출력 형식**: 번역 결과만 출력하십시오. 설명, 인사, 따옴표 등 부가적인 텍스트는 절대 포함하지 마십시오.
        2. **로컬라이징 용어**: 한국 유저들의 실제 게임 용어를 엄격히 사용하십시오.
           - 火力 -> 딜러 / ファスト -> 속공 / 器用 -> 숙련 / リキャスト -> 쿨타임
           - 完凸 -> 풀돌 / 消化 -> 숙제 / 寄生 -> 버스
        3. **약어 유지**: 다음 약어는 일본 서버 컨텍스트 유지를 위해 번역하지 않고 그대로 둡니다.
           - 클래스 및 역할: T, H, D, 狂, 響
           - 콘텐츠 및 모집: NM, EH, M16, 上급, EX
        4. **번역 스타일**: 
           - 문어체가 아닌 자연스러운 한국어 구어체(채팅 스타일)를 사용하십시오.
           - 원문에 없는 주어/목적어를 임의로 추측하여 추가하지 마십시오.
           - 직역보다는 게임 내 상황에 맞는 의역을 우선하되, 원문의 의도를 해치지 마십시오.
       5. **완전성 (매우 중요)**: 원문이 아무리 길거나 복잡한 이모티콘이 포함되어 있어도, 절대 중간에 번역을 끊지 말고 끝까지 빠짐없이 번역하십시오."
        """
    )