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
MODEL_NAME = "rd211/Qwen3-1.7B-Instruct"  # Base model from HF
MASTER_MODEL_DIR = os.path.join(BASE_DIR, "model_f16")
CLEAN_MODEL_DIR = os.path.join(BASE_DIR, "model_f16_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# --- Training Config ---
MAX_SEQ_LENGTH = 512
MAX_STEPS = 1000
LEARNING_RATE = 2e-4

# --- Instruction ---
INSTRUCTION = (
        "Blue Protocol Star Resonance 일본어 채팅 로그를 자연스러운 한국어 구어체로 번역하세요. "
        "직역을 피하고, 원본에 없는 주어/목적어를 임의로 추가하지 마십시오. "
        "클래스 및 파티 모집 약어(T, H, D, 狂, 響, NM, EH, M16 등)는 일본 서버 컨텍스트에 맞게 그대로 유지하십시오. "
        "특히 게임 고유 용어 및 은어(예: ファスト -> 속공, 器用 -> 숙련, 完凸 -> 풀돌, 消化 -> 숙제)는 "
        "한국 유저들이 실제 사용하는 로컬라이징 용어로 엄격하게 번역하십시오."
    )