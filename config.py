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