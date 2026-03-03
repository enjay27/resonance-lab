import os

# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data Paths ---
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
LORA_DATASET_DIR = os.path.join(BASE_DIR, "lora_dataset")

RAW_LOGS = os.path.join(RAW_DATA_DIR, "bp-training-dataset-raw.jsonl")
PROCESSED_LOGS = os.path.join(PROCESSED_DATA_DIR, "bp-training-dataset-processed.jsonl")

# --- Model Paths ---
MODEL_NAME = "dnotitia/Qwen3-4B-Instruct-2507"  # Base model from HF
MODEL_DIR = os.path.join(BASE_DIR, "model_merged")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
GGUF_OUTPUT_DIR = os.path.join(BASE_DIR, "model_gguf")

# --- Config Paths ---
DATASET_INFO_DIR    = os.path.join(BASE_DIR, "data")
DATASET_INFO_PATH   = os.path.join(BASE_DIR, "data", "dataset_info.json")
SYSTEM_PROMPT_PATH  = os.path.join(BASE_DIR, "data", "system_prompt.txt")
TRAIN_YAML          = os.path.join(BASE_DIR, "configs", "training", "bp_train.yaml")
MERGE_YAML          = os.path.join(BASE_DIR, "configs", "training", "bp_merge.yaml")

# --- GGUF Configs ---
LLAMA_CPP_DIR = os.path.join(BASE_DIR, "llama.cpp")
F16_GGUF  = os.path.join(GGUF_OUTPUT_DIR, "bp-qwen3-f16.gguf")
Q4_GGUF   = os.path.join(GGUF_OUTPUT_DIR, "bp-qwen3-q4_k_m.gguf")

# --- Training Config ---
MAX_SEQ_LENGTH = 512
MAX_STEPS = 1000
LEARNING_RATE = 1e-5

with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
    INSTRUCTION = f.read().strip()