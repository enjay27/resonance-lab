import os

# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Data Paths ---
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
LORA_DATASET_DIR = os.path.join(BASE_DIR, "lora_dataset")
EVAL_DATASET_PATH = os.path.join(BASE_DIR, "data", "eval", "bp-eval-dataset.jsonl")  # original/translated[/category] per line
EVAL_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "eval")  # one report per model + prompt

# RESONANCE_RAW_LOGS points the pipeline at another raw log (e.g. a hand-curated file); a relative
# path is relative to the repo root. validate.py and preprocess.py both read it from here.
RAW_LOGS = os.path.join(BASE_DIR, os.environ.get("RESONANCE_RAW_LOGS", os.path.join(RAW_DATA_DIR, "raw_translated_logs.jsonl")))
PROCESSED_LOGS = os.path.join(PROCESSED_DATA_DIR, "lora_train_data.jsonl")

# --- Model Paths ---
MODEL_NAME = "rd211/Qwen3-1.7B-Instruct"  # Base model from HF
MASTER_MODEL_DIR = os.path.join(BASE_DIR, "model_f16")
CLEAN_MODEL_DIR = os.path.join(BASE_DIR, "model_f16_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# --- LLaMA-Factory pipeline (scripts/llamafactory/) ---
# One profile = one model = configs/llamafactory/<profile>/{train,merge}.yaml.
# Which profile a script uses: --model parameter > RESONANCE_LF_PROFILE (remote jobs) > this default
# (see lf_tools.model_name). LF_PROFILE is that default resolved from the environment only; scripts
# do not read it themselves.
LF_PROFILE_DEFAULT = "hy-mt2-1.8b"  # the fixed model (maintainer, 2026-10-02); TranslateGemma's profiles stay but are not developed
LF_PROFILE = os.environ.get("RESONANCE_LF_PROFILE") or LF_PROFILE_DEFAULT
LF_CONFIG_ROOT = os.path.join(BASE_DIR, "configs", "llamafactory")
LF_DATASET_DIR = os.path.join(BASE_DIR, "data")  # where LLaMA-Factory looks for dataset_info.json
LF_DATASET_INFO_PATH = os.path.join(LF_DATASET_DIR, "dataset_info.json")
LF_DATASET_NAME = "bp_translation"
LF_VAL_DATASET_NAME = LF_DATASET_NAME + "_val"  # the validation rows preprocess.py splits off (valsplit.py)
TRAIN_LOG_NAME = "train_stdout.log"  # in each run's directory (runs.py): train.py writes it, the monitor follows it
LLAMA_CPP_DIR = os.path.join(BASE_DIR, "llama.cpp")
GGUF_OUTPUT_DIR = os.path.join(BASE_DIR, "model_gguf")

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
# --- Data checks (validate.py, preprocess.py) ---
# A file that fails these is the wrong file or a broken export, not a dataset. First guesses, not measurements:
# both stages always print the shares, so calibrate them on the first real raw log.
VALIDATE_MAX_STRUCTURE_ERRORS = 0.01  # share of raw lines that are invalid JSON or lack `original` / `translated`
VALIDATE_MAX_HANGEUL_IN_ORIGINAL = 0.10  # share of rows whose Japanese `original` holds Hangeul (preprocess drops those rows)
PREPROCESS_MAX_SUSPICIOUS = 0.30  # share of usable rows dropped as Hangeul-in-source, JP-in-output or runaway-long output

# --- Dataset on Hugging Face (scripts/fetch_data.py) ---
# configs/hf_dataset.yaml pins repo + revision; the app's per-channel dataset_<CHANNEL>.jsonl files are downloaded
# into HF_DATA_DIR and merged into RAW_LOGS. RESONANCE_RAW_LOGS (a hand-made raw log) switches the stage off.
HF_DATASET_CONFIG = os.path.join(BASE_DIR, "configs", "hf_dataset.yaml")
# A dataset recipe (category -> weight; share = weight / total weight) is one JSON file here; see dataset_recipe.py.
RECIPE_DIR = os.path.join(BASE_DIR, "configs", "datasets")
# The categories a message can be in (roots = Gate 1's choices, children = Gate 1-A's) with the description of each.
CATEGORY_TAXONOMY = os.path.join(BASE_DIR, "configs", "category_taxonomy.json")
# Gate 1's accuracy check: lines hand-labelled with a taxonomy category ({original, category}); gitignored like the eval set.
GATE1_SAMPLE = os.path.join(BASE_DIR, "data", "eval", "gate1-sample.jsonl")
# {key, category} per line, what a recipe selects by (written by Gate 1, or by any script); preprocess.py --categories.
CATEGORIES_FILE = os.path.join(PROCESSED_DATA_DIR, "categories.jsonl")
HF_DATA_DIR = os.path.join(BASE_DIR, "data", "hf")

# --- Experiment tracking (tracking.py) ---
# The NAS server's URL and credentials: gitignored, see .env.mlflow.example and deploy/mlflow/README.md.
MLFLOW_ENV_FILE = os.path.join(BASE_DIR, ".env.mlflow")
MLFLOW_EXPERIMENT = "resonance-lab"  # every training (profile run) is one MLflow run in it (track_records.py)
MLFLOW_EVAL_EXPERIMENT = "resonance-lab-eval"  # the per-sample evaluation runs (scripts/mlflow_genai_eval.py), apart from the trainings
EVAL_MAX_NEW_TOKENS = 256  # greedy decoding, batch 1; recorded with every eval run (track_records.eval_records)

# The local queue of runs not yet sent to the MLflow server (run_queue.py): gitignored, written before anything is sent.
RUN_QUEUE_PATH = os.path.join(BASE_DIR, ".run.result.backup.jsonl")  # the journal (run_queue.py)
RUN_QUEUE_LEGACY = os.path.join(BASE_DIR, ".run.result.backup.json")  # the old TinyDB file: read once, renamed .migrated
RUN_QUEUE_FILES = os.path.join(BASE_DIR, ".run.result.backup.files")  # the small files (reports, trainer log) of queued runs
