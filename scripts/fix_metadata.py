import os
import sys

from safetensors import safe_open
from safetensors.torch import save_file

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MASTER_MODEL_DIR, CLEAN_MODEL_DIR

model_path = MASTER_MODEL_DIR
output_path = CLEAN_MODEL_DIR
os.makedirs(output_path, exist_ok=True)

if not os.path.exists(model_path):
    print(f"[ERROR] Source model directory not found: {model_path}")
    sys.exit(1)

# 1. Process the weights
for filename in os.listdir(model_path):
    if filename.endswith(".safetensors"):
        input_file = os.path.join(model_path, filename)
        output_file = os.path.join(output_path, filename)

        tensors = {}
        with safe_open(input_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                # REMOVE the classification head that's causing the error
                if key == "score.weight":
                    print(f"Skipping {key} in {filename}...")
                    continue
                tensors[key] = f.get_tensor(key)

        save_file(tensors, output_file)
        print(f"Saved cleaned {filename}")

# 2. Copy the config files (including your hacked config.json)
import shutil
for filename in os.listdir(model_path):
    if not filename.endswith(".safetensors"):
        shutil.copy(os.path.join(model_path, filename), os.path.join(output_path, filename))

print("\nDone! Now run the GGUF conversion on 'model_f16_clean'.")