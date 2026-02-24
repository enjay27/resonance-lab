import os
from safetensors import safe_open
from safetensors.torch import save_file

model_path = "../model_f16"  # The folder you are trying to convert
output_path = "../model_f16_clean"
os.makedirs(output_path, exist_ok=True)

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