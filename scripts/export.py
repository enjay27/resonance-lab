import os
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MERGE_YAML, GGUF_OUTPUT_DIR, LLAMA_CPP_DIR, MODEL_DIR, F16_GGUF, Q4_GGUF


def run(cmd, desc):
    print(f"\n[→] {desc}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Failed: {desc}")
        sys.exit(1)

def export():
    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)

    # 1. Merge LoRA
    run(
        f"llamafactory-cli export {MERGE_YAML}",
        "Merging LoRA into base model"
    )

    # 2. Convert to F16 GGUF
    run(
        f"python {LLAMA_CPP_DIR}/convert_hf_to_gguf.py {MODEL_DIR} "
        f"--outfile {F16_GGUF} --outtype f16",
        "Converting to F16 GGUF"
    )

    # 3. Quantize to Q4_K_M
    quantize_bin = os.path.join(LLAMA_CPP_DIR, "build", "bin", "Release", "llama-quantize.exe")
    run(
        f"{quantize_bin} {F16_GGUF} {Q4_GGUF} Q4_K_M",
        "Quantizing to Q4_K_M"
    )

    print(f"\n✅ GGUF ready: {Q4_GGUF}")

if __name__ == "__main__":
    export()