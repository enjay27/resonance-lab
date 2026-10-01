import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import GGUF_OUTPUT_DIR, LF_PROFILE, LLAMA_CPP_DIR
from lf_tools import convert_command, gguf_paths, load_profile, quantize_binary, quantize_command, run


def export():
    profile = load_profile(LF_PROFILE)
    if not os.path.isdir(profile.merged_dir):
        print(f"[ERROR] No merged model at {profile.merged_dir}. Run the Merge LoRA stage first.")
        sys.exit(1)
    try:
        quantize = quantize_binary(LLAMA_CPP_DIR)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    os.makedirs(GGUF_OUTPUT_DIR, exist_ok=True)
    f16, q4 = gguf_paths(profile.name, GGUF_OUTPUT_DIR)
    run(convert_command(LLAMA_CPP_DIR, profile.merged_dir, f16), "Converting to F16 GGUF")
    run(quantize_command(quantize, f16, q4), "Quantizing to Q4_K_M")
    print(f"\nGGUF ready: {q4}")


if __name__ == "__main__":
    export()
