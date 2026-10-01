import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import GGUF_OUTPUT_DIR, LLAMA_CPP_DIR
from lf_tools import convert_command, gguf_paths, profile_from_args, quantize_binary, quantize_command, run
from runs import read_merge_record
from stage_tracking import fail_stage, open_tracker, record_stage, resume_stage
from track_records import gguf_tags


def export(argv=None):
    profile, _ = profile_from_args(argv, "Convert the merged model to a q4_k_m GGUF.")
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
    tracker = open_tracker()
    resume_stage(tracker, read_merge_record(profile.merged_dir))  # the run this merged model was made in
    try:
        run(convert_command(LLAMA_CPP_DIR, profile.merged_dir, f16), "Converting to F16 GGUF")
        run(quantize_command(quantize, f16, q4), "Quantizing to Q4_K_M")
    except SystemExit:
        fail_stage(tracker, "gguf")
        raise
    record_stage(tracker, gguf_tags(q4))
    print(f"\nGGUF ready: {q4}")


if __name__ == "__main__":
    export()
