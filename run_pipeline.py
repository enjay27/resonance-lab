import subprocess
import time
import sys
import os
import shutil
import platform
import torch

from config import BASE_DIR


def log_diagnostic(stage, status, elapsed=None):
    """Prints a structured diagnostic message for each pipeline stage."""
    timestamp = time.strftime("%H:%M:%S")
    time_str = f" | Time: {elapsed:.2f}s" if elapsed is not None else ""
    color = "\033[92m" if status == "SUCCESS" else "\033[91m"
    reset = "\033[0m"
    print(f"[{timestamp}] {stage:<20} | {color}{status:<8}{reset}{time_str}")

def check_system():
    """Diagnostic info: Checks the environment before starting."""
    print("--- System Diagnostic ---")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("GPU: NOT FOUND (Check CUDA drivers)")
    print("-" * 25 + "\n")

def run_step(name, script_path):
    """Executes a single python script and tracks its performance."""
    if not os.path.exists(script_path):
        log_diagnostic(name, "MISSING")
        return False

    start_time = time.perf_counter()
    try:
        # Run the script as a sub-process
        result = subprocess.run([sys.executable, script_path], check=True, capture_output=False)
        elapsed = time.perf_counter() - start_time
        log_diagnostic(name, "SUCCESS", elapsed)
        return True
    except subprocess.CalledProcessError:
        elapsed = time.perf_counter() - start_time
        log_diagnostic(name, "FAILED", elapsed)
        return False

def main():
    check_system()
    total_start = time.perf_counter()

    # Define the pipeline stages based on your scripts folder
    pipeline = [
        ("Validate", os.path.join(BASE_DIR, "scripts", "validate.py")),
        ("Preprocessing", os.path.join(BASE_DIR, "scripts", "preprocess.py")),
        ("Dataset Split", os.path.join(BASE_DIR, "scripts", "split_dataset.py")),
        ("Fine-Tuning", os.path.join(BASE_DIR, "scripts", "train.py")),
        ("Metadata Fix", os.path.join(BASE_DIR, "scripts", "fix_metadata.py")),
        ("Evaluation", os.path.join(BASE_DIR, "scripts", "eval.py")),
    ]

    for name, path in pipeline:
        success = run_step(name, path)
        if not success:
            print(f"\n[!] Pipeline halted at {name}. Check logs.")
            sys.exit(1)

    total_elapsed = time.perf_counter() - total_start
    print(f"\n" + "="*40)
    print(f"PIPELINE COMPLETE | Total Time: {total_elapsed/60:.2f} minutes")
    print("="*40)

if __name__ == "__main__":
    main()