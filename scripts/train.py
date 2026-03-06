import os
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_YAML  # ← remove DATASET_INFO_DIR, never used


def train():
    if not os.path.exists(TRAIN_YAML):
        print(f"[ERROR] Training config not found: {TRAIN_YAML}")
        sys.exit(1)

    log_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "train_stdout.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    print(f"Training started — raw logs → {log_path}")
    print(f"Monitor: python scripts/watch_training.py")

    with open(log_path, 'w', encoding='utf-8', buffering=1) as log_file:  # ← buffering=1 = line buffered
        result = subprocess.run(
            ["llamafactory-cli", "train", TRAIN_YAML],
            stdout=log_file,
            stderr=log_file,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",  # ← forces subprocess to not buffer
            },
        )

    if result.returncode != 0:
        print(f"\n[ERROR] Training failed. Last 50 lines from {log_path}:")
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f.readlines()[-50:]:
                print(line, end="")
        sys.exit(1)

    print("\n✅ Training complete.")

if __name__ == "__main__":
    train()