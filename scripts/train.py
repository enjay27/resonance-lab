import os
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TRAIN_YAML, DATASET_INFO_DIR


def train():
    if not os.path.exists(TRAIN_YAML):
        print(f"[ERROR] Training config not found: {TRAIN_YAML}")
        sys.exit(1)

    result = subprocess.run(
        [
            "llamafactory-cli", "train", TRAIN_YAML
        ],
        check=True
    )

if __name__ == "__main__":
    train()