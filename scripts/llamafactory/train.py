import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import LF_PROFILE, OUTPUT_DIR
from lf_tools import load_profile, run_logged, train_command

LOG_PATH = os.path.join(OUTPUT_DIR, "train_stdout.log")


def train():
    profile = load_profile(LF_PROFILE)
    print(f"Profile {profile.name}: {profile.base_model} -> {profile.adapter_dir}")
    run_logged(train_command(profile.train_yaml), LOG_PATH, "Training")
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
