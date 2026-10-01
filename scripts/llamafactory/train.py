import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import TRAIN_STDOUT_LOG
from lf_tools import profile_from_args, run_logged, train_command


def train(argv=None):
    profile, _ = profile_from_args(argv, "Fine-tune the model profile with LLaMA-Factory.")
    print(f"Profile {profile.name}: {profile.base_model} -> {profile.adapter_dir}")
    print("Watch it from a second terminal: python scripts/llamafactory/watch_training.py")
    run_logged(train_command(profile.train_yaml), TRAIN_STDOUT_LOG, "Training")
    print("\nTraining complete.")


if __name__ == "__main__":
    train()
