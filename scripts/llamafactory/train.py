import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import TRAIN_LOG_NAME
from lf_tools import check_training_data, profile_from_args, run_logged, train_command
from manifest import ManifestError
from runs import finish_run, start_run


def train(argv=None):
    profile, _ = profile_from_args(argv, "Fine-tune the model profile with LLaMA-Factory.")
    try:
        check_training_data(profile)
    except ManifestError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    run = start_run(profile.adapter_dir, profile.name)  # a fresh directory: LLaMA-Factory resumes from the last checkpoint it finds
    print(f"Profile {profile.name}: {profile.base_model} -> {run.dir}")
    print("Watch it from a second terminal: python scripts/llamafactory/watch_training.py")
    try:
        run_logged(train_command(profile.train_yaml, run.dir), os.path.join(run.dir, TRAIN_LOG_NAME), "Training")
    except SystemExit:
        finish_run(run.dir, "failed")
        raise
    finish_run(run.dir, "complete")
    print(f"\nTraining complete: run {run.id}.")


if __name__ == "__main__":
    train()
