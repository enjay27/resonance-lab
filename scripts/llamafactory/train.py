import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import TRAIN_LOG_NAME
from lf_tools import check_training_data, profile_from_args, run_logged, train_command, train_config
from manifest import ManifestError
from runs import finish_run, start_run
from stage_tracking import finish_training, open_tracker, start_training, training_context, training_status


def train(argv=None):
    profile, _ = profile_from_args(argv, "Fine-tune the model profile with LLaMA-Factory.")
    try:
        check_training_data(profile)
    except ManifestError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    run = start_run(profile.adapter_dir, profile.name)  # a fresh directory: LLaMA-Factory resumes from the last checkpoint it finds
    print(f"Profile {profile.name}: {profile.base_model} -> {run.dir}")
    tracker = open_tracker()  # never raises; does nothing when tracking is off
    start_training(tracker, profile, run, train_config(profile), **training_context())
    print("Watch it from a second terminal: python scripts/llamafactory/watch_training.py")
    try:
        run_logged(train_command(profile.train_yaml, run.dir), os.path.join(run.dir, TRAIN_LOG_NAME), "Training")
    except (SystemExit, KeyboardInterrupt) as e:  # a failed command, or Ctrl+C: the run is closed either way (KILLED in MLflow)
        finish_run(run.dir, "failed")
        finish_training(tracker, run.dir, training_status(e), profile.train_yaml)
        raise
    finish_run(run.dir, "complete")
    finish_training(tracker, run.dir, "FINISHED", profile.train_yaml)
    print(f"\nTraining complete: run {run.id}.")


if __name__ == "__main__":
    train()
