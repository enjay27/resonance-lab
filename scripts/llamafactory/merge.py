import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lf_tools import merge_command, profile_from_args, run
from runs import RunError, read_run, resolve_adapter, write_merge_record
from stage_tracking import fail_stage, open_tracker, record_stage, resume_stage
from track_records import merge_tags


def merge(argv=None):
    profile, rest = profile_from_args(argv, "Merge the LoRA adapter into the base model.")
    parser = argparse.ArgumentParser(description="Merge options.")
    parser.add_argument("--run", help="id of the training run to merge (default: the latest complete run)")
    args = parser.parse_args(rest)
    try:
        adapter = resolve_adapter(profile.adapter_dir, args.run)
    except RunError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    tracker = open_tracker()
    resume_stage(tracker, read_run(adapter))  # the training's run, by the id its run.json names (none for a pre-runs adapter)
    try:
        run(merge_command(profile.merge_yaml, adapter), f"Merging the LoRA adapter {adapter} into the base model")
    except SystemExit:
        fail_stage(tracker, "merge")
        raise
    record_stage(tracker, merge_tags(adapter))
    write_merge_record(profile.merged_dir, profile.name, adapter)
    print(f"\nMerged model: {profile.merged_dir}")


if __name__ == "__main__":
    merge()
