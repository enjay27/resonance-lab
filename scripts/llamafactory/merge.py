import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lf_tools import merge_command, profile_from_args, run


def merge(argv=None):
    profile, _ = profile_from_args(argv, "Merge the LoRA adapter into the base model.")
    if not os.path.isdir(profile.adapter_dir):
        print(f"[ERROR] No trained adapter at {profile.adapter_dir}. Run the Fine-Tuning stage first.")
        sys.exit(1)
    run(merge_command(profile.merge_yaml), "Merging the LoRA adapter into the base model")
    print(f"\nMerged model: {profile.merged_dir}")


if __name__ == "__main__":
    merge()
