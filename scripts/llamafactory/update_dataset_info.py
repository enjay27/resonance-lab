import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import LF_DATASET_INFO_PATH
from lf_tools import dataset_info_for_processed_logs


def main():
    info = dataset_info_for_processed_logs()
    with open(LF_DATASET_INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"dataset_info.json written -> {LF_DATASET_INFO_PATH}")
    for name, entry in info.items():
        print(f"  {name}: {entry['file_name']}")


if __name__ == "__main__":
    main()
