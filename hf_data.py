"""The training data as it lives on Hugging Face. Pure, no network, no torch (the download is scripts/fetch_data.py).

The app (resonance-stream) exports one `dataset_<CHANNEL>.jsonl` per chat channel; they are stored in a dataset repo
and fetched at a pinned revision, so a run always knows which version of the data it trained on. The channel files are
merged into the one raw log that validate.py and preprocess.py read.
"""

import glob
import json
import os
import re
from datetime import datetime, timezone

import yaml

from manifest import file_sha256

DEFAULT_INCLUDE = "dataset_*.jsonl"


class FetchError(Exception):
    """The dataset config or the downloaded files cannot be used."""


def load_config(path):
    """{repo, revision, include} from configs/hf_dataset.yaml (repo / revision are None while unset)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return {"repo": raw.get("repo"), "revision": raw.get("revision"), "include": raw.get("include") or DEFAULT_INCLUDE}


def download_command(cfg, local_dir):
    """The `hf download` argument list for the pinned revision; refuses an unpinned one."""
    if not cfg.get("revision"):
        raise FetchError("configs/hf_dataset.yaml has a repo but no revision: run `python scripts/fetch_data.py --pin`")
    return ["hf", "download", cfg["repo"], "--repo-type", "dataset", "--revision", cfg["revision"],
            "--include", cfg["include"], "--local-dir", local_dir]


def channel_files(local_dir, include=DEFAULT_INCLUDE):
    """The downloaded channel files, in name order (so the merged log does not depend on the file system)."""
    return sorted(glob.glob(os.path.join(local_dir, include)))


def merge_channels(files, out_path):
    """Concatenate the channel files' lines into `out_path`; returns {rows, files: {name: sha256}, raw_sha256}."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as out:
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        out.write(line.rstrip("\r\n") + "\n")  # a channel's last line may lack its newline
                        rows += 1
    return {"rows": rows, "files": {os.path.basename(p): file_sha256(p) for p in files}, "raw_sha256": file_sha256(out_path)}


def read_state(path):
    """What the last fetch wrote (None when there was none, or it is unreadable)."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def write_state(path, cfg, merged):
    state = {"repo": cfg["repo"], "revision": cfg["revision"], "include": cfg["include"], **merged,
             "fetched": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def is_current(state, cfg, raw_path):
    """True when the raw log is what the last fetch wrote for this repo and revision (and has not been edited since)."""
    if not state or state.get("repo") != cfg["repo"] or state.get("revision") != cfg["revision"]:
        return False
    try:
        return state.get("raw_sha256") == file_sha256(raw_path)
    except FileNotFoundError:
        return False


def with_revision(text, revision):
    """The yaml text with its `revision:` line set to `revision` (comments and the rest are kept)."""
    out, count = re.subn(r"(?m)^revision:.*$", f"revision: {revision}", text, count=1)
    if not count:
        raise FetchError("configs/hf_dataset.yaml has no `revision:` line")
    return out
