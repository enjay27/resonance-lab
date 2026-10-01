"""The sidecar that says how `lora_train_data.jsonl` was made. Pure, no torch.

preprocess.py writes `lora_train_data.meta.json` next to its output. Every profile reads the same training file, so
without it a model can silently be trained on the wrong prompt (Hy trained on TranslateGemma's instruction), or on a
file edited after it was cleaned. update_dataset_info.py and train.py check it first; the same record is what the
experiment tracker logs as the run's data fingerprint.
"""

import hashlib
import json
import os
from datetime import datetime, timezone


class ManifestError(Exception):
    """The training file is missing its manifest, was changed since, or was made for another prompt style."""


def manifest_path(data_path):
    return os.path.splitext(data_path)[0] + ".meta.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(data_path, fmt, style, reverse, raw_path, counts, eval_set, eval_lines):
    """Record how `data_path` was made; returns the manifest's path.

    `eval_set` is the eval file whose lines were kept out (None when nothing was excluded), `eval_lines` how many.
    """
    meta = {
        "format": fmt,
        "style": style,
        "reverse": bool(reverse),
        "raw_sha256": file_sha256(raw_path),
        "data_sha256": file_sha256(data_path),
        "counts": counts,
        "eval_set": os.path.basename(eval_set) if eval_set else None,
        "eval_lines_excluded_from": eval_lines if eval_set else 0,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = manifest_path(data_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return path


def remove_manifest(data_path):
    """Delete a manifest that belongs to an older version of the data file (preprocess is about to rewrite it)."""
    try:
        os.remove(manifest_path(data_path))
    except FileNotFoundError:
        pass


def read_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def require_for_style(data_path, style):
    """The manifest of `data_path`, after checking it was made for the prompt `style` and has not changed since."""
    path = manifest_path(data_path)
    try:
        meta = read_manifest(path)
    except (FileNotFoundError, ValueError):
        raise ManifestError(f"{path} is missing or unreadable: run the Preprocessing stage (preprocess.py) first") from None
    if meta.get("format") != "pair":
        raise ManifestError(f"{data_path} was made with --format {meta.get('format')}; LLaMA-Factory reads the pair format")
    made_for = meta.get("style")
    if made_for != style:
        shown = made_for if made_for else "none (the raw line, no instruction)"
        raise ManifestError(f"{data_path} was made with prompt style {shown}, this model trains on style {style}")
    try:
        unchanged = meta.get("data_sha256") == file_sha256(data_path)
    except FileNotFoundError:
        raise ManifestError(f"{data_path} not found: run the Preprocessing stage (preprocess.py) first") from None
    if not unchanged:
        raise ManifestError(f"{data_path} has changed since preprocess.py wrote it; run the Preprocessing stage again")
    return meta
