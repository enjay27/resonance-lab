"""Pure helpers of the llamafactory pipeline: profiles, dataset_info.json, command lines.

Kept apart from the stage scripts so the data gate can test them without LLaMA-Factory,
torch or a GPU installed. Commands are argument lists (no shell string), run from the
repo root because the yaml files use paths relative to it.
"""

import os
import subprocess
import sys
from typing import NamedTuple

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import (  # noqa: E402
    BASE_DIR,
    LF_CONFIG_ROOT,
    LF_DATASET_DIR,
    LF_DATASET_NAME,
    PROCESSED_LOGS,
)


class Profile(NamedTuple):
    name: str
    train_yaml: str
    merge_yaml: str
    base_model: str
    template: str
    dataset: str
    adapter_dir: str  # where train writes the LoRA adapter
    merged_dir: str  # where merge writes the full model


def _read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def available_profiles(root=LF_CONFIG_ROOT):
    """Names of the profile folders that have both a train.yaml and a merge.yaml."""
    if not os.path.isdir(root):
        return []
    return sorted(
        name
        for name in os.listdir(root)
        if all(os.path.isfile(os.path.join(root, name, f"{part}.yaml")) for part in ("train", "merge"))
    )


def load_profile(name, root=LF_CONFIG_ROOT):
    if name not in available_profiles(root):
        raise ValueError(f"unknown profile {name!r}; choose one of: {', '.join(available_profiles(root)) or '(none)'}")
    train_yaml = os.path.join(root, name, "train.yaml")
    merge_yaml = os.path.join(root, name, "merge.yaml")
    train, merge = _read_yaml(train_yaml), _read_yaml(merge_yaml)
    return Profile(
        name=name,
        train_yaml=train_yaml,
        merge_yaml=merge_yaml,
        base_model=train["model_name_or_path"],
        template=train["template"],
        dataset=train["dataset"],
        adapter_dir=os.path.join(BASE_DIR, train["output_dir"]),
        merged_dir=os.path.join(BASE_DIR, merge["export_dir"]),
    )


def dataset_info(file_name, name=LF_DATASET_NAME):
    """The dataset_info.json content: `name` reads the original/translated columns of `file_name`
    (relative to the dataset dir, forward slashes)."""
    return {name: {"file_name": file_name, "columns": {"prompt": "original", "response": "translated"}}}


def dataset_info_for_processed_logs():
    file_name = os.path.relpath(PROCESSED_LOGS, LF_DATASET_DIR).replace(os.sep, "/")
    return dataset_info(file_name)


def train_command(train_yaml):
    return ["llamafactory-cli", "train", train_yaml]


def merge_command(merge_yaml):
    return ["llamafactory-cli", "export", merge_yaml]


def gguf_paths(profile_name, gguf_dir):
    """(F16, Q4_K_M) GGUF file paths for a profile."""
    return (
        os.path.join(gguf_dir, f"bp-{profile_name}-f16.gguf"),
        os.path.join(gguf_dir, f"bp-{profile_name}-q4_k_m.gguf"),
    )


def convert_command(llama_cpp_dir, merged_dir, out_gguf):
    return [
        sys.executable,
        os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py"),
        merged_dir,
        "--outfile",
        out_gguf,
        "--outtype",
        "f16",
    ]


def quantize_binary(llama_cpp_dir):
    """The llama-quantize built under `llama_cpp_dir`, whichever generator built it."""
    candidates = [
        os.path.join(llama_cpp_dir, "build", "bin", "Release", "llama-quantize.exe"),  # Windows, MSVC
        os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize.exe"),  # Windows, Ninja / MinGW
        os.path.join(llama_cpp_dir, "build", "bin", "llama-quantize"),  # Linux / macOS
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("llama-quantize not found; build llama.cpp first. Looked for:\n  " + "\n  ".join(candidates))


def quantize_command(binary, f16_gguf, q4_gguf):
    return [binary, f16_gguf, q4_gguf, "Q4_K_M"]


def tail(path, n):
    """The last `n` lines of a text file ([] when it does not exist)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()[-n:]
    except FileNotFoundError:
        return []


def run(cmd, description, **kwargs):
    """Run a command from the repo root; exit 1 when it fails so the pipeline stops."""
    print(f"\n[->] {description}")
    try:
        result = subprocess.run(cmd, cwd=BASE_DIR, **kwargs)
    except FileNotFoundError as e:
        print(f"[ERROR] {description}: {e}")
        sys.exit(1)
    if result.returncode != 0:
        print(f"[ERROR] Failed: {description}")
        sys.exit(1)


def run_logged(cmd, log_path, description, tail_lines=50):
    """Run a long command with its output going to `log_path`; on failure show the log's tail."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    print(f"\n[->] {description} -- raw output goes to {log_path}")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    try:
        with open(log_path, "w", encoding="utf-8", buffering=1) as log:  # line buffered, so a monitor can follow it
            result = subprocess.run(cmd, cwd=BASE_DIR, stdout=log, stderr=log, env=env)
    except FileNotFoundError as e:
        print(f"[ERROR] {description}: program not found ({e}). Is it installed in this environment?")
        sys.exit(1)
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed. Last {tail_lines} lines of {log_path}:")
        print("\n".join(tail(log_path, tail_lines)))
        sys.exit(1)
