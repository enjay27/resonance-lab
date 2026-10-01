"""Pure helpers of the llamafactory pipeline: profiles, dataset_info.json, command lines.

Kept apart from the stage scripts so the data gate can test them without LLaMA-Factory,
torch or a GPU installed. Commands are argument lists (no shell string), run from the
repo root because the yaml files use paths relative to it.
"""

import argparse
import json
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
    LF_PROFILE_DEFAULT,
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


def model_name(cli_value, environ=None):
    """The profile to use: the --model parameter, else RESONANCE_LF_PROFILE (set on remote jobs),
    else the default. An empty value counts as unset."""
    environ = os.environ if environ is None else environ
    return cli_value or environ.get("RESONANCE_LF_PROFILE") or LF_PROFILE_DEFAULT


def add_model_argument(parser):
    parser.add_argument("--model", help="model profile, a folder of configs/llamafactory/ "
                        "(default: $RESONANCE_LF_PROFILE, else " + LF_PROFILE_DEFAULT + ")")


def profile_from_args(argv, description):
    """(profile, remaining argv): reads `--model <profile>` from argv; the rest is for the caller."""
    parser = argparse.ArgumentParser(description=description)
    add_model_argument(parser)
    args, rest = parser.parse_known_args(argv)
    return load_profile(model_name(args.model)), rest


def dataset_info(file_name, name=LF_DATASET_NAME):
    """The dataset_info.json content: `name` reads the original/translated columns of `file_name`
    (relative to the dataset dir, forward slashes)."""
    return {name: {"file_name": file_name, "columns": {"prompt": "original", "response": "translated"}}}


def dataset_info_for_processed_logs():
    file_name = os.path.relpath(PROCESSED_LOGS, LF_DATASET_DIR).replace(os.sep, "/")
    return dataset_info(file_name)


# What the model saw as a training example, per LLaMA-Factory template: the raw line as the user
# turn, no system prompt. The tokenizer adds <bos> itself, as in training.
TRAINING_PROMPTS = {
    "gemma3": "<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n",
    # Hy-MT2 (templates registered in LLaMA-Factory from Tencent's hy_dense_template.py); BOS comes
    # from the template prefix. The 1.8B and 7B differ: 1.8B has User/Assistant tokens, 7B only <|extra_0|>.
    "hy_dense_1_8b": "<｜hy_User｜>{text}<｜hy_Assistant｜>",
    "hy_dense_7b": "{text}<|extra_0|>",
}


def training_prompt(template, text):
    if template not in TRAINING_PROMPTS:
        raise ValueError(f"no training prompt for template {template!r}; known: {', '.join(sorted(TRAINING_PROMPTS))}")
    return TRAINING_PROMPTS[template].format(text=text)


def with_bos(ids, bos_id):
    """Token ids with the BOS in front. Training's template prefix always put it there; some tokenizers
    (gemma3) add it when encoding, others (Hy-MT2) do not."""
    if bos_id is None or (ids and ids[0] == bos_id):
        return list(ids)
    return [bos_id, *ids]


def first_pairs(path, count):
    """The first `count` (original, translated) rows of a `--format pair` training file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path} not found; run the Preprocessing stage (preprocess.py --format pair) first")
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "original" not in row or "translated" not in row:
                raise ValueError(f"{path}: row has no original/translated columns; it must be made with preprocess.py --format pair")
            pairs.append((row["original"], row["translated"]))
            if len(pairs) == count:
                break
    return pairs


def generate_inputs(encoding):
    """The tokenizer output narrowed to what model.generate accepts: some tokenizers (Hy-MT2) also
    return token_type_ids, which generate rejects."""
    return {k: encoding[k] for k in ("input_ids", "attention_mask") if k in encoding}


def translategemma_messages(text, source="ja", target="ko"):
    """The structured message TranslateGemma's own chat template turns into its long English prompt."""
    return [{"role": "user", "content": [{"type": "text", "source_lang_code": source, "target_lang_code": target, "text": text}]}]


# Hy-MT2's documented default translation prompt (README, "Default Translation"), target fixed to Korean.
HY_TRANSLATE_PROMPT = (
    "Translate the following text into Korean. Note that you should only output the translated "
    "result without any additional explanation:\n\n{text}"
)


def chat_messages(template, text):
    """Messages for tokenizer.apply_chat_template: the model's own documented prompt, which is not
    the training format (see training_prompt)."""
    if template == "gemma3":
        return translategemma_messages(text)
    if template in ("hy_dense_1_8b", "hy_dense_7b"):
        return [{"role": "user", "content": HY_TRANSLATE_PROMPT.format(text=text)}]
    known = ["gemma3", "hy_dense_1_8b", "hy_dense_7b"]
    raise ValueError(f"no chat prompt for template {template!r}; known: {', '.join(sorted(known))}")


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
