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
from manifest import ManifestError, require_for_style, val_path  # noqa: E402
from prompts import build_prompt, style_for_template  # noqa: E402
from config import (  # noqa: E402
    BASE_DIR,
    LF_CONFIG_ROOT,
    LF_DATASET_DIR,
    LF_DATASET_NAME,
    LF_VAL_DATASET_NAME,
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
    adapter_dir: str  # the yaml's output_dir: train writes each run's adapter in a subdirectory of it (runs.py)
    merged_dir: str  # where merge writes the full model


def _read_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_config(profile):
    """The profile's train.yaml as a dict (what the tracker records as the recipe)."""
    return _read_yaml(profile.train_yaml)


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


FAST_SUFFIX = "-fast"  # a model's fast profile is its folder name + this (configs/llamafactory/<model>-fast/)


def model_name(cli_value, environ=None, fast=False):
    """The profile to use: the --model parameter, else RESONANCE_LF_PROFILE (set on remote jobs),
    else the default. An empty value counts as unset. `fast` (--fast) picks that model's fast profile."""
    environ = os.environ if environ is None else environ
    name = cli_value or environ.get("RESONANCE_LF_PROFILE") or LF_PROFILE_DEFAULT
    return name + FAST_SUFFIX if fast and not name.endswith(FAST_SUFFIX) else name


def add_model_argument(parser):
    parser.add_argument("--model", help="model profile, a folder of configs/llamafactory/ "
                        "(default: $RESONANCE_LF_PROFILE, else " + LF_PROFILE_DEFAULT + ")")
    parser.add_argument("--fast", action="store_true", help="use that model's fast profile (<model>-fast: packing, a bigger batch, "
                        "evaluation every few steps) instead of the full one")


def profile_from_args(argv, description):
    """(profile, remaining argv): reads `--model <profile>` from argv; the rest is for the caller."""
    parser = argparse.ArgumentParser(description=description)
    add_model_argument(parser)
    args, rest = parser.parse_known_args(argv)
    return load_profile(model_name(args.model, fast=args.fast)), rest


def dataset_info(file_name, name=LF_DATASET_NAME):
    """The dataset_info.json content: `name` reads the original/translated columns of `file_name`
    (relative to the dataset dir, forward slashes)."""
    return {name: {"file_name": file_name, "columns": {"prompt": "original", "response": "translated"}}}


def dataset_info_for_processed_logs():
    """The training rows and the validation rows preprocess.py writes, as the two datasets train.yaml names."""
    def relative(path):
        return os.path.relpath(path, LF_DATASET_DIR).replace(os.sep, "/")

    return {**dataset_info(relative(PROCESSED_LOGS)), **dataset_info(relative(val_path(PROCESSED_LOGS)), LF_VAL_DATASET_NAME)}


def check_training_data(profile, data_path=PROCESSED_LOGS):
    """The manifest of the training file, after checking it was made for `profile`'s prompt style and is unchanged.

    Raises ManifestError (naming the profile and the command that fixes it) otherwise.
    """
    style = style_for_template(profile.template)
    try:
        return require_for_style(data_path, style)
    except ManifestError as e:
        raise ManifestError(
            f"profile {profile.name} (template {profile.template}): {e}\n"
            f"  fix: python scripts/preprocess.py --format pair --prompt auto --model {profile.name}"
        ) from None


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


# The rest of a full training example, per template (confirmed with scripts/llamafactory/inspect_pair.py on the
# real tokenizers, 2026-10-01): the BOS the template prefix adds, and what follows the answer. gemma3 closes the
# turn inside the response; the Hy templates have efficient_eos, so LLaMA-Factory appends the eos token itself.
TRAINING_BOS = {
    "gemma3": "<bos>",
    "hy_dense_1_8b": "<｜hy_begin▁of▁sentence｜>",
    "hy_dense_7b": "<|startoftext|>",
}
TRAINING_AFTER_ANSWER = {
    "gemma3": "<end_of_turn>\n",
    "hy_dense_1_8b": "<｜hy_place▁holder▁no▁2｜>",
    "hy_dense_7b": "<|eos|>",
}
# The 1.8B template puts <｜hy_Assistant｜> in the trained response, not in the masked prompt.
_ANSWER_STARTS_WITH = {"hy_dense_1_8b": "<｜hy_Assistant｜>"}


def training_example(template, original, translated):
    """(masked, trained): the text of one training example, as special tokens. `masked` is the prompt (loss
    ignores it), `trained` the response the model is trained to produce."""
    if template not in TRAINING_BOS:
        raise ValueError(f"no training example for template {template!r}; known: {', '.join(sorted(TRAINING_BOS))}")
    prompt = TRAINING_BOS[template] + training_prompt(template, original)
    start = _ANSWER_STARTS_WITH.get(template, "")
    if start:
        prompt = prompt.removesuffix(start)
    return prompt, start + translated + TRAINING_AFTER_ANSWER[template]


def training_user_text(template, text):
    """The user text of a training example: the line behind the instruction of the template's prompt style
    (what `preprocess.py --prompt auto` writes into the `original` column)."""
    return build_prompt(style_for_template(template), "ja-ko", text)


def chat_messages(template, text):
    """Messages for tokenizer.apply_chat_template: the model's own documented prompt, which is not
    the training format (see training_prompt)."""
    if template == "gemma3":
        return translategemma_messages(text)
    if template in ("hy_dense_1_8b", "hy_dense_7b"):
        return [{"role": "user", "content": build_prompt("hy", "ja-ko", text)}]
    known = ["gemma3", "hy_dense_1_8b", "hy_dense_7b"]
    raise ValueError(f"no chat prompt for template {template!r}; known: {', '.join(sorted(known))}")


def repo_relative(path):
    """`path` relative to the repo root with forward slashes: how the yaml files (and the overrides) name directories."""
    return os.path.relpath(path, BASE_DIR).replace(os.sep, "/")


def train_command(train_yaml, output_dir=None):
    """`llamafactory-cli train`; `output_dir` (this run's directory, see runs.py) overrides the yaml's."""
    return ["llamafactory-cli", "train", train_yaml] + ([f"output_dir={repo_relative(output_dir)}"] if output_dir else [])


def merge_command(merge_yaml, adapter_dir=None):
    """`llamafactory-cli export`; `adapter_dir` (the run to merge) overrides the yaml's adapter_name_or_path."""
    return ["llamafactory-cli", "export", merge_yaml] + ([f"adapter_name_or_path={repo_relative(adapter_dir)}"] if adapter_dir else [])


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
