"""Which stages each training pipeline runs, in order.

A pipeline is a training backend. `run_pipeline.py --pipeline <name>` runs its stages;
each stage is a script run as a subprocess that stops the pipeline when it exits non-zero.
Kept free of torch/CUDA imports so the data gate can test it.
"""

import os
from typing import NamedTuple

from config import BASE_DIR

DEFAULT = "llamafactory"


class Stage(NamedTuple):
    name: str
    path: str  # the script, run as a subprocess
    args: tuple = ()  # extra command-line arguments for it


def _script(*parts):
    return os.path.join(BASE_DIR, "scripts", *parts)


PIPELINES = {
    # LLaMA-Factory SFT + LoRA, one model per profile (configs/llamafactory/<profile>/;
    # RESONANCE_LF_PROFILE picks it). Ends in a q4_k_m GGUF for resonance-stream.
    "llamafactory": [
        Stage("Fetch Data", _script("fetch_data.py")),
        Stage("Validate", _script("validate.py")),
        Stage("Preprocessing", _script("preprocess.py"), ("--format", "pair", "--prompt", "auto")),
        Stage("Update Dataset", _script("llamafactory", "update_dataset_info.py")),
        Stage("Fine-Tuning", _script("llamafactory", "train.py")),
        Stage("Merge LoRA", _script("llamafactory", "merge.py")),
        Stage("Export GGUF", _script("llamafactory", "gguf.py")),
        # Last: it only reads the merged model and prints a report, so it must not block the export.
        Stage("Evaluation", _script("llamafactory", "eval.py")),
    ],
    # Qwen3 1.7B, LoRA with unsloth, merged to F16 (fix_metadata drops the classifier head).
    "unsloth": [
        Stage("Fetch Data", _script("fetch_data.py")),
        Stage("Validate", _script("validate.py")),
        Stage("Preprocessing", _script("preprocess.py")),
        Stage("Dataset Split", _script("unsloth", "split_dataset.py")),
        Stage("Fine-Tuning", _script("unsloth", "train.py")),
        Stage("Metadata Fix", _script("unsloth", "fix_metadata.py")),
        Stage("Evaluation", _script("unsloth", "eval.py")),
    ],
}


def stages(name):
    """The Stage list (name, script path, extra arguments) of a pipeline."""
    if name not in PIPELINES:
        raise ValueError(f"unknown pipeline {name!r}; choose one of: {', '.join(sorted(PIPELINES))}")
    return list(PIPELINES[name])
