"""Which stages each training pipeline runs, in order.

A pipeline is a training backend. `run_pipeline.py --pipeline <name>` runs its stages;
each stage is a script run as a subprocess that stops the pipeline when it exits non-zero.
Kept free of torch/CUDA imports so the data gate can test it.
"""

import os

from config import BASE_DIR

DEFAULT = "unsloth"


def _script(*parts):
    return os.path.join(BASE_DIR, "scripts", *parts)


PIPELINES = {
    # Qwen3 1.7B, LoRA with unsloth, merged to F16 (fix_metadata drops the classifier head).
    "unsloth": [
        ("Validate", _script("validate.py")),
        ("Preprocessing", _script("preprocess.py")),
        ("Dataset Split", _script("split_dataset.py")),
        ("Fine-Tuning", _script("train.py")),
        ("Metadata Fix", _script("fix_metadata.py")),
        ("Evaluation", _script("eval.py")),
    ],
}


def stages(name):
    """The (stage name, script path) list of a pipeline."""
    if name not in PIPELINES:
        raise ValueError(f"unknown pipeline {name!r}; choose one of: {', '.join(sorted(PIPELINES))}")
    return list(PIPELINES[name])
