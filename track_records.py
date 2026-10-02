"""What each stage records in the experiment tracker. Pure, no torch, no mlflow: the model-part scripts load their files,
call these, and hand the result to `tracker.Tracker` -- so the records are tested on any OS.

One training = one tracker run, named `<profile>-<run id>` (runs.py). Merge, GGUF and eval, which are run by hand as
separate commands, land in the same run: the id is derived from the run id the merged model's `resonance_run.json` names,
so nothing has to be passed between the commands. Params are fixed once logged, so only the training logs them; what a
later stage can repeat with another value (eval prompt, GGUF) is a tag.
"""

import json
import os
import re

import tracking
from config import EVAL_MAX_NEW_TOKENS

_UNSAFE_ID = re.compile(r"[^A-Za-z0-9_-]")


def read_json(path):
    """A stage's json file (train_results.json, a manifest, ...); None when it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def run_identity(profile_name, run_id):
    """(tracker local id, run name) of a training run. The id is what the offline queue keys on (letters, digits, - and _)."""
    name = f"{profile_name}-{run_id}"
    return _UNSAFE_ID.sub("-", name), name


def stage_run_id(record):
    """The tracker local id of the run a record names ({'run', 'profile'}: run.json or resonance_run.json); None when
    there is none -- an adapter trained before runs existed."""
    if not record or not record.get("run"):
        return None
    return run_identity(record["profile"], record["run"])[0]


def train_records(profile_name, base_model, template, train_cfg, fetch_state, manifest, run_id, git=None, packages=None):
    """(params, tags) for the start of a training."""
    params = tracking.train_params(profile_name, base_model, template, train_cfg)
    tags = {"stage": "train", "run.id": run_id, **tracking.dataset_tags(fetch_state, manifest)}
    if manifest:
        params.update(tracking.data_params(manifest))
        tags.update(tracking.prompt_fingerprint(manifest.get("style"), manifest.get("reverse")))
    tags.update(git or {})
    tags.update(packages or {})
    return params, {key: tracking.clean_value(value, tracking.MAX_TAG_VALUE) for key, value in tags.items()}


def train_result_records(results, state):
    """(metrics, tags) from the trainer's train_results.json and trainer_state.json (either may be None)."""
    return tracking.train_result_metrics(results, state), tracking.trainer_tags(state, results)


def merge_tags(adapter_dir):
    return {"stage.merge": "done", "merge.adapter": os.path.basename(os.path.normpath(adapter_dir))}


def gguf_tags(path):
    """Stage tag plus the GGUF's size and hash; {} when the file does not exist."""
    info = tracking.gguf_info(path)
    return {"stage.gguf": "done", **info} if info else {}


def eval_records(report, comet=None, prompt_mode=None):
    """(metrics, tags) for an eval: the scores, and how the text was generated (the numbers mean nothing without it)."""
    tags = {"stage": "eval", "eval.max_new_tokens": str(EVAL_MAX_NEW_TOKENS), "eval.decoding": "greedy", "eval.batch_size": "1"}
    if prompt_mode:
        tags["eval.prompt"] = prompt_mode
    return tracking.eval_metrics(report, comet), tags
