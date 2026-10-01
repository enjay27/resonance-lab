"""What a run records in MLflow: the pure part. No mlflow import, no network, no torch.

The server runs on the maintainer's NAS (deploy/mlflow/); its URL and credentials live in the gitignored `.env.mlflow`.
These helpers turn what the pipeline already produces (profile yaml, preprocess manifest, HF fetch state, eval report,
trainer files) into the parameters, tags and metrics of a run, within MLflow's limits. Sending them (and keeping them
when the NAS is down) is a separate layer; here nothing can fail because the network does.
"""

import hashlib
import json
import math
import os
import re
import subprocess
from importlib import metadata
from typing import NamedTuple, Optional

import prompts
from config import BASE_DIR, MLFLOW_ENV_FILE
from manifest import file_sha256

# MLflow's limits (mlflow.utils.validation, 3.16): a name may hold only these characters, up to 250 of them.
MAX_KEY = 250
MAX_PARAM_VALUE = 6000
MAX_TAG_VALUE = 8000
MAX_PARAMS_TAGS_PER_BATCH = 100
MAX_METRICS_PER_BATCH = 1000
_BAD_KEY_CHARS = re.compile(r"[^A-Za-z0-9_\-. :/]")
_OFF = ("0", "false", "off", "no")


# --- settings -----------------------------------------------------------------------------------------------------


class Settings(NamedTuple):
    uri: str
    username: Optional[str] = None
    password: Optional[str] = None


def load_env_file(path):
    """KEY=VALUE lines of a `.env` style file ({} when there is none); comments, blank and malformed lines are skipped."""
    values = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def tracking_settings(environ=None, env_file=MLFLOW_ENV_FILE):
    """The NAS server to log to: environment variables over the `.env.mlflow` file; None when tracking is off.

    Off when there is no URL or RESONANCE_MLFLOW is 0/false/off. Raises ValueError for a URL that is not http(s):
    a path or `file:` URL would quietly start a local store, and the point is the one server.
    """
    environ = os.environ if environ is None else environ
    merged = load_env_file(env_file)
    merged.update({key: environ[key] for key in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_USERNAME", "MLFLOW_TRACKING_PASSWORD",
                                                  "RESONANCE_MLFLOW") if environ.get(key)})
    if merged.get("RESONANCE_MLFLOW", "").lower() in _OFF:
        return None
    uri = merged.get("MLFLOW_TRACKING_URI")
    if not uri:
        return None
    if not re.match(r"https?://", uri):
        raise ValueError(f"MLFLOW_TRACKING_URI must be the NAS server's http(s) URL, not {uri!r}")
    return Settings(uri, merged.get("MLFLOW_TRACKING_USERNAME") or None, merged.get("MLFLOW_TRACKING_PASSWORD") or None)


def client_environment(settings):
    """The environment variables the MLflow client reads. Short timeout and one retry: its defaults (120 s, many
    retries with backoff) would stall a training run for minutes when the NAS is off. Telemetry is off."""
    env = {
        "MLFLOW_TRACKING_URI": settings.uri,
        "MLFLOW_DISABLE_TELEMETRY": "true",
        "DO_NOT_TRACK": "true",
        "MLFLOW_HTTP_REQUEST_TIMEOUT": "10",
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES": "1",
        "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR": "1",
    }
    if settings.username:
        env["MLFLOW_TRACKING_USERNAME"] = settings.username
    if settings.password:
        env["MLFLOW_TRACKING_PASSWORD"] = settings.password
    return env


def describe(settings):
    """One line for the log: where, as whom, never the password."""
    return f"MLflow {settings.uri} as {settings.username or 'anonymous'}"


# --- parameters ---------------------------------------------------------------------------------------------------


def clean_key(key):
    return _BAD_KEY_CHARS.sub("_", str(key))[:MAX_KEY]


def clean_value(value, limit=MAX_PARAM_VALUE):
    """`value` as MLflow text: JSON-ish (true/false/null, lists as JSON), cut to `limit` characters."""
    if isinstance(value, bool):
        text = "true" if value else "false"
    elif value is None:
        text = "null"
    elif isinstance(value, (list, tuple, dict)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def flatten_params(mapping, prefix=""):
    """A nested mapping (a yaml) as {dotted.key: text}, within MLflow's limits."""
    flat = {}
    for key, value in mapping.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=name + "."))
        else:
            flat[clean_key(name)] = clean_value(value)
    return flat


def train_params(profile_name, base_model, template, train_cfg):
    """The training recipe: the whole train.yaml flattened, plus what the profile is and the effective batch."""
    params = {"profile": profile_name, "base_model": base_model, "template": template, **flatten_params(train_cfg)}
    per_device, accumulation = train_cfg.get("per_device_train_batch_size"), train_cfg.get("gradient_accumulation_steps")
    if per_device and accumulation:
        params["effective_batch"] = str(per_device * accumulation)
    return params


# --- what the data was --------------------------------------------------------------------------------------------


def dataset_tags(fetch_state, manifest):
    """Tags that say which data a run trained on: the HF revision (data/hf/fetch_state.json), the files' hashes and the
    eval overlap (the preprocess manifest). Either may be missing: a local raw log has no HF dataset."""
    tags = {}
    if fetch_state:
        repo, revision = fetch_state.get("repo"), fetch_state.get("revision")
        tags.update({"dataset.repo": repo, "dataset.revision": revision,
                     "dataset.url": f"https://huggingface.co/datasets/{repo}/tree/{revision}"})
        if fetch_state.get("fetched"):
            tags["dataset.fetched"] = fetch_state["fetched"]
    else:
        tags["dataset.source"] = "local raw log"
    if manifest:
        counts = manifest.get("counts", {})
        tags.update({
            "data.raw_sha256": manifest.get("raw_sha256"),
            "data.train_sha256": manifest.get("data_sha256"),
            "data.val_sha256": manifest.get("val_sha256"),
        })
        if manifest.get("eval_set"):
            tags["data.eval_set"] = manifest["eval_set"]
            tags["data.eval_overlap"] = str(counts.get("eval overlap", 0) + counts.get("eval overlap (near)", 0))
    return {key: clean_value(value, MAX_TAG_VALUE) for key, value in tags.items() if value is not None}


_COUNT_KEYS = {"total": "data.rows_total", "passed": "data.rows_passed", "validation": "data.rows_validation"}


def data_params(manifest):
    """How the training file was made: layout, direction mix, validation share, row counts and the drop reasons."""
    counts = manifest.get("counts", {})
    params = {"data.format": manifest.get("format"), "data.reverse": manifest.get("reverse"),
              "data.val_fraction": manifest.get("val_fraction")}
    params.update({name: counts[key] for key, name in _COUNT_KEYS.items() if key in counts})
    params.update({f"data.drop.{reason}": n for reason, n in counts.items() if reason not in _COUNT_KEYS})
    return {clean_key(key): clean_value(value) for key, value in params.items() if value is not None}


def prompt_fingerprint(style, reverse):
    """Style, direction mix and a hash of the instruction texts trained on: resonance-stream must send the same text."""
    if not style:
        return {"prompt.style": "none", "prompt.reverse": "false" if not reverse else "true", "prompt.sha256": "none"}
    texts = prompts.STYLES[style]
    used = {direction: texts[direction] for direction in (("ja-ko", "ko-ja") if reverse else ("ja-ko",))}
    digest = hashlib.sha256(json.dumps(used, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return {"prompt.style": style, "prompt.reverse": "true" if reverse else "false", "prompt.sha256": digest}


# --- results ------------------------------------------------------------------------------------------------------


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def eval_metrics(report, comet=None):
    """The eval report (eval_metrics.evaluate) as MLflow metrics: scores, rates, and per-category counts."""
    n = report["n"]
    metrics = {
        "eval.n": n,
        "eval.chrf": report["standard"]["chrf"], "eval.bleu": report["standard"]["bleu"], "eval.ter": report["standard"]["ter"],
        "eval.jp_leak_rate": report["jp_leakage"] / n, "eval.think_leak_rate": report["think_leakage"] / n,
        "eval.exact_match_rate": report["exact_match"] / n, "eval.discord_violations": report["discord_violations"],
        "eval.term_total": report["term_total"],
    }
    if report["term_total"]:
        metrics["eval.term_accuracy"] = report["term_hits"] / report["term_total"]
    if comet is not None:
        metrics["eval.comet"] = comet
    for name, stats in report["categories"].items():
        for field in ("total", "jp_leak", "term_miss", "discord_viol"):
            metrics[f"eval.cat.{name}.{field}"] = stats[field]
    return {clean_key(key): float(value) for key, value in metrics.items()}


_RESULT_KEYS = {"train_runtime": "train.runtime_s", "train_samples_per_second": "train.samples_per_s",
                "train_steps_per_second": "train.steps_per_s", "train_loss": "train.loss", "epoch": "train.epochs",
                "total_flos": "train.total_flos"}


def train_result_metrics(results, state):
    """Metrics from the trainer's `train_results.json` and `trainer_state.json` (either may be None)."""
    metrics = {}
    for key, name in _RESULT_KEYS.items():
        if results and _number(results.get(key)):
            metrics[name] = float(results[key])
    if state:
        if _number(state.get("global_step")):
            metrics["train.global_step"] = float(state["global_step"])
        evals = [row for row in state.get("log_history", []) if _number(row.get("eval_loss"))]
        if evals:
            best = min(evals, key=lambda row: row["eval_loss"])
            metrics["eval.best_loss"] = float(best["eval_loss"])
            if _number(best.get("step")):
                metrics["eval.best_loss_step"] = float(best["step"])
    return metrics


def trainer_tags(state):
    """The checkpoint load_best_model_at_end chose (the folder name)."""
    best = (state or {}).get("best_model_checkpoint")
    return {"train.best_checkpoint": re.split(r"[\\/]", best)[-1]} if best else {}


_STEP_KEYS = (("loss", "loss"), ("lr", "learning_rate"), ("eval_loss", "eval_loss"), ("epoch", "epoch"), ("grad_norm", "grad_norm"))


def _seconds(text):
    """'H:MM:SS' (LLaMA-Factory's elapsed_time) as seconds; 0 when it is not that."""
    try:
        hours, minutes, seconds = (int(part) for part in str(text).split(":"))
    except ValueError:
        return 0
    return hours * 3600 + minutes * 60 + seconds


def step_metrics(lines, start_ms):
    """The points of a `trainer_log.jsonl` as (name, value, step, timestamp_ms), time = the run's start + elapsed_time.

    Sent after training instead of live (HF's MLflow callback), so a run made while the NAS was off has its curves too.
    """
    points = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or not isinstance(row.get("current_steps"), int):
            continue
        timestamp = start_ms + _seconds(row.get("elapsed_time")) * 1000
        for key, name in _STEP_KEYS:
            if _number(row.get(key)):
                points.append((name, float(row[key]), row["current_steps"], timestamp))
    return points


def chunks(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


# --- where and what ran -------------------------------------------------------------------------------------------


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=15, check=True).stdout.strip()


def git_info(cwd=BASE_DIR):
    """The commit, branch and whether the tree had uncommitted changes ({} outside a repo or without git)."""
    try:
        return {"git.commit": _git(cwd, "rev-parse", "HEAD"), "git.branch": _git(cwd, "rev-parse", "--abbrev-ref", "HEAD"),
                "git.dirty": "true" if _git(cwd, "status", "--porcelain") else "false"}
    except (OSError, subprocess.SubprocessError):
        return {}


def package_versions(names=("torch", "transformers", "peft", "trl", "llamafactory", "sacrebleu", "mlflow-skinny")):
    """{pkg.<name>: version} of the installed packages among `names`."""
    versions = {}
    for name in names:
        try:
            versions[f"pkg.{name}"] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def gguf_info(path):
    """Size and sha256 of the GGUF (the file itself is never sent); {} when it does not exist."""
    try:
        return {"gguf.size_bytes": str(os.path.getsize(path)), "gguf.sha256": file_sha256(path)}
    except FileNotFoundError:
        return {}
