"""The logic behind notebooks/parameter_test.ipynb: one parameter test (train -> merge -> eval) and a sweep of them.

The notebook is thin: every stage is an existing script run as a subprocess (`lifecycle` builds the argument lists,
`run_command` runs one and streams its output), so nothing of train/merge/eval is reimplemented here. What is here is pure
and tested: the parameter set, the commands, reading `trainer_log.jsonl` into curves, the sweep's bookkeeping and the text
of the decision. Kept free of torch/mlflow imports so the data gate can test it.
"""

import dataclasses
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections import deque
from typing import NamedTuple

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "llamafactory"))
import pipelines
from config import BASE_DIR
from lf_tools import load_profile, model_name, train_config, training_overrides
from runs import latest_run, read_run

EVAL_PROMPTS = ("training", "chat-template")  # the choices of scripts/llamafactory/eval.py --prompt (a test checks it)
DATA_STAGES = [stage.name for stage in pipelines.stages(pipelines.DEFAULT)[:4]]  # Fetch Data .. Update Dataset
TRAINER_LOG = "trainer_log.jsonl"


# --- the parameter set ---------------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ParamSet:
    """One parameter test. `model` None = $RESONANCE_LF_PROFILE / the default; `lr` / `epochs` None = the profile's own
    (only these two can be overridden for now: lf_tools.training_overrides)."""

    model: str | None = None
    fast: bool = True
    lr: float | None = None
    epochs: float | None = None
    prompts: tuple = ("training",)

    @property
    def profile(self):
        return model_name(self.model, fast=self.fast)

    def label(self):
        parts = [f"{name}={value:g}" for name, value in (("lr", self.lr), ("epochs", self.epochs)) if value is not None]
        return " ".join(parts) or "profile defaults"

    def validate(self):
        training_overrides(self.lr, self.epochs)  # ValueError for a non-positive or non-finite number
        if not self.prompts:
            raise ValueError("at least one eval prompt is needed")
        for prompt in self.prompts:
            if prompt not in EVAL_PROMPTS:
                raise ValueError(f"unknown eval prompt {prompt!r}; choose from: {', '.join(EVAL_PROMPTS)}")


def sweep_params(defaults, entries):
    """The parameter sets of a sweep: each entry (a dict of ParamSet fields, e.g. {"lr": 4e-4}) laid over `defaults`."""
    fields = {field.name for field in dataclasses.fields(ParamSet)}
    sets = []
    for entry in entries:
        unknown = set(entry) - fields
        if unknown:
            raise ValueError(f"unknown parameter {', '.join(sorted(unknown))}; use: {', '.join(sorted(fields))}")
        entry = {**entry, **({"prompts": tuple(entry["prompts"])} if "prompts" in entry else {})}
        params = dataclasses.replace(defaults, **entry)
        params.validate()
        sets.append(params)
    return sets


# --- the commands --------------------------------------------------------------------------------------------------

class Command(NamedTuple):
    name: str
    argv: list


def _model_arguments(params):
    return (["--model", params.model] if params.model else []) + (["--fast"] if params.fast else [])


def lifecycle(params, refresh_data=False, python=None):
    """The commands of one parameter test, in order: [the data stages,] train, merge, one eval per prompt. Each is a
    script of the repo run with `python` (the notebook's own interpreter: the project venv), from the repo root."""
    python = python or sys.executable
    model = _model_arguments(params)
    scripts = os.path.join(BASE_DIR, "scripts", "llamafactory")
    commands = []
    if refresh_data:
        commands += [Command(stage, [python, os.path.join(BASE_DIR, "run_pipeline.py"), *model, "--only", stage]) for stage in DATA_STAGES]
    train = [python, os.path.join(scripts, "train.py"), *model]
    if params.lr is not None:
        train += ["--lr", repr(float(params.lr))]
    if params.epochs is not None:
        train += ["--epochs", repr(float(params.epochs))]
    commands.append(Command("Fine-Tuning", train))
    commands.append(Command("Merge LoRA", [python, os.path.join(scripts, "merge.py"), *model]))
    for prompt in params.prompts:
        commands.append(Command(f"Evaluation ({prompt})", [python, os.path.join(scripts, "eval.py"), *model, "--prompt", prompt]))
    return commands


class StageFailed(Exception):
    """A command exited non-zero (or could not start): the lifecycle stops here, nothing is retried."""

    def __init__(self, stage, returncode, tail):
        self.stage, self.returncode, self.tail = stage, returncode, list(tail)
        how = "could not start" if returncode is None else f"failed (exit {returncode})"
        super().__init__(f"{stage} {how}. Last lines of its output:\n" + "\n".join(self.tail))


def terminate_tree(proc, grace_seconds=30):
    """Stop `proc` and everything it started (train.py starts llamafactory-cli, which would train on without it).
    POSIX: SIGINT to the process group first, so a stage can close its MLflow run as KILLED, then SIGKILL after the grace.
    Windows: `taskkill /T /F`, there is no gentler way to reach a child console program from a notebook kernel."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, check=False)
    else:
        try:
            os.killpg(proc.pid, signal.SIGINT)
            proc.wait(timeout=grace_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait()


def run_command(command, emit=print, tail_lines=50, grace_seconds=30):
    """Run one command from the repo root, calling `emit(line)` for each line of its output as it arrives. Returns the seconds
    it took; raises StageFailed on a non-zero exit. Ctrl+C / a kernel interrupt stops the whole process tree first."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    started = time.perf_counter()
    try:
        proc = subprocess.Popen(command.argv, cwd=BASE_DIR, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", start_new_session=sys.platform != "win32")
    except FileNotFoundError as e:
        raise StageFailed(command.name, None, [f"program not found: {e.filename or command.argv[0]}"]) from e
    tail = deque(maxlen=tail_lines)
    try:
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            tail.append(line)
            emit(line)
        proc.wait()
    except BaseException:  # KeyboardInterrupt included: the child must not outlive the cell
        terminate_tree(proc, grace_seconds)
        raise
    finally:
        proc.stdout.close()
    if proc.returncode != 0:
        raise StageFailed(command.name, proc.returncode, tail)
    return time.perf_counter() - started


# --- the curves ----------------------------------------------------------------------------------------------------

class Curves(NamedTuple):
    train: list  # (step, loss)
    evals: list  # (step, eval loss)

    @property
    def best_eval(self):
        """(step, loss) of the lowest eval loss, None without an evaluation."""
        return min(self.evals, key=lambda point: point[1]) if self.evals else None

    @property
    def still_falling(self):
        """The best eval loss is the last evaluation: the run ended before the loss turned up (underfit)."""
        return bool(self.evals) and self.best_eval == self.evals[-1]


def read_curves(path):
    """The loss and eval-loss curves of a run's trainer_log.jsonl (LLaMA-Factory's): empty when the file is missing; lines that
    are not JSON are skipped. A row with `eval_loss` and no `loss` is an evaluation (the rule of watch_training.py)."""
    train, evals = [], []
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        return Curves([], [])
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or "current_steps" not in row:
            continue
        if "eval_loss" in row and "loss" not in row:
            evals.append((row["current_steps"], row["eval_loss"]))
        elif "loss" in row:
            train.append((row["current_steps"], row["loss"]))
    return Curves(train, evals)


def run_curves(adapter_dir):
    """{run id: Curves} of every run under an adapter directory (oldest first), for plotting a sweep offline."""
    try:
        names = sorted(name for name in os.listdir(adapter_dir) if os.path.isdir(os.path.join(adapter_dir, name)))
    except FileNotFoundError:
        return {}
    return {name: read_curves(os.path.join(adapter_dir, name, TRAINER_LOG)) for name in names
            if read_run(os.path.join(adapter_dir, name)) is not None}


# --- one parameter test, and a sweep of them -----------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class Result:
    params: ParamSet
    status: str  # "done" | "failed"
    run_id: str | None = None
    best_eval_loss: float | None = None
    best_step: int | None = None
    train_loss: float | None = None  # the last logged training loss (MLflow's `train.loss` is the run's mean: not comparable)
    seconds: float | None = None
    still_falling: bool = False
    error: str | None = None


def run_lifecycle(params, refresh_data=False, emit=print, run=run_command, adapter_dir=None):
    """Run one parameter test: every command of `lifecycle`, stopping at the first failure (StageFailed). Returns the Result
    of the training run it made, read from that run's trainer_log.jsonl."""
    params.validate()
    adapter_dir = adapter_dir or load_profile(params.profile).adapter_dir
    seconds = 0.0
    for command in lifecycle(params, refresh_data=refresh_data):
        emit(f"\n[->] {command.name}")
        seconds += run(command, emit=emit)
    return summarize_latest_run(params, adapter_dir, seconds)


def summarize_latest_run(params, adapter_dir=None, seconds=None):
    """The Result of the newest training run under the profile's adapter directory, read from its trainer_log.jsonl (the
    notebook's stage-by-stage cells call it after Fine-Tuning)."""
    adapter_dir = adapter_dir or load_profile(params.profile).adapter_dir
    run_dir = latest_run(adapter_dir)
    curves = read_curves(os.path.join(run_dir, TRAINER_LOG)) if run_dir else Curves([], [])
    best = curves.best_eval
    return Result(params, "done", os.path.basename(run_dir) if run_dir else None, best[1] if best else None, best[0] if best else None,
                  curves.train[-1][1] if curves.train else None, seconds, curves.still_falling)


def run_sweep(param_sets, execute):
    """Run each parameter set with `execute(params) -> Result`, one after another. A StageFailed ends the sweep (the rest
    never starts) and is recorded as a failed Result; any other exception, Ctrl+C included, propagates."""
    results = []
    for params in param_sets:
        try:
            results.append(execute(params))
        except StageFailed as e:
            results.append(Result(params, "failed", error=str(e)))
            break
    return results


def best_result(results):
    """The finished result with the lowest best eval loss (None when none finished)."""
    finished = [r for r in results if r.status == "done" and r.best_eval_loss is not None]
    return min(finished, key=lambda r: r.best_eval_loss) if finished else None


# --- the decision and the notes ------------------------------------------------------------------------------------

def _yaml_number(value):
    """A number as the profile yaml wants it: PyYAML reads 1e-05 as text, 1.0e-05 as a float."""
    text = format(float(value), "g")
    mantissa, e, exponent = text.partition("e")
    return f"{mantissa}.0{e}{exponent}" if e and "." not in mantissa else text


def decision_text(results, profile_name, config):
    """What to do with the best result: the lines to change in the profile's train.yaml (`config`, its current values)."""
    best = best_result(results)
    if best is None:
        return "No finished run with an eval loss yet: nothing to decide."
    finished = sum(1 for r in results if r.status == "done")
    lines = [f"Best of {finished} finished run(s): {best.params.label()} -> eval loss {best.best_eval_loss:.4f} at step {best.best_step} "
             f"(run {best.run_id})."]
    changes = []
    for key, chosen in (("learning_rate", best.params.lr), ("num_train_epochs", best.params.epochs)):
        if chosen is not None and not math.isclose(float(chosen), float(config[key])):
            changes.append(f"  {key}: {_yaml_number(chosen)}   (now {_yaml_number(config[key])})")
    if changes:
        lines += [f"Change in configs/llamafactory/{profile_name}/train.yaml:", *changes]
    else:
        lines.append("The profile already has these values; nothing to change.")
    if best.still_falling:
        lines.append("Eval loss was still falling at its last evaluation: the run ended before the minimum (more epochs or a higher lr).")
    if profile_name.endswith("-fast"):
        lines.append(f"Confirm on the full profile ({profile_name.removesuffix('-fast')}, both eval prompts) before setting the base profile; "
                     "the change itself is a normal PR (tests first).")
    return "\n".join(lines)


def results_markdown(results):
    """The sweep as a GitHub table for the memory notes (the eval scores are in compare_runs' table, from MLflow)."""
    def cell(value, fmt="{}"):
        return "-" if value is None else fmt.format(value)

    rows = ["| params | run | best eval loss | @step | last train loss | min | status | note |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        note = "still falling" if r.still_falling else " ".join((r.error or "").split()[:12])
        rows.append("| " + " | ".join([
            r.params.label(), cell(r.run_id), cell(r.best_eval_loss, "{:.4f}"), cell(r.best_step), cell(r.train_loss, "{:.4f}"),
            cell(None if r.seconds is None else r.seconds / 60, "{:.1f}"), r.status, note.replace("|", "/"),
        ]) + " |")
    return "\n".join(rows)


def profile_config(profile_name):
    """The current values of a profile's train.yaml (for `decision_text`)."""
    return train_config(load_profile(profile_name))


def kernel_warning(executable, base_dir=BASE_DIR):
    """A message when `executable` (the notebook's kernel) is not the project's .venv: the stages would run with the wrong
    packages. None when it is."""
    venv = os.path.normcase(os.path.abspath(os.path.join(base_dir, ".venv"))) + os.sep
    if os.path.normcase(os.path.abspath(executable)).startswith(venv):
        return None
    return (f"The kernel runs {executable}, not the project's .venv ({os.path.join(base_dir, '.venv')}): the stages would run with the "
            "wrong packages. Install requirements-llamafactory.txt and requirements-notebook.txt in .venv, then start `jupyter lab` from that venv "
            "(or register it once with `python -m ipykernel install --user --name resonance-lab` and pick that kernel).")
