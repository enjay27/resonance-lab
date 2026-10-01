"""Live training monitor for the llamafactory pipeline: `python scripts/llamafactory/watch_training.py`.

Follows LLaMA-Factory's `trainer_log.jsonl` (loss, lr, epoch, eval loss, elapsed/remaining time: speed and
ETA come from it alone, so a restarted monitor shows them at once) and `train_stdout.log` (grad norm only; the
tqdm bar is a fallback) and redraws a table once a second. Ctrl+C quits. Wants a terminal about
120 columns wide (narrower ones crop the columns). Built on `rich`, so it looks the same on Windows, Linux and macOS; the parsing and bookkeeping (`TrainingState`) is pure
and unit-tested, only `main` and the GPU query touch the outside world.
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from collections import deque

import yaml
from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from runs import run_files
from lf_tools import profile_from_args

SMOOTH_WINDOW = 20
SPEED_WINDOW = 50  # steps the speed is averaged over (elapsed_time has a 1 s resolution)
OVERFIT_MARGIN = 1.02  # eval loss this much above its best, with the train loss lower: overfit signal
SPIKE_THRESHOLD = 10.0

# train_stdout.log: the trainer's dict line, and the tqdm progress bar.
STDOUT_RE = re.compile(r"'loss':\s*([\d.]+).*?'grad_norm':\s*([\d.]+).*?'learning_rate':\s*([\d.e+-]+)")
TQDM_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[[\d:]+<[\d:]+,\s*([\d.]+)s/it\]")

COLUMNS = ("Step", "Epoch", "Loss", "Smooth", "Delta", "PPL", "Status", "Grad", "GradSt", "VRAM", "GPU%", "s/step", "samp/s", "LR")


# --- classification ------------------------------------------------------------------------------


def classify_loss(loss):
    if loss is None:
        return "?"
    if loss < 0.1:
        return "very low"  # one step's loss says nothing about overfitting: see TrainingState.overfit_signal
    if loss < 0.3:
        return "great"
    if loss < 0.8:
        return "good"
    if loss < 1.5:
        return "learning"
    return "underfit"


def classify_grad(grad):
    if grad is None:
        return "?"
    if grad > 10:
        return "spike"
    if grad > 5:
        return "high"
    return "normal"


def loss_style(loss):
    if loss is None:
        return "white"
    if loss < 0.8:
        return "green"  # great / good
    if loss < 1.5:
        return "cyan"  # learning
    return "red"  # underfit


def grad_style(grad):
    if grad is None:
        return "white"
    if grad > 10:
        return "red"
    if grad > 5:
        return "yellow"
    return "green"


def format_eta(seconds):
    if seconds is None or seconds < 0:
        return "N/A"
    h, m, s = int(seconds // 3600), int(seconds % 3600 // 60), int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    if m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# --- gpu + throughput ----------------------------------------------------------------------------


def _number(text, cast):
    """`cast(text)`, or None for nvidia-smi's "[N/A]" / "[Not Supported]"."""
    return None if text.startswith("[") else cast(text)


def parse_gpu_stats(text):
    """(used GiB, total GiB, utilization %, temperature C, power W) of the first GPU from nvidia-smi csv, or Nones.
    The last two are None when the csv has only three fields or the card does not report them."""
    try:
        fields = text.strip().splitlines()[0].replace(" ", "").split(",")
        used, total, util = float(fields[0]) / 1024, float(fields[1]) / 1024, int(fields[2])
        temp = _number(fields[3], int) if len(fields) > 3 else None
        power = _number(fields[4], float) if len(fields) > 4 else None
        return used, total, util, temp, power
    except (IndexError, ValueError):
        return None, None, None, None, None


def get_gpu_stats():
    none = (None,) * 5
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return none
    return parse_gpu_stats(result.stdout) if result.returncode == 0 else none


def parse_elapsed(text):
    """Seconds of LLaMA-Factory's elapsed_time / remaining_time ("0:25:01", "1 day, 0:00:05"), or None."""
    if not isinstance(text, str):
        return None
    try:
        days = 0
        if "day" in text:
            head, text = text.split(",", 1)
            days = int(head.split()[0])
        h, m, s = (int(part) for part in text.strip().split(":"))
        return days * 86400 + h * 3600 + m * 60 + s
    except ValueError:
        return None


def finish_time(remaining_seconds, now):
    """Local clock time ("15:42") training ends, `remaining_seconds` after `now` (epoch seconds)."""
    if remaining_seconds is None:
        return "N/A"
    return time.strftime("%H:%M", time.localtime(now + remaining_seconds))


def tokens_per_step(train_yaml):
    """Upper bound of tokens one optimizer step sees: batch x accumulation x cutoff (single GPU)."""
    with open(train_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return batch_per_step(train_yaml) * cfg.get("cutoff_len", 1)


def batch_per_step(train_yaml):
    """Samples one optimizer step sees: batch x accumulation (single GPU)."""
    with open(train_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("per_device_train_batch_size", 1) * cfg.get("gradient_accumulation_steps", 1)


# --- state ---------------------------------------------------------------------------------------


class TrainingState:
    """Everything the monitor knows, built up from the two logs. No I/O."""

    def __init__(self, tokens_per_step=0, smooth_window=SMOOTH_WINDOW, spike_threshold=SPIKE_THRESHOLD, max_rows=200,
                 samples_per_step=0, speed_window=SPEED_WINDOW):
        self.tokens_per_step = tokens_per_step
        self.samples_per_step = samples_per_step
        self._speed_points = deque(maxlen=speed_window + 1)  # (step, elapsed seconds) of the newest train rows
        self.planned_steps = None
        self.elapsed_seconds = None
        self.remaining_seconds = None
        self.evals = []  # {"step", "epoch", "eval_loss", "train_smooth"}
        self.spike_threshold = spike_threshold
        self.rows = deque(maxlen=max_rows)
        self.gpu_now = (None, None)  # (temperature C, power W) of the newest row
        self.loss_window = deque(maxlen=smooth_window)
        self.total_steps = 0
        self.spike_count = 0
        self.best_loss = math.inf
        self.best_step = 0
        self.prev_loss = None
        self.smooth = None
        self.last_step = None
        self.tqdm_latest = None  # (current, total, sec per it)
        # train_stdout.log values wait here until the matching trainer_log row arrives, in order
        self._grads, self._lrs, self._sec_per_it = deque(), deque(), deque()
        self._trainer_seen = 0
        self._stdout_seen = 0

    def ingest_stdout(self, lines):
        """Take the lines of train_stdout.log (all of them; already-seen ones are skipped)."""
        for line in lines[self._stdout_seen :]:
            m = STDOUT_RE.search(line)
            if m:
                self._grads.append(float(m.group(2)))
                self._lrs.append(float(m.group(3)))
            t = TQDM_RE.search(line)
            if t:
                self._sec_per_it.append(float(t.group(3)))
                self.tqdm_latest = (int(t.group(1)), int(t.group(2)), float(t.group(3)))
        self._stdout_seen = len(lines)

    def ingest_trainer_log(self, lines, gpu=lambda: (None,) * 5):
        """Take the lines of trainer_log.jsonl (all of them; already-seen ones are skipped)."""
        for line in lines[self._trainer_seen :]:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                self._ingest_entry(d, gpu)
        self._trainer_seen = len(lines)

    def _ingest_entry(self, d, gpu):
        if d.get("total_steps"):
            self.planned_steps = d["total_steps"]
        elapsed, remaining = parse_elapsed(d.get("elapsed_time")), parse_elapsed(d.get("remaining_time"))
        if elapsed is not None:
            self.elapsed_seconds = elapsed
        if remaining is not None:
            self.remaining_seconds = remaining

        if "eval_loss" in d and "loss" not in d:
            eval_loss = d["eval_loss"]
            eval_ppl = math.exp(min(eval_loss, 10)) if eval_loss else None
            self.rows.append({"type": "eval", "epoch": d.get("epoch", 0), "eval_loss": eval_loss, "eval_ppl": eval_ppl})
            self.evals.append({"step": d.get("current_steps", self.last_step), "epoch": d.get("epoch", 0),
                               "eval_loss": eval_loss, "train_smooth": self.smooth})
            return
        if "loss" not in d or d.get("epoch", 0) < 0.01:
            return

        self.total_steps += 1
        step = d.get("current_steps", self.total_steps)
        loss = d["loss"]
        grad = self._grads.popleft() if self._grads else None
        lr = self._lrs.popleft() if self._lrs else None
        lr = d.get("lr", lr)  # the trainer log has it; stdout is only the fallback
        if self._sec_per_it:  # only used for the fallback ETA (tqdm_latest); keep the queue from growing
            self._sec_per_it.popleft()

        self.loss_window.append(loss)
        self.smooth = sum(self.loss_window) / len(self.loss_window)
        delta = loss - self.prev_loss if self.prev_loss is not None else None
        self.prev_loss = loss
        if grad is not None and grad > self.spike_threshold:
            self.spike_count += 1
        if loss < self.best_loss:
            self.best_loss, self.best_step = loss, step
        self.last_step = step

        used, total, util, temp, power = (tuple(gpu()) + (None, None))[:5]
        self.gpu_now = (temp, power)
        sec_per_step = self._sec_per_step(step, elapsed)
        samples_per_s = self.samples_per_step / sec_per_step if sec_per_step and self.samples_per_step else None
        tps = self.tokens_per_step / sec_per_step if sec_per_step and self.tokens_per_step else None
        self.rows.append(
            {
                "type": "train",
                "step": step,
                "epoch": d.get("epoch", 0),
                "loss": loss,
                "smooth": self.smooth,
                "delta": delta,
                "ppl": math.exp(min(loss, 10)),
                "grad": grad,
                "lr": lr,
                "vram_alloc": used,
                "vram_total": total,
                "gpu_util": util,
                "sec_per_step": sec_per_step,
                "samples_per_s": samples_per_s,
                "tps": tps,
            }
        )

    def _sec_per_step(self, step, elapsed):
        """Seconds per optimizer step over the newest window, from elapsed_time; None until two timed rows exist."""
        if elapsed is None:
            return None
        self._speed_points.append((step, elapsed))
        (first_step, first_time), (last_step, last_time) = self._speed_points[0], self._speed_points[-1]
        if last_step > first_step and last_time > first_time:
            return (last_time - first_time) / (last_step - first_step)
        return None

    @property
    def best_eval(self):
        """(lowest eval loss, its step), or None before the first eval."""
        if not self.evals:
            return None
        best = min(self.evals, key=lambda ev: ev["eval_loss"])
        return best["eval_loss"], best["step"]

    @property
    def overfit_signal(self):
        """Eval loss is above its best while the train loss is below what it was at that best: the model is learning
        the training lines, not the task. Needs two evals."""
        if len(self.evals) < 2:
            return False
        best = min(self.evals, key=lambda ev: ev["eval_loss"])
        last = self.evals[-1]
        if last is best or best["train_smooth"] is None or last["train_smooth"] is None:
            return False
        return last["eval_loss"] > best["eval_loss"] * OVERFIT_MARGIN and last["train_smooth"] < best["train_smooth"]

    @property
    def eta_text(self):
        if self.remaining_seconds is not None:
            return format_eta(self.remaining_seconds)
        if not self.tqdm_latest:
            return "N/A"
        current, total, sec_per_it = self.tqdm_latest
        return format_eta((total - current) * sec_per_it)

    @property
    def progress_text(self):
        if self.planned_steps and self.last_step is not None:
            return f"{self.last_step}/{self.planned_steps}"
        if self.tqdm_latest and self.tqdm_latest[1] > 0:
            return f"{self.tqdm_latest[0]}/{self.tqdm_latest[1]}"
        return f"{self.last_step if self.last_step is not None else self.total_steps}/?"


def _read_lines(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []


def poll(state, trainer_log, stdout_log, gpu=get_gpu_stats):
    """Read both log files into `state` (a log that does not exist yet is just empty)."""
    state.ingest_stdout(_read_lines(stdout_log))  # first: its grad/lr values belong to the rows below
    state.ingest_trainer_log(_read_lines(trainer_log), gpu)


# --- rendering -----------------------------------------------------------------------------------


def _fmt(value, spec, missing="N/A"):
    return missing if value is None else format(value, spec)


def render(state, log_name, visible_rows=None, now=None):
    """A rich renderable: title, the newest rows as a table, a summary line."""
    rows = list(state.rows)[-visible_rows:] if visible_rows else list(state.rows)
    table = Table(*COLUMNS, header_style="bold", show_edge=False, pad_edge=False)
    for row in rows:
        if row["type"] == "eval":
            table.add_row(
                "EVAL", _fmt(row["epoch"], ".2f"), _fmt(row["eval_loss"], ".4f"), "", "", _fmt(row["eval_ppl"], ".3f"),
                "eval_loss", "", "", "", "", "", "", "", style="bold magenta",
            )  # fmt: skip
            continue
        delta = row["delta"]
        vram = "N/A" if row["vram_alloc"] is None else f"{row['vram_alloc']:.1f}/{row['vram_total']:.0f}G"
        table.add_row(
            str(row["step"]),
            _fmt(row["epoch"], ".2f"),
            _fmt(row["loss"], ".4f"),
            _fmt(row["smooth"], ".4f"),
            "N/A" if delta is None else f"{delta:+.3f}",
            _fmt(row["ppl"], ".3f"),
            Text(classify_loss(row["loss"]), style=f"bold {loss_style(row['loss'])}"),
            _fmt(row["grad"], ".3f"),
            Text(classify_grad(row["grad"]), style=grad_style(row["grad"])),
            vram,
            "N/A" if row["gpu_util"] is None else f"{row['gpu_util']}%",
            _fmt(row["sec_per_step"], ".2f"),
            _fmt(row["samples_per_s"], ".1f"),
            _fmt(row["lr"], ".2e"),
        )

    current = state.last_step if state.last_step is not None else state.total_steps
    spike_pct = state.spike_count / current * 100 if current > 0 else 0
    best = "N/A" if state.best_loss == math.inf else f"{state.best_loss:.4f}@{state.best_step}"
    now = time.time() if now is None else now
    elapsed = "N/A" if state.elapsed_seconds is None else format_eta(state.elapsed_seconds)
    footer = (
        f"ETA: {state.eta_text} (done ~{finish_time(state.remaining_seconds, now)}) | elapsed: {elapsed} | "
        f"progress: {state.progress_text} | best: {best} | "
        f"spikes: {state.spike_count}/{current} ({spike_pct:.1f}%) | smooth: {_fmt(state.smooth, '.4f')}"
    )
    lines = [Text(footer, style="bold")]

    if state.evals:
        last, (best_loss, best_step) = state.evals[-1], state.best_eval
        gap = "" if state.smooth is None else f" | gap to train: {last['eval_loss'] - state.smooth:+.3f}"
        evals = f"eval: {last['eval_loss']:.4f}@{last['step']} (best {best_loss:.4f}@{best_step}){gap}"
        lines.append(Text(evals, style="bold magenta"))
        if state.overfit_signal:
            lines.append(Text("OVERFIT? eval loss is above its best while the train loss kept falling: an earlier "
                              "checkpoint is probably better (load_best_model_at_end picks it)", style="bold red"))

    newest = rows[-1] if rows and rows[-1]["type"] == "train" else next((r for r in reversed(rows) if r["type"] == "train"), None)
    speed = []
    if newest and newest["sec_per_step"]:
        speed.append(f"{newest['sec_per_step']:.2f} s/step")
        if newest["samples_per_s"]:
            speed.append(f"{newest['samples_per_s']:.1f} samples/s")
        if newest["tps"]:
            speed.append(f"~{newest['tps']:.0f} tok/s (upper bound: batch x cutoff)")
    temp, power = state.gpu_now
    gpu = [text for text in (f"{temp}C" if temp is not None else None, f"{power:.0f}W" if power is not None else None) if text]
    if speed or gpu:
        lines.append(Text(" | ".join(speed + (["GPU " + " ".join(gpu)] if gpu else [])), style="bold"))
    title = Text(f"Training Monitor - {log_name}   ETA: {state.eta_text}   [Ctrl+C] quit", style="bold")
    return Group(title, table, *lines)


def main(argv=None):
    profile, _ = profile_from_args(argv, "Live training monitor.")
    new_state = lambda: TrainingState(tokens_per_step=tokens_per_step(profile.train_yaml), samples_per_step=batch_per_step(profile.train_yaml))  # noqa: E731
    state, followed = new_state(), None
    console = Console()
    try:
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                trainer_log, stdout_log = run_files(profile.adapter_dir)  # the latest run: a new training is picked up
                if (trainer_log, stdout_log) != followed:
                    if followed is not None:
                        state = new_state()
                    followed = (trainer_log, stdout_log)
                    console.print(f"Following {trainer_log} and {stdout_log}")
                poll(state, trainer_log, stdout_log)
                live.update(render(state, os.path.basename(os.path.dirname(trainer_log)), visible_rows=max(console.height - 6, 5)))
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
