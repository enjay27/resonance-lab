"""Live training monitor for the llamafactory pipeline: `python scripts/llamafactory/watch_training.py`.

Follows LLaMA-Factory's `trainer_log.jsonl` (loss, epoch, eval) and `train_stdout.log` (grad norm,
learning rate, tqdm speed) and redraws a table once a second. Ctrl+C quits. Wants a terminal about
110 columns wide (narrower ones crop the columns). Built on `rich`, so it looks the same on Windows, Linux and macOS; the parsing and bookkeeping (`TrainingState`) is pure
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
from config import TRAIN_STDOUT_LOG
from lf_tools import profile_from_args

SMOOTH_WINDOW = 20
SPIKE_THRESHOLD = 10.0

# train_stdout.log: the trainer's dict line, and the tqdm progress bar.
STDOUT_RE = re.compile(r"'loss':\s*([\d.]+).*?'grad_norm':\s*([\d.]+).*?'learning_rate':\s*([\d.e+-]+)")
TQDM_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[[\d:]+<[\d:]+,\s*([\d.]+)s/it\]")

COLUMNS = ("Step", "Epoch", "Loss", "Smooth", "Delta", "PPL", "Status", "Grad", "GradSt", "VRAM", "GPU%", "tok/s", "LR")


# --- classification ------------------------------------------------------------------------------


def classify_loss(loss):
    if loss is None:
        return "?"
    if loss < 0.1:
        return "overfit!"
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
    if loss < 0.1:
        return "yellow"  # overfit
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


def parse_gpu_stats(text):
    """(used GiB, total GiB, utilization %) of the first GPU from nvidia-smi csv, or Nones."""
    try:
        used, total, util = text.strip().splitlines()[0].replace(" ", "").split(",")
        return float(used) / 1024, float(total) / 1024, int(util)
    except (IndexError, ValueError):
        return None, None, None


def get_gpu_stats():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None, None, None
    return parse_gpu_stats(result.stdout) if result.returncode == 0 else (None, None, None)


def tokens_per_step(train_yaml):
    """Upper bound of tokens one optimizer step sees: batch x accumulation x cutoff (single GPU)."""
    with open(train_yaml, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg.get("per_device_train_batch_size", 1) * cfg.get("gradient_accumulation_steps", 1) * cfg.get("cutoff_len", 1)


# --- state ---------------------------------------------------------------------------------------


class TrainingState:
    """Everything the monitor knows, built up from the two logs. No I/O."""

    def __init__(self, tokens_per_step=0, smooth_window=SMOOTH_WINDOW, spike_threshold=SPIKE_THRESHOLD, max_rows=200):
        self.tokens_per_step = tokens_per_step
        self.spike_threshold = spike_threshold
        self.rows = deque(maxlen=max_rows)
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

    def ingest_trainer_log(self, lines, gpu=lambda: (None, None, None)):
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
        if "eval_loss" in d and "loss" not in d:
            eval_loss = d["eval_loss"]
            eval_ppl = math.exp(min(eval_loss, 10)) if eval_loss else None
            self.rows.append({"type": "eval", "epoch": d.get("epoch", 0), "eval_loss": eval_loss, "eval_ppl": eval_ppl})
            return
        if "loss" not in d or d.get("epoch", 0) < 0.01:
            return

        self.total_steps += 1
        step = d.get("current_steps", self.total_steps)
        loss = d["loss"]
        grad = self._grads.popleft() if self._grads else None
        lr = self._lrs.popleft() if self._lrs else None
        sec_per_it = self._sec_per_it.popleft() if self._sec_per_it else None

        self.loss_window.append(loss)
        self.smooth = sum(self.loss_window) / len(self.loss_window)
        delta = loss - self.prev_loss if self.prev_loss is not None else None
        self.prev_loss = loss
        if grad is not None and grad > self.spike_threshold:
            self.spike_count += 1
        if loss < self.best_loss:
            self.best_loss, self.best_step = loss, step
        self.last_step = step

        used, total, util = gpu()
        tps = self.tokens_per_step / sec_per_it if sec_per_it and self.tokens_per_step else None
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
                "tps": tps,
            }
        )

    @property
    def eta_text(self):
        if not self.tqdm_latest:
            return "N/A"
        current, total, sec_per_it = self.tqdm_latest
        return format_eta((total - current) * sec_per_it)

    @property
    def progress_text(self):
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


def render(state, log_name, visible_rows=None):
    """A rich renderable: title, the newest rows as a table, a summary line."""
    rows = list(state.rows)[-visible_rows:] if visible_rows else list(state.rows)
    table = Table(*COLUMNS, header_style="bold", show_edge=False, pad_edge=False)
    for row in rows:
        if row["type"] == "eval":
            table.add_row(
                "EVAL", _fmt(row["epoch"], ".2f"), _fmt(row["eval_loss"], ".4f"), "", "", _fmt(row["eval_ppl"], ".3f"),
                "eval_loss", "", "", "", "", "", "", style="bold magenta",
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
            _fmt(row["tps"], ".0f"),
            _fmt(row["lr"], ".2e"),
        )

    current = state.last_step if state.last_step is not None else state.total_steps
    spike_pct = state.spike_count / current * 100 if current > 0 else 0
    best = "N/A" if state.best_loss == math.inf else f"{state.best_loss:.4f}@{state.best_step}"
    footer = (
        f"ETA: {state.eta_text} | progress: {state.progress_text} | best: {best} | "
        f"spikes: {state.spike_count}/{current} ({spike_pct:.1f}%) | smooth: {_fmt(state.smooth, '.4f')}"
    )
    title = Text(f"Training Monitor - {log_name}   ETA: {state.eta_text}   [Ctrl+C] quit", style="bold")
    return Group(title, table, Text(footer, style="bold"))


def main(argv=None):
    profile, _ = profile_from_args(argv, "Live training monitor.")
    trainer_log = os.path.join(profile.adapter_dir, "trainer_log.jsonl")
    state = TrainingState(tokens_per_step=tokens_per_step(profile.train_yaml))
    console = Console()
    print(f"Following {trainer_log} and {TRAIN_STDOUT_LOG}")
    try:
        with Live(console=console, refresh_per_second=2) as live:
            while True:
                poll(state, trainer_log, TRAIN_STDOUT_LOG)
                live.update(render(state, os.path.basename(trainer_log), visible_rows=max(console.height - 6, 5)))
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
