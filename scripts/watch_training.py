import json
import re
import time
import os
import sys
import math
import curses
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BASE_DIR

# --- Config ---
LOG_FILE     = os.path.join(BASE_DIR, "outputs", "translategemma_lora", "trainer_log.jsonl")
STDOUT_LOG   = os.path.join(BASE_DIR, "outputs", "train_stdout.log")
SMOOTH_WIN   = 20
SPIKE_THRES  = 10.0

# --- State ---
loss_window  = deque(maxlen=SMOOTH_WIN)
prev_loss    = None
spike_count  = 0
total_steps  = 0
seen_lines   = 0
stdout_lines = 0
best_loss    = float('inf')
best_step    = 0
prev_step    = None
prev_time    = None
grad_cache   = {}
smooth       = None

STDOUT_RE = re.compile(
    r"'loss':\s*([\d.]+).*?"
    r"'grad_norm':\s*([\d.]+).*?"
    r"'learning_rate':\s*([\d.e+-]+)"
)
TQDM_RE = re.compile(r"\|\s*(\d+)/(\d+)\s*\[[\d:]+<[\d:]+,\s*([\d.]+)s/it\]")

HEADER = (
    f"{'Step':>6} | {'Epoch':>6} | {'Loss':>7} | {'Smooth':>7} | {'Delta':>7} | "
    f"{'PPL':>6} | {'Status':<11} | {'Grad':>7} | {'GradSt':<8} | "
    f"{'VRAM':>9} | {'GPU%':>5} | {'tok/s':>6} | LR"
)


def classify_loss(loss):
    if loss is None: return "?          "
    if loss < 0.1:   return "overfit!   "
    if loss < 0.3:   return "great      "
    if loss < 0.8:   return "good       "
    if loss < 1.5:   return "learning   "
    return               "underfit   "


def classify_grad(grad):
    if grad is None: return "?      "
    if grad > 10:    return "spike  "
    if grad > 5:     return "high   "
    return               "normal "


def get_gpu_stats():
    try:
        import subprocess as sp
        result = sp.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            raw   = result.stdout.strip().replace(" ", "")
            parts = raw.split(",")
            used  = float(parts[0]) / 1024
            total = float(parts[1]) / 1024
            util  = int(parts[2])
            return used, total, util
    except Exception:
        pass
    return None, None, None


def parse_stdout_grads():
    global stdout_lines
    if not os.path.exists(STDOUT_LOG):
        return
    with open(STDOUT_LOG, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    for line in lines[stdout_lines:]:
        m = STDOUT_RE.search(line)
        if m:
            grad_cache.setdefault("grad_queue", []).append(float(m.group(2)))
            grad_cache.setdefault("lr_queue",   []).append(float(m.group(3)))
        t = TQDM_RE.search(line)
        if t:
            grad_cache.setdefault("tps_queue", []).append(float(t.group(3)))
            # Always overwrite with latest tqdm state — no queue needed
            grad_cache["tqdm_latest"] = (int(t.group(1)), int(t.group(2)), float(t.group(3)))
    stdout_lines = len(lines)


def pop_grad():
    queue = grad_cache.get("grad_queue", [])
    return queue.pop(0) if queue else None


def pop_lr():
    queue = grad_cache.get("lr_queue", [])
    return queue.pop(0) if queue else None


def pop_tps():
    # sec/it from tqdm -> tok/s
    queue = grad_cache.get("tps_queue", [])
    if not queue:
        return None
    sec_per_it = queue.pop(0)
    if sec_per_it <= 0:
        return None
    tokens_per_step = 8 * 256  # effective_batch * cutoff_len
    return tokens_per_step / sec_per_it


def get_tqdm_latest():
    # Returns most recent (current, total, sec_per_it) from tqdm
    return grad_cache.get("tqdm_latest", None)


def format_eta(seconds):
    if seconds is None or seconds < 0:
        return "N/A"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h{m:02d}m"
    elif m > 0:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def get_status_color(loss):
    """Color only the status text, not the whole row."""
    if loss is None:  return curses.color_pair(0)
    if loss < 0.1:    return curses.color_pair(3)  # yellow — overfit
    if loss < 0.3:    return curses.color_pair(2)  # green  — great
    if loss < 0.8:    return curses.color_pair(2)  # green  — good
    if loss < 1.5:    return curses.color_pair(4)  # cyan   — learning
    return                   curses.color_pair(1)  # red    — underfit


def get_grad_color(grad):
    if grad is None:  return curses.color_pair(0)
    if grad > 10:     return curses.color_pair(1)  # red    — spike
    if grad > 5:      return curses.color_pair(3)  # yellow — high
    return                   curses.color_pair(2)  # green  — normal


# --- Row buffer: keep last N rows to redraw on resize ---
row_buffer = deque(maxlen=200)
eval_buffer = deque(maxlen=20)
catching_up  = True  # True while replaying historical logs — skip tok/s calc
eta_str      = "N/A"   # estimated time remaining
tqdm_current = 0
tqdm_total   = 0
tqdm_sec_it  = 0.0


def draw_screen(stdscr):
    global prev_loss, smooth, total_steps, seen_lines, stdout_lines
    global spike_count, best_loss, best_step, prev_step, prev_time

    curses.curs_set(0)
    stdscr.nodelay(True)

    # Colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_RED,     -1)  # underfit
    curses.init_pair(2, curses.COLOR_GREEN,   -1)  # great/good
    curses.init_pair(3, curses.COLOR_YELLOW,  -1)  # warning
    curses.init_pair(4, curses.COLOR_CYAN,    -1)  # learning
    curses.init_pair(5, curses.COLOR_WHITE,   -1)  # header
    curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # eval

    HEADER_ROWS = 4   # title + separator + column names + separator
    FOOTER_ROWS = 3   # separator + stats + separator

    while True:
        # Handle keypress — q to quit
        key = stdscr.getch()
        if key == ord('q'):
            break

        height, width = stdscr.getmaxyx()
        content_rows = height - HEADER_ROWS - FOOTER_ROWS

        parse_stdout_grads()

        # --- Read new log lines ---
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            for line in lines[seen_lines:]:
                try:
                    d = json.loads(line)

                    # Eval row
                    if 'eval_loss' in d and 'loss' not in d:
                        eval_loss = d.get('eval_loss')
                        eval_ppl  = math.exp(min(eval_loss, 10)) if eval_loss else None
                        eval_buffer.append({
                            "epoch": d.get('epoch', 0),
                            "eval_loss": eval_loss,
                            "eval_ppl": eval_ppl,
                        })
                        row_buffer.append({"type": "eval", "eval_loss": eval_loss,
                                           "eval_ppl": eval_ppl, "epoch": d.get('epoch', 0)})
                        continue

                    if 'loss' not in d:
                        continue
                    if d.get("epoch", 0) < 0.01:
                        continue

                    total_steps += 1
                    step  = d.get("current_steps", total_steps)
                    epoch = d.get("epoch", 0)
                    loss  = d.get("loss")
                    grad  = pop_grad()   # from train_stdout.log
                    lr    = pop_lr()     # from train_stdout.log

                    if loss is not None:
                        loss_window.append(loss)
                    smooth    = sum(loss_window) / len(loss_window) if loss_window else None
                    delta     = (loss - prev_loss) if (prev_loss is not None and loss is not None) else None
                    prev_loss = loss
                    ppl       = math.exp(min(loss, 10)) if loss is not None else None

                    if grad is not None and grad > SPIKE_THRES:
                        spike_count += 1
                    if loss is not None and loss < best_loss:
                        best_loss = loss
                        best_step = step

                    tps = pop_tps()  # from tqdm sec/it in train_stdout.log

                    # ETA from latest tqdm state
                    tqdm_info = get_tqdm_latest()
                    if tqdm_info:
                        tqdm_current, tqdm_total, tqdm_sec_it = tqdm_info
                        remaining_steps = tqdm_total - tqdm_current
                        eta_seconds = remaining_steps * tqdm_sec_it
                        eta_str = format_eta(eta_seconds)

                    prev_time = time.time()
                    prev_step = step

                    vram_alloc, vram_total, gpu_util = get_gpu_stats()

                    row_buffer.append({
                        "type":       "train",
                        "step":       step,
                        "epoch":      epoch,
                        "loss":       loss,
                        "smooth":     smooth,
                        "delta":      delta,
                        "ppl":        ppl,
                        "grad":       grad,
                        "lr":         lr,
                        "vram_alloc": vram_alloc,
                        "vram_total": vram_total,
                        "gpu_util":   gpu_util,
                        "tps":        tps,
                    })

                except Exception:
                    pass

            seen_lines = len(lines)
            catching_up = False  # done replaying history, now live

        # --- Draw ---
        stdscr.erase()

        # Title row
        eta_display = eta_str if "eta_str" in dir() else "N/A"
        title = f"  Training Monitor — {os.path.basename(LOG_FILE)}   ETA: {eta_str}   [q] quit"
        stdscr.addstr(0, 0, title[:width-1], curses.color_pair(5) | curses.A_BOLD)

        # Separator
        stdscr.addstr(1, 0, "=" * min(120, width-1), curses.color_pair(5))

        # Pinned column header
        stdscr.addstr(2, 0, HEADER[:width-1], curses.color_pair(5) | curses.A_BOLD)

        # Separator
        stdscr.addstr(3, 0, "─" * min(120, width-1), curses.color_pair(5))

        # Content rows — show last N rows that fit
        visible_rows = list(row_buffer)[-content_rows:]
        for i, row in enumerate(visible_rows):
            screen_row = HEADER_ROWS + i
            if screen_row >= height - FOOTER_ROWS:
                break

            if row["type"] == "eval":
                eval_loss = row.get("eval_loss")
                eval_ppl  = row.get("eval_ppl")
                text = (
                    f"  EVAL | epoch {row['epoch']:.2f} | "
                    f"eval_loss: {eval_loss:.4f} | eval_ppl: {eval_ppl:.3f}"
                    if eval_loss else "  EVAL"
                )
                try:
                    stdscr.addstr(screen_row, 0, text[:width-1], curses.color_pair(6) | curses.A_BOLD)
                except curses.error:
                    pass
                continue

            loss       = row.get("loss")
            smooth_val = row.get("smooth")
            delta      = row.get("delta")
            ppl        = row.get("ppl")
            grad       = row.get("grad")
            lr         = row.get("lr", 0)
            vram_alloc = row.get("vram_alloc")
            vram_total = row.get("vram_total")
            gpu_util   = row.get("gpu_util")
            tps        = row.get("tps")

            loss_str   = f"{loss:.4f}"       if loss        is not None else "  N/A "
            smooth_str = f"{smooth_val:.4f}" if smooth_val  is not None else "  N/A "
            delta_str  = (f"+{delta:.3f}" if delta > 0 else f"{delta:.3f}") if delta is not None else "  N/A "
            ppl_str    = f"{ppl:.3f}"        if ppl         is not None else "  N/A"
            grad_str   = f"{grad:.3f}"       if grad        is not None else "  N/A "
            lr_str     = f"{lr:.2e}"         if lr                      else "  N/A "
            vram_str   = f"{vram_alloc:.1f}/{vram_total:.0f}G" if vram_alloc is not None else "   N/A   "
            util_str   = f"{gpu_util}%"      if gpu_util    is not None else "  N/A"
            tps_str    = f"{tps:.0f}"        if tps         is not None else "  N/A"
            step_str   = str(row.get("step", "?"))
            epoch_str  = f"{row.get('epoch', 0):.2f}"
            status     = classify_loss(loss)
            grad_st    = classify_grad(grad)

            text = (
                f"{step_str:>6} | {epoch_str:>6} | {loss_str:>7} | {smooth_str:>7} | "
                f"{delta_str:>7} | {ppl_str:>6} | {status} | "
                f"{grad_str:>7} | {grad_st:<8} | "
                f"{vram_str:>9} | {util_str:>5} | {tps_str:>6} | {lr_str}"
            )

            # Draw row in white, then recolor status and grad columns
            try:
                # Full row in default color
                stdscr.addstr(screen_row, 0, text[:width-1], curses.color_pair(0))

                # Find and recolor status field in-place
                status_col = text.find(status)
                if status_col >= 0:
                    stdscr.addstr(screen_row, status_col,
                                  status[:11],
                                  get_status_color(loss) | curses.A_BOLD)

                # Recolor grad status field
                gradst_col = text.find(grad_st)
                if gradst_col >= 0 and gradst_col != status_col:
                    stdscr.addstr(screen_row, gradst_col,
                                  grad_st[:8],
                                  get_grad_color(grad))
            except curses.error:
                pass

        # --- Pinned footer ---
        footer_row = height - FOOTER_ROWS
        try:
            stdscr.addstr(footer_row, 0, "─" * min(120, width-1), curses.color_pair(5))
        except curses.error:
            pass

        current_step = prev_step if prev_step is not None else total_steps
        spike_pct  = spike_count / current_step * 100 if current_step > 0 else 0
        smooth_val = f"{smooth:.4f}" if smooth else "N/A"
        progress_str = f"{tqdm_current}/{tqdm_total}" if tqdm_total > 0 else f"{current_step}/?"
        footer_text = (
            f"  ETA: {eta_str} | "
            f"progress: {progress_str} | "
            f"best: {best_loss:.4f}@{best_step} | "
            f"spikes: {spike_count}/{current_step} ({spike_pct:.1f}%) | "
            f"smooth: {smooth_val}"
        )
        try:
            stdscr.addstr(footer_row + 1, 0, footer_text[:width-1], curses.color_pair(5) | curses.A_BOLD)
            stdscr.addstr(footer_row + 2, 0, "─" * min(120, width-1), curses.color_pair(5))
        except curses.error:
            pass

        stdscr.refresh()
        time.sleep(1)


def main():
    try:
        curses.wrapper(draw_screen)
    except KeyboardInterrupt:
        pass
    finally:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()