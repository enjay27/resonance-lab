import json
import math
import re
import time

import pytest
from rich.console import Console

import config
import lf_tools
import watch_training as wt


def trainer_line(step, loss, epoch=0.5, **extra):
    return json.dumps({"current_steps": step, "loss": loss, "epoch": epoch, **extra})


def stdout_line(loss, grad, lr):
    return f"{{'loss': {loss}, 'grad_norm': {grad}, 'learning_rate': {lr}, 'epoch': 0.1}}"


def tqdm_line(current, total, sec_per_it):
    return f" 10%|#         | {current}/{total} [00:10<01:30,  {sec_per_it}s/it]"


# --- classification --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "loss, label",
    [(None, "?"), (0.05, "very low"), (0.1, "great"), (0.29, "great"), (0.3, "good"), (0.79, "good"), (0.8, "learning"), (1.49, "learning"), (1.5, "underfit"), (4.0, "underfit")],
)
def test_classify_loss(loss, label):
    assert wt.classify_loss(loss) == label


@pytest.mark.parametrize("grad, label", [(None, "?"), (0.5, "normal"), (5, "normal"), (5.01, "high"), (10, "high"), (10.01, "spike")])
def test_classify_grad(grad, label):
    assert wt.classify_grad(grad) == label


@pytest.mark.parametrize("loss, style", [(None, "white"), (0.05, "green"), (0.2, "green"), (0.5, "green"), (1.0, "cyan"), (2.0, "red")])
def test_loss_style(loss, style):
    assert wt.loss_style(loss) == style


@pytest.mark.parametrize("grad, style", [(None, "white"), (1, "green"), (6, "yellow"), (11, "red")])
def test_grad_style(grad, style):
    assert wt.grad_style(grad) == style


@pytest.mark.parametrize("seconds, text", [(None, "N/A"), (-1, "N/A"), (7, "7s"), (65, "1m05s"), (3725, "1h02m")])
def test_format_eta(seconds, text):
    assert wt.format_eta(seconds) == text


# --- gpu + throughput -------------------------------------------------------------------------


def test_parse_gpu_stats():
    assert wt.parse_gpu_stats("5120, 12288, 87, 71, 142.5\n") == (5.0, 12.0, 87, 71, 142.5)


def test_parse_gpu_stats_without_temperature_and_power_still_works():
    assert wt.parse_gpu_stats("5120, 12288, 87\n") == (5.0, 12.0, 87, None, None)


def test_parse_gpu_stats_treats_not_supported_power_as_missing():
    assert wt.parse_gpu_stats("5120, 12288, 87, 71, [N/A]\n") == (5.0, 12.0, 87, 71, None)


@pytest.mark.parametrize("text", ["", "garbage", "1,2", "a,b,c"])
def test_parse_gpu_stats_rejects_garbage(text):
    assert wt.parse_gpu_stats(text) == (None, None, None, None, None)


@pytest.mark.parametrize("text, seconds", [("0:25:01", 1501), ("1:02:03", 3723), ("1 day, 0:00:05", 86405), ("garbage", None), (None, None)])
def test_parse_elapsed(text, seconds):
    assert wt.parse_elapsed(text) == seconds


def test_finish_time_is_the_local_clock_time_remaining_seconds_from_now():
    now = 1_700_000_000
    assert wt.finish_time(0, now) == time.strftime("%H:%M", time.localtime(now))
    assert wt.finish_time(3600, now) == time.strftime("%H:%M", time.localtime(now + 3600))
    assert wt.finish_time(None, now) == "N/A"


def test_batch_per_step_is_batch_times_accumulation():
    profile = lf_tools.load_profile("hy-mt2-1.8b")
    assert wt.batch_per_step(profile.train_yaml) == 2 * 4


def test_tokens_per_step_comes_from_the_profile_yaml():
    # translategemma-4b: batch 2 x accumulation 4 x cutoff 256 -- not a hard-coded guess.
    profile = lf_tools.load_profile(config.LF_PROFILE)
    assert wt.tokens_per_step(profile.train_yaml) == 2 * 4 * 256


# --- state: the trainer log ---------------------------------------------------------------------


def test_train_rows_carry_loss_ppl_smooth_and_delta():
    state = wt.TrainingState(tokens_per_step=1000)

    state.ingest_trainer_log([trainer_line(1, 2.0), trainer_line(2, 1.0)])

    first, second = state.rows
    assert (first["step"], first["loss"], first["delta"]) == (1, 2.0, None)
    assert second["delta"] == pytest.approx(-1.0)
    assert second["smooth"] == pytest.approx(1.5)
    assert second["ppl"] == pytest.approx(math.exp(1.0))


def test_ppl_is_capped_for_huge_losses():
    state = wt.TrainingState()
    state.ingest_trainer_log([trainer_line(1, 50.0)])
    assert state.rows[0]["ppl"] == pytest.approx(math.exp(10))


def test_rows_before_the_first_hundredth_of_an_epoch_are_skipped():
    state = wt.TrainingState()
    state.ingest_trainer_log([trainer_line(1, 2.0, epoch=0.001), trainer_line(2, 2.0, epoch=0.02)])
    assert [r["step"] for r in state.rows] == [2]


def test_eval_rows_are_kept_apart_from_train_rows():
    state = wt.TrainingState()

    state.ingest_trainer_log([json.dumps({"eval_loss": 0.9, "epoch": 1.0}), trainer_line(1, 1.0)])

    assert [r["type"] for r in state.rows] == ["eval", "train"]
    assert state.rows[0]["eval_ppl"] == pytest.approx(math.exp(0.9))
    assert state.total_steps == 1  # an eval row is not a training step


def test_lines_are_only_read_once_when_the_log_grows():
    state = wt.TrainingState()
    lines = [trainer_line(1, 1.0)]

    state.ingest_trainer_log(lines)
    lines.append(trainer_line(2, 0.9))
    state.ingest_trainer_log(lines)

    assert [r["step"] for r in state.rows] == [1, 2]


def test_broken_and_foreign_lines_are_ignored():
    state = wt.TrainingState()
    state.ingest_trainer_log(["{not json", "[1, 2]", json.dumps({"something": "else"}), trainer_line(1, 1.0)])
    assert [r["step"] for r in state.rows] == [1]


def test_smoothing_uses_a_sliding_window():
    state = wt.TrainingState(smooth_window=2)
    state.ingest_trainer_log([trainer_line(i, loss) for i, loss in enumerate([4.0, 2.0, 1.0], start=1)])
    assert state.rows[-1]["smooth"] == pytest.approx(1.5)  # mean of the last two


def test_best_loss_is_tracked_with_its_step():
    state = wt.TrainingState()
    state.ingest_trainer_log([trainer_line(1, 2.0), trainer_line(2, 0.7), trainer_line(3, 0.9)])
    assert (state.best_loss, state.best_step) == (0.7, 2)


# --- state: stdout (grad / lr / tqdm fallback) -----------------------------------------------


def test_grad_and_lr_from_stdout_pair_with_train_rows_in_order():
    state = wt.TrainingState()
    state.ingest_stdout([stdout_line(2.0, 1.5, "1e-05"), stdout_line(1.0, 12.0, "9e-06")])

    state.ingest_trainer_log([trainer_line(1, 2.0), trainer_line(2, 1.0)])

    assert [(r["grad"], r["lr"]) for r in state.rows] == [(1.5, 1e-05), (12.0, 9e-06)]
    assert state.spike_count == 1  # 12.0 > 10


def test_a_row_without_stdout_data_has_no_grad():
    state = wt.TrainingState()
    state.ingest_trainer_log([trainer_line(1, 2.0)])
    assert state.rows[0]["grad"] is None and state.rows[0]["lr"] is None


def test_eta_and_progress_fall_back_to_the_tqdm_line():
    state = wt.TrainingState(tokens_per_step=1000)
    state.ingest_stdout([tqdm_line(10, 100, 2.0)])

    state.ingest_trainer_log([trainer_line(10, 1.0)])

    assert state.eta_text == "3m00s"  # 90 steps left x 2 s
    assert state.progress_text == "10/100"


def test_stdout_lines_are_only_read_once_when_the_log_grows():
    state = wt.TrainingState()
    lines = [stdout_line(2.0, 1.0, "1e-05")]

    state.ingest_stdout(lines)
    state.ingest_stdout(lines)
    state.ingest_trainer_log([trainer_line(1, 2.0), trainer_line(2, 1.0)])

    assert [r["grad"] for r in state.rows] == [1.0, None]


# --- state: speed, progress and lr from the trainer log itself (survives a monitor restart) ----------


def timed(step, seconds, loss=1.0, **extra):
    h, m, s = seconds // 3600, seconds % 3600 // 60, seconds % 60
    return trainer_line(step, loss, total_steps=100, elapsed_time=f"{h}:{m:02d}:{s:02d}", **extra)


def test_speed_comes_from_elapsed_time_without_any_stdout():
    state = wt.TrainingState(tokens_per_step=1000, samples_per_step=8)

    state.ingest_trainer_log([timed(10, 10), timed(30, 30)])  # 20 steps in 20 s

    row = state.rows[-1]
    assert row["sec_per_step"] == pytest.approx(1.0)
    assert row["samples_per_s"] == pytest.approx(8.0)
    assert row["tps"] == pytest.approx(1000.0)


def test_the_first_row_has_no_speed_yet():
    state = wt.TrainingState(samples_per_step=8)
    state.ingest_trainer_log([timed(10, 10)])
    assert state.rows[0]["sec_per_step"] is None and state.rows[0]["samples_per_s"] is None


def test_speed_uses_only_the_newest_window_of_steps():
    state = wt.TrainingState(samples_per_step=8, speed_window=2)

    # steps 1-2 are slow (10 s each), the next ones fast (1 s each)
    state.ingest_trainer_log([timed(1, 0), timed(2, 10), timed(3, 20), timed(4, 21), timed(5, 22)])

    assert state.rows[-1]["sec_per_step"] == pytest.approx(1.0)  # window of 2 steps: 3 -> 5 = 2 s


def test_speed_is_the_same_whether_the_log_is_read_at_once_or_step_by_step():
    lines = [timed(i, i * 2) for i in range(1, 30)]
    live, restarted = wt.TrainingState(samples_per_step=8), wt.TrainingState(samples_per_step=8)
    for i in range(1, len(lines) + 1):
        live.ingest_trainer_log(lines[:i])
    restarted.ingest_trainer_log(lines)

    assert live.rows[-1]["samples_per_s"] == restarted.rows[-1]["samples_per_s"] is not None


def test_lr_prefers_the_trainer_log_over_stdout():
    state = wt.TrainingState()
    state.ingest_stdout([stdout_line(1.0, 1.5, "9e-09")])

    state.ingest_trainer_log([trainer_line(1, 1.0, lr=2e-05)])

    assert state.rows[0]["lr"] == 2e-05 and state.rows[0]["grad"] == 1.5


def test_progress_elapsed_and_remaining_come_from_the_trainer_log():
    state = wt.TrainingState()

    state.ingest_trainer_log([trainer_line(50, 1.0, total_steps=200, elapsed_time="0:10:00", remaining_time="0:30:00")])

    assert state.progress_text == "50/200"
    assert state.elapsed_seconds == 600 and state.remaining_seconds == 1800
    assert state.eta_text == "30m00s"


# --- state: eval loss and the overfit signal -----------------------------------------------------------


def eval_line(step, eval_loss, epoch=1.0):
    return json.dumps({"current_steps": step, "eval_loss": eval_loss, "epoch": epoch})


def test_evals_keep_their_step_and_the_train_loss_at_that_time():
    state = wt.TrainingState()

    state.ingest_trainer_log([trainer_line(1, 0.5), eval_line(1, 0.7)])

    [ev] = state.evals
    assert (ev["step"], ev["eval_loss"], ev["train_smooth"]) == (1, 0.7, pytest.approx(0.5))
    assert state.best_eval == (0.7, 1)


def test_best_eval_is_the_lowest_eval_loss():
    state = wt.TrainingState()
    state.ingest_trainer_log([trainer_line(1, 0.5), eval_line(100, 0.9), eval_line(200, 0.6), eval_line(300, 0.8)])
    assert state.best_eval == (0.6, 200)


def test_overfit_signal_needs_eval_loss_up_while_train_loss_is_down():
    state = wt.TrainingState(smooth_window=1)
    state.ingest_trainer_log([trainer_line(1, 0.5), eval_line(100, 0.6)])
    assert state.overfit_signal is False  # one eval is no trend

    state.ingest_trainer_log([trainer_line(1, 0.5), eval_line(100, 0.6), trainer_line(2, 0.2), eval_line(200, 0.9)])
    assert state.overfit_signal is True


def test_eval_loss_rising_with_train_loss_also_rising_is_not_overfit():
    state = wt.TrainingState(smooth_window=1)
    state.ingest_trainer_log([trainer_line(1, 0.3), eval_line(100, 0.6), trainer_line(2, 0.9), eval_line(200, 0.9)])
    assert state.overfit_signal is False


# --- polling the files ---------------------------------------------------------------------------


def test_poll_reads_both_logs_and_survives_missing_ones(tmp_path):
    state = wt.TrainingState()
    trainer, stdout = tmp_path / "trainer_log.jsonl", tmp_path / "train_stdout.log"

    wt.poll(state, str(trainer), str(stdout), gpu=lambda: (None, None, None))  # nothing exists yet
    assert list(state.rows) == []

    trainer.write_text(trainer_line(1, 1.0) + "\n", encoding="utf-8")
    stdout.write_text(stdout_line(1.0, 2.5, "1e-05") + "\n", encoding="utf-8")
    wt.poll(state, str(trainer), str(stdout), gpu=lambda: (4.0, 12.0, 90))

    row = state.rows[0]
    assert (row["grad"], row["vram_alloc"], row["vram_total"], row["gpu_util"]) == (2.5, 4.0, 12.0, 90)


# --- rendering ------------------------------------------------------------------------------------


def render_text(state, width=160):
    console = Console(record=True, width=width, force_terminal=False)
    console.print(wt.render(state, "trainer_log.jsonl"))
    return console.export_text()


def test_render_shows_header_rows_eval_and_footer():
    state = wt.TrainingState(tokens_per_step=1000)
    state.ingest_stdout([stdout_line(0.5, 12.0, "1e-05"), tqdm_line(2, 10, 1.0)])
    state.ingest_trainer_log([trainer_line(1, 0.5), json.dumps({"eval_loss": 0.4, "epoch": 1.0})])

    text = render_text(state)

    for expected in ("Training Monitor", "trainer_log.jsonl", "Step", "Loss", "good", "spike", "EVAL", "eval_loss", "best: 0.5000@1", "spikes: 1/1"):
        assert expected in text, expected


def test_render_footer_shows_eval_overfit_speed_finish_time_and_gpu():
    state = wt.TrainingState(smooth_window=1, tokens_per_step=1000, samples_per_step=8)
    state.ingest_trainer_log([
        timed(10, 10, 0.5), eval_line(10, 0.6),
        timed(30, 30, 0.2, remaining_time="0:20:00"), eval_line(30, 0.9),
    ], gpu=lambda: (4.0, 12.0, 90, 71, 142.5))

    text = render_text(state)

    for expected in ("eval: 0.9000@30", "best 0.6000@10", "OVERFIT?", "8.0 samples/s", "1.00 s/step", "71C", "142W", "done ~", "elapsed: 30s"):
        assert expected in text, expected


def test_render_of_a_healthy_run_has_no_overfit_warning():
    state = wt.TrainingState(smooth_window=1)
    state.ingest_trainer_log([trainer_line(1, 0.5), eval_line(1, 0.6), trainer_line(2, 0.4), eval_line(2, 0.5)])
    assert "OVERFIT?" not in render_text(state)


def test_render_of_an_empty_state_does_not_crash():
    assert "Training Monitor" in render_text(wt.TrainingState())


def test_render_keeps_only_the_newest_rows():
    state = wt.TrainingState(max_rows=3)
    state.ingest_trainer_log([trainer_line(i, 1.0) for i in range(1, 11)])
    steps = re.findall(r"^(\d+)\s+│", render_text(state), flags=re.MULTILINE)
    assert steps == ["8", "9", "10"]


def test_render_can_show_fewer_rows_than_the_state_keeps():
    state = wt.TrainingState()
    state.ingest_trainer_log([trainer_line(i, 1.0) for i in range(1, 11)])

    console = Console(record=True, width=160, force_terminal=False)
    console.print(wt.render(state, "log", visible_rows=2))

    assert re.findall(r"^(\d+)\s+│", console.export_text(), flags=re.MULTILINE) == ["9", "10"]
