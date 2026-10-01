# Plan: MLflow experiment tracking (maintainer, 2026-10-01: "I would use mlflow ... I'll start from a new session")

**Status: plan only, nothing implemented.** Per CLAUDE.md the next session presents this (adjusted) and waits for the maintainer's OK before touching code.
Why: the first runs (`roadmap/first-training-run-2026-10-01.md`) were compared by pasting `train_results.json`, eval `.txt` reports and PowerShell
output into chat and then into markdown. MLflow replaces that: every run keeps its params, metrics, data fingerprint and reports, comparable in the UI
and — for LLM agents — through `mlflow.search_runs` without a browser.

## Do these first (small, independent of MLflow; they decide what a "run" means)
1. **Train/eval overlap check** (`first-training-run` §Caution): are the eval originals in `lora_train_data.jsonl`? If yes, an `--exclude-eval` filter in
   `preprocess.py` (tested, default on when `data/eval/bp-eval-dataset.jsonl` exists, count printed), then retrain. Otherwise MLflow would record
   contaminated numbers as history.
2. **`-fast` profiles: `eval_steps`/`save_steps` scaled to their ~90 steps** (they are 100, so no eval loss exists); base profile: 2 epochs is enough
   (eval loss minimum at step 1000).

## What exists that MLflow plugs into (checked in the repo, 2026-10-01)
- `.gitignore` already ignores `mlruns/`, `artifacts/`, `.mlflow/`; `requirements-unsloth.txt` has a commented `# mlflow`. Nothing else.
- Run entry points (all take `--model <profile>`, parameter > `RESONANCE_LF_PROFILE` env): `scripts/llamafactory/train.py` (runs `llamafactory-cli train <yaml>`
  via `lf_tools.train_command`), `merge.py`, `gguf.py`, `eval.py`, and `run_pipeline.py` (passes `RESONANCE_LF_PROFILE` to every stage through `stage_env`).
- LLaMA-Factory (pinned commit `ce9dc9e`, transformers 4.57.1) writes `trainer_log.jsonl`, `train_results.json`, `trainer_state.json` in the adapter dir; it
  accepts `report_to=mlflow` (HF Trainer's MLflow callback: logs training args as params, `loss`/`eval_loss`/`learning_rate`/`grad_norm` per step as metrics) and
  `llamafactory-cli train cfg.yaml key=value` overrides. Not verified against this pin — check first.
- `eval_metrics.evaluate()` returns one dict (chrF/BLEU/TER in `standard`, `jp_leakage`, `term_hits/total`, `discord_violations`, `exact_match`, `categories`); COMET separate.
- `preprocess.transform_for_lora()` returns the per-reason counts but only prints them.

## Design
**Tracking store.** Local: SQLite file `mlflow.db` in the repo root (gitignored; the file store `mlruns/` is the deprecated backend), UI with
`mlflow ui --backend-store-uri sqlite:///mlflow.db`. Remote jobs: the standard `MLFLOW_TRACKING_URI` env var (a server URL); credentials only via env
(`MLFLOW_TRACKING_USERNAME/PASSWORD`), never committed. Same rule as `--model`: nothing hard-coded in scripts; `config.py` holds the default store URI.
**Tracking is optional.** Off when `mlflow` is not importable or `RESONANCE_MLFLOW=0`; the pipeline must behave exactly as today without it.
**One run = one training (profile).** Experiment = `resonance-lab` (or per model family). Run name `<profile>-<yyyymmdd-hhmm>`. Later stages (merge, eval, gguf),
which the maintainer runs by hand on Windows as separate commands, must land in the SAME run:
- `train.py` creates the run and writes its id to `<adapter dir>/mlflow_run_id.txt`; `merge.py`/`gguf.py`/`eval.py --model X` read that file and resume the
  run (`mlflow.start_run(run_id=...)`). Under `run_pipeline.py` the id travels as `MLFLOW_RUN_ID` in `stage_env` (the HF callback resumes it too).
- Eval metrics are logged with a tag `eval_prompt=chat-template|training`, so several evals of one model sit in one run without overwriting.
**What is logged**
| kind | content |
|---|---|
| params | profile, `base_model`, `template`, prompt style (`--prompt`), `--reverse`, the whole `train.yaml` flattened (lr, epochs, batch, accumulation, cutoff, lora r/alpha, packing, liger, flash_attn), effective batch |
| tags | git commit + dirty flag, LLaMA-Factory commit / transformers / torch versions (from `pip`), host, `data_sha` (sha256 of `lora_train_data.jsonl`), `eval_sha` (eval set), `eval_overlap` (count, from the leak check), `pipeline` |
| data | preprocess report counts (total, passed, per-reason) — `preprocess.py` writes `data/processed/preprocess_report.json` so train.py can log it |
| train metrics | per step from the HF callback (`report_to=mlflow`); at the end `train_runtime`, `train_samples_per_second`, steps/s, `total_flos` from `train_results.json`, best eval loss + step from `trainer_state.json` |
| eval metrics | chrF, BLEU, TER, COMET (when installed), JP leak, think leak, term hits/total, discord violations, exact match; per-category counts as `cat_<name>_<metric>` |
| artifacts | `train.yaml`, `merge.yaml`, the eval report `.txt`, per-sample predictions JSONL (eval set only: it is made-up/hand-written, not player chat), `trainer_log.jsonl`. **Never** training data, weights, checkpoints (`HF_MLFLOW_LOG_ARTIFACTS=0`) or GGUFs (log path + sha256 + size as params) |
**Code layout (CLAUDE.md: new pure logic goes in the data part).** A new `tracking.py` at the root, importing `mlflow` lazily: pure, tested functions
(`flatten_params(yaml)`, `eval_metrics(report)`, `train_result_metrics(...)`, `git_info()`, `file_sha256()`, `run_id_path(profile)`, `read/write_run_id`) and one thin
wrapper `tracked_stage(profile, stage)` (context manager; no-op when disabled). Stage scripts call it in a few lines. Tests use a fake `mlflow` module
(monkeypatch `sys.modules`) plus one `pytest.importorskip("mlflow")` test against a temp SQLite store — `mlflow` is NOT added to `requirements-dev.txt`
(heavy); candidate `mlflow-skinny` only if the maintainer wants the gate to run the real client.

## PRs, in order (one task, one branch, one PR; wait for each merge)
0. the two "do first" items above (own PRs).
1. `tracking.py` + tests + `requirements-llamafactory.txt` pin (`mlflow`, exact version after checking Python 3.13/Windows and the transformers 4.57.1 callback) + `config.py` store default + docs. No stage uses it yet.
2. `train.py`: start/resume the run, params/tags, `report_to=mlflow` override (`train_command(..., extra=[])`, tested), end-of-train metrics, preprocess report + `data_sha`.
3. `eval.py`: eval metrics, report + predictions artifacts, `eval_prompt` tag. `merge.py`/`gguf.py`: stage tags, GGUF sha.
4. `run_pipeline.py --model`: run id through `stage_env`, mark the run FAILED when a stage fails.
5. `scripts/mlflow_compare.py` (or README snippet): print the last N runs as a table (profile, chrF, term acc, runtime, eval-loss min, data_sha) via `mlflow.search_runs` — the agent-readable replacement for pasting reports. Then retire `first-training-run`-style tables from `.memory/` in favour of run ids.
6. Remote job doc: `MLFLOW_TRACKING_URI` + credentials in the job environment; artifact upload size; clock/time zone.

## Decisions to take with the maintainer at the start of the session
- Local SQLite vs a tracking server on the remote machine (or both: local by default, server via env on remote jobs).
- May per-sample predictions (eval set only) go to a shared server? (they are made-up/hand-written lines; training data never goes.)
- One experiment for everything or per model family; run naming.
- Back-fill: log the existing runs (hy-mt2-1.8b, `-fast`, TG-4B, zero-shot) from the saved `outputs/` files, or start clean after the leak fix (recommended: start clean).
- Also track the `unsloth` pipeline? (its `train.py` runs at import; out of scope unless asked.)

## Risks / unknowns
- `mlflow` vs transformers 4.57.1's `MLflowCallback` and Python 3.13 on Windows: unverified; the callback needs `MLFLOW_RUN_ID` resume support.
- `llamafactory-cli ... report_to=mlflow` override syntax and whether LF logs the full args: verify with a 5-step dry run on the GPU machine.
- MLflow param values are limited (~6000 chars) and names restricted: flatten yaml carefully, test it.
- The model part cannot run in a cloud session: stage integration (PRs 2-4) is NOT VERIFIED until a manual run; pure helpers are tested in the gate.
