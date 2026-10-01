# Plan: MLflow experiment tracking (maintainer, 2026-10-01: "I would use mlflow ... I'll start from a new session")

## REVISED 2026-10-01 — maintainer's decisions (these override the "Design" section where they differ)
- **Server on the maintainer's Synology NAS** (Intel Celeron J4025, x86_64, 10 GB RAM, Docker/docker-compose), reached over the LAN from the Windows
  desktop (same gateway). `deploy/mlflow/` holds compose + `.env.example` + README. Server URL and credentials live only in the gitignored
  `.env.mlflow` (committed `.env.mlflow.example` has blank values). Not exposed to the internet; DSM serves port 5000, so MLflow uses 5050;
  MLflow basic-auth, single user; image tag == client version; `--allowed-hosts`; no Projects/model serving/registry/gateway; non-root, read-only
  fs, `cap_drop: ALL`, no `docker.sock`, 1 GB limit; **superseded 2026-10-01: Postgres (`deploy/postgres/`) for records + basic-auth users and MinIO for artifacts, both already on the NAS (maintainer); backup = `pg_dump` + bucket.**
- **Metadata + small files.** Params, tags, metrics, plus small artifacts: eval report `.txt`, per-sample predictions (eval set only: hand-written),
  `train.yaml`/`merge.yaml`, `trainer_log.jsonl`. `--serve-artifacts` on the NAS volume. Never training data, weights, checkpoints, GGUFs.
- **Dataset = public HF repo**, refreshed about every 2 months. Download with `hf download <repo> --repo-type dataset --revision <sha>`;
  `configs/hf_dataset.yaml` pins repo + revision (each refresh is a small PR). Every run records repo id, link, revision, `dataset_version`, the
  sha256 of the files used, row counts and per-reason drops, and the eval-overlap count.
- **Offline queue, write-locally-first.** NoSQL local store: TinyDB, file `.run.result.backup.json` (gitignored) + `.run.result.backup.files/` for
  small attachments. Flow: (1) open a run record at the start of the pipeline / a standalone stage; (2) every stage appends params/tags/metrics to it;
  (3) ping the server (short timeout, one retry); (4) when up, replay pending records oldest first as a queue, stop at the first failure; (5) when
  down, leave them pending. Idempotent: each record has a `local_run_id` tag checked on the server before replaying; later stages resume the stored
  `mlflow_run_id`. Tracking never raises (a dead NAS must not fail or hang a training run).
- **No HF live callback** (`report_to=mlflow`): after training, parse LLaMA-Factory's own `trainer_log.jsonl` and send the curves in `log_batch`
  chunks of <=1000, so online and offline runs share one path. Live view stays with `watch_training.py`.
- Client package: prefer `mlflow-skinny` (check Python 3.13 / Windows / pinned torch). Cloud sessions cannot reach the NAS: `mlflow_compare.py`
  runs on the desktop.
- Run-identity prerequisites (unique adapter dir per run, sidecar `.meta.json`, per-run log) come first: `pipeline-review-2026-10-01.md` #2, #3, #21.
- **PR 1 (server) facts found by running a real MLflow 3.16.1 server (2026-10-01):** basic-auth needs `mlflow[auth]` (Flask-WTF) and
  `MLFLOW_FLASK_SERVER_SECRET_KEY`, so the image is built from PyPI, not the official one; the shipped auth ini defaults to `default_permission = READ`
  (we use `NO_PERMISSIONS`); the admin password can come from `MLFLOW_AUTH_ADMIN_PASSWORD`; **job workers are ON by default**
  (`MLFLOW_SERVER_ENABLE_JOB_EXECUTION`) and are switched off, with the assistant/sandbox; telemetry is switched off; `/health` is the only open URL, `/signup`
  and user creation are 401; a foreign Host header is 403; `log_batch` took 1000 metrics, `start_time` can be backdated, `search_runs` finds a run by tag
  (the replay's idempotence check). Files: `deploy/mlflow/`, `.env.mlflow.example`.
- **PR 2 (`tracking.py`, pure) done 2026-10-01:** `.env.mlflow` loader (env over file; only an http(s) URL is accepted, never a local store), client env with a 10 s timeout /
  1 retry / backoff 1 (the package defaults are 120 s and many retries), params (`flatten_params`, MLflow limits: key 250, param 6000, tag 8000, 100 params+tags and
  1000 metrics per batch), `dataset_tags` (HF repo/revision/url from `data/hf/fetch_state.json`, file hashes and eval overlap from the preprocess manifest),
  `data_params`, `prompt_fingerprint`, `eval_metrics`, `train_result_metrics`, `step_metrics` (points from `trainer_log.jsonl`, timestamped start + elapsed_time),
  `git_info`, `package_versions`, `gguf_info`. **`mlflow-skinny==3.16.1` is enough for the desktop** (checked on Python 3.11 against a real server: params, 1000-metric batch,
  tags, artifact, search by tag; no torch/Flask); the pin equals the server's version (test). Not checked on Python 3.13 / Windows. A refused connection fails in ~1 s;
  a NAS that drops packets takes the 10 s timeout. With `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` the server has no huey job-worker processes (checked).
- **PR 3 (offline queue) done 2026-10-01:** `run_queue.py` (TinyDB document store, atomic temp-file+replace writes because TinyDB's own storage writes in place and a crash would
  corrupt every pending run; a corrupt file is set aside as `.corrupt-<ts>`, never fatal) + `tracker.py` (`sync`, `Tracker`, `NullTracker`, `MlflowAdapter`, `from_environment`).
  A run = a list of events (params, tags, metrics, step_log, artifact, status), each marked sent right after the server has it; `begin()` records locally, pings, replays every
  pending run oldest first and stops at the first failure; a run found on the server by its `local_run_id` tag is not created twice; synced runs are pruned to the newest 20.
  Metric points everywhere are `(name, value, timestamp_ms, step)` = MLflow's `Metric` order. **Run against a real 3.16.1 server (skinny client) the first time exposed a swapped
  timestamp/step that the unit tests had pinned on both sides; a cross-module test now covers it.** Checked end to end: a run recorded with the server off (log line, files copied into
  the queue) arrived complete (params, tags, 300 loss points at the right times, artifact, FINISHED) when the next run started; nothing was sent twice.
- **PR 4a (done, data part, tested): `track_records.py`** assembles each stage's records from the `tracking.py` helpers: `run_identity` (tracker run = `<profile>-<run id>`, no need to store
  an id in `run.json` or pass it through `stage_env`: merge/gguf/eval derive it from the run `resonance_run.json` names, `runs.read_merge_record` + `stage_run_id`; a pre-runs adapter -> None -> no
  tracking), `train_records` (params: recipe + data; tags: dataset revision/hashes, prompt fingerprint, git, packages), `train_result_records`, `merge_tags`, `gguf_tags`, `eval_records`.
  `config.MLFLOW_EXPERIMENT`, `config.EVAL_MAX_NEW_TOKENS`. Params are logged by train only; eval decoding/prompt are tags (a param cannot change once logged). Known limit: two evals with different
  `--prompt` in one run write the same `eval.*` metric keys (history of one metric, tag = the last prompt).
- **Found on the first GPU run (2026-10-01): `report_to` unset = transformers' `all`**, so with `mlflow-skinny` installed the trainer's own MLflow callback started and died on a local
  `sqlite:///mlflow.db` the skinny client cannot open (`UnsupportedModelRegistryStoreURIException`, Fine-Tuning FAILED after 22 s). Every `train.yaml` now has `report_to: none` (a test guards it).
- **PR 4b (done in code 2026-10-01, NOT VERIFIED on the GPU, see `unverified-on-gpu.md`): the thin wiring** via the tested `stage_tracking.py` (`open_tracker` never raises, `start_training`, `finish_training`, `resume_stage`, `record_stage`, `fail_stage`); `run_queue.prune` now also prunes runs that later stages continued. Plan as written: Callers checked: `train.py` (`start_run`/`finish_run`), `merge.py` (`resolve_adapter`/`write_merge_record`), `gguf.py`, `eval.py`
  (hardcodes `max_new_tokens=256` -> use `config.EVAL_MAX_NEW_TOKENS`). `train.py`: `tracker.from_environment()`, `begin(MLFLOW_EXPERIMENT, name, local_run_id)`, params + tags (`git_info`, `package_versions`,
  `read_state(data/hf/fetch_state.json)`, `read_manifest`), after training `step_log(trainer_log.jsonl)` + `train_result_records` (read `train_results.json`/`trainer_state.json` from the run dir) + `finish`,
  `flush`; failure -> `finish("failed")`. `merge.py`/`gguf.py`/`eval.py`: `resume(stage_run_id(read_merge_record(merged_dir)))`, tags/metrics/artifacts (report, predictions), `flush`.
  Every call goes through `Tracker` (never raises); off = `NullTracker`. Failure marking is done in each stage script (not `run_pipeline.py`): the stage that fails marks its run failed. Then PR 5:
  `scripts/mlflow_compare.py`. Pipeline-review items #4, #6-#8 (GGUF eval, term metric, bigger eval set, COMET/pins) follow.
- PR order: 1 server (`deploy/mlflow/` + guard tests), 2 `tracking.py` pure helpers, 3 the queue (TinyDB + sync, fake client in tests), 4 stage
  wiring (model part, NOT VERIFIED), 5 `mlflow_compare.py` + docs.

**Status: PR 1-3 and 4a merged/in review; 4b (stage wiring) next.** Per CLAUDE.md the next session presents this (adjusted) and waits for the maintainer's OK before touching code.
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
