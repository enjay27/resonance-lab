# Not run on the GPU machine

Code or config changed in a session without CUDA. Delete an item once it has run.

- **`scripts/mlflow_compare.py` (2026-10-02):** data part (needs `.env.mlflow` and `mlflow-skinny`, no GPU). Unit-tested with a stand-in client and run against a real MLflow 3.16.1 server with runs recorded through
  the tracker (sorted by eval loss / chrF, killed run hidden, markdown). Not run against the NAS or your real runs: `python scripts\mlflow_compare.py` should list `hy-mt2-1.8b-fast-...` with their eval loss;
  the eval columns stay `-` until a run's eval stage ran (two evals with different `--prompt` in one run share the same metric keys: the table shows the last prompt).

- **`--fast` and the new default profile (2026-10-02):** `--fast` on train/merge/gguf/eval/update_dataset_info/watch_training/inspect_* and `run_pipeline.py` resolves `<model>-fast`; the default profile is
  `hy-mt2-1.8b` now. Unit-tested (name resolution, parsers, `stage_env`), the scripts import. Not run on the GPU machine: `python run_pipeline.py --fast` (default model, ~4 min) and
  `python scripts\llamafactory\train.py --model hy-mt2-1.8b --fast` should both train `hy-mt2-1.8b-fast`; check `eval.py --fast` and `watch_training.py --fast` follow the same run.

- **Profile `translategemma-4b-lr1e-4` (2026-10-02):** YAML only, unit-tested (differs from `translategemma-4b` only in `learning_rate` 1e-4 and its own `outputs/translategemma-lr1e-4_lora` / `model_gemma-lr1e-4_merged`
  folders). Never trained. Why: the lr 1e-5 run was underfit (eval loss 0.888 still falling at the last step, gap +0.04; the Hy profiles use 2e-4). Compare in MLflow: eval-loss curve, best checkpoint, then both eval
  prompts (`eval.py --model translategemma-4b-lr1e-4 [--prompt training]`) against chrF 45.2 / term accuracy 1/21 of the lr 1e-5 run. A higher lr can overfit: watch the train/eval gap.

- **`-fast` eval cadence (2026-10-02):** the three `-fast` profiles now have `eval_steps: 10`, `save_steps: 10` (were 100 = never evaluated in an ~87-step run). Unit-tested (YAML only). Check on the next `-fast` training:
  `eval_loss` rows in `trainer_log.jsonl` / the MLflow curve, the best checkpoint chosen (`train.best_checkpoint` tag), and that the extra evals (199 validation rows) cost seconds, not minutes.

- **Curves, constants, experiment kind (2026-10-02):** `finish_training` writes `<run dir>/curves.jsonl` (the trainer log + `grad_norm` joined from `trainer_state.json`) and sends it as the curves;
  `train.epochs` / `train.global_step` / `train.total_flos` are tags now (they were bar charts); the adapter sets the experiment tag `mlflow.experimentKind=finetuning` on a new experiment and once on an existing one
  without it (a kind chosen in the UI is kept). Unit-tested; the tag logic ran against a real MLflow 3.16.1 server. NOT confirmed: that the web UI then shows the training-runs view with line charts
  (read from the UI's code, not seen) and that `grad_norm` appears in a real run: check on the next training.

- **Tracker rough edges (2026-10-02):** Ctrl+C during `train.py` now closes the run `KILLED` (and `run.json` `failed`); a failed artifact upload no longer stops the run's closing status or the runs after it
  (3 tries, then given up and logged); the run is closed before its artifacts go up; artifacts are `train.yaml`, the manifest and `trainer_state.json` (+ `train_stdout.log` only for a FAILED/KILLED run); tags
  `train.trainable_params` / `train.all_params` come from the trainer's log line. Unit-tested, and the close-then-upload order was run against a real MLflow 3.16.1 server (KILLED + artifact). Never run on the
  GPU machine: press Ctrl+C in a `-fast` training and check the run shows KILLED, then a normal run shows FINISHED with those three artifacts and the two tags. Found on the NAS: the MLflow container's
  `MINIO_ENDPOINT_URL` still held the example address (`192.168.0.10`), so every upload timed out; the real LAN address fixed it.

- **Stage wiring to the tracker (MLflow PR 4b, 2026-10-01):** `train.py`, `merge.py`, `gguf.py` and `llamafactory/eval.py` call `stage_tracking.py` (unit-tested with a recording tracker,
  the real `Tracker` + offline queue and a fake client; never run on the GPU machine). Check with the NAS up and `.env.mlflow` filled: (1) `train.py` creates the run
  `<profile>-<run id>` in experiment `resonance-lab` (params, `dataset.*`/`prompt.*`/`git.*` tags) and, when it ends, the loss curves, `train.*` / `eval.best_loss` metrics and the log
  artifact, status FINISHED; (2) `merge.py`, `gguf.py` and `eval.py --prompt training` add their tags/metrics/report to the SAME run (`stage.merge`/`stage.gguf`/`stage`, `gguf.size_bytes`, `eval.chrf`...);
  (3) with the NAS off (or `RESONANCE_MLFLOW=0`) every stage behaves as before and the next run sends the backlog; (4) the eval's `EVAL_MAX_NEW_TOKENS` is 256, as the hardcoded value was.
  Seen on the maintainer's machine 2026-10-02: a `-fast` training queued its run while the NAS login was wrong (401, training unaffected) and `[tracking] sent 2 run(s) to MLflow` after it was fixed; the run's params/tags were in `.run.result.backup.json`. The loss charts could not be seen until the MLflow server allowed the UI's own origin (CORS; `deploy/mlflow/docker-compose.yml`, redeploy on the NAS). Still to confirm in the UI: curves, `train.*` metrics, stage tags, the report artifact, merge/gguf/eval in the same run.
  Also unchecked: `train_results.json` / `trainer_state.json` land in the run dir (`train_result_records` finds them there), and a failed stage tags `stage.<name>=failed` without closing the run.

- **Offline queue (2026-10-01, `claude/relaxed-pascal-xjeh11`):** `run_queue.py` / `tracker.py` are unit-tested with a fake client and were run end to end against a real MLflow 3.16.1
  server (skinny client, Python 3.11, Linux): offline record, replay, no duplicates, correct metric times. Not run on Windows / Python 3.13 (file replace semantics, `tinydb`),
  and not against the NAS container. Nothing in the pipeline calls the tracker yet (the stage wiring PR does).

- **`mlflow-skinny` on the desktop (2026-10-01, `claude/relaxed-pascal-xjeh11`):** `requirements-llamafactory.txt` now pins `mlflow-skinny==3.16.1` (the NAS server's version). Checked here on
  Python 3.11 only; install it in the llamafactory venv (Python 3.13, Windows) with the other pins and check `pip` resolves it next to torch 2.9.1 / transformers 4.57.1 / trl 0.24.0.
  `tracking.py` itself is unit-tested and sends nothing yet.

- **MLflow on Postgres + MinIO (2026-10-01, `claude/modest-edison-0nvyf5`):** `deploy/postgres/` (compose, `initdb/10-mlflow.sh`) is new; `deploy/mlflow/` moved from SQLite to
  Postgres (records + basic-auth users) and MinIO (artifacts), with `entrypoint.py` rendering the auth config. Checked in a cloud session with a real PostgreSQL 16 (scram; the compose file now pins `postgres:18-alpine`, never run: check MLflow's schema migration on 18 and that the volume on `/var/lib/postgresql` holds `18/docker`), the
  init script, `entrypoint.py` + `mlflow server` 3.16.1 and a moto S3 server: 401/403/200, run + artifact written. Never built or run in Docker: on the NAS check `docker compose up`
  of both projects (Postgres first), the `resonance-db` network join, `user: 70:70` + read-only root for Postgres, that the postgres image's entrypoint passes `PGPASSWORD` to the
  init script (the first NAS run printed `chmod: /var/run/postgresql: Operation not permitted` and `ls: can't open '/docker-entrypoint-initdb.d/'`; fixed by `deploy/postgres/Dockerfile` copying `initdb/` in and a tmpfs with `uid=70`: re-run with `docker compose down -v` first), a real MinIO (bucket policy, path-style), and the README's `curl` checks. This replaces the SQLite-server item below (its Dockerfile/compose no longer exist).

- **MLflow server (2026-10-01, `claude/relaxed-pascal-xjeh11`):** `deploy/mlflow/` was checked as far as a cloud session can: a real MLflow 3.16.1 server with the same flags/env
  (401/403/200 behaviour, logging, artifacts) and `docker compose config`. Never built or run in Docker (no daemon here): on the NAS check the image build
  (`python:3.12-slim` + `mlflow[auth]`), that the container starts with `read_only: true` + tmpfs `/tmp`, the health check, and the README's `curl` checks from the desktop.

- **Runs (2026-10-01, `claude/relaxed-pascal-xjeh11`):** `train.py` now passes `output_dir=outputs/<profile>_lora/<run id>` to `llamafactory-cli train`, `merge.py` passes
  `adapter_name_or_path=<run dir>` to `export`, the monitor follows the latest run. Unit-tested with faked commands only. `key=value` overrides after the yaml and
  the "only weight files make an output_dir non-empty" rule were read in the pinned source (`hparams/parser.py` `read_args` 117-120, 612-623; `CHECKPOINT_NAMES`), never run.
  Check on a real run: training starts in the new dir and `trainer_log.jsonl` lands there, the monitor follows it, `merge.py` merges it and writes
  `resonance_run.json`, and `--from merge` works. Older adapters in `outputs/<profile>_lora/` still merge (legacy path).

- **Fetch Data stage (2026-10-01, `claude/relaxed-pascal-xjeh11`):** `scripts/fetch_data.py` is unit-tested with a faked `hf download`; the real `hf` CLI was never run
  (no HF access in the cloud session). Check on the desktop: set `repo:` in `configs/hf_dataset.yaml`, `python scripts/fetch_data.py --pin`, then run it: the
  `hf download ... --include "dataset_*.jsonl" --local-dir data/hf` flags, the merged `data/raw/raw_translated_logs.jsonl`, and that a second run says "up to date".
  `huggingface_hub`'s `HfApi().dataset_info(repo).sha` is what `--pin` uses. The state (`data/hf/fetch_state.json`) is what MLflow will log as the dataset revision.

- **Data-check limits (2026-10-01, `claude/relaxed-pascal-xjeh11`):** `validate.py` (1% damaged / 10% Hangeul originals) and the `preprocess.py` guard (30% suspicious)
  are unit-tested on made-up rows; the numbers are guesses, never run on the real raw log. Both stages print their shares: read them on the first real
  run and move the limits in `config.py` (or pass `--max-drop`) if they stop a healthy file or let a bad one through.

- **Validation split (2026-10-01, `claude/relaxed-pascal-xjeh11`):** `preprocess.py --format pair` writes `lora_train_data.val.jsonl` (5%, by line hash); every
  `train.yaml` now has `eval_dataset: bp_translation_val` and no `val_size`; `dataset_info.json` has two entries. Unit-tested; `eval_dataset` and its
  exclusion of `val_size` read in the pinned LLaMA-Factory source (`hparams/data_args.py` 34, 171), never run. Check on the first real run that
  training starts, that eval-loss rows appear in `trainer_log.jsonl`, and the Preprocessing report's `Validation rows` is about 5% of `Passed`.
  Eval-loss numbers are NOT comparable with the earlier runs (different validation rows).

- **Manifest check (2026-10-01, `claude/relaxed-pascal-xjeh11`):** `preprocess.py` writes `data/processed/lora_train_data.meta.json`; `update_dataset_info.py`
  and `train.py` refuse to run without a matching one. Unit-tested only. Old processed data has no manifest: run the Preprocessing stage again
  (`python run_pipeline.py` does). Check that a full `run_pipeline.py` still reaches Fine-Tuning, and that `update_dataset_info.py` (now takes `--model`)
  still works as a pipeline stage.

- **`preprocess.py` eval exclusion (2026-10-01, `claude/relaxed-pascal-xjeh11`):** unit-tested (`tests/test_overlap.py`, `test_preprocess.py`); not run on the real
  raw log. Run `preprocess.py` once and read the `eval overlap` / `eval overlap (near)` counts: they are the size of the leak. Then retrain
  and re-run `eval.py` (every earlier score is on contaminated data). Near matches use difflib ratio >=0.9 on lines of >=10 normalised characters.

- **`run_pipeline.py` after `claude/align-project-structure-s21g7q` (2026-10-01):** lint
  removed an unused `shutil` import and the unused `result =` in `run_step` — no
  behaviour change, but never run since. `validate.py` raises its own `ValidationError`
  instead of `huggingface_hub`'s (unit-tested; pipeline halt unchanged).
- **`scripts/unsloth/*` moved (2026-10-01, PR #3):** pure move + one path-depth line per script;
  `run_pipeline.py --pipeline unsloth` never run since. Check: `python run_pipeline.py` reaches
  every stage (the stage scripts' `config` import resolves).
- **`preprocess.py` filters (2026-10-01, `claude/preprocess-filters`):** unit-tested; not run on a
  real raw log. It now drops more rows than before (the branch's filters) — compare the
  Preprocessing Report counts on the real data before training, and re-run `eval.py` after.
- **`llamafactory` pipeline (2026-10-01, `claude/llamafactory-pipeline`):** `update_dataset_info.py`,
  `train.py`, `merge.py`, `gguf.py` ran here only as far as their error paths (missing
  `llamafactory-cli`, adapter, merged model); `lf_tools` is unit-tested. Never run end to end:
  check `python run_pipeline.py` on the Windows/CUDA machine — in particular that
  `llamafactory-cli` accepts the profile yaml from the repo root, that `dataset_info.json`
  is found (`data/`), `flash_attn: fa2` installs, and that `gguf.py` finds `llama-quantize`.
  `requirements-llamafactory.txt` leaves `llamafactory` unpinned — pin what you trained with.
- **`watch_training.py` (2026-10-01, `claude/training-monitor`):** state/render unit-tested; `main`
  ran here against fake logs only. On a real run check that LLaMA-Factory's `trainer_log.jsonl`
  really has `current_steps` / `loss` / `epoch` / `eval_loss` (the branch's monitor assumed so),
  that `train_stdout.log` carries the `'grad_norm'` dict lines and the tqdm bar, and that
  `nvidia-smi` is on the PATH.
- **Eval stage (2026-10-01, `claude/eval-metrics`):** `eval_metrics.py` is unit-tested; the two
  `eval.py` scripts ran here only as far as their error paths (llamafactory) / a parse check
  (unsloth). Never generated anything: check `--prompt chat-template` and `--prompt training` on
  the real merged model (tokenizer call, `enable_thinking`, BOS handling), COMET with `unbabel-comet`,
  and that the unsloth eval's `inputs.shape[-1]` slicing (tensor, not dict) is right.
- **Hy-MT2 profiles (2026-10-01, `claude/next-task-model-selection-nzghmf`):** `configs/llamafactory/hy-mt2-{1.8b,7b}/`,
  Update: step 1 ran locally -- both templates MATCH; neither tokenizer adds BOS (eval `--prompt training` now prepends it, `with_bos`).
  LLaMA-Factory: PyPI 0.9.5 lacks the Hy templates; source commit ce9dc9e0 has them and is now pinned (with transformers 4.57.1, peft 0.18.1, trl 0.24.0). eval.py `chat-template` crashed on `token_type_ids` (Hy tokenizer) -> `generate_inputs`; re-run to confirm. **Confirmed 2026-10-01: eval.py ran on hy-mt2-1.8b (both prompts) and translategemma-4b (chat-template); results in roadmap/zero-shot-results-2026-10-01.md.**
  Local py3.13 raised `linecache._register_code ... 'str' has no attribute 'co_consts'` in a `python -c` run: open.
  `TRAINING_PROMPTS` / `chat_messages` in `lf_tools.py` (unit-tested), `scripts/llamafactory/inspect_template.py`.
  huggingface.co is blocked in cloud sessions, so the strings come from Tencent's README and from
  LLaMA-Factory's `template.py` (templates `hy_dense_1_8b`, `hy_dense_7b` are registered upstream; both read
  from GitHub). **Check locally, in this order:**
  1. `python scripts/llamafactory/inspect_template.py --model hy-mt2-1.8b` (then `hy-mt2-7b`):
     must print `MATCH`; also read whether the tokenizer adds BOS on its own (`eval.py --prompt training`
     encodes with `tokenizer(...)`, so a missing BOS there is an eval-only mismatch).
  2. The installed LLaMA-Factory knows the two template names (`pip show llamafactory`; they are new).
  3. The 1.8B and 7B templates differ (1.8B: `<｜hy_User｜>`/`<｜hy_Assistant｜>`, 7B: `<|extra_0|>`):
     confirm against each model's `tokenizer_config.json` chat template, and that the Hy-MT2 GGUF's embedded
     template (what llama-server uses) agrees.
  4. Hyper-parameters are Tencent's LoRA defaults (r64/α128, q/k/v/o, lr 2e-4), `cutoff_len` 256,
     gradient checkpointing on — not our 4B recipe (lr 1e-5). The 7B in bf16 LoRA may not fit your GPU:
     add `quantization_bit: 4` to its train.yaml if so. Nothing trained.
  5. Recommended sampling (README): temperature 0.7, top_p 0.6, top_k 20, repetition_penalty 1.05;
     `eval.py` uses greedy — fine for comparison, note it.
- **`--model` parameter (2026-10-01):** parsing and precedence are unit-tested and the error paths ran here; train/eval/inspect/watch with a real model not run. `inspect_template.py`'s old `--model` (tokenizer id) is now `--tokenizer`.
- **`*-fast` profiles (2026-10-01):** `translategemma-4b-fast`, `hy-mt2-1.8b-fast`, `hy-mt2-7b-fast` = the base profile with
  packing + Liger + a bigger batch (4B 16x2, 1.8B 32x1, 7B 8x4 with checkpointing), own output/export dirs. Never run.
  Liger on Windows: `pip install liger-kernel` fails (wants `triton`, only `triton-windows` exists) -> install `triton-windows<3.6` then `liger-kernel --no-deps` (**verified locally 2026-10-01:** `triton-windows<3.6` -> triton 3.5.1, `import liger_kernel` works with torch 2.9.1; the `pynvml` FutureWarning is harmless), else set `enable_liger_kernel: false`. Check; `flash_attn: auto`
  picks fa2/sdpa; the batch fits VRAM; then compare time (watch_training tok/s) and eval chrF against the base profile.
  `python run_pipeline.py --model hy-mt2-1.8b-fast`. If fa2 works, add `neat_packing: true`.
- **`python -c` crashes on the maintainer's Python 3.13** with `linecache._register_code ... 'str' has no attribute 'co_consts'` (seen with `-c` only; the same imports from a .py file work). Run checks from a file, not `-c`. Cause not investigated.
- **`scripts/llamafactory/inspect_pair.py` (2026-10-01; run locally: the API calls work, the eos was missing -> fixed, re-run to see MATCH):** prints the training example LLaMA-Factory builds (masked prompt, trained
  response, EOS, tokens vs cutoff) via `template.encode_oneturn`. `first_pairs` is tested; the LLaMA-Factory calls are written from
  memory of its API (`get_template_and_fix_tokenizer`, `DataArguments`, `encode_oneturn`) and never ran. Run:
  `python scripts/llamafactory/inspect_pair.py --model hy-mt2-1.8b --rows 3` (needs `data/processed/lora_train_data.jsonl` from
  `preprocess.py --format pair`). Check: the response ends with the model's end-of-turn token; the prompt column is the raw line.
- **`preprocess.py --prompt auto` (2026-10-01):** unit-tested only. The llamafactory pipeline's Preprocessing stage now writes the
  instruction into `original` (training data changes -> a retrain needs the eval). Check on the real raw log:
  `python scripts/preprocess.py --format pair --prompt auto --model hy-mt2-1.8b [--reverse]`, then
  `python scripts/llamafactory/inspect_pair.py --model hy-mt2-1.8b` (MATCH, tokens vs cutoff_len 256 -- TG's instruction is ~60 tokens
  longer than the line, cutoff 128 may truncate: raise it). `eval.py --prompt training` now wraps the line in the style too.
- **`--reverse` rule + TG `cutoff_len` 256 (2026-10-01):** the rule is from `aab6b66` and unit-tested, never run on the real raw log. Check the
  Preprocessing Report on it (reverse rows are counted in Total/Passed, reasons shared) against the shipped processed file's row count.
  `translategemma-4b(-fast)` train.yaml now has `cutoff_len: 256` (the shipped recipe), which changes tokens/step in the monitor.
- **`watch_training.py` rework (2026-10-01):** speed/ETA/lr/eval now come from `trainer_log.jsonl` (fields confirmed from a real run); unit-tested and
  rendered here on synthetic rows, not watched on a live run. Check on the next training: s/step and samples/s appear after a restart, the
  footer's eval/overfit lines, and `nvidia-smi` temperature/power (`power.draw` may print `[N/A]`: shown as missing).
