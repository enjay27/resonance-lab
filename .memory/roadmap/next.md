# Next — candidates, in order

## Queued by the maintainer (2026-10-02): run queue as an append-only journal — **the next task after the lr 1e-4 profile decision**
Idea: `run_queue.py` stores `.run.result.backup.jsonl`, one line per record, read in batches; a sent line is skipped, and sent lines are dropped at the next start. Agreed refinements
(assessment in the session): mark sent by **appending an ack line** (editing a line in place needs a whole-file rewrite, which is what TinyDB does now and what loses an update when two
processes write); line types `run` (local id, experiment, name, start), `event`, `remote` (server id), `sent`; **keep the run header + `remote_id`** of the newest N runs (merge/GGUF/eval
resume a run days later by it, and `remote_id` stops a double create) and drop only sent event lines and their copied files; ignore a torn last line; read the old `.run.result.backup.json`
once to migrate. Only `run_queue.py` + tests change (data part); the tracker/stage API stays. Not urgent: the current queue works (a 401 run stayed queued and was sent later, 2026-10-02).

## Worked in this order (2026-10-01, maintainer approved): [`pipeline-review-2026-10-01.md`](pipeline-review-2026-10-01.md)
Data safety (eval-overlap exclusion **done**; sidecar manifest, pair/time split, validate vs preprocess, Fetch Data) -> run identity (unique adapter
dir, per-run log, `--from/--only`) -> MLflow PRs (revised in `mlflow-plan.md`) -> eval upgrades -> model changes. The two items below are inside it.

## Start here in the next session (maintainer, 2026-10-01: "I'll start from a new session")
1. **Leak check + fixes** (small): are eval originals in `lora_train_data.jsonl`? (`roadmap/first-training-run-2026-10-01.md` §Caution); `-fast`
   profiles' `eval_steps`/`save_steps`; the base profile needs ~2 epochs. Details: that file.
2. **MLflow tracking**: plan in `roadmap/mlflow-plan.md` (present it, wait for OK, then PR by PR).
3. Then: `[P0]` rule, TG-4B retrained on the same cleaned data as the control, Hy-7B; pick the family, copy the prompt to resonance-stream.
Everything below this block is older history of how the pipeline work was split; the A-D pipeline PRs are merged (#3-#7).

1. **Model choice + prompt** — this repo decides, resonance-stream follows
   (`active-issues/stream-contract.md` §1). Plan: `translator-shortlist-2026-10-01.md`
   (zero-shot eval round, then fine-tune the top family on the app's exact prompt).
   Blocks the re-fine-tune (stream's A4).
2. **Bring `experiment/translategemma`'s features in as a switchable pipeline** (maintainer,
   2026-10-01: keep the Qwen3 pipeline; take only the branch's features; `llamafactory`
   becomes the default). One PR at a time, each merged before the next:
   - **A — registry + move (done):** `pipelines.py`, `--pipeline`, `scripts/unsloth/`.
   - **B — shared preprocess (done):** the branch's filters (Hangeul in source, JP residual,
     10x-length hallucination, recruitment spam, first-occurrence dedup, per-reason report,
     malformed JSON counted not fatal) as `clean_reason`, tested first. **Changes the
     training data** (rows are dropped) — a feature, so re-run `eval.py` after the next
     training. Not taken yet: the branch's `{system,original,translated}` output format
     (its consumer is `update_dataset_info.py`, which reads a different file; decide in C).
   - **C1 — `llamafactory` pipeline (done, default):** `configs/llamafactory/translategemma-4b/`
     (the branch's yaml, dataset renamed `bp_translation`), `scripts/llamafactory/` (`lf_tools.py`
     pure + tested; `update_dataset_info.py`, `train.py`, `merge.py`, `gguf.py`),
     `requirements-llamafactory.txt` / `requirements-unsloth.txt` (separate venvs), `preprocess
     --format pair`, registry `Stage` with args. Deviations from the branch, on purpose: Merge
     runs before any eval (the branch ran eval.py before export.py, but eval loads the merged
     model); commands are argument lists run from the repo root (no `shell=True`);
     `llama-quantize` is found under any `build/bin/` layout (the branch hardcoded
     `Release/llama-quantize.exe`); the dataset is preprocess's own output
     (`processed/lora_train_data.jsonl`) — the branch pointed at a hand-made
     `raw/bp-training-dataset-final.jsonl`; the unused with-system dataset and the system
     prompt file are dropped (training is system-less); GGUF names `bp-<profile>-*.gguf`.
   - **C2 — training monitor (done):** `rich` replacement for the curses `watch_training.py`
     `scripts/llamafactory/watch_training.py`; `TrainingState` (pure, tested) follows
     `outputs/train_stdout.log` and `<adapter dir>/trainer_log.jsonl`. Changed from the branch:
     tok/s uses batch x accumulation x cutoff from the yaml (the branch hard-coded 8 x 256,
     but cutoff is 128); only JSON errors are swallowed (the branch had a bare `except`);
     Ctrl+C instead of `q`; needs a ~110-column terminal.
   - **D — shared eval metrics (done):** `eval_metrics.py` (chrF/BLEU/TER, COMET, JP + think
     leakage, `TERM_DICT` accuracy, discord, exact match, categories, report), used by both
     pipelines' `eval.py`; `text_rules.py` holds the shared JP/Hangeul patterns. Evaluation is the
     LAST stage of both pipelines (it only reads the merged model; a missing eval set must not block
     the GGUF). `llamafactory/eval.py --prompt chat-template|training` measures the prompt gap.
     **Fixed a bug of the branch:** its BLEU/chrF/TER scored only the first sample
     (`active-issues/old-eval-numbers.md`). Discord violations now count samples, not word hits.
   Prompt format is not in the registry yet (one pipeline carries the contract); revisit
   when the model is chosen (`translator-shortlist-2026-10-01.md`).
3. **Ingest the app's per-channel files** (`dataset_<CHANNEL>.jsonl`) instead of one
   hand-made `raw_translated_logs.jsonl` — a data-part script, tested.
4. **Model scripts testable** — `train.py` / `eval.py` / `fix_metadata.py` run at import;
   move their pure parts (formatting the chat text, the `score.weight` filter) into
   functions the data gate can test. Behaviour-preserving, one PR.
5. **`ruff format` baseline** — one formatting-only commit, its hash in
   `.git-blame-ignore-revs`, then `fmt-check` into `just check`. Off by the maintainer's
   call (2026-10-01).

## After the pipeline work (A-D)
- **Re-baseline:** run the shipped TranslateGemma-4B through `llamafactory/eval.py` with both `--prompt`
  modes (the old chrF numbers are invalid) — the first step of the shortlist's zero-shot round.
- A `llamafactory` profile per shortlist candidate (Hy-MT2-1.8B/7B, Gemma 4 E4B, TranslateGemma-12B):
  `train.yaml` + `merge.yaml`, plus a `TRAINING_PROMPTS` entry and an eval prompt for its template.
