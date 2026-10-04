# Next — candidates, in order

## DECISION 2026-10-02 (maintainer): one fixed model, Hy-MT2-1.8B (the 7B later); `hy-mt2-1.8b` is the default profile
TranslateGemma profiles stay in the repo, not developed. Work now, in this order, one PR at a time:
1. **`--fast` flag on every llamafactory script + `run_pipeline.py`, default profile -> `hy-mt2-1.8b` (done).**
2. **`scripts/mlflow_compare.py` (done)**: one table of the runs (profile, lr, epochs, best eval loss, eval scores) for the sweeps; `--sort eval-loss|chrf|term`, `--profile`, `--markdown`, `--all`.
3. **Built 2026-10-02 (PR open; not yet run on the GPU machine): a Jupyter notebook for the whole lifecycle of a parameter test** (parameters -> data -> train -> merge -> eval -> record/compare -> decide, plus a
   sweep loop) — `notebooks/parameter_test.ipynb` + `parameter_test.py`; design, constraints (thin notebook, tested logic in a data-part module, outputs stripped because eval lines are player chat, GPU machine only) and open questions in [`parameter-test-notebook.md`](parameter-test-notebook.md).
4. **Sweeps on `hy-mt2-1.8b-fast` (~4 min each), from the notebook's sweep cell (or by hand):** `train.py --fast --lr L [--epochs E]` (override, recorded in MLflow), then `merge.py --fast`, `eval.py --fast --prompt training`, `mlflow_compare.py --profile hy-mt2-1.8b-fast --sort eval-loss`.
   **First sweep done (lr 5e-5 / 1e-4 / 2e-4: higher is better, 2e-4 best, eval loss still falling at the end; table in the notebook note).** **Second sweep 2026-10-04 (notebook, done): eval loss best at 4e-4 (0.7724) but chrF/term best at 8e-4 (62.9 / 61.9%); 2e-4 x 6 epochs overfits (51.7); table in the notebook note.** Next: confirm 4e-4 and 8e-4 (2e-4 as control) on the full profile (~25 min each, both eval prompts), then set the winner in the profiles.
5. Then the 7B (QLoRA: bf16 does not fit 16 GB), the prompt copy to resonance-stream once Hy beats the shipped model.
6. **Per-sample evaluation in MLflow (`mlflow.genai.evaluate`)** — PR 1 built 2026-10-03 (deterministic per-line scores, `scripts/mlflow_genai_eval.py`, `eval.py` saves the predictions; not yet run on the maintainer's machine). Next, one PR at a time:
   **[2026-10-04: the judge's first job is now dataset quality, Gate 1 categorizes and Gate 2 (the ratio rule) is code only, and `/v1/systemone` is the only backend, so the llama-server-logprobs backend below is dropped: see [`jev-gates-2026-10-04.md`](jev-gates-2026-10-04.md).]** 2a a typed judge with an open model (llama-server logprobs + a Jev-spec backend) and a planted-error probe to pick the model, 2b `--judge`. Architecture + handoff: [`jev-gate-handoff-2026-10-04.md`](jev-gate-handoff-2026-10-04.md) (two-phase: judge into a file, MLflow second). Design, decisions and sources in [`mlflow-genai-eval.md`](mlflow-genai-eval.md).

## Done 2026-10-02 (maintainer's idea): the run queue is an append-only journal, `run_queue.py` / `.run.result.backup.jsonl`
One line per write; a sent event is acknowledged by a `sent` line and dropped (with its copied file) at the next start; the run header + server id stay for resuming (and for `prune`); a torn last line is ignored; two
processes appending lose nothing; the old TinyDB file is migrated once (renamed `.migrated`); TinyDB is no longer a dependency. Details: `mlflow-plan.md`.

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
