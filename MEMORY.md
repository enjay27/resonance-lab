# Active State — resonance-lab

**Index, not the record.** Only what would be *false* the moment it goes stale lives here.

## Now — 2026-10-01

**Prompt: this repo owns it; resonance-stream follows** ([`stream-contract.md`](.memory/active-issues/stream-contract.md)). Shipped TG-4B was trained on
the TranslateGemma instruction + line, both directions; `preprocess.py --prompt auto [--reverse]` + `prompts.py` rebuild those rows (rule from
`aab6b66`). The app also sends a `[P0]` rule and a literal `<bos>` it never trained on. Model is being re-chosen
([`shortlist`](.memory/roadmap/translator-shortlist-2026-10-01.md)).

**State (2026-10-01).** Pipelines switchable (`--model <profile>`; parameter > `RESONANCE_LF_PROFILE` > default); Hy-MT2 profiles + `<model>-fast`
profiles exist. Eval: zero-shot Hy-1.8B chrF 39.3, fine-tuned base 66.7 (25 min), `-fast` 59.3 (4.6 min), shipped TG-4B 41.7 — **likely inflated by eval lines in the training data (now excluded)** ([`first-training-run`](.memory/roadmap/first-training-run-2026-10-01.md)). Old BLEU/chrF numbers invalid ([`old-eval-numbers`](.memory/active-issues/old-eval-numbers.md)).

**Worked in this order** ([`pipeline-review-2026-10-01.md`](.memory/roadmap/pipeline-review-2026-10-01.md), one PR at a time; PRs #25-#33 merged):
**done** = eval-overlap exclusion (re-measure every score above after a retrain), manifest + validation split (re-run Preprocessing; eval loss is not comparable with
old runs), validate/drop-rate guards (limits are guesses: calibrate on the real log), Fetch Data stage, run identity (each training = its own run dir; `run_pipeline.py --from/--only`),
MLflow server files `deploy/mlflow/` (now on the NAS's **Postgres** `deploy/postgres/` + **MinIO**, 2026-10-01), `tracking.py`, offline queue (`run_queue.py`/`tracker.py`, run against a real server).
**Done by the maintainer (2026-10-01):** `repo:` set and pinned (`b37c268f`, repo `enjay27/blue-protocol-star-resonance-chatting-message`), Postgres 18 (`deploy/postgres/`) and the MLflow server (`deploy/mlflow/`, MinIO artifacts) run on the NAS.
The unified `bp-training-dataset-*.jsonl` in that repo is being archived: the repo goes back to the app's per-channel files, so `include` stays `dataset_*.jsonl`.
**Maintainer to do:** fill `.env.mlflow`; check `pip` resolves `mlflow-skinny`/`tinydb` on Python 3.13 / Windows; first real `fetch_data.py` run (after the unified file is archived).
**Done (2026-10-01):** `fetch_data.py --pin` says whether the pin moved; `--check` reports the same and writes nothing (exit 1 when unset/behind). `HfApi` call itself NOT VERIFIED (no network login in cloud).
**Done 2026-10-01:** MLflow PR 4a (`track_records.py`) and 4b (`stage_tracking.py` wired into llamafactory train/merge/gguf/eval; **NOT VERIFIED on the GPU**: [`unverified-on-gpu.md`](.memory/active-issues/unverified-on-gpu.md) has the checklist).
First TG-4B run on cleaned data (lr 1e-5, 3 ep): eval loss 0.888 still falling at the last step, train/eval gap +0.04 = underfit; chrF 45.2, term accuracy 1/21 with `--prompt chat-template` (run `--prompt training` too).
Hy profiles use lr 2e-4. Tracking works end to end (UI shows runs, curves, artifacts; NAS `MINIO_ENDPOINT_URL` must be the NAS's real address). Tracker rough edges fixed: Ctrl+C -> KILLED, a failed artifact cannot block the queue. Queued: run-queue journal ([`next.md`](.memory/roadmap/next.md)). **Profile `translategemma-4b-lr1e-4` exists (only the lr differs from `translategemma-4b`): run it (`run_pipeline.py --model translategemma-4b-lr1e-4`, then `eval.py --prompt training` too) and compare with the lr 1e-5 run (eval loss 0.888, chrF 45.2) in MLflow. Then `scripts/mlflow_compare.py` (PR 5), the journal.** Rule: **graft first** (`graft skeleton <file>` before reading a file, see CLAUDE.md).
Maintainer: HF dataset is the unified file for ~2 months — local `include: "bp-training-dataset-*.jsonl"`, then `fetch_data.py --force` once (replaces the hand-made raw log; records the revision). Index: [`roadmap/next.md`](.memory/roadmap/next.md).

**Model part never runs in a cloud session** (no GPU): every model stage is NOT VERIFIED —
[`unverified-on-gpu.md`](.memory/active-issues/unverified-on-gpu.md). Auto-merge of `claude/*` PRs works.

## Where the detail is

| read | when |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | rules, gates, layout |
| [`.memory/README.md`](.memory/README.md) | which memory file takes what |
| [`.memory/active-issues/`](.memory/active-issues/) | before trusting a doc, a run, or the stream contract |
| [`.memory/reference/`](.memory/reference/) | what each model's training example looks like (`training-examples.md`) |
| [`.memory/roadmap/`](.memory/roadmap/) | what is next, and what proves it done |
| [`.memory/sessions/`](.memory/sessions/) | why a decision was made, including wrong turns |

## Rules
- **Under ~40 lines.** *Now* carries what is next and what is unverified; nothing else.
- **Update *Now* every session**, even when the answer is "unchanged".
