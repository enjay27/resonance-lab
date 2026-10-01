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
**Done by the maintainer (2026-10-01):** `repo:` + `--pin` set (files `bp-training-dataset-*.jsonl`), Postgres 18 (`deploy/postgres/`) and the MLflow server (`deploy/mlflow/`, MinIO artifacts) run on the NAS.
**Maintainer to do:** fill `.env.mlflow`; check `pip` resolves `mlflow-skinny`/`tinydb` on Python 3.13 / Windows; first real `fetch_data.py` run.
**Small, do first next session:** `fetch_data.py --pin` says whether the pin moved (`--check` writes nothing) and `hf_data.DEFAULT_INCLUDE` / yaml comment follow the real file name `bp-training-dataset-*.jsonl`.
**Next session continues here: MLflow PR 4 — wire the stages to the tracker** (design notes in [`mlflow-plan.md`](.memory/roadmap/mlflow-plan.md) "PR 4"; model part, NOT VERIFIED
without the GPU). Then eval upgrades -> model changes. Rule: **graft first** (`graft skeleton <file>` before reading a file, see CLAUDE.md). Index of what comes next: [`roadmap/next.md`](.memory/roadmap/next.md).

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
