# Active State — resonance-lab

**Index, not the record.** Only what would be *false* the moment it goes stale lives here.

## Now — 2026-10-01

**Prompt: this repo owns it; resonance-stream follows** ([`stream-contract.md`](.memory/active-issues/stream-contract.md)). Shipped TG-4B was trained on
the TranslateGemma instruction + line, both directions; `preprocess.py --prompt auto [--reverse]` + `prompts.py` rebuild those rows (rule from
`aab6b66`). The app also sends a `[P0]` rule and a literal `<bos>` it never trained on. Model is being re-chosen
([`shortlist`](.memory/roadmap/translator-shortlist-2026-10-01.md)).

**State (2026-10-01).** Pipelines switchable (`--model <profile>`; parameter > `RESONANCE_LF_PROFILE` > default); Hy-MT2 profiles + `<model>-fast`
profiles exist. Eval: zero-shot Hy-1.8B chrF 39.3, fine-tuned base 66.7 (25 min), `-fast` 59.3 (4.6 min), shipped TG-4B 41.7 — **likely inflated:
eval lines may be in the training data, unchecked** ([`first-training-run`](.memory/roadmap/first-training-run-2026-10-01.md)). Training examples:
[`reference/training-examples.md`](.memory/reference/training-examples.md). Old BLEU/chrF numbers invalid ([`old-eval-numbers`](.memory/active-issues/old-eval-numbers.md)).

**Worked in this order** ([`pipeline-review-2026-10-01.md`](.memory/roadmap/pipeline-review-2026-10-01.md), one PR at a time): data safety
(eval-overlap exclusion **done** — every score above must be re-measured after a retrain; sidecar manifest **done** and validation split **done** — re-run Preprocessing before
training, old data has neither, eval-loss is not comparable with old runs; validate/drop-rate guards **done** (limits are guesses: calibrate on the real log); Fetch Data stage **done** — maintainer: set `repo:` in `configs/hf_dataset.yaml`, then `fetch_data.py --pin`; run identity **done** (each training = its own run dir; `run_pipeline.py --from/--only`); MLflow PR 1 (NAS server files `deploy/mlflow/`) in review — on the NAS: copy, fill `.env`, start, firewall; next: `tracking.py`, the offline queue, stage wiring) -> run identity -> MLflow on the NAS ([`mlflow-plan.md`](.memory/roadmap/mlflow-plan.md), revised
with the maintainer's decisions) -> eval upgrades -> model changes. Index of what comes next: [`roadmap/next.md`](.memory/roadmap/next.md).

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
