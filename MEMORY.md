# Active State — resonance-lab

**Index, not the record.** Only what would be *false* the moment it goes stale lives here.

## Now — 2026-10-01

**Aligned with resonance-stream: done (PRs #1, #2).** Auto-merge of `claude/*` PRs works (#2 merged itself
on green CI) — [`sessions/2026-10-01-align-with-resonance-stream.md`](.memory/sessions/2026-10-01-align-with-resonance-stream.md).

**This repo owns the prompt; resonance-stream follows (maintainer, 2026-10-01).** Shipped
model = TranslateGemma-4B LoRA from `experiment/translategemma` (raw line, gemma3, no
system); the app sends an English instruction + `[P0]` rule + `<bos>` it never trained on.
Model is being re-chosen — [`translator-shortlist-2026-10-01.md`](.memory/roadmap/translator-shortlist-2026-10-01.md);
contract: [`stream-contract.md`](.memory/active-issues/stream-contract.md).

**`translated: null` rows (2026-10-01):** `preprocess.py` skips them (and blank/missing
fields) instead of crashing `split_dataset`. Fixed on `main` only, not on the experiment branch.

**Switchable pipelines (2026-10-01, `claude/pipeline-registry`):** `run_pipeline.py --pipeline unsloth`
(only one so far; `pipelines.py`, scripts in `scripts/unsloth/`). A merged (#3); B (`claude/preprocess-filters`) shared preprocess filters, tested — changes the
training data, re-eval after the next training. Next: C `llamafactory` (becomes default), D eval metrics — [`roadmap/next.md`](.memory/roadmap/next.md) 2. NOT VERIFIED: moved scripts (no GPU).

**Model part never runs in a cloud session** (no GPU). What changed without a run:
[`unverified-on-gpu.md`](.memory/active-issues/unverified-on-gpu.md).

**Next:** [`roadmap/next.md`](.memory/roadmap/next.md).

## Where the detail is

| read | when |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | rules, gates, layout |
| [`.memory/README.md`](.memory/README.md) | which memory file takes what |
| [`.memory/active-issues/`](.memory/active-issues/) | before trusting a doc, a run, or the stream contract |
| [`.memory/roadmap/`](.memory/roadmap/) | what is next, and what proves it done |
| [`.memory/sessions/`](.memory/sessions/) | why a decision was made, including wrong turns |

## Rules
- **Under ~40 lines.** *Now* carries what is next and what is unverified; nothing else.
- **Update *Now* every session**, even when the answer is "unchanged".
