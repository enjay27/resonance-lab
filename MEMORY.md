# Active State — resonance-lab

**Index, not the record.** Only what would be *false* the moment it goes stale lives here.

## Now — 2026-10-01

**This repo owns the prompt; resonance-stream follows (maintainer, 2026-10-01).** Shipped model =
TranslateGemma-4B LoRA (raw line, gemma3, no system); the app sends an English instruction + `[P0]`
rule + `<bos>` it never trained on. Model is being re-chosen —
[`translator-shortlist-2026-10-01.md`](.memory/roadmap/translator-shortlist-2026-10-01.md);
contract: [`stream-contract.md`](.memory/active-issues/stream-contract.md).

**Switchable pipelines (2026-10-01):** `run_pipeline.py --pipeline llamafactory|unsloth` (default
`llamafactory`; `pipelines.py`). Merged: A registry+move (#3), B shared preprocess filters (#4 — drops
more rows, so re-eval after the next training; also skips `translated: null`). C1 `llamafactory` pipeline (#5; profile yaml,
per-pipeline requirements, separate venvs); `claude/training-monitor` (C2): `rich` monitor. Next: D shared
eval metrics — [`roadmap/next.md`](.memory/roadmap/next.md) 2.

**Model part never runs in a cloud session** (no GPU): every model stage is NOT VERIFIED —
[`unverified-on-gpu.md`](.memory/active-issues/unverified-on-gpu.md). Auto-merge of `claude/*` PRs works.

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
