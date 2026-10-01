# Active State — resonance-lab

**Index, not the record.** Only what would be *false* the moment it goes stale lives here.

## Now — 2026-10-01

**This repo owns the prompt; resonance-stream follows (maintainer, 2026-10-01).** Shipped model =
TranslateGemma-4B LoRA (raw line, gemma3, no system); the app sends an English instruction + `[P0]`
rule + `<bos>` it never trained on. Model is being re-chosen —
[`translator-shortlist-2026-10-01.md`](.memory/roadmap/translator-shortlist-2026-10-01.md);
contract: [`stream-contract.md`](.memory/active-issues/stream-contract.md).

**Switchable pipelines: done (2026-10-01, PRs #3-#7).** `run_pipeline.py --pipeline llamafactory|unsloth`
(default `llamafactory`), shared preprocess filters (drop more rows — check the report, re-eval after the
next training), `rich` training monitor, shared eval report. `RESONANCE_RAW_LOGS` = another raw log.
**The branch's BLEU/chrF/TER scored only the first sample — old numbers invalid, re-baseline**
([`old-eval-numbers.md`](.memory/active-issues/old-eval-numbers.md)).
**Next session starts at** [`roadmap/next.md`](.memory/roadmap/next.md) *Start here*: re-baseline on the GPU
machine, then a profile per shortlist candidate.

**Model choice, step 1 (2026-10-01): Hy-MT2 profiles written** (`hy-mt2-1.8b`, `hy-mt2-7b`; lead challenger of the
shortlist). Templates verified locally against both tokenizers (`inspect_template.py`: MATCH; no tokenizer adds BOS, eval now
prepends it). Not trained/evaluated. **Next: rest of the local checklist in** [`unverified-on-gpu.md`](.memory/active-issues/unverified-on-gpu.md)
(`inspect_template.py`), then zero-shot eval vs the re-baselined TranslateGemma-4B. Other candidates: later.

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
