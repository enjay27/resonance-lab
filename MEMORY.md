# Active State — resonance-lab

**Index, not the record.** Only what would be *false* the moment it goes stale lives here.

## Now — 2026-10-01

**This repo owns the prompt; resonance-stream follows (maintainer, 2026-10-01).** Shipped model =
TranslateGemma-4B LoRA (raw line, gemma3, no system); the app sends an English instruction + `[P0]`
rule + `<bos>` it never trained on. Model is being re-chosen —
[`translator-shortlist-2026-10-01.md`](.memory/roadmap/translator-shortlist-2026-10-01.md);
contract: [`stream-contract.md`](.memory/active-issues/stream-contract.md).

**Pipelines + model choice so far (2026-10-01).** Switchable pipelines done (PRs #3-#7); old BLEU/chrF numbers invalid
([`old-eval-numbers.md`](.memory/active-issues/old-eval-numbers.md)). Hy-MT2 profiles `hy-mt2-1.8b|7b` exist, templates verified
on the real tokenizers; zero-shot 1.8B chrF 39.3 vs shipped TG-4B 41.7, Hy needs the instruction prompt
([`zero-shot-results`](.memory/roadmap/zero-shot-results-2026-10-01.md)). Training examples per model:
[`reference/training-examples.md`](.memory/reference/training-examples.md). **Open: the prompt** (instruction vs raw line,
`[P0]`), TG-4B `--prompt training`, Hy-7B zero-shot, first training. Next steps: [`roadmap/next.md`](.memory/roadmap/next.md).

**Selecting a model:** `--model <profile>` on every llamafactory script and `run_pipeline.py` (parameter > `RESONANCE_LF_PROFILE`
> default `translategemma-4b`); `<model>-fast` profiles (packing + Liger + bigger batch) are never run yet.

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
