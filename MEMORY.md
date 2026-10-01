# Active State — resonance-lab

**Index, not the record.** Only what would be *false* the moment it goes stale lives here.

## Now — 2026-10-01

**Aligned with resonance-stream (2026-10-01, `claude/align-project-structure-s21g7q`):**
graft-indexed, CLAUDE.md rules (data part = `just check`, model part = manual GPU run),
`.memory/`, CI (Linux, data gate) + auto-merge of `claude/*` PRs.
**Open:** first CI run is this PR; auto-merge takes effect once it is on `main`
(this PR merged by hand) — [`sessions/2026-10-01-align-with-resonance-stream.md`](.memory/sessions/2026-10-01-align-with-resonance-stream.md).

**This repo owns the prompt; resonance-stream follows (maintainer, 2026-10-01).** Shipped
model = TranslateGemma-4B LoRA from `experiment/translategemma` (raw line, gemma3, no
system); the app sends an English instruction + `[P0]` rule + `<bos>` it never trained on.
Model is being re-chosen — [`translator-shortlist-2026-10-01.md`](.memory/roadmap/translator-shortlist-2026-10-01.md);
contract and the `translated: null` crash: [`stream-contract.md`](.memory/active-issues/stream-contract.md).

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
