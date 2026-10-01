# 2026-10-01 — align with resonance-stream

Asked: apply resonance-stream's project structure and rules here, graft included.
Branch `claude/align-project-structure-s21g7q`.

## Taken over
- graft wiring, unchanged (settings hooks, bootstrap pinned to 0.21.1, helpers,
  `.mcp.json`, skill, `.ignore`). `graft map` indexed all 8 Python files.
- CLAUDE.md shape (parts → gates, guardrails, DoD, version control), workflow-control
  skill, MEMORY.md + `.memory/`, `justfile`, CI + auto-merge.

## Left out
- `release-candidate.yml` / `rc-lib.sh` (app exe), `ui-preview` (UI) — no counterpart.
- `.git-blame-ignore-revs` — no formatting baseline yet.

## Maintainer's calls
- `validate.py`: local `ValidationError` instead of `huggingface_hub`'s (keeps the data
  gate free of model-stack packages).
- `ruff format`: configured, not enforced yet.
- Open a PR; this first one is merged by hand (auto-merge runs from main).

## Choices made here
- Parts: **data** (validate/preprocess/split, CPU, tested) and **model** (train/eval/
  fix_metadata/pipeline, CUDA, manual). Lint is pyflakes-level only (`E9`, `F`); four
  findings in existing code were fixed (unused import/variable, two bare f-strings).
- CI uses Python 3.13 (README); this session ran 3.11.

## Found
- The app's prompt is Gemma-format with an English instruction; training here is Qwen3
  ChatML with a Korean one — `active-issues/stream-contract.md`.
- `translated: null` rows (the app writes them) crash `split_dataset` — same file.

## Follow-up (same day)
- Maintainer: **this repo decides the prompt; resonance-stream follows.** CLAUDE.md and
  `stream-contract.md` were written the other way round at first (prompt "pinned there") —
  corrected.
- The first contract note compared the app with `main`'s Qwen3 pipeline. Wrong baseline:
  the shipped model (TranslateGemma-4B, gist 1.1.0) comes from `experiment/translategemma`
  (LLaMA-Factory, raw line, gemma3). Read that branch; `experiment/qwen3.5` is the step
  between (same LLaMA-Factory move, Qwen3 4B).
- The maintainer's model shortlist is kept in `roadmap/translator-shortlist-2026-10-01.md`.
