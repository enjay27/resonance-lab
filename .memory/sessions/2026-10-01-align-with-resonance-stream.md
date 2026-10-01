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

## Switchable pipelines (same day, later)
- Maintainer: bring `experiment/translategemma`'s *features* onto `main`, keep the Qwen3
  pipeline switchable, default `llamafactory`. A plain merge was rejected: conflicts in
  README/preprocess/split_dataset, and its `config.py` reads `data/system_prompt.txt` at
  import (gitignored) -> every test fails in CI.
- Read the branch from a separate clone indexed with graft (25 symbols): LLaMA-Factory
  train/export/dataset_info, richer `is_clean`, eval metrics, curses monitor.
- Decisions: pipeline = backend, model profile = yaml; folder per pipeline; one
  requirements file per pipeline (trl pins conflict); monitor on `rich` instead of curses
  (OS-neutral); prompt format stays out of the registry for now.
- Step C1 read the branch's pipeline closely and found: eval before export (eval needs the
  merged model that export creates), `shell=True` + a Windows-only `llama-quantize.exe` path,
  and a dataset file (`bp-training-dataset-final.jsonl`) nothing in the pipeline produces.
  Fixed in the port, listed in `roadmap/next.md` C1.
- Test environment trap: the sandbox's `pytest` is a uv tool with its own venv, so a dev
  dependency (pyyaml) is invisible to it until installed there (`uv pip install --python
  <tool venv>/bin/python ...`); CI installs `requirements-dev.txt` and is unaffected.
