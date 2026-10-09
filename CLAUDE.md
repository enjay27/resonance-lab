# Agent Operating Rules — resonance-lab

resonance-lab fine-tunes the translator model of
[resonance-stream](https://github.com/enjay27/resonance-stream): Japanese Blue Protocol:
Star Resonance chat → natural Korean, as a GGUF `q4_k_m` for resonance-stream's llama.cpp
server. The base model is being re-chosen —
`.memory/roadmap/translator-shortlist-2026-10-01.md` is the plan.

**Pipelines are switchable:** `python run_pipeline.py --pipeline <name>` (registry:
`pipelines.py`). `llamafactory` (default; LLaMA-Factory SFT + LoRA, one model per profile in
`configs/llamafactory/<profile>/`, ends in a q4_k_m GGUF) and `unsloth` (Qwen3 1.7B). Each has its
own folder under `scripts/`, its own stage list, and its own requirements file / virtualenv
(`requirements-<pipeline>.txt`; the stacks pin different trl/transformers). They share
`validate.py` and `preprocess.py` (`--format instruction|pair` is the row layout each one reads).
The shipped model (TranslateGemma-4B, gist 1.1.0) was trained on `experiment/translategemma`,
whose features are being brought in — plan and what is left in `.memory/roadmap/next.md`.

The repo is two parts. **Which part you touch decides which gate applies.** That is the
most important thing on this page.

| tree | part | runs on | gate |
|---|---|---|---|
| `scripts/fetch_data.py` `scripts/validate.py` `scripts/preprocess.py` `scripts/unsloth/split_dataset.py` `scripts/llamafactory/lf_tools.py` `scripts/llamafactory/update_dataset_info.py` `scripts/llamafactory/watch_training.py` `scripts/mlflow_compare.py` `scripts/mlflow_genai_eval.py` `scripts/judge_check.py` `judge_client.py` `judge_local.py` `gate_pipeline.py` `gate_compare.py` `glossary.py` `translation_check.py` `translation_assemble.py` `scripts/translate_agents.py` `labeling_tools.py` `scripts/label_lines.py` `judge_prompts.py` `scripts/compare_judges.py` `compare_runs.py` `eval_metrics.py` `genai_eval.py` `text_rules.py` `hf_data.py` `runs.py` `tracking.py` `run_queue.py` `tracker.py` `track_records.py` `stage_tracking.py` `parameter_test.py` `overlap.py` `valsplit.py` `manifest.py` `config.py` `pipelines.py` `run_pipeline.py` `configs/` `deploy/` | **data** — raw chat logs → checked, LoRA-ready train/val JSONL | any OS, CPU | `just check` (ruff lint + pytest) |
| `scripts/unsloth/train.py` `scripts/unsloth/eval.py` `scripts/unsloth/fix_metadata.py` `scripts/llamafactory/train.py` `merge.py` `gguf.py` `eval.py`, GGUF conversion (README) | **model** — fine-tune, merge, clean, evaluate | **Windows + CUDA GPU only** (`requirements-<pipeline>.txt`) | none automated — a manual run, reported |

`just check` needs only `requirements-dev.txt` (`pip install -r requirements-dev.txt`;
`pip install rust-just` for `just`). CI (`.github/workflows/ci.yml`) runs it on Linux on
every push to `main` and every PR. The model part never runs in CI or in a cloud session:
no GPU, and torch/unsloth are not installed there.

**New pure logic goes in the data part**, where it is tested on every OS. Model-part
scripts run at import time (no `main()`); keep anything worth testing out of them.

**This repo owns the prompt.** The prompt the model is trained on is decided here, and
resonance-stream follows it (`crates/core/src/text.rs`, `translation_prompt`) — decision of
the maintainer, 2026-10-01. So a change to the prompt or chat template is made here first,
written down in `.memory/active-issues/stream-contract.md`, and then copied to
resonance-stream in its own PR there; the app must send exactly what training saw.
The training data comes from the app (`dataset_<CHANNEL>.jsonl`: `pid`, `original`,
`translated` — `null` when untranslated, `timestamp`).

---

## Tech Stack

- **Python 3.13** (README; CI runs the data gate on 3.13). Windows for the model part.
- **Gate dev deps** (`requirements-dev.txt`): pytest, ruff (configured in `pyproject.toml`), pyyaml (profile yaml files), rich (the monitor).
- **`llamafactory` pipeline** (`requirements-llamafactory.txt`): LLaMA-Factory SFT + LoRA; profile
  **default and fixed model (maintainer, 2026-10-02): `hy-mt2-1.8b`** = `tencent/Hy-MT2-1.8B` (the 7B later; the TranslateGemma profiles stay, not developed; the shipped app model is still `translategemma-4b` =
  `google/translategemma-4b-it`, template `gemma3`), dataset `bp_translation` (validation: `bp_translation_val`, split by line in `preprocess.py`)
  (`preprocess.py --format pair --prompt auto` puts the template's instruction, from `prompts.py`, before each line; `--reverse` adds ko→ja rows — `.memory/active-issues/stream-contract.md` §1), `cutoff_len` 256, LoRA r=32.
  Profile = `configs/llamafactory/<profile>/{train,merge}.yaml`, chosen by `--model <profile>` (every llamafactory script and `run_pipeline.py`), else `RESONANCE_LF_PROFILE` (remote jobs), else the default;
  `train.py --lr 1e-4 --epochs 2` overrides the profile's learning rate / epochs for one training (sweeps without editing a profile; recorded in MLflow as the params, tag `train.overrides`);
  `--fast` (same scripts) means that model's fast profile `<model>-fast` (packing, bigger batch, eval every 10 steps; ~4 min for the 1.8B), e.g. `run_pipeline.py --fast`.
- **`unsloth` pipeline** (`requirements-unsloth.txt`): unsloth (pinned commit), transformers, peft,
  trl 0.24, bitsandbytes 4-bit, torch 2.10 + CUDA 12.6 (`triton-windows`). Base
  `rd211/Qwen3-1.7B-Instruct` (`config.py`), LoRA r=64, alpha=128.
- **Conversion:** llama.cpp `convert_hf_to_gguf.py` → `llama-quantize q4_k_m` (README; the
  `llamafactory` pipeline's `gguf.py` does it, finding `llama-quantize` under `llama.cpp/build/bin/`).

---

## Repository Layout

Moved to [`docs/repository-layout.md`](docs/repository-layout.md): what every file is for. It is not loaded automatically; read it when you need to find a file.

---

## Conventions

- **Paths and hyper-parameters live in `config.py`**, built from `BASE_DIR`. A script
  never hardcodes a path; it imports it. Two environment overrides exist:
  `RESONANCE_RAW_LOGS` (raw log file; also switches the Fetch Data stage off), `RESONANCE_MLFLOW=0` (tracking off) and `RESONANCE_LF_PROFILE` (llamafactory model profile; the `--model` parameter wins over it).
- **A stage that fails exits non-zero** (`sys.exit(1)` or an exception) — that is how
  `run_pipeline.py` stops. A stage that only prints an error lets the pipeline continue
  on bad data.
- **Scripts import `config` via `sys.path.append(<repo root>)`** (one `dirname` per folder
  level: three in `scripts/<pipeline>/`); tests get the same through `pyproject.toml`
  (`pythonpath` lists the root and each script folder, `scripts/llamafactory` included) and import scripts as modules
  (`import preprocess`). A new script folder goes into `pythonpath`.
- **A new pipeline is a folder under `scripts/` plus an entry in `pipelines.py`**;
  `tests/test_pipelines.py` checks every registered stage script exists.
- JSONL is read and written as UTF-8 with `ensure_ascii=False`.

## Guardrails

- **The plan's evidence** is an impact analysis: `g callers` from the `graft-kade` skill. See
  `.claude/skills/workflow-control/SKILL.md`.
- **Changing the model is a feature, not a refactor.** `INSTRUCTION`, the chat template and a
  training hyper-parameter each change the model: own PR and an eval run. A move/split commit
  changes none of them, nor a file format.
- **No data or weights in git.** Chat logs are players' messages; they stay local.
  Test fixtures are short, made-up lines.

---

## Definition of Done

1. **Model-part changes need a manual run** on the Windows/CUDA machine (the affected
   stage, plus `eval.py` when the model changes); say in the commit body whether it was
   done, or `NOT VERIFIED: model part -- no GPU in this session`.
2. **Record the outcome in the memory tree.** `MEMORY.md` is an index under ~40 lines —
   update its *Now* section. Detail goes in `.memory/` (see its README). Both are updated in
   the task's own branch, before the push, so a merged task never leaves the index behind.

---

## Version Control

- `.github/workflows/auto-merge.yml` (not GitHub auto-merge) merges the PR and deletes its
  `claude/*` branch (never any other branch) once the CI workflow passes on the PR's latest
  commit. If CI fails, fix on the same branch and push again: the run for the new commit decides.
- Only `claude/*` branches auto-merge. `workflow_run` workflows are read from `main`, so a
  change to `auto-merge.yml` itself takes effect after it has been merged once.
- Only a green local gate is pushed.

### Never commit
- Chat logs and datasets (`data/**`, `lora_dataset/`), weights (`*.safetensors`,
  `*.gguf`, `model_f16*/`, `outputs/`), `llama.cpp/`, `unsloth_compiled_cache/`.
- `graft/` (regenerable), `__pycache__/`, virtualenvs, IDE folders.
