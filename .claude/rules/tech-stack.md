---
paths:
  - "scripts/**"
  - "configs/**"
  - "config.py"
  - "pipelines.py"
  - "prompts.py"
  - "run_pipeline.py"
  - "requirements*.txt"
---

# Tech Stack

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
