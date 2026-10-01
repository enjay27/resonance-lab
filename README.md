# resonance-lab
Python project for Resonance Stream: fine-tunes its Japanese -> Korean translator model (pipelines: LLaMA-Factory, unsloth).

## Development
- Rules, gates and layout: [`CLAUDE.md`](CLAUDE.md); current state: [`MEMORY.md`](MEMORY.md)
- `pip install -r requirements-dev.txt` then `just check` (lint + data-stage tests, CPU only)
- `python run_pipeline.py` runs a training pipeline (see *Pipelines*; `pipelines.py` lists them)

## Pipelines
`python run_pipeline.py [--pipeline llamafactory|unsloth]` (default `llamafactory`). Each pipeline has its
own folder under `scripts/`, its own requirements file and needs its **own virtualenv** (the stacks pin
different `trl`/`transformers`).

| pipeline | trains | requirements | stages |
|---|---|---|---|
| `llamafactory` | any model with a profile in `configs/llamafactory/<profile>/` (now `translategemma-4b`; pick with `--model <profile>` on any script or on `run_pipeline.py`; `RESONANCE_LF_PROFILE` is the fallback for remote jobs, the parameter wins) | `requirements-llamafactory.txt` | Validate, Preprocessing, Update Dataset, Fine-Tuning, Merge LoRA, Export GGUF (`model_gguf/bp-<profile>-q4_k_m.gguf`) |
| `unsloth` | Qwen3 1.7B | `requirements-unsloth.txt` | Validate, Preprocessing, Dataset Split, Fine-Tuning, Metadata Fix, Evaluation (+ the shared report), then the manual GGUF steps below |

A new model for the `llamafactory` pipeline is a new folder `configs/llamafactory/<profile>/` with `train.yaml` and
`merge.yaml` (`tests/test_lf_tools.py` checks the two agree on adapter path, base model and template).

## Training data
The raw chat log is `data/raw/raw_translated_logs.jsonl` (rows `original`, `translated`; the app's `translated: null`
rows are skipped). To train on another file, e.g. a hand-curated one, set `RESONANCE_RAW_LOGS` to its path (relative
paths are relative to the repo root): `RESONANCE_RAW_LOGS=data/raw/bp-training-dataset-final.jsonl python run_pipeline.py`.
It still goes through Validate and Preprocessing, so the cleaning rules in `scripts/preprocess.py` apply to it too --
check the Preprocessing Report to see how many rows they drop.

## Evaluation
`data/eval/bp-eval-dataset.jsonl` (gitignored; one JSON per line: `original`, `translated`, optional `category`).
Both pipelines print the same report (chrF/BLEU/TER, COMET if `unbabel-comet` is installed, JP and `<think>`
leakage, game-term accuracy, exact match, per-category counts, every output) via `eval_metrics.py`, and save it
to `outputs/eval/`. `python scripts/llamafactory/eval.py --prompt training` evaluates on the exact training prompt
instead of the model's chat template (the default, which is what resonance-stream sends) -- the gap between the two
is the prompt mismatch in `.memory/active-issues/stream-contract.md`.

## Prerequisites
- Python 3.13
- Windows OS (Linux not tested yet)
- CUDA Toolkit 2.6

## Install (llamafactory pipeline)
- `pip install -r requirements-llamafactory.txt`
- `train.yaml` uses `flash_attn: fa2`; if flash-attn is not installed, set it to `auto` for that profile
- Watch a run from a second terminal: `python scripts/llamafactory/watch_training.py` (Ctrl+C quits; `rich`, any OS)
- GGUF export needs llama.cpp built in `llama.cpp/` (see below; `llama-quantize` is found under `build/bin/`)

## Install (unsloth pipeline)
- `pip install -r requirements-unsloth.txt` (in its own virtualenv)

## Convert to Model (F16 -> GGUF -> q4_k_m) -- unsloth pipeline, by hand
(The llamafactory pipeline does this itself in its last two stages.)
- Make sure model_f16_clean config.json "architectures": "Qwen3ForCausalLM"
- git clone --recursive https://github.com/ggerganov/llama.cpp
- cmake -B build
- cmake --build build --config Release -j
- python llama.cpp/convert_hf_to_gguf.py model_f16_clean --outfile model.f16.gguf
- llama.cpp/build/bin/Release/llama-quantize.exe .\model.f16.gguf model_q4_k_m.gguf q4_k_m