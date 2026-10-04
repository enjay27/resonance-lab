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
| `llamafactory` | any model with a profile in `configs/llamafactory/<profile>/` (the default and fixed model is `hy-mt2-1.8b`; also `hy-mt2-7b`, the `translategemma-4b*` ones and the `-fast` variants; pick with `--model <profile>` on any script or on `run_pipeline.py`, and add `--fast` for that model's fast profile; `RESONANCE_LF_PROFILE` is the fallback for remote jobs, the parameter wins) | `requirements-llamafactory.txt` | Fetch Data, Validate, Preprocessing, Update Dataset, Fine-Tuning, Merge LoRA, Export GGUF (`model_gguf/bp-<profile>-q4_k_m.gguf`) |
| `unsloth` | Qwen3 1.7B | `requirements-unsloth.txt` | Validate, Preprocessing, Dataset Split, Fine-Tuning, Metadata Fix, Evaluation (+ the shared report), then the manual GGUF steps below |

A new model for the `llamafactory` pipeline is a new folder `configs/llamafactory/<profile>/` with `train.yaml` and
`merge.yaml` (`tests/test_lf_tools.py` checks the two agree on adapter path, base model and template).

Part of a pipeline: `--from <stage>` starts at a stage and runs the rest, `--only <stage>` runs just one (a stage
name or a unique prefix, case and `-`/`_`/space ignored: `python run_pipeline.py --model hy-mt2-1.8b --from merge`).

**Every training is a run.** `train.py` writes into a fresh directory `outputs/<profile>_lora/<run id>/` (run id = UTC
time) with `run.json` (running / failed / complete) and `train_stdout.log`; a retrain never resumes an older run's
checkpoint (LLaMA-Factory would). `merge.py` merges the latest *complete* run, or `--run <id>`, and leaves
`resonance_run.json` in the merged model's directory naming the run. The monitor follows the latest run. Adapters
trained before runs existed (directly in `outputs/<profile>_lora/`) are still merged.

## Training data
The app's per-channel `dataset_<CHANNEL>.jsonl` files live in a Hugging Face dataset repo. Set `repo:` in
`configs/hf_dataset.yaml`, run `python scripts/fetch_data.py --pin` once (pins the latest commit; needs `hf auth login`),
and the pipeline's first stage, Fetch Data, downloads that revision into `data/hf/` and merges the channels into the raw
log below -- only when it is missing or the pin changed. Every dataset refresh is a new pin. Without a `repo:` the stage
is skipped and `data/raw/` is used as it is.

The raw chat log is `data/raw/raw_translated_logs.jsonl` (rows `original`, `translated`; the app's `translated: null`
rows are skipped). To train on another file, e.g. a hand-curated one, set `RESONANCE_RAW_LOGS` to its path (relative
paths are relative to the repo root): `RESONANCE_RAW_LOGS=data/raw/bp-training-dataset-final.jsonl python run_pipeline.py`.
It still goes through Validate and Preprocessing, so the cleaning rules in `scripts/preprocess.py` apply to it too --
check the Preprocessing Report to see how many rows they drop.

To train on a chosen mix of message categories instead of every clean line, give the run a **dataset recipe**:
`python run_pipeline.py --recipe balanced` (or `preprocess.py --recipe`; `RESONANCE_RECIPE` is the fallback). A recipe is
`configs/datasets/<name>.json` -- a weight per category, a category's share being its weight / the total weight -- and
needs a categories file (`data/processed/categories.jsonl`, `--categories` on `preprocess.py`: one `{"key", "category"}`
per line, `key` = `dataset_recipe.line_key(original)`). `configs/datasets/example.json` shows the format. The validation
file is the same for every recipe; the recipe, its hash and the lines each category gave are recorded in the manifest and
in MLflow (tags `dataset.recipe*`, params `data.cat.<category>`, a `recipe` column in `scripts/mlflow_compare.py`).

## Evaluation
`data/eval/bp-eval-dataset.jsonl` (gitignored; one JSON per line: `original`, `translated`, optional `category`).
Both pipelines print the same report (chrF/BLEU/TER, COMET if `unbabel-comet` is installed, JP and `<think>`
leakage, game-term accuracy, exact match, per-category counts, every output) via `eval_metrics.py`, and save it
to `outputs/eval/`. `python scripts/llamafactory/eval.py --prompt training` evaluates on the exact training prompt
instead of the model's chat template (the default, which is what resonance-stream sends) -- the gap between the two
is the prompt mismatch in `.memory/active-issues/stream-contract.md`.

The llamafactory `eval.py` also saves the translations to `outputs/eval/<profile>-<prompt>.jsonl`. With tracking on,
`python scripts/mlflow_genai_eval.py --prompt training` then sends every eval line to MLflow as a trace with its own scores
(chrF, JP leakage, term check, `discord`, think leak, exact match) in the experiment `resonance-lab-eval`, tagged with the
training run it came from (no LLM, no API key; needs `pandas` next to `mlflow-skinny`).

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