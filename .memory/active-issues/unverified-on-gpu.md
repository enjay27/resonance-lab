# Not run on the GPU machine

Code or config changed in a session without CUDA. Delete an item once it has run.

- **`run_pipeline.py` after `claude/align-project-structure-s21g7q` (2026-10-01):** lint
  removed an unused `shutil` import and the unused `result =` in `run_step` — no
  behaviour change, but never run since. `validate.py` raises its own `ValidationError`
  instead of `huggingface_hub`'s (unit-tested; pipeline halt unchanged).
- **`scripts/unsloth/*` moved (2026-10-01, PR #3):** pure move + one path-depth line per script;
  `run_pipeline.py --pipeline unsloth` never run since. Check: `python run_pipeline.py` reaches
  every stage (the stage scripts' `config` import resolves).
- **`preprocess.py` filters (2026-10-01, `claude/preprocess-filters`):** unit-tested; not run on a
  real raw log. It now drops more rows than before (the branch's filters) — compare the
  Preprocessing Report counts on the real data before training, and re-run `eval.py` after.
- **`llamafactory` pipeline (2026-10-01, `claude/llamafactory-pipeline`):** `update_dataset_info.py`,
  `train.py`, `merge.py`, `gguf.py` ran here only as far as their error paths (missing
  `llamafactory-cli`, adapter, merged model); `lf_tools` is unit-tested. Never run end to end:
  check `python run_pipeline.py` on the Windows/CUDA machine — in particular that
  `llamafactory-cli` accepts the profile yaml from the repo root, that `dataset_info.json`
  is found (`data/`), `flash_attn: fa2` installs, and that `gguf.py` finds `llama-quantize`.
  `requirements-llamafactory.txt` leaves `llamafactory` unpinned — pin what you trained with.
- **`watch_training.py` (2026-10-01, `claude/training-monitor`):** state/render unit-tested; `main`
  ran here against fake logs only. On a real run check that LLaMA-Factory's `trainer_log.jsonl`
  really has `current_steps` / `loss` / `epoch` / `eval_loss` (the branch's monitor assumed so),
  that `train_stdout.log` carries the `'grad_norm'` dict lines and the tqdm bar, and that
  `nvidia-smi` is on the PATH.
- **Eval stage (2026-10-01, `claude/eval-metrics`):** `eval_metrics.py` is unit-tested; the two
  `eval.py` scripts ran here only as far as their error paths (llamafactory) / a parse check
  (unsloth). Never generated anything: check `--prompt chat-template` and `--prompt training` on
  the real merged model (tokenizer call, `enable_thinking`, BOS handling), COMET with `unbabel-comet`,
  and that the unsloth eval's `inputs.shape[-1]` slicing (tensor, not dict) is right.
- **Hy-MT2 profiles (2026-10-01, `claude/next-task-model-selection-nzghmf`):** `configs/llamafactory/hy-mt2-{1.8b,7b}/`,
  Update: step 1 ran locally -- both templates MATCH; neither tokenizer adds BOS (eval `--prompt training` now prepends it, `with_bos`).
  LLaMA-Factory: PyPI 0.9.5 lacks the Hy templates; source commit ce9dc9e0 has them and is now pinned (with transformers 4.57.1, peft 0.18.1, trl 0.24.0). eval.py `chat-template` crashed on `token_type_ids` (Hy tokenizer) -> `generate_inputs`; re-run to confirm. **Confirmed 2026-10-01: eval.py ran on hy-mt2-1.8b (both prompts) and translategemma-4b (chat-template); results in roadmap/zero-shot-results-2026-10-01.md.**
  Local py3.13 raised `linecache._register_code ... 'str' has no attribute 'co_consts'` in a `python -c` run: open.
  `TRAINING_PROMPTS` / `chat_messages` in `lf_tools.py` (unit-tested), `scripts/llamafactory/inspect_template.py`.
  huggingface.co is blocked in cloud sessions, so the strings come from Tencent's README and from
  LLaMA-Factory's `template.py` (templates `hy_dense_1_8b`, `hy_dense_7b` are registered upstream; both read
  from GitHub). **Check locally, in this order:**
  1. `python scripts/llamafactory/inspect_template.py --model hy-mt2-1.8b` (then `hy-mt2-7b`):
     must print `MATCH`; also read whether the tokenizer adds BOS on its own (`eval.py --prompt training`
     encodes with `tokenizer(...)`, so a missing BOS there is an eval-only mismatch).
  2. The installed LLaMA-Factory knows the two template names (`pip show llamafactory`; they are new).
  3. The 1.8B and 7B templates differ (1.8B: `<｜hy_User｜>`/`<｜hy_Assistant｜>`, 7B: `<|extra_0|>`):
     confirm against each model's `tokenizer_config.json` chat template, and that the Hy-MT2 GGUF's embedded
     template (what llama-server uses) agrees.
  4. Hyper-parameters are Tencent's LoRA defaults (r64/α128, q/k/v/o, lr 2e-4), `cutoff_len` 256,
     gradient checkpointing on — not our 4B recipe (lr 1e-5). The 7B in bf16 LoRA may not fit your GPU:
     add `quantization_bit: 4` to its train.yaml if so. Nothing trained.
  5. Recommended sampling (README): temperature 0.7, top_p 0.6, top_k 20, repetition_penalty 1.05;
     `eval.py` uses greedy — fine for comparison, note it.
- **`--model` parameter (2026-10-01):** parsing and precedence are unit-tested and the error paths ran here; train/eval/inspect/watch with a real model not run. `inspect_template.py`'s old `--model` (tokenizer id) is now `--tokenizer`.
- **`*-fast` profiles (2026-10-01):** `translategemma-4b-fast`, `hy-mt2-1.8b-fast`, `hy-mt2-7b-fast` = the base profile with
  packing + Liger + a bigger batch (4B 16x2, 1.8B 32x1, 7B 8x4 with checkpointing), own output/export dirs. Never run.
  Liger on Windows: `pip install liger-kernel` fails (wants `triton`, only `triton-windows` exists) -> install `triton-windows<3.6` then `liger-kernel --no-deps` (**verified locally 2026-10-01:** `triton-windows<3.6` -> triton 3.5.1, `import liger_kernel` works with torch 2.9.1; the `pynvml` FutureWarning is harmless), else set `enable_liger_kernel: false`. Check; `flash_attn: auto`
  picks fa2/sdpa; the batch fits VRAM; then compare time (watch_training tok/s) and eval chrF against the base profile.
  `python run_pipeline.py --model hy-mt2-1.8b-fast`. If fa2 works, add `neat_packing: true`.
- **`python -c` crashes on the maintainer's Python 3.13** with `linecache._register_code ... 'str' has no attribute 'co_consts'` (seen with `-c` only; the same imports from a .py file work). Run checks from a file, not `-c`. Cause not investigated.
- **`scripts/llamafactory/inspect_pair.py` (2026-10-01):** prints the training example LLaMA-Factory builds (masked prompt, trained
  response, EOS, tokens vs cutoff) via `template.encode_oneturn`. `first_pairs` is tested; the LLaMA-Factory calls are written from
  memory of its API (`get_template_and_fix_tokenizer`, `DataArguments`, `encode_oneturn`) and never ran. Run:
  `python scripts/llamafactory/inspect_pair.py --model hy-mt2-1.8b --rows 3` (needs `data/processed/lora_train_data.jsonl` from
  `preprocess.py --format pair`). Check: the response ends with the model's end-of-turn token; the prompt column is the raw line.
