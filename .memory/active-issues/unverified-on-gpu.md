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
