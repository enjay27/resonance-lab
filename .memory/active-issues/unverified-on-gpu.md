# Not run on the GPU machine

Code or config changed in a session without CUDA. Delete an item once it has run.

- **`run_pipeline.py` after `claude/align-project-structure-s21g7q` (2026-10-01):** lint
  removed an unused `shutil` import and the unused `result =` in `run_step` — no
  behaviour change, but never run since. `validate.py` raises its own `ValidationError`
  instead of `huggingface_hub`'s (unit-tested; pipeline halt unchanged).
