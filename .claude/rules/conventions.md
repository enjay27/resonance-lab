---
paths:
  - "**/*.py"
  - "tests/**"
  - "pyproject.toml"
---

# Conventions

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
