# Per-sample evaluation in MLflow (`mlflow.genai.evaluate`) — idea, 2026-10-02

**Status: idea + feasibility checked, NOT started. The maintainer starts from a new session. Present the design below, get the answers to the open questions, then build (tests first).**
Raised by the maintainer after looking at MLflow 3.16's GenAI view (Overview -> Quality: "No assessments available; monitor quality metrics from scorers"; Evaluation runs): "mlflow new feature: evaluate llm.
Use this as a tracing metric." Its sample code uses `mlflow.genai.evaluate(data=..., predict_fn=..., scorers=[Correctness()])`.

## What it would give
Today a run has one number per metric (`eval.chrf`, ...) plus a text report in `outputs/eval/`. With this, every eval line (51 now) is a **trace** with its own **assessments**: the Japanese source, our translation, the
reference and the scores (chrF, JP leakage, term hit, `discord` violation, exact match). In the UI: open/sort lines by their score, compare two evaluation runs side by side (which lines got better after an lr change),
run-level means (`chrf/mean`, ...). It targets the failure patterns seen in the first TG eval (`草` -> `고블린`, `ww` left as `ww`, `杖` -> `도검`, `discord` -> `디스코드`).

## Feasibility (checked in a cloud session against a real MLflow 3.16.1 server; the web UI was NOT seen)
- `mlflow.genai.evaluate(data=[{"inputs": {"text": jp}, "outputs": prediction, "expectations": {"expected_response": reference}}], scorers=[...])` with **precomputed outputs and no `predict_fn`** works.
  Custom scorers via `@scorer` return float/bool (deterministic: sacrebleu chrF, a JP regex) — **no LLM judge, no API key, no data leaves the LAN**. Result: a run tagged `mlflow.runType=genai_evaluate`, one trace per row
  with assessments (`chrf` 5.43, `jp_leak` True, ...), metrics `chrf/mean`, `jp_leak/mean`. The built-in scorers (`Correctness()`, ...) are LLM judges: not used (they would send chat lines to a provider).
- **`mlflow-skinny==3.16.1` (the project's pin) works only with `pandas` installed** (`ModuleNotFoundError: pandas` otherwise). LLaMA-Factory pulls pandas through `datasets`, so the maintainer's venv probably has it:
  check `python -c "import pandas"`; if not, add a pin to `requirements-llamafactory.txt` (test `test_the_queue_needs_no_database_package`-style guard if it becomes a requirement).
- Probe used (re-run it first, it takes a minute): `mlflow.set_tracking_uri(<server>); mlflow.set_experiment("genai-probe"); @scorer def chrf(outputs, expectations) -> float: return sacrebleu.sentence_chrf(outputs,
  [expectations["expected_response"]]).score; evaluate(data=[...], scorers=[chrf])`; then `mlflow.search_traces(...)` showed 2 traces with their assessments.
- **Unverified:** whether Overview -> Quality fills from these assessments as a trend over runs (expected, not seen); how the Evaluation runs page renders them; behaviour with the NAS's basic-auth (the fluent API reads the same
  `MLFLOW_TRACKING_*` variables `tracking.client_environment` sets, so it should work).

## Proposed design
1. **A separate script, not part of the eval stage:** `python scripts\mlflow_genai_eval.py [--model M] [--fast] [--prompt training|chat-template]`, run after `eval.py`. `evaluate` uses MLflow's fluent API and needs the server
   at that moment; the tracker is offline-first and never raises into a stage (`tracker.py`, `stage_tracking.py`). Putting it inside `eval.py` would break that guarantee.
2. **`eval.py` also saves the predictions** as JSONL next to the report (`outputs/eval/<profile>-<prompt>.jsonl`: `original`, `reference`/`translated`, `prediction`, `category`, raw output). Only saving is added to the
   model-part script; generation stays as it is. Keep the logic (row building, scorers) in a new **data-part** module (e.g. `genai_eval.py`, pure, tested) — the scorers wrap what `eval_metrics.py` already has:
   `standard_metrics` (chrF/BLEU/TER), `has_jp`, `think_leaked`, `term_results`, `discord_violation`, exact match; the eval category as trace metadata.
3. **One evaluation run per (training run, prompt)**, tagged `training_run` (the tracker local id `<profile>-<run id>`, from the merged model's `resonance_run.json` via `runs.read_merge_record` + `track_records.stage_run_id`),
   `profile`, `eval.prompt`, `dataset.revision`. This also removes the known limitation that two `--prompt`s in one training run share the `eval.*` metric keys.
4. **The existing numbers stay:** the training run keeps its `eval.*` metrics, so `mlflow_compare.py` is unchanged. This adds the per-sample view; it replaces nothing.
5. Out of scope for now: an LLM judge (naturalness) — would need a judge model (maybe a local llama.cpp server later), a privacy decision, and a prompt; MLflow "Datasets" registration; tracing live translations
   from resonance-stream (the Rust app cannot be traced from this repo).

## Open questions for the maintainer (answer before building)
1. Is "per-sample evaluation in MLflow" what "use this as a tracing metric" meant? (If it meant tracing live translations in the app, this is a different task.)
2. Separate script after `eval.py` (recommended) or automatic at the end of `eval.py`?
3. Keep the LLM judge out for now (recommended)?

## Plan when started (CLAUDE.md: plan first, tests first, one PR at a time)
- PR 1 (data part + a small model-part change): `genai_eval.py` (rows from the predictions JSONL, scorers) + tests; `eval.py` saves the JSONL (NOT VERIFIED on the GPU: say so); `scripts/mlflow_genai_eval.py`
  (thin wrapper, `client=`/`evaluate` injectable for tests like `mlflow_compare.py`); docs/checklist.
- Verify against a real MLflow 3.16.1 server in the cloud session (scratch env: `pip install mlflow==3.16.1 sacrebleu`; for skinny also `pandas`), then the maintainer runs it on the NAS and looks at Evaluation runs and Overview -> Quality.
