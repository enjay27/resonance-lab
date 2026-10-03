# Per-sample evaluation in MLflow (`mlflow.genai.evaluate`) — started 2026-10-03

**Status: PR 1 (deterministic per-line scores) built, NOT yet run on the maintainer's machine/NAS. PR 2a/2b (a typed judge) are designed and decided, not started; each starts only after the previous PR has merged.**
Raised by the maintainer after looking at MLflow 3.16's GenAI view (Overview -> Quality: "No assessments available; monitor quality metrics from scorers"; Evaluation runs): "use this as a tracing metric".

## What it gives
Today a run has one number per metric (`eval.chrf`, ...) plus a text report in `outputs/eval/`. With this, every eval line is a **trace** with its own **assessments** (chrF, JP leakage, term check, `discord`, think leak, exact
match), run means (`chrf/mean`, ...), and two evaluation runs can be compared line by line. It targets the failure patterns of the first TG eval (`草` -> `고블린`, `ww` left as `ww`, `杖` -> `도검`, `discord` -> `디스코드`).

## PR 1 — what was built (2026-10-03)
- `genai_eval.py` (data part, pure, tested): the predictions JSONL (`prediction_rows`, `write_predictions`, `read_predictions`, `predictions_path`), `evaluation_data` (inputs `{text, category}`, outputs `{translation, raw_output}`,
  expectations `{expected_response}`), the scores (`SCORERS`: chrf = sentence chrF, jp_leak, think_leak, term_ok (None when the line has no term), discord_violation, exact_match) and `build_scorers` (the only place that imports mlflow).
  A test compares the per-line scores with `eval_metrics.evaluate`'s run-level counts so a trace cannot disagree with the report.
- `scripts/mlflow_genai_eval.py [--model M] [--fast] [--prompt training|chat-template] [--predictions FILE]`: reads `outputs/eval/<profile>-<prompt>.jsonl`, runs the evaluation in the experiment `resonance-lab-eval`
  (`config.MLFLOW_EVAL_EXPERIMENT`, apart from the trainings), tags the run `profile`, `eval.prompt`, `eval.n`, `training_run` (`<profile>-<run id>` from the merged model's `resonance_run.json`; absent for a pre-runs model).
  A separate script, not part of `eval.py`: `evaluate` needs the server now, the stages' tracker is offline-first and never raises. Errors exit 1 with a message; a failed tag is only a warning.
- `scripts/llamafactory/eval.py` (model part) also saves the predictions JSONL beside its report. Generation unchanged. **NOT VERIFIED on the GPU.**
- `requirements-llamafactory.txt` lists `pandas` (the skinny package does not install it; a test guards it).
- **Verified here against a real MLflow 3.16.1 server with the pinned `mlflow-skinny==3.16.1` client (+ pandas, sacrebleu), Python 3.11:** 3 made-up lines -> 3 traces with the right assessments, metrics `chrf/mean`, `jp_leak/mean`, ...,
  `term_ok/mean` over the lines that have a term only (None is skipped), run type `genai_evaluate`, tags set afterwards through `MlflowClient.set_tag`. Dict outputs work. Found by running it: `lf_tools` is not on the path of a
  script in `scripts/` (fixed). **Not seen:** the web UI (Evaluation runs page, Overview -> Quality), the NAS's basic-auth, Python 3.13 / Windows.
- Maintainer's check: `python scripts\llamafactory\eval.py --prompt training` (now also writes the JSONL), then `python scripts\mlflow_genai_eval.py --prompt training`; look at Evaluation runs in `resonance-lab-eval`.

## PR 2a/2b — the judge (decided 2026-10-03, not started)
Why: the checks above are deterministic; they cannot say "meaning kept", "negation dropped", "something invented". A judge answers typed yes/no questions with a **probability** (the Jev idea: typed answers, calibrated
confidence). Sources read: TypeSafe's Jev post (hosted API, early access, no string output), Benchmark Heaven's JevBench list (Jev-class systems; open ones such as Winnow-12B / Cygnet (Gemma-4-12B base), JevK5 / decider-4b
(Qwen3.5-4B base), OpenSourceJev (Qwen3.5-4B Q4_K_M, llama.cpp, MIT)), and the LinkedIn "Jev in the loop" post (a yes/no gate caught planted errors: wrong number, dropped negation, omission, invention; a proof of concept).
- **Decision (maintainer): open-source models, not the hosted Jev** (privacy: only the hand-written eval set would ever be sent anywhere; nothing is). Serve with **llama-server** or anything that runs on Windows.
- **Decision: both backends behind one interface.** (1) Own client for llama-server: a one-token `true`/`false` answer, `logprobs`, softmax over the two candidates = p(true). No extra dependency, any GGUF. (2) A client for
  Jev-spec servers: OpenRouter/TypeSafe `POST /api/alpha/decisions` (`{model, state, questions: {id: {type: "noul", instructions, criteria: {true, false}}}}` -> `answers[id]` = a number, >= 0.5 is true) and
  OpenSourceJev's `POST /v1/systemone` (Noul/Choice/Score). llama-server has no such endpoint. Do NOT copy `GeekLinkDev/jev-subtitle-translator` (GPL-3.0): the request shape is read from it, the code is ours.
- **Questions are binary defects, not 1-5 scores:** meaning kept, negation preserved, numbers/names preserved, nothing invented, nothing omitted, Japanese left, game term correct. Each is one assessment per trace
  (`judge.<name>` = p(defect)); the run mean is a defect rate. Log the judge model, backend, prompt hash and calibration temperature as tags: scores of different judges are not comparable.
- **PR 2a** (data part + a script run on the PC): `judge.py` (questions, prompts, probability parser, `Judge` interface with an injected client, "unscored" instead of a failed run) and a **planted-error probe**: take eval lines,
  plant one error each (swap a number, delete a negation, drop a clause, add a sentence), score each candidate's recall and false-alarm rate on our language pair, plus `--check-server` (does the llama-server build return
  `logprobs`?). First candidates: one 12B (Winnow-12B Q8 or Cygnet; ~13 GB at Q8, so it runs after the translator is unloaded on a 16 GB GPU) and one 4B Qwen (JevK5 v0.3 or decider-4b v2); also a plain instruct GGUF, to see
  whether the Jev tuning pays off. NOT VERIFIED here (no GPU, no model).
- **PR 2b:** `--judge` in `mlflow_genai_eval.py`, with the model PR 2a picked. Opt-in.
- **Not planned:** judging inside the training loop (the trainer's eval is cross-entropy, a callback would take GPU memory and could stall the training). Optional PR 3 instead: score the saved checkpoints after training and
  log a judge-score-vs-step curve. Another idea, not started: use a judge to filter noisy pairs out of the training data.
- Sequence on the PC: train -> merge -> `eval.py` (predictions) -> `mlflow_genai_eval.py [--judge]`. The translator and the judge must not share the GPU at the same time.

## Open
- Whether Overview -> Quality fills from these assessments as a trend over runs (expected, not seen).
- Which judge model: decided by the probe (PR 2a), not by the generic JevBench score.
