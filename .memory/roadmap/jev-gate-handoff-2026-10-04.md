> **Superseded for Gate 1 (2026-10-04) by [`jev-gate1-handoff-2026-10-04.md`](jev-gate1-handoff-2026-10-04.md):** the first use of a judge is dataset quality (categorising messages) on `/v1/systemone`; the logprob backend below is dropped. This file's translation-fidelity design (ja -> ko) stays possible later.

# Handoff: the Jev gate (typed judge), architecture for PR 2a — 2026-10-04

**For the next session.** Read this, then `.memory/roadmap/mlflow-genai-eval.md` (the decisions of 2026-10-03, which this builds on, does not replace) and `CLAUDE.md`.
Per CLAUDE.md the first turn is **plan only**: present this architecture (adjusted by what the code shows), list the decisions in §7 for the maintainer, wait for the OK, then TDD (§9).
Nothing below is built. Everything was designed in a cloud session: no GPU, no llama-server, no judge model was run.

## 1 · What the gate is, and what it is not
The deterministic per-line scores (`genai_eval.py`: chrF, jp_leak, term_ok, discord, exact) cannot say "meaning kept", "negation dropped", "something invented". The **gate** answers
typed **yes/no defect questions** about one eval line with a **probability**, from an open model served locally (decided 2026-10-03: no hosted Jev, privacy: only the hand-written eval set would
ever leave the process, and nothing does). `p_defect` per question per line -> an MLflow assessment `judge.<question>`; its run mean is a defect rate; a threshold turns a line into pass/flag.
- It **measures**; in PR 2a it enforces nothing (no exit code, no blocked stage). A pass/fail threshold needs calibration first (probe, §6).
- It is **not** in the training loop (decided) and **not** a translator: it never rewrites. It must not share the GPU with the translator (a 12B Q8 judge is ~13 GB of 16).

## 2 · Data flow (the new parts are marked +)
```
eval.py (model part, exists)                     outputs/eval/<profile>-<prompt>.jsonl    {original, reference, prediction, raw_output, category}
        |
        v
+ scripts/judge_pass.py  --judge-url ... --backend llama-server|jev-spec   (needs the judge server up, the translator unloaded)
        |   judge.py: for every line x every applicable question -> Judge.ask() -> p_defect | unscored
        v
+ outputs/eval/<profile>-<prompt>.judge-<judge id>.jsonl      one row per line: {index, verdicts: {question: {p, status}}}  + a header row with the judge tags
        |
        v
scripts/mlflow_genai_eval.py --judge <that file>        (exists; gains --judge: no network, no GPU, the scorers only READ the file)
        -> traces with assessments judge.<question> (+ the existing ones), run tags judge.model / judge.backend / judge.prompt_hash / judge.temperature
```
**Decision proposed (D2): a two-phase design, judging first into a file, MLflow second.** Why: the judge needs the GPU and a server; `mlflow.genai.evaluate` needs the NAS; the stages' rule
(`mlflow_genai_eval.py` docstring) is that a flaky dependency must not fail the other. A cache file makes the pass resumable and re-scorable (a new threshold or temperature needs no new inference), and
makes MLflow scoring deterministic and offline. The alternative (scorers calling the server inside `evaluate`) is simpler to write and worse in every other way.

## 3 · Modules (data part unless marked; each pure, injected client, no network in tests)
| file | content |
|---|---|
| `judge.py` | `Question` (name, instructions, `criteria={true: "...", false: "..."}`, `applies(row)`), `QUESTIONS`, `build_state(row)` / `build_prompt(question, state)`, `Verdict(p_defect: float \| None, status: "scored"\|"unscored", reason)`, the `Judge` protocol (`ask(question, state) -> float`, `.tags() -> dict`), `JudgeError`, `judge_line(judge, row, questions)` (a `JudgeError` or an unusable answer becomes `unscored`, never an exception), `judge_rows(...)`, `prompt_hash()` |
| `judge_probability.py` | pure maths: `p_true(top_logprobs, true_tokens, false_tokens, temperature=1.0)` = softmax over the two candidate groups (all spellings: `true`, ` true`, `True`, `yes`...), `None` when the two candidates hold < `MIN_MASS` of the probability (then `unscored`); `calibrate(temperature)`; `rate(verdicts, threshold)` (defect rate = share >= threshold; mean p) |
| `judge_backends.py` | `LlamaServerJudge` (OpenAI-compatible `/v1/chat/completions` with `logprobs`+`top_logprobs`, `max_tokens=1`, temperature 0; or `/completion` with `n_probs`: which one is **unverified**, see §8) and `JevSpecJudge` (`POST /api/alpha/decisions` {model, state, questions:{id:{type:"noul", instructions, criteria}}} -> `answers[id]` (a number, >= 0.5 true) and OpenSourceJev's `POST /v1/systemone`). Both take an injected `post(url, payload) -> dict` (stdlib `urllib`, no new dependency). **No hosted/OpenRouter client (D4)**: it would need an API key and sends data out |
| `judge_probe.py` | the planted-error probe: `plant(kind, ko_text) -> str \| None` for `number_swap`, `negation_drop`, `clause_drop`, `sentence_add` (`None` = does not apply to this line), `probe_scores(results)` = recall per kind and false-alarm rate on untouched lines at a threshold; `candidates_table` as markdown |
| `genai_eval.py` (edit) | `judge_scorers(path, scorer=None)`: one MLflow scorer per question reading the judgments file (returns `p_defect` or `None` for unscored, like `term_ok`); `build_scorers` unchanged without a judge |
| `scripts/judge_pass.py` (data part, run on the PC) | CLI: `--model/--fast/--prompt` (same resolution as `mlflow_genai_eval.py`), `--backend`, `--judge-url`, `--judge-model`, `--temperature`, `--check-server`; writes the judgments file; exit 1 with a message when the server does not answer or has no `logprobs` |
| `scripts/judge_probe.py` (data part, run on the PC) | CLI over the same backends: builds planted candidates from the pool (§6), prints `candidates_table`; one run per candidate model |
| `scripts/mlflow_genai_eval.py` (edit, **PR 2b**) | `--judge FILE`: adds `judge_scorers`, sets the `judge.*` tags |
| `config.py` (edit) | `JUDGE_URL` default `http://127.0.0.1:8081`, `JUDGE_MIN_MASS`, `JUDGE_THRESHOLD = 0.5`. Paths from `EVAL_OUTPUT_DIR`; a script never hardcodes one |

## 4 · The questions (binary defects: `true` = the defect is present)
Source = the Japanese line, translation = the model's Korean. Each prompt is `state` first (source, translation), then the question, so llama-server's prompt cache reuses the shared prefix across the questions of one line.
| name | defect asked | needs |
|---|---|---|
| `meaning_changed` | the translation says something different from the source | source, translation |
| `negation_flipped` | a negation is dropped, added or flipped | source, translation |
| `number_or_name_changed` | a number, name or ID differs | source, translation |
| `invented_content` | the translation adds information the source does not have | source, translation |
| `omitted_content` | information of the source is missing | source, translation |
| `japanese_left` | **control**: the translation still has Japanese | translation (ground truth: `jp_leak`) |
| `game_term_wrong` | **control**: a game term of `TERM_DICT` is rendered otherwise | source, translation, the expected term (ground truth: `term_ok`) |
The two **controls** have a deterministic answer already, so every judged run also measures the judge's agreement with it for free. `applies(row)` skips a question the line cannot raise (no digit in the source -> no
`number_or_name_changed`; no term -> no `game_term_wrong`), which is `None` in MLflow, like `term_ok`. **D1:** the judge sees source + translation only, **not the reference**: a reference-aware judge would
punish a correct but differently worded translation. The reference is only a hint in `game_term_wrong`.

## 5 · Probability and calibration
One-token answer; read `top_logprobs`; sum the probability of every spelling of the two labels; `p_defect = P(true) / (P(true) + P(false))`; if the two together hold less than `JUDGE_MIN_MASS` (say 0.5)
the answer is not a yes/no: `unscored` with a reason. A temperature `T` divides the logits before the softmax; **T is fitted on the probe** (not guessed) and logged as `judge.temperature`.
Scores of different judges (model, backend, prompt hash, T) are not comparable: they are tags of the file and of the run, and `judge_pass.py` refuses to overwrite a file with other tags.

## 6 · The probe (picks the judge model; the generic JevBench score does not)
Take lines known to be good (the **references**, human-edited: free of the translator's noise), plant one defect each, and ask every question of every candidate:
- `number_swap` (change a digit), `negation_drop` (remove 안 / 못 / 지 않- / 없- by rule), `clause_drop` (cut at `, ` or a sentence end; only for lines with two clauses), `sentence_add` (append a fixed unrelated sentence).
- **Recall** = share of planted lines flagged by the matching question; **false alarm** = share of untouched lines flagged by any of the five judged questions; both at 0.5 and at the fitted T.
- **Limit found while designing:** the eval set has 51 lines and most are short chat (`ㅋㅋㅋ`, `orz`): few lines have a digit, a negation or two clauses, so some kinds may apply to < 10 lines. **D3: pool = the eval set plus the
  validation split `lora_train_data.val.jsonl`** (not used for training; stays local, never committed). The probe reports `n` per kind so a thin cell is visible.
- Candidates (maintainer's list): one 12B (Winnow-12B Q8 or Cygnet; Gemma-4-12B base: needs llama.cpp >= June 2026 per the shortlist), one 4B Qwen (JevK5 v0.3 or decider-4b v2), OpenSourceJev, and a plain instruct GGUF
  as the baseline for "does the Jev tuning pay off". `--check-server` first: does this llama-server build return `logprobs`.

## 7 · Decisions for the maintainer (proposal in bold; state them in the plan)
D1 reference-free judging (**yes**). D2 judge into a file, MLflow second (**yes**). D3 probe pool = eval + validation split (**yes**). D4 no hosted client, Jev-spec only for local servers (**yes**).
D5 keep the 7 questions, two of them as controls (**yes**). D6 no pass/fail in 2a: report rates only; thresholds set after the probe (**yes**).

## 8 · Unverified / risks (say so in the commit body)
- llama.cpp server details are **from memory, not checked**: whether `/v1/chat/completions` returns `logprobs` in the maintainer's build (b8157 Vulkan ships in resonance-stream; the PC's llama.cpp may differ), the
  shape of `top_logprobs` vs `/completion`'s `completion_probabilities`, and how `true`/`false` tokenise (leading space, capitals): hence `--check-server`, and the spellings list in §5. Verify against the real server first;
  the tests use recorded-shape fixtures, so they pin the parser, not the server.
- Jev-spec shapes (`/api/alpha/decisions`, `/v1/systemone`) are from the 2026-10-03 reading of the public pages; do **not** copy `GeekLinkDev/jev-subtitle-translator` (GPL-3.0).
- Korean negation/clause rules for the probe are heuristics: they can plant a non-defect (a double negation). The probe's own tests pin the rules; a human glance at 20 planted lines before trusting a recall number.
- 357 calls (51 lines x 7) per pass is minutes on a 4B and a few minutes on a 12B; a 12B Q8 plus the translator does not fit 16 GB: run the pass after `eval.py`, translator unloaded.

## 9 · PR slicing and tests (TDD: each failing test first)
**PR 2a** (data part; gate `just check`; the probe/pass runs on the PC afterwards): `judge.py`, `judge_probability.py`, `judge_backends.py`, `judge_probe.py`, `scripts/judge_pass.py`, `scripts/judge_probe.py`, `config.py` constants, `judge_scorers`
(maybe split the scorer into PR 2b). Tests to write first: `p_true` over tokenisation variants and the `MIN_MASS` -> unscored case; `judge_line` turns a `JudgeError`/garbage into `unscored`; `applies`; prompt contains source and translation and
not the reference (D1); `prompt_hash` changes with the instructions; each backend against a fake `post` with a recorded-shape response and an error; `plant` per kind incl. `None` for a line it cannot apply to; `probe_scores`;
the judgments file round trip and the tag-mismatch refusal; the CLIs (`main(argv, client)` pattern of `mlflow_genai_eval.py`), including the exit 1 messages. **PR 2b:** `--judge` + `judge_scorers` + tags; a test against the stand-in client like `tests/test_mlflow_genai_eval.py`.
One PR at a time, each merged before the next (CLAUDE.md).

## 10 · State of the repo at handoff (2026-10-04)
- Branch `claude/modest-thompson-x8l5yb` has **commits not yet in a PR** (the maintainer asked for none): the notebook for Jupyter Lab + `RUN_SWEEP` switch, `terminate_tree` reports a failed taskkill (`StopFailed`),
  `Result.scores` (chrF/term in the sweep table and the decision text), and the notes of the second lr sweep. `just check` equivalent: ruff + pytest, 732 passed. **Open the PR first** (or ask), then start the Jev branch from `main` (`git fetch origin main && git checkout -B claude/<name> origin/main`) only after it merges.
- Per-sample MLflow eval PR 1 (`genai_eval.py`, `scripts/mlflow_genai_eval.py`) is merged (#52) but **never run on the maintainer's machine/NAS**: step 1.0 below.
- Hy-MT2-1.8B `-fast` results so far (51-line eval set, noisy): 2e-4 -> chrF 58.1 / term 33%; 4e-4 -> 0.7724 eval loss (best), chrF 53.8 / 43%; 8e-4 -> chrF 62.9 / 62% (best generation scores); 2e-4 x 6 epochs overfits. Details: `parameter-test-notebook.md`.
- Unverified on Windows: the interrupt fix, the sweep-table scores (`unverified-on-gpu.md`).

## 11 · The rest of the roadmap (2026-10-04 list; owner in brackets: Claude = data part here, PC = the maintainer's GPU machine)
**Jev:** 1.0 verify PR 1 on the PC: `eval.py --fast --prompt training`, then `mlflow_genai_eval.py --fast --prompt training`, look at "Evaluation runs" in `resonance-lab-eval` and Overview -> Quality [PC]; 1.1-1.3 PR 2a [Claude]; 1.4 probe the candidates [PC]; 1.5 PR 2b [Claude, PC]; 1.6 optional: judge score vs checkpoint step, judge-filtered training pairs.
**Hy tuning (notebook):** epochs 2/3/4 at 4e-4 and 8e-4 [PC]; decide whether `-fast` is the final recipe or a proxy (the full profile has batch 8, no packing, 1605 steps: **a `-fast` lr does not transfer**; the full run at 2e-4 had its eval minimum at epoch ~1.9) [maintainer]; `--set key=value` overrides for LoRA r/alpha, warmup, batch/packing, `cutoff_len` [Claude]; `--reverse` / `--prompt` toggles + 25/50/100% learning curve [Claude, PC]; packing confound #16 [PC].
**Eval quality (`pipeline-review`):** #6 term accuracy is a substring test; #7 grow the eval set to 200-300 lines + bootstrap CI (lines from the maintainer); #8 COMET never ran (`unbabel-comet` in no requirements), pin `sacrebleu`/`rich`, BLEU `13a` on Korean; #9 record decoding settings; #4 GGUF eval stage with llama-server (bf16 -> q4_k_m gap; settles resonance-stream's A4 double BOS) [model part]; #18 imatrix; #5 time holdout.
**Data:** #12 dedup exact only / half-width `ID:` spam filter; #13 label provenance (is `translated` the app's own output? share of human-edited rows) [maintainer's answer]; calibrate the data-check limits on the real raw log; `repo:` in `configs/hf_dataset.yaml` is `null` (notebook prints it) -> set it, `fetch_data.py --pin` [maintainer].
**Shipping:** the exact prompt per model incl. the `[P0]` placeholder rule and the derived Hy ko->ja prompt, pinned in `stream-contract.md`; copy it to resonance-stream (its own PR there) once Hy beats the shipped model on the GGUF eval; Hy-7B with QLoRA (bf16 does not fit 16 GB; profile `hy-mt2-7b` exists, never run); pin model revisions / `trust_remote_code` (#17).
**Housekeeping:** model scripts testable (`train.py`/`eval.py`/`fix_metadata.py` run at import); the `ruff format` baseline is off by the maintainer's call.

## 12 · First actions in the new session
1. `git status`, `git log --oneline -6`; read this file, `mlflow-genai-eval.md`, `genai_eval.py`, `scripts/mlflow_genai_eval.py` (graft skeleton first).
2. Present the plan of §3-§9 with the decisions of §7 (workflow-control step 1: impact = new modules + `genai_eval.py` + `config.py`; part = data; gate = `just check`; nothing reaches resonance-stream), wait for the OK.
3. Branch from an up-to-date `main` once this branch's PR is merged; tests first.
