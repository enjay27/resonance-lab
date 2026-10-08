# Agent Operating Rules — resonance-lab

resonance-lab fine-tunes the translator model of
[resonance-stream](https://github.com/enjay27/resonance-stream): Japanese Blue Protocol:
Star Resonance chat → natural Korean, as a GGUF `q4_k_m` for resonance-stream's llama.cpp
server. The base model is being re-chosen —
`.memory/roadmap/translator-shortlist-2026-10-01.md` is the plan.

**Pipelines are switchable:** `python run_pipeline.py --pipeline <name>` (registry:
`pipelines.py`). `llamafactory` (default; LLaMA-Factory SFT + LoRA, one model per profile in
`configs/llamafactory/<profile>/`, ends in a q4_k_m GGUF) and `unsloth` (Qwen3 1.7B). Each has its
own folder under `scripts/`, its own stage list, and its own requirements file / virtualenv
(`requirements-<pipeline>.txt`; the stacks pin different trl/transformers). They share
`validate.py` and `preprocess.py` (`--format instruction|pair` is the row layout each one reads).
The shipped model (TranslateGemma-4B, gist 1.1.0) was trained on `experiment/translategemma`,
whose features are being brought in — plan and what is left in `.memory/roadmap/next.md`.

The repo is two parts. **Which part you touch decides which gate applies.** That is the
most important thing on this page.

| tree | part | runs on | gate |
|---|---|---|---|
| `scripts/fetch_data.py` `scripts/validate.py` `scripts/preprocess.py` `scripts/unsloth/split_dataset.py` `scripts/llamafactory/lf_tools.py` `scripts/llamafactory/update_dataset_info.py` `scripts/llamafactory/watch_training.py` `scripts/mlflow_compare.py` `scripts/mlflow_genai_eval.py` `scripts/judge_check.py` `judge_client.py` `judge_local.py` `gate_pipeline.py` `gate_compare.py` `glossary.py` `translation_check.py` `translation_assemble.py` `scripts/translate_agents.py` `labeling_tools.py` `scripts/label_lines.py` `judge_prompts.py` `scripts/compare_judges.py` `compare_runs.py` `eval_metrics.py` `genai_eval.py` `text_rules.py` `hf_data.py` `runs.py` `tracking.py` `run_queue.py` `tracker.py` `track_records.py` `stage_tracking.py` `parameter_test.py` `overlap.py` `valsplit.py` `manifest.py` `config.py` `pipelines.py` `run_pipeline.py` `configs/` `deploy/` | **data** — raw chat logs → checked, LoRA-ready train/val JSONL | any OS, CPU | `just check` (ruff lint + pytest) |
| `scripts/unsloth/train.py` `scripts/unsloth/eval.py` `scripts/unsloth/fix_metadata.py` `scripts/llamafactory/train.py` `merge.py` `gguf.py` `eval.py`, GGUF conversion (README) | **model** — fine-tune, merge, clean, evaluate | **Windows + CUDA GPU only** (`requirements-<pipeline>.txt`) | none automated — a manual run, reported |

`just check` needs only `requirements-dev.txt` (`pip install -r requirements-dev.txt`;
`pip install rust-just` for `just`). CI (`.github/workflows/ci.yml`) runs it on Linux on
every push to `main` and every PR. The model part never runs in CI or in a cloud session:
no GPU, and torch/unsloth are not installed there.

**New pure logic goes in the data part**, where it is tested on every OS. Model-part
scripts run at import time (no `main()`); keep anything worth testing out of them.

**This repo owns the prompt.** The prompt the model is trained on is decided here, and
resonance-stream follows it (`crates/core/src/text.rs`, `translation_prompt`) — decision of
the maintainer, 2026-10-01. So a change to the prompt or chat template is made here first,
written down in `.memory/active-issues/stream-contract.md`, and then copied to
resonance-stream in its own PR there; the app must send exactly what training saw.
The training data comes from the app (`dataset_<CHANNEL>.jsonl`: `pid`, `original`,
`translated` — `null` when untranslated, `timestamp`).

---

## Tech Stack

- **Python 3.13** (README; CI runs the data gate on 3.13). Windows for the model part.
- **Gate dev deps** (`requirements-dev.txt`): pytest, ruff, pyyaml (profile yaml files), rich (the monitor).
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
- **Gate tooling:** pytest, ruff (`pyproject.toml`), `just`.

---

## Repository Layout

```
.claude/              skills/: workflow-control, season-data (the runbook for labelling + translating a season's chat with agents)
.memory/              working memory; see .memory/README.md
.github/workflows/    CI (data gate) + auto-merge
justfile              the gates as commands
config.py             every path and hyper-parameter; INSTRUCTION (system prompt)
eval_metrics.py       shared eval scoring + report (chrF/BLEU/TER, COMET, JP/think leakage, terms); tested
genai_eval.py         per-sample eval for `mlflow.genai.evaluate`: the predictions JSONL, the per-line scores (chrF, JP leak, term, discord, ...), `build_scorers`; tested
text_rules.py         JP / Hangeul patterns shared by preprocess and the eval metrics
hf_data.py            the HF dataset: config, `hf download` command, merge of the per-channel files (each row tagged with its `channel`, `MERGE_FORMAT`), fetch state; tested (no network)
runs.py               one training = one run dir `<output_dir>/<run id>/` + run.json status; merge takes the latest complete run; tested
tracking.py           what a run records in MLflow, pure: .env.mlflow settings, params/tags/metrics from yaml, manifest, fetch state, eval report, trainer files (no mlflow import); tested
run_queue.py          the local record of every run: an append-only JSON-lines journal `.run.result.backup.jsonl` (one line per write; sent events acknowledged by a `sent` line and dropped at the next start, the run header + server id kept; the old TinyDB file is migrated once); written BEFORE anything is sent; tested
track_records.py      what each stage records (params/tags/metrics for train, merge, gguf, eval; the tracker run id `<profile>-<run id>`), pure, no mlflow; tested
stage_tracking.py     the stages' use of the tracker (open_tracker never raises; start/finish_training, resume_stage, record_stage, fail_stage); tested
compare_runs.py       the experiment's runs as one table (profile, lr, epochs, best eval loss, eval scores, data revision), sorted/filtered; pure; tested
tracker.py            sends queued runs to the NAS oldest-first, resumable, never raises into a training; `from_environment()` -> Tracker or NullTracker; tested with a fake client
judge_local.py        the same judge WITHOUT llama-server: `LocalKevClient` (a `SystemOneClient` whose `post` runs the Kev model loaded in this process; own venv `.venv-kev`, requirements-kev.txt); the loader is injected, only `load_kev` imports torch/kev (the part that is NOT tested here); tested with a stub engine
judge_prompts.py     what the Gate judge is asked, apart from the taxonomy: a variant (`configs/judge_prompts/<name>.json`: `instructions`, per-root `descriptions`, `drop`) gives the question's instructions and options; `default` = no file = the taxonomy's root descriptions as they are; every variant is another judge id (journals and probes never mix); `categorize.py --judge-prompt NAME`; pure; tested against the real taxonomy
gate_compare.py      which Gate judge is better: a probe per judge saved as data/eval/gate1-compare/<label>.json (answers keyed by line sha1, never chat text), one table on the CURRENT labelled sample (accuracy / coverage / precision at a cutoff, argmax accuracy with a Wilson interval, coverage at a target precision, speed, per pair: who is right where they differ); pure; tested
gate_pipeline.py     the logic of notebooks/gate_pipeline.ipynb: draft of the labelled sample (stratified, judge pre-labels, `?` until a human labels it), channel mix, recipe preview, decision text, does-the-model-fit-the-card; pure; tested
judge_client.py       client of llama-server's `POST /v1/systemone` (a decision model answers typed questions): `SystemOneClient.choice` / `check_server`, `JudgeError`; injected `post`, stdlib only; fixtures recorded from a real server; tested
categorizer.py        Gate 1 baseline: rules put a chat line in a root category of the taxonomy, or nothing; pure; tested
gate_judge.py         Gate 1 with a judge (llama-server decision model): the question (`choice_options` from the taxonomy, `INSTRUCTIONS`, `judge_id` = model + prompt hash), `GateJudge` (cutoff on the margin, cached answers), `cutoff_sweep`, the resumable journal (`read_journal`, `trim_torn_tail`) and `categories_from` (the categories file is derived from the journal for a cutoff); pure; tested with `tests/judge_stub.py`
gate_eval.py          scores a categorizer on a hand-labelled sample (accuracy, coverage, precision / recall per root, confusions); pure; tested
taxonomy.py           configs/category_taxonomy.json: the categories (roots, children, descriptions), `coverage`: eval lines per root; pure; tested
dataset_recipe.py     a recipe (configs/datasets/<name>.json: category -> weight) + a categories file choose the training lines, seeded and nested; pure; `preprocess.py --recipe` applies it; tested
glossary.py           configs/glossary/<season>.json (the translation glossary as data: `required` names, `banned` renderings, `fixes`) + the version of docs/translation-glossary.md; `doc_version` ties the two; pure; tested
translation_check.py  is an agent's output complete, clean (no kana/kanji left), numbers kept, terms verbatim, in line with the glossary? rows in, problems out; pure; tested
translation_assemble.py  translating with agents, minus the agents: choose the lines (guild/non-Japanese skipped by default), batches, rounds (latest wins), the glossary's fixes, term table, the brief, `require_current` (assemble REFUSES when the glossary document changed since `prepare`); pure; tested
labeling_tools.py    labelling a season with agents, minus the agents: distinct lines of the raw logs, batches + brief (docs/labeling-guide.md), the check of the agents' labels, labels.jsonl, the judge's sample (dev/test by a hash of the line; labels the taxonomy lacks -> configs/label_map.json), the counts table; pure; tested
overlap.py            is a training line also an eval line? (normalised + near-duplicate); preprocess drops them; tested
valsplit.py           which lines are validation: sha1 of the normalised line, so pairs/variants stay together and lines keep their side as data grows; tested
manifest.py           lora_train_data.meta.json: how the training file was made (style, reverse, shas, counts); update_dataset_info/train check it; tested
parameter_test.py     the logic of notebooks/parameter_test.ipynb: ParamSet, the stage commands, run_command (streams, stops the process tree on interrupt), curves from trainer_log.jsonl, sweep, decision text; tested
pipelines.py          registry: pipeline name -> ordered stage scripts (no torch; tested)
prompts.py            instruction text per model family and direction (translategemma, hy); used by preprocess + eval; tested
run_pipeline.py       --pipeline <name>: runs its stages in order, stops at the first failure; --from/--only <stage> run part of it
scripts/
  fetch_data.py         shared data: `hf download` the app's dataset_<CHANNEL>.jsonl at the revision pinned in configs/hf_dataset.yaml, merge -> raw log;
                          only when missing/changed; skipped without a repo or with RESONANCE_RAW_LOGS; `--pin` writes the latest commit; `--force`
  mlflow_compare.py     data: `python scripts/mlflow_compare.py [--profile hy] [--sort eval-loss|chrf|term] [--markdown] [--all]` prints compare_runs' table from the NAS's MLflow
  mlflow_genai_eval.py  data: `python scripts/mlflow_genai_eval.py [--model M] [--prompt ...]` evaluates `eval.py`'s saved predictions line by line in MLflow (traces + assessments, experiment `resonance-lab-eval`)
  label_lines.py        data: `python scripts/label_lines.py prepare|check|assemble|export|report --season S1 ...` the agent labelling workflow in data/labeling/<season>/ (gitignored): batches + brief, checks, labels.jsonl, judge-sample.jsonl, labels.meta.json (guide version, git sha); assemble refuses when docs/labeling-guide.md changed since prepare
  translate_agents.py   data: `python scripts/translate_agents.py prepare|check|assemble|revise|report --season S1 ...` the agent translation workflow in data/translation/<season>/ (gitignored): batches + brief per round, checks, final.jsonl + terms.tsv + final.meta.json (glossary version, git sha); agents are started by the session (skill season-data)
  compare_judges.py     data: `python scripts/compare_judges.py [--cutoff X] [--only a,b]` compares the saved probes (made with `categorize.py --probe ... --save-probe LABEL`) on the labelled sample; needs no model
  judge_check.py        data: `python scripts/judge_check.py [--url U]` is the judge server up, new enough, serving a decision model? (exit 1 with the reason)
  categorize.py         data: `python scripts/categorize.py [--judge-url URL | --judge-local [RUN]] [--judge-prompt NAME] [--cutoff X] [--use-channel]] [--probe [SAMPLE] [--save-probe LABEL]]` raw log -> data/processed/categories.jsonl (rule baseline, or the judge: resumable via data/processed/gate1_judge.jsonl; falls back to the rules with a warning when the server is unreachable), or score the baseline / the judge (+ cutoff sweep) on a labelled sample; `tests/test_judge_live.py` runs against a real server when `RESONANCE_JUDGE_URL` is set
  validate.py           shared data: raw-log sanity gate (empty/missing file, >1% damaged lines, >10% Hangeul in `original` -> ValidationError; limits in config.py)
  preprocess.py         shared data: raw {original, translated} -> {instruction, input, output};
                          `clean_reason` drops empty/untranslated, Hangeul-in-source,
                          JP-left-in-output, 10x-long, recruitment-spam, duplicate rows
                          and counts each reason; exits 1 when >30% of the usable rows are suspicious (--max-drop);
                          `--recipe` keeps only the lines a dataset recipe selects by category weight (dataset_recipe.py)
  unsloth/              pipeline `unsloth` (Qwen3 1.7B)
    split_dataset.py      data: dedup by input, shuffle (seed 42), train/val -> lora_dataset/
    train.py              model: LoRA fine-tune (unsloth), merge -> model_f16/
    fix_metadata.py       model: drop `score.weight` -> model_f16_clean/
    eval.py               model: demo lines, then the shared eval report (when data/eval/ has a dataset)
  llamafactory/         pipeline `llamafactory` (default)
    lf_tools.py           data: profiles, dataset_info.json, command lines (argument lists), run helpers
    update_dataset_info.py  data: writes data/dataset_info.json -> the processed pair file
    train.py              model: `llamafactory-cli train` into a fresh run dir (`output_dir=` override), log -> <run dir>/train_stdout.log
    merge.py              model: `llamafactory-cli export` (adapter -> full model)
    gguf.py               model: convert to F16 GGUF, quantize to q4_k_m -> model_gguf/
    eval.py               model: generate on the eval set (--prompt chat-template|training), shared report
    watch_training.py     data: live training monitor (`rich`); TrainingState parses the two logs, tested
notebooks/            parameter_test.ipynb: one parameter test / a sweep, train -> merge -> eval -> compare -> decide (Jupyter Lab from the project .venv, requirements-notebook.txt); gate_pipeline.ipynb: the dataset pipeline's Gate steps, raw log -> judge (local Kev or llama-server, `BACKEND`) -> draft sample -> probe -> cutoff -> full pass -> recipe preview (Jupyter Lab from `.venv-kev`); both committed with outputs CLEARED (tests/test_notebooks.py)
configs/llamafactory/<profile>/   train.yaml + merge.yaml per model (tests check they agree)
configs/label_map.json  labels of the labeling guide the taxonomy lacks -> the taxonomy path the judge sees (tests keep it in step with docs/labeling-guide.md); configs/glossary/     the translation glossary as data, one JSON per season (tests keep it in step with docs/translation-glossary.md); configs/judge_prompts/ judge-facing wordings of the Gate question (default, clean, clean-no-other, v2, v2-no-other); configs/datasets/     dataset recipes (category -> weight); configs/category_taxonomy.json: the categories + descriptions (tests check they agree)
deploy/mlflow/        the MLflow tracking server for the maintainer's NAS (Dockerfile, compose, basic_auth.ini, .env.example, README); tests/test_deploy_mlflow.py guards it
docs/                 versioned guides, edited per season: translation-glossary.md (official + decided Korean game terms, translation rules), labeling-guide.md (how a chat line gets its category); tests/test_docs.py pins header, changelog and taxonomy coverage
tests/                pytest for the data part; conftest.py has the JSONL fixtures
data/raw/ data/processed/   stage inputs/outputs (config.py paths) -- GITIGNORED
```

Model and dataset outputs (`lora_dataset/`, `model_f16*/`, `outputs/`, `*.gguf`,
`*.safetensors`, `llama.cpp/`) are gitignored. Never commit them.

---

## Conventions

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

## Guardrails

- **Plan first.** Do not modify scripts, config, manifests or CI on the first turn of a
  task. Present an impact analysis (`g callers` from the `graft-kade` skill is the evidence) and wait for
  explicit confirmation. See `.claude/skills/workflow-control/SKILL.md`.
- **Refactors do not change behaviour.** A move/split commit changes no logic, no
  hyper-parameter, no prompt and no file format. Changing `INSTRUCTION`, the chat
  template or a training hyper-parameter changes the model — that is a feature, with its
  own PR and an eval run.
- **Zero hardcoded credentials.** No HF tokens or keys in committed files.
- **No data or weights in git.** Chat logs are players' messages; they stay local.
  Test fixtures are short, made-up lines.
- **TDD for every task flow.** Test first: write the failing unit test that pins the
  wanted behaviour, run it and see it fail for the right reason, then write the code that
  makes it pass, then refactor with the tests green. A bug fix starts with a test that
  reproduces the bug. If a change cannot be unit-tested (GPU / model code), say so in the
  commit body.
- **Auto-correction restraint.** Self-correct at most **2** times, then stop and ask.
- **Never report a gate as passed when it could not run.** A cloud session cannot train
  or evaluate; say so (`NOT VERIFIED: ...`).

---

## Definition of Done

0. **Test first.** New behaviour or a bug fix has its failing unit test before its code.
1. **Run the gate for every part touched** (table above): `just check`.
2. **Model-part changes need a manual run** on the Windows/CUDA machine (the affected
   stage, plus `eval.py` when the model changes); say in the commit body whether it was
   done, or `NOT VERIFIED: model part -- no GPU in this session`.
3. **Record the outcome in the memory tree.** `MEMORY.md` is an index under ~40 lines —
   update its *Now* section. Detail goes in `.memory/` (see its README).
4. **Push the branch and open the PR** — see *Version Control*; CI merges it when green.

---

## Version Control

**One task, one branch, one PR.** Claude runs the whole flow without being asked.

1. **Start.** Every new task gets its own branch from an up-to-date `main`:
   `git checkout main && git pull && git checkout -b claude/<short-task-name>`.
   Never commit to `main`. A follow-up to a merged task is a new task: new branch.
2. **During the task, commit freely** -- as many local commits as help. Unpushed history may
   be tidied (`git commit --amend`, or `git reset --soft <base>` + one commit to squash).
   Never rewrite history that is already pushed.
3. **Finish = test, then push.** When the task is done, run the gate for every part
   touched and fix failures. Only a green local gate is pushed:
   `git push -u origin claude/<name>`. A check that could not run here is named in the
   last commit body (`NOT VERIFIED: train.py -- no GPU in this session`) and left to a
   manual run.
4. **Open the PR** against `main` (check for a PR template first). Do not merge it by
   hand: `.github/workflows/auto-merge.yml` merges it and deletes its `claude/*` branch
   (never any other branch) once the CI workflow passes on the PR's latest commit. If CI
   fails, fix on the same branch and push again -- the run for the new commit decides.
   Never skip, disable or edit a test/gate to get green.

### Several tasks in one session

1. **One PR at a time, in order.** Finish a task (gate green, pushed), open its PR, then
   **wait until the PR is merged**. Do not start the next task, or push anything for it,
   before that.
2. **CI failed?** Fix it first, on the same branch, and push again. Retry until the PR
   merges; if a failure is not this PR's (red on `main` too), say so on the PR.
3. **Before the next task, check it is really done:** the PR is closed as *merged* and its
   `claude/*` branch is gone. Then start from `main` again: `git fetch origin main &&
   git checkout -B claude/<next> origin/main`. A later task never stacks on an unmerged one.
4. Waiting is done with the PR event subscription (`subscribe_pr_activity`) and a
   check-in (`send_later`), not with `sleep` loops. Update `MEMORY.md` in each task's own
   branch, so a merged task never leaves the index behind.

```bash
git status            # check BEFORE -A, never after
git add -A && git commit
```

- **Commit subject states the point of the change**, not the files touched
  (`Data stages are unit-tested on any OS`, not `add tests`). The body says what
  changed, why, and **what is verified vs open**.
- `MEMORY.md` and `.memory/` updates go in the branch, before the push.
- **Claude never commits work it did not do.** Pre-existing changes stay untouched.
- Only `claude/*` branches auto-merge. `workflow_run` workflows are read from `main`, so a
  change to `auto-merge.yml` itself takes effect after it has been merged once.

### Never commit
- Secrets, `.env`, HF tokens.
- Chat logs and datasets (`data/**`, `lora_dataset/`), weights (`*.safetensors`,
  `*.gguf`, `model_f16*/`, `outputs/`), `llama.cpp/`, `unsloth_compiled_cache/`.
- `graft/` (regenerable), `__pycache__/`, virtualenvs, IDE folders.
- A half-applied tree "to save progress". Use a branch.
