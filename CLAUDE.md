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
| `scripts/fetch_data.py` `scripts/validate.py` `scripts/preprocess.py` `scripts/unsloth/split_dataset.py` `scripts/llamafactory/lf_tools.py` `scripts/llamafactory/update_dataset_info.py` `scripts/llamafactory/watch_training.py` `scripts/mlflow_compare.py` `compare_runs.py` `eval_metrics.py` `text_rules.py` `hf_data.py` `runs.py` `tracking.py` `run_queue.py` `tracker.py` `track_records.py` `stage_tracking.py` `overlap.py` `valsplit.py` `manifest.py` `config.py` `pipelines.py` `run_pipeline.py` `configs/` `deploy/` | **data** — raw chat logs → checked, LoRA-ready train/val JSONL | any OS, CPU | `just check` (ruff lint + pytest) |
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
.claude/              graft wiring (hooks, helpers), skills/: graft, workflow-control
.memory/              working memory; see .memory/README.md
.github/workflows/    CI (data gate) + auto-merge
justfile              the gates as commands
config.py             every path and hyper-parameter; INSTRUCTION (system prompt)
eval_metrics.py       shared eval scoring + report (chrF/BLEU/TER, COMET, JP/think leakage, terms); tested
text_rules.py         JP / Hangeul patterns shared by preprocess and the eval metrics
hf_data.py            the HF dataset: config, `hf download` command, merge of the per-channel files, fetch state; tested (no network)
runs.py               one training = one run dir `<output_dir>/<run id>/` + run.json status; merge takes the latest complete run; tested
tracking.py           what a run records in MLflow, pure: .env.mlflow settings, params/tags/metrics from yaml, manifest, fetch state, eval report, trainer files (no mlflow import); tested
run_queue.py          the local record of every run: an append-only JSON-lines journal `.run.result.backup.jsonl` (one line per write; sent events acknowledged by a `sent` line and dropped at the next start, the run header + server id kept; the old TinyDB file is migrated once); written BEFORE anything is sent; tested
track_records.py      what each stage records (params/tags/metrics for train, merge, gguf, eval; the tracker run id `<profile>-<run id>`), pure, no mlflow; tested
stage_tracking.py     the stages' use of the tracker (open_tracker never raises; start/finish_training, resume_stage, record_stage, fail_stage); tested
compare_runs.py       the experiment's runs as one table (profile, lr, epochs, best eval loss, eval scores, data revision), sorted/filtered; pure; tested
tracker.py            sends queued runs to the NAS oldest-first, resumable, never raises into a training; `from_environment()` -> Tracker or NullTracker; tested with a fake client
overlap.py            is a training line also an eval line? (normalised + near-duplicate); preprocess drops them; tested
valsplit.py           which lines are validation: sha1 of the normalised line, so pairs/variants stay together and lines keep their side as data grows; tested
manifest.py           lora_train_data.meta.json: how the training file was made (style, reverse, shas, counts); update_dataset_info/train check it; tested
pipelines.py          registry: pipeline name -> ordered stage scripts (no torch; tested)
prompts.py            instruction text per model family and direction (translategemma, hy); used by preprocess + eval; tested
run_pipeline.py       --pipeline <name>: runs its stages in order, stops at the first failure; --from/--only <stage> run part of it
scripts/
  fetch_data.py         shared data: `hf download` the app's dataset_<CHANNEL>.jsonl at the revision pinned in configs/hf_dataset.yaml, merge -> raw log;
                          only when missing/changed; skipped without a repo or with RESONANCE_RAW_LOGS; `--pin` writes the latest commit; `--force`
  mlflow_compare.py     data: `python scripts/mlflow_compare.py [--profile hy] [--sort eval-loss|chrf|term] [--markdown] [--all]` prints compare_runs' table from the NAS's MLflow
  validate.py           shared data: raw-log sanity gate (empty/missing file, >1% damaged lines, >10% Hangeul in `original` -> ValidationError; limits in config.py)
  preprocess.py         shared data: raw {original, translated} -> {instruction, input, output};
                          `clean_reason` drops empty/untranslated, Hangeul-in-source,
                          JP-left-in-output, 10x-long, recruitment-spam, duplicate rows
                          and counts each reason; exits 1 when >30% of the usable rows are suspicious (--max-drop)
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
configs/llamafactory/<profile>/   train.yaml + merge.yaml per model (tests check they agree)
deploy/mlflow/        the MLflow tracking server for the maintainer's NAS (Dockerfile, compose, basic_auth.ini, .env.example, README); tests/test_deploy_mlflow.py guards it
tests/                pytest for the data part; conftest.py has the JSONL fixtures
data/raw/ data/processed/   stage inputs/outputs (config.py paths) -- GITIGNORED
graft/                graft's generated cards -- GITIGNORED, regenerable (`graft build`)
```

Model and dataset outputs (`lora_dataset/`, `model_f16*/`, `outputs/`, `*.gguf`,
`*.safetensors`, `llama.cpp/`) are gitignored. Never commit them.

---

## Using graft (the repo is indexed) — graft first, as the index of every file

**Rule: before you open a source file, get its index from graft; read only the span you need.** The index is the
file's map; the file itself is the last thing to open, and only in pieces. (`.claude/skills/graft/SKILL.md` has the
full tool list.)

1. **Know a file?** `graft skeleton <file>`: every signature with its `file:line` span, ~10x cheaper than the file.
   Then `Read` with `offset`/`limit` for just that span. Never `Read` a whole code file, or `grep`/`rg` for it, first.
2. **Don't know where it lives?** `graft ask "<task>" --source` (code inlined at each hit). **Need every
   occurrence?** `graft grep "<literal>"` (ask is ranked top-N and misses some).
3. **Before moving, renaming, splitting or changing a signature:** `graft callers <sym> --depth all`. Editing the
   primary file and stopping is the classic miss. `config.py` constants are imported by name from every script —
   `graft grep "<NAME>"` before renaming one.
4. **The graph follows your edits** (PostToolUse hook); `graft build` if it looks stale. `graft/` is gitignored.
5. **Not indexed** (yaml, md, json, requirements, `.github/`, `.memory/`): read those directly. Files you are about
   to rewrite whole, or under ~50 lines, may be read directly too. Say so when graft is missing or empty instead of
   silently falling back to grep.
6. **Close a turn that used graft with the savings tally line** its hook asks for (`🌱 graft saved ~N tokens ...`).

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

- **Graft first.** Index a file with `graft skeleton` before reading it; see *Using graft*.
- **Plan first.** Do not modify scripts, config, manifests or CI on the first turn of a
  task. Present an impact analysis (graft `callers` output is the evidence) and wait for
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
