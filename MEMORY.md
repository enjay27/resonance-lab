# Active State — resonance-lab

**Index, not the record.** Only what would be *false* the moment it goes stale lives here.

## Now — 2026-10-02

**Fixed model: Hy-MT2-1.8B** (maintainer, 2026-10-02; the 7B later; TranslateGemma profiles stay but are not developed). `hy-mt2-1.8b` is the default profile; `--fast` on any llamafactory script or
`run_pipeline.py` = that model's `-fast` profile (~4 min, eval every 10 steps). Plan and order: [`roadmap/next.md`](.memory/roadmap/next.md): `python scripts\mlflow_compare.py --sort eval-loss` compares the runs; sweep with `train.py --fast --lr 1e-4 [--epochs 2]` (no profile edit), then confirm on the full profile.

**Prompt: this repo owns it; resonance-stream follows** ([`stream-contract.md`](.memory/active-issues/stream-contract.md)). The shipped app model is TG-4B (instruction + line, both directions; the app also sends a `[P0]` rule and a
literal `<bos>` it never trained on). Hy's prompt is in `prompts.py`; copy it to resonance-stream only once Hy beats the shipped model on the eval ([`shortlist`](.memory/roadmap/translator-shortlist-2026-10-01.md)).

**Evals so far** (old BLEU/chrF invalid: [`old-eval-numbers`](.memory/active-issues/old-eval-numbers.md)): TG-4B on cleaned data, lr 1e-5, 3 ep: eval loss 0.888 still falling (underfit), chrF 45.2, term accuracy 1/21 (`--prompt chat-template`).
Hy-1.8B at lr 2e-4 overfits (train 0.15 vs eval 0.69, best epoch ~1.9); older Hy numbers (chrF 66.7 / `-fast` 59.3) were likely inflated by eval lines in the training data (now excluded) — re-measure.

**Pipeline state:** switchable pipelines/profiles; eval-overlap exclusion, manifest + validation split, validate/drop-rate guards (limits are guesses), Fetch Data (`--pin`/`--check`; HF dataset pinned `b37c268f`, the
unified file for ~2 months: local `include: "bp-training-dataset-*.jsonl"`, `fetch_data.py --force` once), run identity, MLflow end to end (NAS server, offline queue, `track_records.py`/`stage_tracking.py`; UI shows runs, curves,
artifacts; NAS `MINIO_ENDPOINT_URL` must be the NAS's real address; Ctrl+C -> KILLED; a failed artifact cannot block the queue; the offline queue is an append-only journal `.run.result.backup.jsonl`, TinyDB dropped). `-fast` profiles evaluate every 10 steps.
**Run once on the GPU machine under Jupyter (2026-10-04; Run All with the sweep off finished; sweep done: 4e-4 best eval loss 0.7724, 8e-4 best chrF 62.9 / term 61.9%, 2e-4x6 overfits; next: 4e-4 vs 8e-4 on the full profile with both prompts): `notebooks/parameter_test.ipynb` (`jupyter lab` from the `.venv`; whole lifecycle of a parameter test + a sweep loop; logic in `parameter_test.py`; commit it with outputs cleared)** — [`roadmap/parameter-test-notebook.md`](.memory/roadmap/parameter-test-notebook.md) (also holds the first lr sweep: 2e-4 > 1e-4 > 5e-5, eval loss still falling; next, in the notebook's sweep cell: 4e-4, 8e-4, 2e-4 with 6 epochs). **Per-sample evaluation in MLflow, PR 1 built 2026-10-03** (`genai_eval.py`, `scripts/mlflow_genai_eval.py`, `eval.py` saves the predictions JSONL; verified against a real MLflow 3.16.1 server, NOT yet run on your machine/NAS or seen in the UI); next a typed judge with an open model (**architecture + handoff: `roadmap/jev-gate-handoff-2026-10-04.md`**) (llama-server logprobs + a Jev-spec backend, a planted-error probe picks the model) — [`roadmap/mlflow-genai-eval.md`](.memory/roadmap/mlflow-genai-eval.md). Rule: **graft first** (`graft skeleton <file>` before reading a file, see CLAUDE.md). Index: [`roadmap/next.md`](.memory/roadmap/next.md).

**Jev (decision model) use, decided 2026-10-04 (**Next session (maintainer, 2026-10-04): TWO judge methods — with llama-server (built) and WITHOUT it (the model loaded in Python, `jaredpalmer/kev`) — and a notebook of the whole Gate pipeline first, without llama-server: plan [`roadmap/local-judge-and-gate-notebook-plan-2026-10-04.md`](.memory/roadmap/local-judge-and-gate-notebook-plan-2026-10-04.md) (decisions D1-D9 made, notebook = the DATASET pipeline, default judge **Kev-9B** (~18 GB bf16: check the card); **P0 spike done and P1 built 2026-10-04: `judge_local.py`, `judge_check.py --local`, `requirements-kev.txt` (own `.venv-kev`); kev-0.8b answers in-process on CPU; Windows+CUDA and 9B on the PC NOT VERIFIED; **P2 built: `categorize.py --judge-local`, `judge_id` kind (the journals of the two methods never mix)**; **P3 built: `gate_pipeline.py`; P4 built: `notebooks/gate_pipeline.ipynb` (the dataset pipeline's Gate steps; `BACKEND` local|server; run for real on CPU with kev-0.8b, all cells OK; Windows+CUDA, Kev-9B/4B, the server branch and any accuracy NOT VERIFIED). **Judge comparison built (`--save-probe`, `scripts/compare_judges.py`, notebook 6b); Kev-4B and 9B bf16 load on the maintainer's GPU.** Next [PC]: draft + correct the sample, one probe per judge (4B, 9B-Q8_0, 9B-Q4_K_M on llama-server), compare; then G1-4, G2-3**). Gate 1 on llama-server: [`roadmap/jev-gate1-handoff-2026-10-04.md`](.memory/roadmap/jev-gate1-handoff-2026-10-04.md).** Gate 1 baseline built (G1-1: `categorizer.py`, `scripts/categorize.py`, `--probe`); **G1-2 built 2026-10-04: `judge_client.py` + `scripts/judge_check.py`, fixtures recorded from a real llama-server (Julia-1 reads Japanese; handoff §11), D1-D6 approved; D6 built (merged raw rows carry `channel`; the next `fetch_data.py` re-merges by itself). **G1-3 built 2026-10-04: `gate_judge.py`, `categorize.py --judge-url/--cutoff/--use-channel` (resumable journal; categories derived from it), `--probe ... --judge-url` with a cutoff sweep; first real probe on the 98 made-up lines: Julia-1 (144M) collapses to `chat` (unusable for 9 roots), Kev-4B Q4 works (78% at cutoff 0, precision 89% at 0.3) — directional only (handoff §13). Next [PC]: hand-label ~200 real lines, run the probe on the GPU, pick the cutoff; then G1-4 (children), G2-3.** Eval set `category` = the taxonomy roots (maintainer decided; per-category chrF / term accuracy / coverage in the eval report, relabel `data/eval/` by hand). Categories + first recommended weights (a guess, no data): [`roadmap/dataset-categories-2026-10-04.md`](.memory/roadmap/dataset-categories-2026-10-04.md) (`configs/category_taxonomy.json`, `configs/datasets/balanced-v1*.json`). Gate 2 built: `dataset_recipe.py` (recipe = category weights, share = weight / total) and `preprocess.py --recipe/--categories`, `run_pipeline.py --recipe`, MLflow tags/params/compare column (G2-2a/2b; notebook cells G2-3 still to do); Gate 1 not started):** Gate 1 = Jev categorizes raw messages (`/v1/systemone` `choice`); **Gate 2 = the dataset ratio rule is CODE ONLY, no Jev** (maintainer) — [`roadmap/jev-gates-2026-10-04.md`](.memory/roadmap/jev-gates-2026-10-04.md).

**Model part never runs in a cloud session** (no GPU): every model stage is NOT VERIFIED —
[`unverified-on-gpu.md`](.memory/active-issues/unverified-on-gpu.md). Auto-merge of `claude/*` PRs works.

## Where the detail is

| read | when |
|---|---|
| [`CLAUDE.md`](CLAUDE.md) | rules, gates, layout |
| [`.memory/README.md`](.memory/README.md) | which memory file takes what |
| [`.memory/active-issues/`](.memory/active-issues/) | before trusting a doc, a run, or the stream contract |
| [`.memory/reference/`](.memory/reference/) | what each model's training example looks like (`training-examples.md`) |
| [`.memory/roadmap/`](.memory/roadmap/) | what is next, and what proves it done |
| [`.memory/sessions/`](.memory/sessions/) | why a decision was made, including wrong turns |

## Rules
- **Under ~40 lines.** *Now* carries what is next and what is unverified; nothing else.
- **Update *Now* every session**, even when the answer is "unchanged".
