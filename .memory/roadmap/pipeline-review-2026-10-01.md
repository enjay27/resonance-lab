# Training-flow review (2026-10-01) and the order it is worked in

Read from the code (no GPU in the session): `validate.py`, `preprocess.py`, `lf_tools.py`, the stage scripts, `eval.py`, `eval_metrics.py`, the profile
yamls and `first-training-run-2026-10-01.md`. Maintainer approved the order below ("proceed with your recommended order"); one task = one branch =
one PR, each merged before the next (CLAUDE.md). `pid` is resonance-stream's sequential message id and the app already strips user info on export,
so the privacy item (14) is closed.

## Findings (status: `todo` / `done <PR>` / `model` = needs a GPU run and an eval)
| # | finding | status |
|---|---|---|
| 1 | eval lines may be in the training data; nothing excludes them | **done** — `overlap.py`, `preprocess.py --eval-set/--keep-eval`; exact (normalised) + near (>=0.9, >=10 chars) |
| 2 | fixed `output_dir` per profile: LLaMA-Factory resumes the old checkpoint on a retrain (**confirmed in the pinned source**, `hparams/parser.py` 619-628: `get_last_checkpoint` -> `resume_from_checkpoint`) | todo — unique dir per run |
| 3 | `lora_train_data.jsonl` is shared by all profiles and does not say which prompt style / `--reverse` made it | **done** — `manifest.py`; `preprocess.py` writes `lora_train_data.meta.json`, `update_dataset_info.py` and `train.py` refuse a missing/changed/other-style file |
| 4 | eval runs the bf16 merged model, not the q4_k_m GGUF that ships | todo — GGUF eval stage (`llama-server`), log the bf16->q4 delta; also settles stream's A4 (double BOS) |
| 5 | `val_size: 0.05` is a random row split: reverse rows and near-duplicates leak into validation | **done** (hash split; time holdout NOT done) — `valsplit.py`: sha1 of the normalised line, ~5%, to `lora_train_data.val.jsonl`; yamls use `eval_dataset: bp_translation_val` instead of `val_size`. Open: a time holdout (newest ~2 months) is still a choice for the maintainer |
| 6 | term accuracy is a substring test (`ウルト -> 궁` hits 궁금) | todo |
| 7 | 51 eval samples / 21 term checks: differences are noise | todo — grow to 200-300, bootstrap CI |
| 8 | COMET never ran (`unbabel-comet` in no requirements); `sacrebleu`/`rich` unpinned in the llamafactory file; BLEU uses `13a` on Korean | todo |
| 9 | eval decoding (batch 1, `max_new_tokens=256`) is not recorded | with MLflow |
| 10 | `validate.py` halts on ONE Hangeul line though `preprocess` drops such rows; JSON errors uncounted | todo — fail on structure/threshold |
| 11 | no drop-rate guard in preprocess | todo — threshold, report saved as the sidecar |
| 12 | dedup: exact only, first wins; `total` counts reverse rows; spam filter only half-width `ID:` | todo |
| 13 | label provenance: if `translated` is the app's model output we train on its own errors | ask / record share of human-edited rows |
| 14 | privacy of the public HF dataset | closed (pid is a sequence id, user info stripped by the app) |
| 15 | TG-4B lr 1e-5 with LoRA r32/a32 vs Hy 2e-4 r64/a128; Hy overfits after ~2 epochs | model — control run TG at 1e-4, Hy 2 epochs |
| 16 | `-fast`: eval/save steps (100) > ~90 steps; packing without `neat_packing` confounds the comparison | model |
| 17 | `trust_remote_code: true` everywhere, no `model_revision` | model — pin revisions, trust only where needed |
| 18 | Q4_K_M without imatrix | model — decide with the GGUF eval |
| 19 | ~4.3k training rows | learning-curve run (25/50/100%) |
| 20 | `run_pipeline.py` has no `--from/--only` | todo |
| 21 | one fixed `train_stdout.log` | todo — per run |
| 22 | no Fetch Data stage | todo — `hf download <repo> --repo-type dataset --revision <sha>` first, revision pinned in `configs/hf_dataset.yaml` |

## Order
1. **Data safety** (data part, gate `just check`): #1 (done), #3, #5, #10/#11, #22.
2. **Run identity**: #2, #21, #20 — the same wiring MLflow needs.
3. **MLflow** PRs 1-5 (`mlflow-plan.md`, as revised there).
4. **Eval upgrades**: #6, #7, #8, #4.
5. **Model changes, each its own PR with an eval run**: #15, #16, #17, #18, #19.
