# Next — candidates, in order

1. **Model choice + prompt** — this repo decides, resonance-stream follows
   (`active-issues/stream-contract.md` §1). Plan: `translator-shortlist-2026-10-01.md`
   (zero-shot eval round, then fine-tune the top family on the app's exact prompt).
   Blocks the re-fine-tune (stream's A4).
2. **Bring `experiment/translategemma`'s features in as a switchable pipeline** (maintainer,
   2026-10-01: keep the Qwen3 pipeline; take only the branch's features; `llamafactory`
   becomes the default). One PR at a time, each merged before the next:
   - **A — registry + move (done):** `pipelines.py`, `--pipeline`, `scripts/unsloth/`.
   - **B — shared preprocess (done):** the branch's filters (Hangeul in source, JP residual,
     10x-length hallucination, recruitment spam, first-occurrence dedup, per-reason report,
     malformed JSON counted not fatal) as `clean_reason`, tested first. **Changes the
     training data** (rows are dropped) — a feature, so re-run `eval.py` after the next
     training. Not taken yet: the branch's `{system,original,translated}` output format
     (its consumer is `update_dataset_info.py`, which reads a different file; decide in C).
   - **C — `llamafactory` pipeline:** `configs/training/*.yaml` (one yaml = one model
     profile, so Hy-MT2 / Gemma 4 / Qwen3.5 from the shortlist are config files), `train.py`
     (wraps `llamafactory-cli`), `export.py` (merge -> F16 GGUF -> q4_k_m),
     `update_dataset_info.py`, a training monitor on `rich` (OS-neutral; replaces the
     curses `watch_training.py`, helpers `classify_loss`/`classify_grad` tested), `config.py`
     loads the system prompt lazily (the branch's import-time read breaks CI), one
     requirements file per pipeline (trl 0.24.0 vs 0.29.0 conflict -> separate venvs),
     default switches to `llamafactory`.
   - **D — shared eval metrics:** chrF, COMET, JP leakage, think leakage, term accuracy
     (`TERM_DICT`) as a tested module both pipelines' `eval.py` use, so numbers compare.
   Prompt format is not in the registry yet (one pipeline carries the contract); revisit
   when the model is chosen (`translator-shortlist-2026-10-01.md`).
3. **Ingest the app's per-channel files** (`dataset_<CHANNEL>.jsonl`) instead of one
   hand-made `raw_translated_logs.jsonl` — a data-part script, tested.
4. **Model scripts testable** — `train.py` / `eval.py` / `fix_metadata.py` run at import;
   move their pure parts (formatting the chat text, the `score.weight` filter) into
   functions the data gate can test. Behaviour-preserving, one PR.
5. **`ruff format` baseline** — one formatting-only commit, its hash in
   `.git-blame-ignore-revs`, then `fmt-check` into `just check`. Off by the maintainer's
   call (2026-10-01).
