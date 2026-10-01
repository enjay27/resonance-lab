# Next — candidates, in order

1. **Prompt contract with resonance-stream** — decide the format (see
   `active-issues/stream-contract.md` §1). Blocks the re-fine-tune (stream's A4).
2. **`translated: null` rows** — test + fix in preprocess (§2 of the same file).
3. **Ingest the app's per-channel files** (`dataset_<CHANNEL>.jsonl`) instead of one
   hand-made `raw_translated_logs.jsonl` — a data-part script, tested.
4. **Model scripts testable** — `train.py` / `eval.py` / `fix_metadata.py` run at import;
   move their pure parts (formatting the chat text, the `score.weight` filter) into
   functions the data gate can test. Behaviour-preserving, one PR.
5. **`ruff format` baseline** — one formatting-only commit, its hash in
   `.git-blame-ignore-revs`, then `fmt-check` into `just check`. Off by the maintainer's
   call (2026-10-01).
