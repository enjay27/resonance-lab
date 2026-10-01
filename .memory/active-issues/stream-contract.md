# resonance-stream contract — mismatches found

Found 2026-10-01 while aligning the repo with resonance-stream. Nothing here is fixed;
each is a decision for the maintainer.

## 1 · Prompt format: the app does not send what this repo trains on

- **resonance-stream** (`crates/core/src/text.rs`, `translation_prompt`, pinned by
  `translation_prompt_is_pinned`) sends a **Gemma-style** raw prompt:
  `<bos><start_of_turn>user\n` + an **English** instruction (placeholders `[P0]`...,
  "Produce only the Korean translation") + the line + `<end_of_turn>\n<start_of_turn>model\n`.
  Its comment says it "must match `make_prompt()` of the fine-tuning".
- **this repo** (`train.py`, `eval.py`) trains **Qwen3** with `tokenizer.apply_chat_template`
  (ChatML `<|im_start|>`), system = the **Korean** `config.INSTRUCTION`, no placeholder rule.
  There is no `make_prompt()` here.
- So either the app's prompt comes from an older (Gemma) fine-tune whose code is not in
  this repo, or a Qwen3 GGUF from here is served with the wrong template.
- Links to resonance-stream's open item **A4** (double `<bos>`, waits on the re-fine-tune;
  `.memory/roadmap/review-2026-09-30-round2.md` there): the re-fine-tune is this repo's job.
- **Check that closes it:** one prompt format, written down in both repos, and a test
  here that pins the exact training text of one sample against resonance-stream's
  `PINNED_PROMPT`.

## 2 · Untranslated rows crash `split_dataset`

- resonance-stream writes `dataset_<CHANNEL>.jsonl` rows `{pid, original, translated,
  timestamp}`, `translated` = `null` when the line was never translated.
- `preprocess.py` copies `null` into `output`; `split_dataset.py` then does
  `line_data.get("output", "").strip()` → `AttributeError` (reproduced 2026-10-01).
- `config.RAW_LOGS` is `data/raw/raw_translated_logs.jsonl` — one file; the app writes one
  file per channel. How they are merged/filtered today is undocumented (manual?).
- **Check that closes it:** a test with a `translated: null` row, then the fix (skip it in
  preprocess, most likely).
