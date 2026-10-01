# Contract with resonance-stream

**Decision (maintainer, 2026-10-01): this repo owns the prompt.** resonance-stream changes
its `translation_prompt` (`crates/core/src/text.rs`) to whatever this repo trains on, in
its own PR, after the training format is settled here.

## 1 · The shipped model never saw the prompt the app sends — open

Shipped: TranslateGemma-4B + LoRA (gist model 1.1.0), trained on `experiment/translategemma`
(not on `main`). Training there, from `configs/training/bp_train.yaml` and
`scripts/update_dataset_info.py`:

```
<start_of_turn>user
遺跡1Fから　29k↑　＠T1<end_of_turn>
<start_of_turn>model
```

(`template: gemma3`, dataset `bp_translation_nosystem` = no system prompt, user turn is the
raw line, `cutoff_len: 128`; LLaMA-Factory adds BOS once.)

The app sends (`translation_prompt`, pinned by `translation_prompt_is_pinned` there):

```
<bos><start_of_turn>user
You are a professional Japanese (ja) to Korean (ko) translator. ... placeholders [P0] ...
遺跡1Fから　29k↑　＠T1<end_of_turn>
<start_of_turn>model
```

- The English instruction and the `[P0]` placeholder rule were never in training.
- The literal `<bos>` plus llama-server's own BOS is very likely a double BOS training never
  had — resonance-stream's open item **A4**.
- `main`'s Qwen3 pipeline (ChatML + Korean `config.INSTRUCTION`) is a third format; it is
  not what ships.

**Next (per `roadmap/translator-shortlist-2026-10-01.md`):** pick the model family on the
eval set, then train on one written-down prompt — the exact text the app will send, with
`[P0]`-masked lines in part of the data — and pin it here with a test (the exact training
text of one sample). resonance-stream then copies that text and its pin.

## 2 · Untranslated rows crashed the data stages — fixed on `main`

The app writes `translated: null` for lines never translated. `main`'s `preprocess.py`
copies the `null` into `output`, and `split_dataset.py` then fails
(`None.strip()` → `AttributeError`, reproduced 2026-10-01). `experiment/translategemma`'s
`preprocess.py` has the same bug (`data.get("translated", "").strip()` on `null`).
**Fixed on `main` (2026-10-01):** `preprocess.py` skips rows whose `original` or
`translated` is missing, `null` or blank and reports how many (tests in
`tests/test_preprocess.py`). `experiment/translategemma` still has it.
