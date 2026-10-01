# Contract with resonance-stream

**Decision (maintainer, 2026-10-01): this repo owns the prompt.** resonance-stream changes
its `translation_prompt` (`crates/core/src/text.rs`) to whatever this repo trains on, in
its own PR, after the training format is settled here.

## 1 · What the shipped model was trained on — corrected 2026-10-01, open items below

**Correction.** This section used to say the shipped model was trained on the raw Japanese line. That was read from
`experiment/translategemma`'s yaml and dataset names (`bp_translation_nosystem` = "no system prompt"), without the data file.
The maintainer's sample of the file the branch actually trained on (`raw/bp-training-dataset-final.jsonl`, hand-made, **not**
the output of that branch's `preprocess.py`) shows:

- `original` already holds the **TranslateGemma instruction + the line**; `translated` is the answer. So the user turn was
  (gemma3 template, LLaMA-Factory adds BOS once):

```
<start_of_turn>user
You are a professional Japanese (ja) to Korean (ko) translator. Your goal is to accurately convey the meaning and nuances of the original Japanese text while adhering to Korean grammar, vocabulary, and cultural sensitivities.
Produce only the Korean translation, without any additional explanations or commentary. Please translate the following Japanese text into Korean:
{line}<end_of_turn>
<start_of_turn>model
{korean}<end_of_turn>
```
- The file is **bidirectional**: every pair also appears reversed (ko→ja, instruction "...Korean (ko) to Japanese (ja)
  translator ... Please translate the following Korean text into Japanese:", answer = the Japanese line). Both directions
  are in the same file, doubling the rows.
- That instruction is the text TranslateGemma's own chat template builds from language codes, so `eval.py --prompt chat-template`
  for TG-4B **is** its training prompt: the baseline in `roadmap/zero-shot-results-2026-10-01.md` is like-for-like, and
  `--prompt training` (raw line) was never TG's training format.

The app sends (`translation_prompt`, pinned by `translation_prompt_is_pinned` there): `<bos><start_of_turn>user\n` + the
same instruction + the `[P0]` placeholder rule + the line. Remaining differences from training:

- the `[P0]` placeholder rule was never in training;
- the literal `<bos>` plus llama-server's own BOS is very likely a double BOS training never had — resonance-stream's open item **A4**
  (still unchecked);
- `main`'s Qwen3 pipeline (ChatML + Korean `config.INSTRUCTION`) is a third format; it is not what ships.

**This repo's pipeline does not yet reproduce that training file.** `preprocess.py --format pair` writes the raw line only (no
instruction) and has no reverse direction; its filters (`Hangeul in original`, `JP residual in translation`) would drop every
ko→ja row anyway. Where the final file came from (a one-off script?) is not in the repo — ask the maintainer.

**Next:** decide per model the exact prompt (instruction text, directions, `[P0]`), build it in a data-part stage (test first), pin
one sample's full text here, then resonance-stream copies it.

## 2 · Untranslated rows crashed the data stages — fixed on `main`

The app writes `translated: null` for lines never translated. `main`'s `preprocess.py`
copies the `null` into `output`, and `split_dataset.py` then fails
(`None.strip()` → `AttributeError`, reproduced 2026-10-01). `experiment/translategemma`'s
`preprocess.py` has the same bug (`data.get("translated", "").strip()` on `null`).
**Fixed on `main` (2026-10-01):** `preprocess.py` skips rows whose `original` or
`translated` is missing, `null` or blank and reports how many (tests in
`tests/test_preprocess.py`). `experiment/translategemma` still has it.
