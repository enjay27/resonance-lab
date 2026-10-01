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

**This repo now builds that file's rows (2026-10-01):** `preprocess.py --format pair --prompt auto [--reverse]`, instruction
texts in `prompts.py` (TranslateGemma's copied from the maintainer's sample, Hy's = its documented default prompt), pinned in
`tests/test_prompts.py`. The pipeline's Preprocessing stage runs `--prompt auto` (style from `--model`'s template), no `--reverse`.
**Reverse rule (found 2026-10-01 in `experiment/translategemma` commit `aab6b66`, "process bidirectual training", pushed late):**
the ko→ja row is tried only for a row whose ja→ko direction passed, with its own filters, counted separately, sharing the
`seen_inputs` set: Korean (the input) must hold no kana/kanji, Japanese (the answer) no Hangeul — both can never fire after the
forward filters passed —, answer not 10x longer than the input, no recruitment spam (>150 chars, 2+ `ID:`), and the Korean input
must not already be a seen input. `preprocess.py --reverse` now does exactly that. The same commit raised `cutoff_len` 128 → 256
(our `translategemma-4b` profile had copied the older 128; fixed). The maintainer's sample, though, had a long recruitment line
without a reverse partner, which this rule would write: the sample was probably trimmed — unconfirmed.
Hy's ko→ja prompt is derived (README template, target "Japanese"), unverified.

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
