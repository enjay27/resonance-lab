# What each model is trained on — one example per template

Reference for humans and LLM agents. Verified 2026-10-01 on the maintainer's machine with
`python scripts/llamafactory/inspect_pair.py --model <profile>` (LLaMA-Factory `ce9dc9e`, real tokenizers).
Lines below are placeholders (`{original}` = the Japanese line, `{translated}` = the Korean one); **real chat rows are never
archived** (guardrail: no player data in git). The tested copy of these strings is `lf_tools.training_example` /
`tests/test_lf_tools.py`; if that test and this file disagree, the test wins. Regenerate with the script when a template,
LLaMA-Factory version or profile changes, and update both.

Data in every case: `data/processed/lora_train_data.jsonl` rows `{original, translated}` (`preprocess.py --format pair`),
mapped by `data/dataset_info.json` to prompt = `original`, response = `translated`. Same file for all profiles.
The user turn is **the raw Japanese line only** — no instruction, no system prompt (the decision for Hy is open:
`active-issues/stream-contract.md` §1, `roadmap/zero-shot-results-2026-10-01.md`).

| profile(s) | template | base model | cutoff_len | eos token |
|---|---|---|---|---|
| `translategemma-4b` (+`-fast`) | `gemma3` | `google/translategemma-4b-it` | 128 | `<end_of_turn>` (replaces eos) |
| `hy-mt2-1.8b` (+`-fast`) | `hy_dense_1_8b` | `tencent/Hy-MT2-1.8B` | 256 | `<｜hy_place▁holder▁no▁2｜>` |
| `hy-mt2-7b` (+`-fast`) | `hy_dense_7b` | `tencent/Hy-MT2-7B` | 256 | `<|eos|>` |

## `gemma3` — TranslateGemma-4B (`efficient_eos=False`)
```
MASKED : <bos><start_of_turn>user\n{original}<end_of_turn>\n<start_of_turn>model\n
TRAINED: {translated}<end_of_turn>\n
```
## `hy_dense_1_8b` — Hy-MT2-1.8B (`efficient_eos=True`)
```
MASKED : <｜hy_begin▁of▁sentence｜><｜hy_User｜>{original}
TRAINED: <｜hy_Assistant｜>{translated}<｜hy_place▁holder▁no▁2｜>
```
`<｜hy_Assistant｜>` is in the trained part, not the prompt (the template's user slot has no assistant marker). The
inference prompt ends with it, so the model continues with the answer: same text, no mismatch.
## `hy_dense_7b` — Hy-MT2-7B (`efficient_eos=True`)
```
MASKED : <|startoftext|>{original}<|extra_0|>
TRAINED: {translated}<|eos|>
```
No role tokens: `<|extra_0|>` ends the user text.

## Things that are easy to get wrong
- **The eos of the Hy templates is not in the template output**: `encode_oneturn` shows the response without it; LLaMA-Factory's
  supervised processor appends it (`efficient_eos`). `inspect_pair.py` appends it the same way (written from memory of that code
  — not checked against LLaMA-Factory's source in a cloud session).
- **BOS:** the template prefix adds it in training; Hy's tokenizers do not add it on their own (`tokenizer(...)` gives no BOS),
  gemma's does. `eval.py --prompt training` prepends it (`lf_tools.with_bos`).
- **Inference prompts differ from the training prompt** for the app/eval: the model's own chat template with its instruction
  (`lf_tools.chat_messages`): Hy = "Translate the following text into Korean. Note that you should only output the translated
  result without any additional explanation:\n\n{original}"; TranslateGemma = language codes -> its long English prompt.
- Token counts seen (real rows, 1 short / 1 medium chat line / 1 long recruitment line): Hy-1.8B 21 / 63 / 120, Hy-7B 24 / 76 / 152,
  Gemma 28 / 55 / 114 tokens; `cutoff_len` 128 for Gemma is close for long recruitment lines — check truncation if rows grow.
