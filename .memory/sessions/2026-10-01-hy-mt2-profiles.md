# 2026-10-01 — Hy-MT2 profiles (model choice, step 1)

Task: "prepare next task, choosing new model". Decision (maintainer): Hy-MT2 only for now; verify the chat
template with the real model, else note it for a local run.

- **Could not download the model:** `huggingface.co` returns 403 from the cloud proxy; `raw.githubusercontent.com`
  works. So the template was read from Tencent's README (`train/README.md`: templates `hy_dense_1_8b` /
  `hy_dense_7b`) and from LLaMA-Factory `src/llamafactory/data/template.py` — they are already registered upstream.
- **Finding:** the two sizes use different formats. 1.8B: `bos + <｜hy_User｜>{text}<｜hy_Assistant｜>`, stop
  `<｜hy_place▁holder▁no▁2｜>`; 7B: `bos + {text}<|extra_0|>`, stop `<|eos|>`. One prompt text still serves both
  (the instruction goes inside the user turn); only the template wrapper differs.
- Hy-MT2's documented prompt: "Translate the following text into Korean. Note that you should only output the
  translated result without any additional explanation:\n\n{text}" — used for `eval.py --prompt chat-template`.
  No default system prompt. Recommended sampling is not greedy (see unverified-on-gpu.md).
- `eval.py` now picks the message via `lf_tools.chat_messages(template, text)` (gemma3 unchanged).
- Open for the prompt decision (`stream-contract.md` §1): Hy is instruction-tuned, so "raw line only" (what
  `training_prompt` encodes) vs the English instruction is a real choice, not a given as with TranslateGemma.
