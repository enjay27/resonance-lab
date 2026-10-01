# Zero-shot round, first numbers (2026-10-01, maintainer's GPU machine, 51-sample eval set)

Shared eval report (`eval_metrics.py`, correct per-set chrF/BLEU/TER). COMET not installed (`pip install unbabel-comet`).
Greedy decoding, LLaMA-Factory `ce9dc9e`, transformers 4.57.1.

| model | prompt | chrF | BLEU | TER | JP leak | term acc | Discord→디스코드 | exact |
|---|---|---|---|---|---|---|---|---|
| TranslateGemma-4B **fine-tuned (shipped)** | chat-template (app-like, never trained on) | 41.70 | 48.19 | 63.03 | 1/51 | 2/21 | 7 | 8/51 |
| Hy-MT2-1.8B **zero-shot, no training** | chat-template (Hy's English instruction) | 39.34 | 42.10 | 74.55 | 1/51 | 0/21 | 4 | 6/51 |
| Hy-MT2-1.8B zero-shot | training (raw line, no instruction) | 5.02 | 3.40 | 456.97 | 43/51 | 0/21 | 0 | 2/51 |

## What it says
- **Hy-MT2-1.8B untrained is within 2.4 chrF of the fine-tuned 4B** at less than half the size — a strong base,
  as the shortlist hoped. TER is worse (74.6 vs 63.0): wordier / politer ("-요", "-습니다") than the casual references.
- **Raw line without an instruction is unusable zero-shot** (84% JP leakage): Hy-MT2 answers like a chatbot, in
  Japanese or Chinese. Unlike TranslateGemma, Hy needs the instruction in the prompt, or fine-tuning on the raw
  format. Weighs toward **training on the instruction prompt** (what the app already sends, minus the TG wording) —
  decision still open in `active-issues/stream-contract.md` §1.
- Domain gaps are what fine-tuning must fix: game terms 0/21 (`リキャスト`→리캐스트 not 쿨타임, `杖`→지팡이 not 법사,
  `盾`→방패 not 탱커, `火力`→화력 not 딜러, `消化`→소화 not 숙제), register (polite vs casual), `草`→풀 not ㅋㅋㅋ,
  `草草草wwww`→"짧게 짧게 짧게 wwww" (hallucination), `ww` left as is. Discord kept as Latin in 1/5 alphabet lines.
- Hy was better than the shipped model on keeping `Discord` (4 vs 7 violations) and on `†…†` / `T@1` lines.
- **Correction (same day):** the shipped TG-4B was trained with the TranslateGemma instruction in the user turn, both directions
  (`active-issues/stream-contract.md` §1), so its `chat-template` row **is** its training prompt and the comparison is
  like-for-like; `--prompt training` (raw line) is not its format and need not be run. The rest of this bullet is superseded:
  ~~the shipped TG-4B was trained on the raw line, and its
  `chat-template` row is the prompt it never saw (the mismatch in `stream-contract.md` §1). Its fair number is
  `--prompt training`, **not yet run** — run it before reading "TG beats Hy" out of this table. 51 samples: differences
  of a few points are noise.~~ (Still true: 51 samples, a few points are noise.)

## Next
1. Maintainer: `hy-mt2-7b` zero-shot (`chat-template`); `pip install unbabel-comet` for COMET.
2. Decide the prompt (instruction + `[P0]` placeholder lines?), then a data-part change that writes the instruction into the
   training prompt per profile (today `--format pair` writes the raw line, no instruction, one direction), test first.
3. Fine-tune `hy-mt2-1.8b` on it, eval, compare with TG-4B trained the same way.
