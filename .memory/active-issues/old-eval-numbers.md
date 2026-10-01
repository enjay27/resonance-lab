# BLEU / chrF / TER from experiment/translategemma's eval.py are not trustworthy

Found 2026-10-01 while moving its metrics into `eval_metrics.py` (step D).

- That `eval.py` built `references = [[ref1], [ref2], ...]` and called
  `sacrebleu.corpus_chrf(predictions, references)`. sacrebleu expects ONE reference set
  `[[ref1, ref2, ...]]`; the one-element-lists form is accepted without an error but scores
  **only the first sample**. Reproduced with sacrebleu 2.6.0: first line right + rest garbage
  -> chrF 100.0 (correct 42.06); first line garbage + rest right -> 0.0 (correct 63.85).
- So any BLEU / chrF / TER number printed by that script (including numbers behind the
  TranslateGemma-4B 1.1.0 decision) describes one eval line, not the eval set. COMET, JP leakage,
  term accuracy, exact match and the category breakdown were computed per sample and are fine.
- `eval_metrics.standard_metrics` passes the references correctly; `tests/test_eval_metrics.py`
  pins the case. **Re-run the shipped model through `scripts/llamafactory/eval.py` before using
  it as the baseline** in the shortlist's zero-shot round.
- BLEU is 0 when no line has 4 tokens (corpus BLEU needs 4-grams); trust chrF/COMET for short chat lines.
