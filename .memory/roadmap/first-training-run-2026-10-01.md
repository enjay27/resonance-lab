# First training run: hy-mt2-1.8b (base profile), 2026-10-01, maintainer's GPU machine (16 GB)

Data: `preprocess.py --format pair --prompt auto --model hy-mt2-1.8b` (Hy instruction + line, ja→ko only, no `--reverse`), LLaMA-Factory `ce9dc9e`.
Profile `hy-mt2-1.8b`: LoRA r64/α128, lr 2e-4, 3 epochs, batch 2 x accumulation 4, cutoff 256, no packing.

| | |
|---|---|
| steps | 1605 (3 epochs x 535, effective batch 8); 226 validation rows (5%) |
| train runtime | 25 min 08 s — 1.064 steps/s, **8.5 samples/s** |
| GPU while training | VRAM ~8 GB of 16, GPU utilisation only 26–50% (the batch is small: the `-fast` profile should raise it) |
| train loss | ~0.07–0.32 in the last steps (smoothed ~0.15); run mean 0.505 |
| eval loss at step 1600 | **0.693** — far above the train loss: overfit signal, see below |

## Reading
- Train ≈ 0.15 vs eval ≈ 0.69 after 3 epochs at lr 2e-4: the model memorises the training lines. `load_best_model_at_end: true`, so the merge
  should use the best-eval checkpoint (check `trainer_state.json` `best_model_checkpoint`). Open: the eval-loss curve (does it bottom out near
  epoch 1–2?); if so, fewer epochs or a lower lr for the Hy profiles — a hyper-parameter change, decided with the eval reports.
- `-fast` has effective batch 32: about a quarter of the steps for the same epochs, so its eval loss is not directly comparable; compare the
  eval reports (chrF etc.) instead.
- The monitor's old `best: …@step` was the lowest single-step train loss and means nothing; now the footer shows eval loss and an overfit warning.
- `trainer_log.jsonl` (this LLaMA-Factory version) rows: `current_steps, total_steps, loss, lr, epoch, percentage, elapsed_time ("H:MM:SS"),
  remaining_time`; eval rows: `current_steps, total_steps, eval_loss, epoch, elapsed_time, remaining_time`. `train_stdout.log` holds mojibake
  tqdm blocks (cosmetic).

## Eval reports (51-sample set, `eval.py --prompt chat-template`, greedy) — same day
| model | chrF | BLEU | TER | JP leak | term acc | Discord→디스코드 | exact |
|---|---|---|---|---|---|---|---|
| shipped TG-4B (fine-tuned, older data) | 41.70 | 48.19 | 63.03 | 1/51 | 2/21 | 7 | 8/51 |
| Hy-MT2-1.8B zero-shot | 39.34 | 42.10 | 74.55 | 1/51 | 0/21 | 4 | 6/51 |
| **hy-mt2-1.8b** fine-tuned (base profile, 1605 steps, 25:08) | **66.74** | 64.52 | 44.24 | 0/51 | 16/21 (76%) | 0 | 14/51 |
| **hy-mt2-1.8b-fast** (packing+Liger, ~90 steps, 4:36) | 59.33 | 58.92 | 54.55 | 1/51 | 13/21 (62%) | 4 | 11/51 |

Eval loss, base profile (validation split, 226 rows): 0.897@100 → 0.728@500 → 0.7252@600 → 0.686@800 → **0.6656@1000 (epoch 1.87, the minimum)** → 0.70@1100 (epoch
boundary) → 0.69 flat to 0.6926@1600. So the 3rd epoch adds nothing; ~2 epochs is enough. `-fast` has **no eval-loss rows**: its `eval_steps`/`save_steps`
(100) exceed its ~90 steps — a mistake in the profile, to fix (scale them to the step count).

## Caution — probable train/eval overlap (unchecked; `preprocess.py` now excludes eval lines, so retrain before trusting any number)
The held-out validation loss is poor (0.67–0.70) yet the eval set scores chrF 66.7 with exact matches of idiosyncratic references (`ばんわ`→`존밤!`,
`ウルト溜まった`→`궁 찼다!`, `器用特化で組んでる`→`숙련 특화로 맞추고 있어`). That pattern fits eval lines being in the training data. If so, 66.7 vs the shipped
model's 41.7 is not a fair comparison (the shipped TG-4B was trained on older data). Check: do the eval originals occur in `lora_train_data.jsonl`? If yes: exclude
them in `preprocess.py` (tested), retrain, re-evaluate, and retrain TG-4B on the same cleaned data as the control.
