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
