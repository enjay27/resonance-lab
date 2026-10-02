# A Jupyter notebook for the whole lifecycle of a parameter test — idea, 2026-10-02 (maintainer)

**Status: BUILT 2026-10-02 (PR from `claude/great-babbage-i0rage`): `notebooks/parameter_test.ipynb` + `parameter_test.py` (tested). Not yet run on the GPU machine: see `active-issues/unverified-on-gpu.md`.** Why: a parameter test is eight manual steps in PowerShell and a table to read; the sweeps showed it is the thing repeated most.

## The lifecycle one notebook run should cover ("one parameter test")
1. **Set the parameters** in one cell: model profile (default `hy-mt2-1.8b`), `fast`, `lr`, `epochs`, eval prompt(s) (`training`, `chat-template`).
2. **Data (only when it changed):** Fetch Data -> Validate -> Preprocessing -> Update Dataset (the pipeline's first four stages; `run_pipeline.py --only/--from`), showing the drop report and the data revision.
3. **Train:** `train.py --fast --lr L --epochs E` (live: stream its output; the monitor `watch_training.py` is a terminal UI, so show the loss/eval-loss curve from `trainer_log.jsonl` instead). Ctrl+C must stay KILLED.
4. **Merge:** `merge.py --fast`. 5. **Eval:** `eval.py --fast --prompt <p>` for each prompt (an evaluation per prompt: the metric keys of two prompts in one run collide, see `mlflow_compare.py`'s note).
6. **Record + compare:** the run is already in MLflow (params, `train.overrides` tag, curves, `eval.*`); show `compare_runs.build_rows(...)` for the profile, sorted by eval loss / chrF, and plot the eval-loss curves of the runs in the sweep
   (read from MLflow, or from `outputs/<profile>_lora/<run id>/trainer_log.jsonl` offline).
7. **A sweep loop:** a list of parameter sets (e.g. `[{"lr": 4e-4}, {"lr": 8e-4}, {"lr": 2e-4, "epochs": 6}]`) run one after another, each a full lifecycle, then the comparison table and a markdown block for the memory notes.
8. **Decide:** a cell that prints "best so far" and the exact line to change in the profile (and the commit to make) — the change itself stays a normal PR (tests first).
Later parameters (not overridable yet — only `learning_rate` and `num_train_epochs` are, `lf_tools.training_overrides`): LoRA rank/alpha, warmup, batch/packing, `cutoff_len`, and the **data** parameters (`preprocess.py --reverse`, `--prompt`)
which need steps 2 again. A generic allow-listed `--set key=value` would be the next override PR.

## Design constraints (this repo's rules decide them)
- **Thin notebook, tested logic:** the stages are the existing scripts run as subprocesses with the venv's python (`sys.executable`): no reimplementation of train/merge/eval. Anything pure (building the command lists from a parameter
  set, reading `trainer_log.jsonl` into curves, formatting the decision text, the sweep loop's bookkeeping) goes in a **data-part module** (e.g. `parameter_test.py`) with unit tests (CLAUDE.md: new pure logic goes in the data part).
- **No data in git:** a notebook's saved outputs would hold eval lines (player chat). Commit notebooks **with outputs stripped**, and guard it with a test (every `notebooks/*.ipynb` has empty `outputs` and no `execution_count`),
  or use `nbstripout`/jupytext percent format; decide with the maintainer. `.gitignore` the executed copies.
- **Runs on the GPU machine only** (Windows, the project `.venv`): the cloud session cannot run it. State `NOT VERIFIED: model part -- no GPU in this session`; the data-part module + a notebook-structure test are what the gate checks.
- **Environment:** `ipykernel` (+ `nbformat`, `matplotlib` or the plotting lib chosen) in an optional `requirements-notebook.txt`, not in the training requirements. Check the kernel is the project venv (`sys.executable` ends in `.venv\Scripts\python.exe`):
  a venv built by an older Python than the base install broke `python -c` earlier with `linecache._register_code ... 'str' has no attribute 'co_consts'` — fixed with `python -m venv --upgrade .venv` (base Python path, venv deactivated).
- Failure handling follows the stage rule: a failing stage stops the lifecycle with its tail log; nothing is retried silently.

## Answers of the maintainer (2026-10-02) and what was built
1. PyCharm notebook support (live output = line-by-line streaming into the cell; plots = inline matplotlib). 2. `.ipynb`, outputs stripped, guarded by `tests/test_notebooks.py`; executed copies `*.executed.ipynb` are gitignored.
3. One notebook. 4. The sweep runs only the `training` prompt unless a parameter set (or `PROMPTS`) lists `chat-template` too.
Built: `parameter_test.py` (`ParamSet`, `lifecycle`, `run_command` with process-tree stop on interrupt, `read_curves`/`run_curves`, `run_lifecycle`/`run_sweep`/`best_result`, `decision_text`, `results_markdown`, `kernel_warning`);
`notebooks/parameter_test.ipynb` (parameters, data, train, merge, eval, curves, MLflow compare, sweep, decide); `requirements-notebook.txt` (ipykernel, matplotlib).
Left out / next: an interrupted cell on Windows is a hard `taskkill` (the MLflow run is probably left RUNNING, not KILLED; unverified); the compare cell shows MLflow's chrF/term table but the sweep's local table has eval loss only;
only `lr`/`epochs` are sweepable (the generic `--set key=value` override is the next PR).

## Open questions that were asked (kept for the record)
1. Where does it run: PyCharm's notebook support, or `jupyter lab` in a terminal? (affects the live-output cell and the plotting choice)
2. Stored how: `.ipynb` with stripped outputs (guard test), or jupytext percent-format `.py` (clean diffs, opened as a notebook)?
3. One notebook for the whole lifecycle, or a small set (`01_data`, `02_parameter_test`, `03_compare`)?
4. Should the sweep loop also run both eval prompts for every parameter set (doubles the eval time, small), or only `training` unless asked?

## First sweep (2026-10-02, the baseline the notebook must reproduce) — `hy-mt2-1.8b-fast`, 3 epochs, eval prompt `training`, data `b37c268f`, 87 steps, eval every 10 steps
| lr | best eval loss (@step) | train loss (run mean) | chrF | term acc | JP leak |
|---|---|---|---|---|---|
| 2e-4 | 0.8317 (80) | 1.2382 | 58.1 | 33.3% | 0.0% |
| 1e-4 | 0.9859 (80) | 1.5467 | 49.8 | 19.0% | 0.0% |
| 5e-5 | 1.2144 (80) | 1.9582 | 36.6 | 14.3% | 0.0% |
Reading: higher lr is better on every metric and 2e-4 is the edge of the sweep; the best eval is at step 80 = the last evaluation, so eval loss was still falling when the runs ended (underfit at 3 epochs). `train loss` is the run mean (early
high steps included), not a final loss: do not read a train/eval gap from it. Training is reproducible (three 2e-4 runs: identical train loss 1.2382); the noise is the 51-line eval set. The fast profile has effective batch 32 + packing =
~87 updates, far fewer than the full profile, so it may want a higher lr or more epochs. **Next sweep:** `--lr 4e-4`, `--lr 8e-4` (watch `grad_norm`/loss spikes), `--lr 2e-4 --epochs 6` (where does eval loss turn up?). Then confirm the best on the full
`hy-mt2-1.8b` (~25 min) with both eval prompts, and set the winner in the base + `-fast` profiles (a PR with tests). The profile default is 2e-4 already.
