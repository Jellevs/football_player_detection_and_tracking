# XGBoost Merger Experiments

Tier-1 inference-side experiments from `give-me-all-the-adaptive-coral.md`.
All five scripts re-use the cached attribute tracklets
(`output/cache/cache_attributes_SNPT-*.pkl`) so no detection/tracking/attribute
prediction is re-run. Only the splitter + merger are executed per run.

## Split policy — IMPORTANT

- **All sweeps/tuning run on the `valid` split** (the default).
- **Only the single winner of each experiment** is re-evaluated on `test`.

This avoids test-set overfitting. The runner sets `settings.EVAL_SPLIT` and
`settings.DATA_ROOT` at runtime, so `settings.py` can stay unchanged.

To re-evaluate a winner on test:

```python
from experiments.exp01_merge_threshold_sweep import main
main(split="test", thresholds=[0.45])   # just the winning threshold
```

## Running

Each script is runnable as a module:

```bash
python -m experiments.exp01_merge_threshold_sweep
python -m experiments.exp02_linkage_method
python -m experiments.exp03_hard_constraint_thresholds
python -m experiments.exp04_hard_constraint_ablation
python -m experiments.exp05_ensemble_inference
```

Recommended order: EXP-1 → EXP-2 (uses EXP-1 winner) → EXP-3 → EXP-4 → EXP-5.

## Output

Every run writes:
- MOT files under `evaluation/SNPT/<EXP_ID>__<run_name>__<split>/data/`.
- HOTA summary under the same folder (`pedestrian_summary.txt`).
- One row in `experiments/results.csv` with HOTA / DetA / AssA / IDF1 / MOTA.

Re-running a script will reuse any run whose summary file already exists, so
interrupted sweeps can resume without recomputing.

## Experiments

| ID | Script | What it sweeps |
|----|--------|----------------|
| EXP-1 | `exp01_merge_threshold_sweep.py` | `merge_threshold ∈ {0.2 … 0.7}` with defaults everywhere else |
| EXP-2 | `exp02_linkage_method.py` | `linkage ∈ {average, single, complete, ward}` at EXP-1 winner's threshold |
| EXP-3 | `exp03_hard_constraint_thresholds.py` | `jersey_entropy`, `team_consistency`, `team_confidence` independently, then joint best |
| EXP-4 | `exp04_hard_constraint_ablation.py` | disable each hard constraint (all_on / no_jersey / no_team / no_both) |
| EXP-5 | `exp05_ensemble_inference.py` | baseline vs tuned single seed vs 5-seed ensemble |

## Adding a new experiment

1. Create `expNN_<name>.py`.
2. Instantiate `ExperimentRunner(exp_id="EXPNN_<name>", split="valid")`.
3. Call `runner.prepare()` once, then `runner.run(...)` for each config.
4. At the end, pass collected `RunResult`s to `summarize_results(...)`.

The merger mutates tracklets in place, so the runner deep-copies the cached
post-splitter output before every `run(...)` call.
