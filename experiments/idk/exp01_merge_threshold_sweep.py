"""
EXP-1 — Merge-threshold sweep (inference only).

Sweep the hierarchical-clustering cut threshold on the VALIDATION split using
the current default XGBoost model. No retraining.

Run:
    python -m experiments.exp01_merge_threshold_sweep

After the sweep, pick the best HOTA and re-run just that one on the test
split by calling main(split="test", thresholds=[best]).
"""

from experiments.idk.runner import (
    ExperimentRunner,
    default_xgboost_factory,
    summarize_results,
)


EXP_ID = "EXP1_merge_threshold"


def main(
    split: str = "valid",
    thresholds=(0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7),
):
    runner = ExperimentRunner(exp_id=EXP_ID, split=split)
    runner.prepare()

    results = []
    for t in thresholds:
        run_name = f"t{t:.2f}"
        factory  = default_xgboost_factory(merge_threshold=t)
        res      = runner.run(run_name, factory, config={"merge_threshold": t})
        results.append(res)

    summarize_results(results)


if __name__ == "__main__":
    main()
