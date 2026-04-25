"""
EXP-3 — Hard-constraint threshold sweep (inference only).

Sweeps each of the three hard-constraint thresholds independently, holding the
other two at their defaults:

    jersey_entropy_threshold        (default 0.15)
    team_consistency_threshold      (default 0.9)
    team_confidence_threshold       (default 0.6)

All sweeps run on the VALIDATION split. After the per-axis sweeps, the script
optionally combines the top value of each axis (`run_joint_best=True`) to
check whether gains are independent.

Run:
    python -m experiments.exp03_hard_constraint_thresholds
"""

from experiments.idk.runner import (
    ExperimentRunner,
    default_xgboost_factory,
    summarize_results,
)


EXP_ID = "EXP3_hard_constraint_thresholds"

DEFAULT_JERSEY_ENTROPY     = 0.15
DEFAULT_TEAM_CONSISTENCY   = 0.9
DEFAULT_TEAM_CONFIDENCE    = 0.6

JERSEY_ENTROPY_GRID    = (0.10, 0.15, 0.20, 0.30)
TEAM_CONSISTENCY_GRID  = (0.7, 0.8, 0.9, 0.95)
TEAM_CONFIDENCE_GRID   = (0.4, 0.5, 0.6, 0.7)


def main(split: str = "valid", run_joint_best: bool = True):
    runner = ExperimentRunner(exp_id=EXP_ID, split=split)
    runner.prepare()

    results = []

    # --- axis 1: jersey_entropy_threshold ---
    for v in JERSEY_ENTROPY_GRID:
        run_name = f"jersey_entropy_{v:.2f}"
        factory  = default_xgboost_factory(jersey_entropy_threshold=v)
        res = runner.run(run_name, factory, config={"jersey_entropy_threshold": v})
        results.append(res)

    # --- axis 2: team_consistency_threshold ---
    for v in TEAM_CONSISTENCY_GRID:
        run_name = f"team_consistency_{v:.2f}"
        factory  = default_xgboost_factory(team_consistency_threshold=v)
        res = runner.run(run_name, factory, config={"team_consistency_threshold": v})
        results.append(res)

    # --- axis 3: team_confidence_threshold ---
    for v in TEAM_CONFIDENCE_GRID:
        run_name = f"team_confidence_{v:.2f}"
        factory  = default_xgboost_factory(team_confidence_threshold=v)
        res = runner.run(run_name, factory, config={"team_confidence_threshold": v})
        results.append(res)

    # --- optional joint run at per-axis best values ---
    if run_joint_best:
        best = {"jersey_entropy_threshold": DEFAULT_JERSEY_ENTROPY,
                "team_consistency_threshold": DEFAULT_TEAM_CONSISTENCY,
                "team_confidence_threshold": DEFAULT_TEAM_CONFIDENCE}
        for axis, grid, key in [
            ("jersey_entropy_threshold",    JERSEY_ENTROPY_GRID,   "jersey_entropy_"),
            ("team_consistency_threshold",  TEAM_CONSISTENCY_GRID, "team_consistency_"),
            ("team_confidence_threshold",   TEAM_CONFIDENCE_GRID,  "team_confidence_"),
        ]:
            axis_results = [r for r in results if r.run_name.startswith(key)]
            if axis_results:
                winner = max(axis_results, key=lambda r: r.metrics.get("HOTA", -1))
                best[axis] = winner.config[axis]

        run_name = "joint_best"
        factory  = default_xgboost_factory(**best)
        res = runner.run(run_name, factory, config=best)
        results.append(res)

    summarize_results(results)


if __name__ == "__main__":
    main()
