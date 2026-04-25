"""
EXP-4 — Hard-constraint ablation.

Disables each hard constraint in `XGBoostMerger._build_distance_matrix` one at
a time (and all together) to quantify how much each one contributes to HOTA.

Temporal-overlap is mandatory (physical impossibility) and stays on.

Disabling mechanism (no merger code change needed):
    - jersey conflict: set jersey_entropy_threshold = 0.0 so
      `both_conf = (e1 < thr and e2 < thr)` is always False.
    - team   conflict: set team_consistency_threshold = 1.1 so
      `both_cons = (c1 >= thr and c2 >= thr)` is always False (consistencies
      are in [0, 1]).

Run:
    python -m experiments.exp04_hard_constraint_ablation
"""

from experiments.idk.runner import (
    ExperimentRunner,
    default_xgboost_factory,
    summarize_results,
)


EXP_ID = "EXP4_constraint_ablation"

JERSEY_ON  = 0.15   # default
JERSEY_OFF = 0.0    # always-False — disables jersey hard constraint
TEAM_ON    = 0.9    # default
TEAM_OFF   = 1.1    # always-False — disables team hard constraint


ABLATIONS = [
    ("all_on",         JERSEY_ON,  TEAM_ON,  "baseline — both hard constraints active"),
    ("no_jersey",      JERSEY_OFF, TEAM_ON,  "disable jersey hard constraint only"),
    ("no_team",        JERSEY_ON,  TEAM_OFF, "disable team hard constraint only"),
    ("no_jersey_team", JERSEY_OFF, TEAM_OFF, "disable both jersey and team hard constraints"),
]


def main(split: str = "valid"):
    runner = ExperimentRunner(exp_id=EXP_ID, split=split)
    runner.prepare()

    results = []
    for name, jersey_thr, team_thr, note in ABLATIONS:
        factory = default_xgboost_factory(
            jersey_entropy_threshold=jersey_thr,
            team_consistency_threshold=team_thr,
        )
        res = runner.run(
            run_name=name,
            merger_factory=factory,
            config={
                "note": note,
                "jersey_entropy_threshold":   jersey_thr,
                "team_consistency_threshold": team_thr,
            },
        )
        results.append(res)

    summarize_results(results)


if __name__ == "__main__":
    main()
