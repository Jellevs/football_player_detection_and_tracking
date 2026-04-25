"""
EXP-2 — Linkage method comparison (inference only).

Compare the four hierarchical-clustering linkage methods at the EXP-1 winner's
threshold. Defaults to merge_threshold=0.5 (the current default) — after
EXP-1 finishes, re-run with `best_threshold=<winner>`.

Run:
    python -m experiments.exp02_linkage_method
"""

from experiments.idk.runner import (
    ExperimentRunner,
    default_xgboost_factory,
    summarize_results,
)


EXP_ID = "EXP2_linkage"


def main(
    split: str = "valid",
    linkage_methods=("average", "single", "complete", "ward"),
    best_threshold: float = 0.5,
):
    runner = ExperimentRunner(exp_id=EXP_ID, split=split)
    runner.prepare()

    results = []
    for linkage in linkage_methods:
        run_name = f"{linkage}_t{best_threshold:.2f}"
        factory  = default_xgboost_factory(
            merge_threshold=best_threshold,
            linkage_method=linkage,
        )
        res = runner.run(
            run_name,
            factory,
            config={"linkage": linkage, "merge_threshold": best_threshold},
        )
        results.append(res)

    summarize_results(results)


if __name__ == "__main__":
    main()
