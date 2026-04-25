"""
EXP-5 — Ensemble inference.

Compares three XGBoost inference configurations on the VALIDATION split:
    1. baseline        : weights/xgboost_0_neg_ratio/xgboost_merger.json (current main.py default)
    2. tuned_seed0     : weights/xgboost_tuned/xgboost_merger_seed0.json (single tuned seed)
    3. ensemble_5seed  : average of seeds 0..4 from weights/xgboost_tuned

Requires:
    - The XGBoost tuned ensemble must share the same feature schema as the
      baseline. `xgboost_merger_meta.json` in the tuned folder should be used
      for ensemble runs; the single-seed tuned run uses its own meta file.

Run:
    python -m experiments.exp05_ensemble_inference
"""

from experiments.idk.runner import (
    ExperimentRunner,
    DEFAULT_MODEL_DIR,
    TUNED_MODEL_DIR,
    summarize_results,
)
from tracklets.xgboost_merger import XGBoostMerger


EXP_ID = "EXP5_ensemble"


def _factory(model_paths, meta_path):
    def _make():
        return XGBoostMerger(
            model_path=model_paths,
            meta_path=meta_path,
            merge_threshold=0.5,
            linkage_method="average",
            jersey_entropy_threshold=0.15,
            team_consistency_threshold=0.9,
            team_confidence_threshold=0.6,
        )
    return _make


def main(split: str = "valid", n_seeds: int = 5):
    runner = ExperimentRunner(exp_id=EXP_ID, split=split)
    runner.prepare()

    configs = []

    # 1. baseline (current main.py default)
    configs.append((
        "baseline_0_neg_ratio",
        [DEFAULT_MODEL_DIR / "xgboost_merger.json"],
        DEFAULT_MODEL_DIR / "xgboost_merger_meta.json",
    ))

    # 2. tuned single seed
    configs.append((
        "tuned_seed0",
        [TUNED_MODEL_DIR / "xgboost_merger_seed0.json"],
        TUNED_MODEL_DIR / "xgboost_merger_meta.json",
    ))

    # 3. tuned n-seed ensemble
    ensemble_paths = [
        TUNED_MODEL_DIR / f"xgboost_merger_seed{i}.json"
        for i in range(n_seeds)
    ]
    configs.append((
        f"tuned_ensemble_{n_seeds}seed",
        ensemble_paths,
        TUNED_MODEL_DIR / "xgboost_merger_meta.json",
    ))

    results = []
    for run_name, model_paths, meta_path in configs:
        missing = [p for p in model_paths if not p.exists()]
        if missing or not meta_path.exists():
            print(f"[{EXP_ID}:{run_name}] SKIP — missing: {missing or meta_path}")
            continue
        factory = _factory(model_paths, meta_path)
        res = runner.run(
            run_name,
            factory,
            config={
                "model_paths": [str(p) for p in model_paths],
                "meta_path":   str(meta_path),
            },
        )
        results.append(res)

    summarize_results(results)


if __name__ == "__main__":
    main()
