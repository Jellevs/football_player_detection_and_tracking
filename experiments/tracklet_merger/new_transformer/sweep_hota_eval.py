"""
sweep_hota_eval.py

End-to-end HOTA evaluation for the new transformer and ensemble mergers.

Sweeps:
  - Merge threshold (0.2 to 0.8)
  - Linkage method (average, complete, ward)
  - Normalization method (none, rank, zscore)
  - Ensemble blend weight (0.0 to 1.0)

Reuses ExperimentRunner infrastructure from experiments/idk/runner.py.

Usage:
    # Sweep standalone transformer
    python -m experiments.tracklet_merger.new_transformer.sweep_hota_eval --mode transformer

    # Sweep ensemble (transformer + XGBoost blend)
    python -m experiments.tracklet_merger.new_transformer.sweep_hota_eval --mode ensemble

    # Quick evaluation at a single config
    python -m experiments.tracklet_merger.new_transformer.sweep_hota_eval --mode transformer --threshold 0.5

Output:
    experiments/tracklet_merger/new_transformer/sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from itertools import product
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import settings
from experiments.idk.runner import ExperimentRunner, summarize_results

from experiments.tracklet_merger.new_transformer.merger import NewTransformerMerger
from experiments.tracklet_merger.new_transformer.ensemble import EnsembleMerger

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR       = Path(__file__).resolve().parent
OUTPUT_CSV       = SCRIPT_DIR / "sweep_results.csv"
TRANSFORMER_DIR  = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\new_transformer")
XGBOOST_DIR      = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\xgboost_0_neg_ratio")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "mode", "blend_weight", "merge_threshold", "linkage_method",
    "normalization", "HOTA", "DetA", "AssA", "IDF1", "MOTA",
    "split", "timestamp",
]


def append_result(row: dict) -> None:
    write_header = not OUTPUT_CSV.exists()
    with open(OUTPUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# Merger factories
# ---------------------------------------------------------------------------

def transformer_factory(
    merge_threshold: float = 0.5,
    linkage_method: str = "average",
    normalization_method: str = "none",
):
    """Create a zero-arg factory for NewTransformerMerger."""
    model_path = TRANSFORMER_DIR / "best_model.pt"
    meta_path  = TRANSFORMER_DIR / "meta.json"

    def _make():
        return NewTransformerMerger(
            model_path=model_path,
            meta_path=meta_path,
            merge_threshold=merge_threshold,
            linkage_method=linkage_method,
            normalization_method=normalization_method,
        )
    return _make


def ensemble_factory(
    blend_weight: float = 0.5,
    merge_threshold: float = 0.5,
    linkage_method: str = "average",
    normalization_method: str = "none",
):
    """Create a zero-arg factory for EnsembleMerger."""
    tf_model_path = TRANSFORMER_DIR / "best_model.pt"
    tf_meta_path  = TRANSFORMER_DIR / "meta.json"
    xgb_model_path = XGBOOST_DIR / "xgboost_merger.json"
    xgb_meta_path  = XGBOOST_DIR / "xgboost_merger_meta.json"

    def _make():
        return EnsembleMerger(
            transformer_model_path=tf_model_path,
            transformer_meta_path=tf_meta_path,
            xgboost_model_path=xgb_model_path,
            xgboost_meta_path=xgb_meta_path,
            blend_weight=blend_weight,
            merge_threshold=merge_threshold,
            linkage_method=linkage_method,
            normalization_method=normalization_method,
        )
    return _make


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(
    mode: str,
    eval_split: str = "valid",
    thresholds: List[float] = None,
    linkage_methods: List[str] = None,
    normalizations: List[str] = None,
    blend_weights: List[float] = None,
):
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    if linkage_methods is None:
        linkage_methods = ["average", "complete"]
    if normalizations is None:
        normalizations = ["none", "rank"]
    if blend_weights is None:
        blend_weights = [0.0, 0.3, 0.5, 0.7, 1.0]

    # Build experiment grid
    if mode == "transformer":
        configs = list(product(thresholds, linkage_methods, normalizations))
        n_total = len(configs)
        print(f"\n{'='*60}")
        print(f"  New Transformer HOTA Sweep")
        print(f"  Split: {eval_split}")
        print(f"  Configs: {n_total} ({len(thresholds)} thresholds x "
              f"{len(linkage_methods)} linkages x {len(normalizations)} norms)")
        print(f"{'='*60}\n")
    elif mode == "ensemble":
        configs = list(product(blend_weights, thresholds, linkage_methods, normalizations))
        n_total = len(configs)
        print(f"\n{'='*60}")
        print(f"  Ensemble (Transformer + XGBoost) HOTA Sweep")
        print(f"  Split: {eval_split}")
        print(f"  Configs: {n_total}")
        print(f"{'='*60}\n")
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Prepare runner
    runner = ExperimentRunner(
        exp_id=f"new_tf_{mode}",
        split=eval_split,
    )
    runner.prepare()

    results = []
    for i, cfg in enumerate(configs, 1):
        if mode == "transformer":
            threshold, link, norm = cfg
            blend = None
            run_name = f"tf_t{threshold}_l{link}_n{norm}"
            factory = transformer_factory(
                merge_threshold=threshold,
                linkage_method=link,
                normalization_method=norm,
            )
        else:
            blend, threshold, link, norm = cfg
            run_name = f"ens_b{blend}_t{threshold}_l{link}_n{norm}"
            factory = ensemble_factory(
                blend_weight=blend,
                merge_threshold=threshold,
                linkage_method=link,
                normalization_method=norm,
            )

        print(f"\n--- [{i}/{n_total}] {run_name} ---")
        t0 = time.time()

        result = runner.run(
            run_name=run_name,
            merger_factory=factory,
            config={
                "mode": mode,
                "blend_weight": blend,
                "merge_threshold": threshold,
                "linkage_method": link,
                "normalization": norm,
            },
        )
        dt = time.time() - t0

        append_result({
            "mode": mode,
            "blend_weight": blend if blend is not None else "",
            "merge_threshold": threshold,
            "linkage_method": link,
            "normalization": norm,
            "HOTA": result.metrics.get("HOTA", ""),
            "DetA": result.metrics.get("DetA", ""),
            "AssA": result.metrics.get("AssA", ""),
            "IDF1": result.metrics.get("IDF1", ""),
            "MOTA": result.metrics.get("MOTA", ""),
            "split": eval_split,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
        results.append(result)
        print(f"  HOTA={result.metrics.get('HOTA', '?'):.3f}  "
              f"AssA={result.metrics.get('AssA', '?'):.3f}  ({dt:.1f}s)")

    if results:
        summarize_results(results, sort_by="HOTA")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="HOTA sweep for the new transformer / ensemble merger"
    )
    parser.add_argument(
        "--mode", type=str, default="transformer",
        choices=["transformer", "ensemble"],
        help="Which merger to sweep (default: transformer)"
    )
    parser.add_argument(
        "--split", type=str, default="valid",
        choices=["train", "valid", "test"],
        help="Dataset split (default: valid)"
    )
    parser.add_argument(
        "--threshold", type=float, nargs="+", default=None,
        help="Merge threshold(s) to sweep (default: 0.3 0.4 0.5 0.6 0.7)"
    )
    parser.add_argument(
        "--linkage", type=str, nargs="+", default=None,
        help="Linkage method(s) to sweep (default: average complete)"
    )
    parser.add_argument(
        "--norm", type=str, nargs="+", default=None,
        help="Normalization method(s) (default: none rank)"
    )
    parser.add_argument(
        "--blend", type=float, nargs="+", default=None,
        help="Blend weight(s) for ensemble (default: 0.0 0.3 0.5 0.7 1.0)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_sweep(
        mode=args.mode,
        eval_split=args.split,
        thresholds=args.threshold,
        linkage_methods=args.linkage,
        normalizations=args.norm,
        blend_weights=args.blend,
    )


if __name__ == "__main__":
    main()
