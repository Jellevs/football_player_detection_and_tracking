"""
Sweep merge thresholds for transformer connectors using the full pipeline.

Runs: pairwise scoring → distance matrix → HAC clustering → HOTA evaluation
for each threshold, and reports the best threshold per model.

Toggle which models to sweep below. No argparse needed.

Usage:
    python -m experiments.tracklet_merger.transformers.sweep_hota_threshold
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from experiments.idk.runner import ExperimentRunner


# -----------------------------------------------------------------------
# Toggle: set to True to include a model in the sweep
# -----------------------------------------------------------------------

SWEEP_SIAMESE_CLS      = True
SWEEP_CROSS_ATTENTION   = False
SWEEP_HYBRID            = True
SWEEP_PAIRWISE_MLP      = True

# -----------------------------------------------------------------------
# Sweep settings
# -----------------------------------------------------------------------

EVAL_SPLIT = "valid"                          # "valid" for tuning, "test" for final
THRESHOLDS = [round(t, 2) for t in
              [x * 0.05 for x in range(6, 19)]]  # 0.30 to 0.80 step 0.05
LINKAGE    = "average"
SKIP_EXISTING = True                          # reuse results already on disk

# -----------------------------------------------------------------------
# Model paths (best run per architecture, real+synthetic data)
# -----------------------------------------------------------------------

_TRANSFORMERS = project_root / "experiments" / "tracklet_merger" / "transformers"

MODEL_CONFIGS = {
    "siamese_cls": {
        "model_path": _TRANSFORMERS / "siamese_cls" / "output" / "siamese_cls_simple_synth" / "best_model.pt",
        "meta_path":  _TRANSFORMERS / "siamese_cls" / "output" / "siamese_cls_simple_synth" / "transformer_merger_meta.json",
    },
    "cross_attention": {
        "model_path": _TRANSFORMERS / "cross_attention" / "output" / "cross_attention_simple_synth" / "best_model.pt",
        "meta_path":  _TRANSFORMERS / "cross_attention" / "output" / "cross_attention_simple_synth" / "transformer_merger_meta.json",
    },
    "hybrid": {
        "model_path": _TRANSFORMERS / "hybrid" / "output" / "hybrid_simple_synth" / "best_model.pt",
        "meta_path":  _TRANSFORMERS / "hybrid" / "output" / "hybrid_simple_synth" / "transformer_merger_meta.json",
    },
    "pairwise_mlp": {
        "model_path": _TRANSFORMERS / "pairwise_mlp" / "output" / "pairwise_mlp_xgb_matched" / "best_model.pt",
    },
}


# -----------------------------------------------------------------------
# Merger factories
# -----------------------------------------------------------------------

def siamese_cls_factory(threshold: float):
    from tracklets.transformer_merger import TransformerMerger
    cfg = MODEL_CONFIGS["siamese_cls"]
    def _make():
        return TransformerMerger(
            model_path=cfg["model_path"],
            meta_path=cfg["meta_path"],
            merge_threshold=threshold,
            linkage_method=LINKAGE,
        )
    return _make


def cross_attention_factory(threshold: float):
    from tracklets.cross_attention_merger import CrossAttentionMerger
    cfg = MODEL_CONFIGS["cross_attention"]
    def _make():
        return CrossAttentionMerger(
            model_path=cfg["model_path"],
            meta_path=cfg["meta_path"],
            merge_threshold=threshold,
            linkage_method=LINKAGE,
        )
    return _make


def hybrid_factory(threshold: float):
    from tracklets.hybrid_merger import HybridMerger
    cfg = MODEL_CONFIGS["hybrid"]
    def _make():
        return HybridMerger(
            model_path=cfg["model_path"],
            meta_path=cfg["meta_path"],
            merge_threshold=threshold,
            linkage_method=LINKAGE,
        )
    return _make


def pairwise_mlp_factory(threshold: float):
    from tracklets.pairwise_mlp_merger import PairwiseMLPMerger
    cfg = MODEL_CONFIGS["pairwise_mlp"]
    def _make():
        return PairwiseMLPMerger(
            model_path=cfg["model_path"],
            merge_threshold=threshold,
            linkage_method=LINKAGE,
        )
    return _make


# -----------------------------------------------------------------------
# Sweep one model
# -----------------------------------------------------------------------

def sweep_model(model_name: str, factory_fn, runner: ExperimentRunner):
    """Sweep all thresholds for a single model and return results."""
    print(f"\n{'='*68}")
    print(f"  Sweeping: {model_name}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"{'='*68}")

    results = []
    for t in THRESHOLDS:
        run_name = f"{model_name}_t{t:.2f}"
        factory = factory_fn(t)
        config = {"model": model_name, "merge_threshold": t, "linkage": LINKAGE}

        result = runner.run(
            run_name=run_name,
            merger_factory=factory,
            config=config,
            reuse_if_exists=SKIP_EXISTING,
        )
        results.append((t, result.metrics))

    # Print summary
    print(f"\n  {'Threshold':>10}  {'HOTA':>6}  {'DetA':>6}  {'AssA':>6}  {'IDF1':>6}  {'MOTA':>6}")
    print(f"  {'-'*52}")

    best_hota = 0.0
    best_t = None
    for t, m in results:
        hota = m.get("HOTA", 0.0)
        marker = ""
        if hota > best_hota:
            best_hota = hota
            best_t = t
            marker = " *"
        print(f"  {t:10.2f}  {m.get('HOTA', 0):6.2f}  {m.get('DetA', 0):6.2f}  "
              f"{m.get('AssA', 0):6.2f}  {m.get('IDF1', 0):6.2f}  {m.get('MOTA', 0):6.2f}{marker}")

    print(f"\n  Best: threshold={best_t:.2f}, HOTA={best_hota:.2f}")
    return results, best_t, best_hota


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    models_to_sweep = []
    if SWEEP_SIAMESE_CLS:
        models_to_sweep.append(("siamese_cls", siamese_cls_factory))
    if SWEEP_CROSS_ATTENTION:
        models_to_sweep.append(("cross_attention", cross_attention_factory))
    if SWEEP_HYBRID:
        models_to_sweep.append(("hybrid", hybrid_factory))
    if SWEEP_PAIRWISE_MLP:
        models_to_sweep.append(("pairwise_mlp", pairwise_mlp_factory))

    if not models_to_sweep:
        print("No models selected. Set at least one SWEEP_* flag to True.")
        return

    print(f"Sweep config: split={EVAL_SPLIT}, linkage={LINKAGE}, "
          f"thresholds={THRESHOLDS[0]:.2f}..{THRESHOLDS[-1]:.2f} "
          f"(step {THRESHOLDS[1] - THRESHOLDS[0]:.2f})")
    print(f"Models: {[name for name, _ in models_to_sweep]}")

    runner = ExperimentRunner(exp_id="transformer_hota_sweep", split=EVAL_SPLIT)
    runner.prepare()

    all_results = {}
    for model_name, factory_fn in models_to_sweep:
        results, best_t, best_hota = sweep_model(model_name, factory_fn, runner)
        all_results[model_name] = {
            "results": results,
            "best_threshold": best_t,
            "best_hota": best_hota,
        }

    # Final summary
    print(f"\n{'='*68}")
    print(f"  SWEEP SUMMARY ({EVAL_SPLIT})")
    print(f"{'='*68}")
    print(f"  {'Model':<25}  {'Best threshold':>14}  {'HOTA':>6}")
    print(f"  {'-'*50}")
    for name, info in all_results.items():
        print(f"  {name:<25}  {info['best_threshold']:>14.2f}  {info['best_hota']:>6.2f}")


if __name__ == "__main__":
    main()
