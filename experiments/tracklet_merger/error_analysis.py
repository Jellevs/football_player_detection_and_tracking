"""
Error analysis: compare merger error types (false merges vs missed merges).

Runs each merger on cached split tracklets and compares against the oracle
to understand WHY transformers achieve similar AUC but lower HOTA than XGBoost.

For each merger, we compute:
  - False merges: pairs merged by the model but NOT by the oracle (different GT identity)
  - Missed merges: pairs merged by the oracle but NOT by the model (same GT identity)
  - Frame-weighted versions: how many frames are affected by each error type

Usage:
    Step 1: Run main.py once with CACHE_SPLIT_TRACKLETS=True to save split tracklets
    Step 2: python experiments/tracklet_merger/error_analysis.py
"""

import pickle
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set

# Add project root to path FIRST, and remove script dir to avoid
# shadowing HuggingFace 'transformers' with local transformers/ folder
project_root = Path(__file__).resolve().parents[2]
script_dir = str(Path(__file__).resolve().parent)
sys.path.insert(0, str(project_root))
if script_dir in sys.path:
    sys.path.remove(script_dir)

import settings
from tracklets.tracklet import Tracklet

# Mergers
from tracklets.xgboost_merger import XGBoostMerger
from tracklets.cross_attention_merger import CrossAttentionMerger
from tracklets.hybrid_merger import HybridMerger
from tracklets.transformer_merger import TransformerMerger
from tracklets.oracle_merger import OracleMerger


# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch")

CACHE_DIR = PROJECT_ROOT / "output" / "cache_split_tracklets"

XGBOOST_MODEL = PROJECT_ROOT / "experiments" / "tracklet_merger" / "xgboost" / "sweep_results" / "neg3_minlen0_purity0.6_splitTrue" / "xgboost_merger.json"
XGBOOST_META  = PROJECT_ROOT / "experiments" / "tracklet_merger" / "xgboost" / "sweep_results" / "neg3_minlen0_purity0.6_splitTrue" / "xgboost_merger_meta.json"

CROSS_ATTN_MODEL = PROJECT_ROOT / "experiments" / "tracklet_merger" / "transformers" / "cross_attention" / "output" / "cross_attention_real_synth" / "best_model.pt"
CROSS_ATTN_META  = PROJECT_ROOT / "experiments" / "tracklet_merger" / "transformers" / "cross_attention" / "output" / "cross_attention_real_synth" / "transformer_merger_meta.json"

SIAMESE_CLS_MODEL = PROJECT_ROOT / "experiments" / "tracklet_merger" / "transformers" / "siamese_cls" / "output" / "siamese_cls_real_synth" / "best_model.pt"
SIAMESE_CLS_META  = PROJECT_ROOT / "experiments" / "tracklet_merger" / "transformers" / "siamese_cls" / "output" / "siamese_cls_real_synth" / "transformer_merger_meta.json"

HYBRID_MODEL = PROJECT_ROOT / "experiments" / "tracklet_merger" / "transformers" / "hybrid" / "output" / "hybrid_real_synth" / "best_model.pt"
HYBRID_META  = PROJECT_ROOT / "experiments" / "tracklet_merger" / "transformers" / "hybrid" / "output" / "hybrid_real_synth" / "transformer_merger_meta.json"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "tracklet_merger" / "error_analysis_output"


# ── Helpers ────────────────────────────────────────────────────────────

def get_gt_identity(tracklet: Tracklet) -> int:
    """Get the majority GT identity of a tracklet from its gt_attributes."""
    gt_ids = tracklet.gt_attributes.get('track_ids', [])
    if not gt_ids:
        return -1
    # Filter out NaN/None
    valid = [int(g) for g in gt_ids if g is not None and not (isinstance(g, float) and np.isnan(g))]
    if not valid:
        return -1
    return Counter(valid).most_common(1)[0][0]


def build_frame_to_tracklet_map(tracklets: Dict) -> Dict[int, int]:
    """Map each frame to its tracklet ID."""
    frame_map = {}
    for tid, t in tracklets.items():
        for f in t.frames:
            frame_map[f] = tid
    return frame_map


def extract_merge_pairs(pre_merge: Dict, post_merge: Dict) -> Tuple[Set[Tuple[int,int]], Dict[int, int]]:
    """
    Determine which pre-merge tracklet pairs were merged together.

    For each post-merge tracklet, find which pre-merge tracklets
    contributed frames to it. Any two pre-merge tracklets that ended up in
    the same post-merge tracklet were "merged".

    Returns:
        merged_pairs: set of (tid_a, tid_b) tuples (sorted) that were merged
        pre_to_post: mapping from pre-merge tracklet ID to post-merge tracklet ID
    """
    # Build frame -> pre-merge tracklet ID map
    pre_frame_map = build_frame_to_tracklet_map(pre_merge)

    # For each post-merge tracklet, find which pre-merge tracklets contributed
    pre_to_post = {}
    post_to_pre = defaultdict(set)

    for post_tid, post_t in post_merge.items():
        for frame in post_t.frames:
            if frame in pre_frame_map:
                pre_tid = pre_frame_map[frame]
                pre_to_post[pre_tid] = post_tid
                post_to_pre[post_tid].add(pre_tid)

    # Extract merged pairs
    merged_pairs = set()
    for post_tid, pre_tids in post_to_pre.items():
        pre_list = sorted(pre_tids)
        for i in range(len(pre_list)):
            for j in range(i + 1, len(pre_list)):
                merged_pairs.add((pre_list[i], pre_list[j]))

    return merged_pairs, pre_to_post


def deep_copy_tracklets(tracklets: Dict) -> Dict:
    """Deep copy tracklets so each merger gets its own copy to modify."""
    result = {}
    for tid, t in tracklets.items():
        new_t = Tracklet(
            track_id=t.track_id,
            frames=list(t.frames),
            scores=list(t.scores),
            bboxes=list(t.bboxes),
            embeddings=list(t.embeddings) if t.embeddings else [],
            parent_id=t.parent_id,
        )
        new_t.gt_attributes = {k: list(v) for k, v in t.gt_attributes.items()}
        new_t.pred_attributes = {k: list(v) for k, v in t.pred_attributes.items()}
        new_t.final_jersey = t.final_jersey
        new_t.final_team = t.final_team
        result[tid] = new_t
    return result


def analyze_errors(
    split_tracklets: Dict,
    merged_tracklets: Dict,
    oracle_tracklets: Dict,
    model_name: str,
) -> Dict:
    """
    Compare a model's merge decisions against the oracle.

    Returns dict with:
        - false_merges: pairs merged by model but not oracle (wrong merges)
        - missed_merges: pairs merged by oracle but not model (missed opportunities)
        - correct_merges: pairs merged by both
        - correct_non_merges: pairs not merged by either
        - frame-weighted impact of each error type
    """
    # Get merge pairs for model and oracle
    model_pairs, _ = extract_merge_pairs(split_tracklets, merged_tracklets)
    oracle_pairs, _ = extract_merge_pairs(split_tracklets, oracle_tracklets)

    # Classify
    correct_merges = model_pairs & oracle_pairs
    false_merges = model_pairs - oracle_pairs
    missed_merges = oracle_pairs - model_pairs

    # All possible pairs
    all_tids = sorted(split_tracklets.keys())
    all_possible = set()
    for i in range(len(all_tids)):
        for j in range(i + 1, len(all_tids)):
            all_possible.add((all_tids[i], all_tids[j]))
    correct_non_merges = all_possible - model_pairs - oracle_pairs

    # Frame-weighted impact: how many frames are affected by each error
    def pair_frame_weight(tid_a, tid_b):
        return len(split_tracklets[tid_a].frames) + len(split_tracklets[tid_b].frames)

    false_merge_frames = sum(pair_frame_weight(a, b) for a, b in false_merges)
    missed_merge_frames = sum(pair_frame_weight(a, b) for a, b in missed_merges)
    correct_merge_frames = sum(pair_frame_weight(a, b) for a, b in correct_merges)

    # GT identity analysis of false merges
    false_merge_details = []
    for a, b in sorted(false_merges):
        gt_a = get_gt_identity(split_tracklets[a])
        gt_b = get_gt_identity(split_tracklets[b])
        n_frames = pair_frame_weight(a, b)
        false_merge_details.append({
            'pair': (a, b),
            'gt_a': gt_a, 'gt_b': gt_b,
            'frames_affected': n_frames,
        })
    # Sort by frames affected (most damaging first)
    false_merge_details.sort(key=lambda x: -x['frames_affected'])

    results = {
        'model': model_name,
        'n_split_tracklets': len(split_tracklets),
        'n_output_tracklets': len(merged_tracklets),
        'n_oracle_tracklets': len(oracle_tracklets),
        'total_possible_pairs': len(all_possible),
        'oracle_merges': len(oracle_pairs),
        'model_merges': len(model_pairs),
        'correct_merges': len(correct_merges),
        'false_merges': len(false_merges),
        'missed_merges': len(missed_merges),
        'correct_non_merges': len(correct_non_merges),
        'false_merge_frames': false_merge_frames,
        'missed_merge_frames': missed_merge_frames,
        'correct_merge_frames': correct_merge_frames,
        'merge_precision': len(correct_merges) / max(len(model_pairs), 1),
        'merge_recall': len(correct_merges) / max(len(oracle_pairs), 1),
        'false_merge_details': false_merge_details[:20],  # top 20 most damaging
    }

    return results


def print_results(results: Dict):
    """Pretty-print error analysis results."""
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  {results['model']}")
    print(bar)
    print(f"  Split tracklets:      {results['n_split_tracklets']}")
    print(f"  Output tracklets:     {results['n_output_tracklets']} (oracle: {results['n_oracle_tracklets']})")
    print(f"  Total possible pairs: {results['total_possible_pairs']}")
    print()
    print(f"  Oracle merges:        {results['oracle_merges']}")
    print(f"  Model merges:         {results['model_merges']}")
    print()
    print(f"  Correct merges:       {results['correct_merges']}")
    print(f"  FALSE MERGES:         {results['false_merges']}  ({results['false_merge_frames']} frames affected)")
    print(f"  MISSED MERGES:        {results['missed_merges']}  ({results['missed_merge_frames']} frames affected)")
    print()
    print(f"  Merge precision:      {results['merge_precision']:.4f}")
    print(f"  Merge recall:         {results['merge_recall']:.4f}")

    if results['false_merge_details']:
        print(f"\n  Top false merges (most damaging):")
        for i, d in enumerate(results['false_merge_details'][:10]):
            print(f"    {i+1}. Pair {d['pair']}: GT {d['gt_a']} vs {d['gt_b']}, "
                  f"{d['frames_affected']} frames affected")
    print(bar)


def print_comparison_table(all_results: List[Dict]):
    """Print a comparison table across all models."""
    bar = "=" * 90
    print(f"\n{bar}")
    print(f"  MERGER ERROR COMPARISON (aggregated across all sequences)")
    print(bar)
    print(f"  {'Model':<30} {'Merges':>7} {'Correct':>8} {'False':>7} {'Missed':>7} "
          f"{'Prec':>6} {'Rec':>6} {'FalseFr':>8} {'MissFr':>8}")
    print(f"  {'-'*82}")

    for r in all_results:
        print(f"  {r['model']:<30} {r['model_merges']:>7} {r['correct_merges']:>8} "
              f"{r['false_merges']:>7} {r['missed_merges']:>7} "
              f"{r['merge_precision']:>6.3f} {r['merge_recall']:>6.3f} "
              f"{r['false_merge_frames']:>8} {r['missed_merge_frames']:>8}")
    print(bar)


def aggregate_results(per_seq_results: List[Dict], model_name: str) -> Dict:
    """Aggregate per-sequence results into totals."""
    agg = {
        'model': model_name,
        'n_split_tracklets': sum(r['n_split_tracklets'] for r in per_seq_results),
        'n_output_tracklets': sum(r['n_output_tracklets'] for r in per_seq_results),
        'n_oracle_tracklets': sum(r['n_oracle_tracklets'] for r in per_seq_results),
        'total_possible_pairs': sum(r['total_possible_pairs'] for r in per_seq_results),
        'oracle_merges': sum(r['oracle_merges'] for r in per_seq_results),
        'model_merges': sum(r['model_merges'] for r in per_seq_results),
        'correct_merges': sum(r['correct_merges'] for r in per_seq_results),
        'false_merges': sum(r['false_merges'] for r in per_seq_results),
        'missed_merges': sum(r['missed_merges'] for r in per_seq_results),
        'false_merge_frames': sum(r['false_merge_frames'] for r in per_seq_results),
        'missed_merge_frames': sum(r['missed_merge_frames'] for r in per_seq_results),
        'correct_merge_frames': sum(r['correct_merge_frames'] for r in per_seq_results),
        'false_merge_details': [],
    }
    agg['merge_precision'] = agg['correct_merges'] / max(agg['model_merges'], 1)
    agg['merge_recall'] = agg['correct_merges'] / max(agg['oracle_merges'], 1)
    return agg


# ── Main ───────────────────────────────────────────────────────────────

def save_split_tracklets(split_tracklets: Dict, sequence_name: str):
    """Save split tracklets to cache for use by error analysis."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"split_tracklets_{sequence_name}.pkl"
    with open(cache_path, "wb") as f:
        pickle.dump(split_tracklets, f)
    print(f"  Cached split tracklets -> {cache_path}")


def load_split_tracklets(sequence_name: str) -> Dict:
    """Load cached split tracklets."""
    cache_path = CACHE_DIR / f"split_tracklets_{sequence_name}.pkl"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No cached split tracklets for {sequence_name}.\n"
            f"Expected: {cache_path}\n"
            f"Run main.py first with the cache save line enabled:\n"
            f"  from experiments.tracklet_merger.error_analysis import save_split_tracklets\n"
            f"  save_split_tracklets(splitted_tracklets, sequence)"
        )
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build mergers
    mergers = {}

    if XGBOOST_MODEL.exists():
        mergers['XGBoost'] = XGBoostMerger(
            model_path=XGBOOST_MODEL,
            meta_path=XGBOOST_META,
            merge_threshold=0.7,
        )

    if CROSS_ATTN_MODEL.exists():
        mergers['Cross-Attention'] = CrossAttentionMerger(
            model_path=CROSS_ATTN_MODEL,
            meta_path=CROSS_ATTN_META,
            merge_threshold=0.75,
        )

    if SIAMESE_CLS_MODEL.exists():
        mergers['Siamese CLS'] = TransformerMerger(
            model_path=SIAMESE_CLS_MODEL,
            meta_path=SIAMESE_CLS_META,
            merge_threshold=0.7,
        )

    if HYBRID_MODEL.exists():
        mergers['Hybrid'] = HybridMerger(
            model_path=HYBRID_MODEL,
            meta_path=HYBRID_META,
            merge_threshold=0.65,
        )

    oracle = OracleMerger(gt_root=settings.DATA_ROOT)

    sequences = sorted([d.name for d in settings.DATA_ROOT.iterdir() if d.is_dir()])

    # Check that caches exist
    missing = [s for s in sequences if not (CACHE_DIR / f"split_tracklets_{s}.pkl").exists()]
    if missing:
        print(f"ERROR: Missing cached split tracklets for {len(missing)} sequences:")
        for s in missing[:5]:
            print(f"  - {s}")
        print(f"\nAdd this to main.py after split_tracklets():")
        print(f"    from experiments.tracklet_merger.error_analysis import save_split_tracklets")
        print(f"    save_split_tracklets(splitted_tracklets, sequence)")
        print(f"\nThen run main.py once to generate caches.")
        sys.exit(1)

    # Collect per-sequence results for each model
    all_model_results = {name: [] for name in mergers}

    for sequence in sequences:
        print(f"\n{'#'*70}")
        print(f"  Sequence: {sequence}")
        print(f"{'#'*70}")

        # Load cached split tracklets
        split_trks = load_split_tracklets(sequence)
        print(f"  Loaded {len(split_trks)} split tracklets from cache")

        # Run oracle
        oracle_copy = deep_copy_tracklets(split_trks)
        oracle_result = oracle.merge(oracle_copy, sequence_name=sequence)

        # Run each merger and analyze
        for name, merger in mergers.items():
            print(f"\n  Running {name}...")
            merger_copy = deep_copy_tracklets(split_trks)
            merged = merger.merge(merger_copy)
            results = analyze_errors(split_trks, merged, oracle_result, name)
            print_results(results)
            all_model_results[name].append(results)

    # Aggregate and compare
    aggregated = []
    for name in mergers:
        agg = aggregate_results(all_model_results[name], name)
        aggregated.append(agg)

    print_comparison_table(aggregated)

    # Save results
    save_data = {}
    for agg in aggregated:
        save_agg = {k: v for k, v in agg.items() if k != 'false_merge_details'}
        save_data[agg['model']] = save_agg

    output_path = OUTPUT_DIR / "error_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved -> {output_path}")


if __name__ == "__main__":
    main()
