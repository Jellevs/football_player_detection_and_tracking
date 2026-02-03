import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)


def find_optimal_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Find threshold that maximizes F1 score."""
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_f1 = 0
    best_thresh = 0.5

    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return float(best_thresh)


def evaluate_model(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
) -> Dict:
    """
    Compute all evaluation metrics.

    Args:
        labels: Ground truth binary labels
        scores: Predicted probabilities
        threshold: Decision threshold. If None, uses optimal threshold from F1.

    Returns:
        Dictionary of metrics
    """
    if threshold is None:
        threshold = find_optimal_threshold(labels, scores)

    predictions = (scores >= threshold).astype(int)

    metrics = {
        "threshold": threshold,
        "auc_roc": roc_auc_score(labels, scores),
        "avg_precision": average_precision_score(labels, scores),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"] = cm.ravel()

    return metrics


def print_evaluation_report(metrics: Dict, title: str = "Evaluation"):
    """Print a formatted evaluation report."""
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")
    print(f"  AUC-ROC:          {metrics['auc_roc']:.4f}")
    print(f"  Avg Precision:    {metrics['avg_precision']:.4f}")
    print(f"  Threshold:        {metrics['threshold']:.3f}")
    print(f"  Accuracy:         {metrics['accuracy']:.4f}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}")
    print(f"  F1:               {metrics['f1']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                  No Merge  Merge")
    print(f"  Actual No Merge   {metrics['tn']:5d}  {metrics['fp']:5d}")
    print(f"  Actual Merge      {metrics['fn']:5d}  {metrics['tp']:5d}")
    print(f"{'='*50}")


def evaluate_per_sequence(
    labels: np.ndarray,
    scores: np.ndarray,
    sequence_labels: np.ndarray,
    threshold: float,
) -> Dict[str, Dict]:
    """Evaluate metrics per sequence."""
    results = {}
    for seq in np.unique(sequence_labels):
        mask = sequence_labels == seq
        if mask.sum() < 2:
            continue
        seq_labels = labels[mask]
        seq_scores = scores[mask]
        # Need both classes present for AUC
        if len(np.unique(seq_labels)) < 2:
            results[seq] = {
                "n_samples": int(mask.sum()),
                "n_positive": int(seq_labels.sum()),
                "auc_roc": float("nan"),
            }
            continue
        results[seq] = evaluate_model(seq_labels, seq_scores, threshold=threshold)
        results[seq]["n_samples"] = int(mask.sum())
        results[seq]["n_positive"] = int(seq_labels.sum())

    return results
