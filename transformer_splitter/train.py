"""
Training script for the per-frame split-point transformer.

Usage:
    python transformer_splitter/train.py

Prerequisite:
    Run generate_split_data.py for train, valid, and test splits first.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
)

from transformer_splitter.config import SplitTransformerConfig
from transformer_splitter.model import SplitPointTransformer
from transformer_splitter.dataset import SplitDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\splitter_data")
CACHE_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache")
SAVE_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\transformer_splitter")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_samples(split_name: str):
    path = DATA_DIR / f"samples_{split_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\nRun generate_split_data.py for the '{split_name}' split first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def make_dataloader(samples, config, cache_root, training=True):
    ds = SplitDataset(
        sample_metadata=samples,
        cache_root=cache_root,
        t_max=config.t_max,
        max_frame_value=config.max_frame_value,
        split_label_radius=config.split_label_radius,
        training=training,
        frame_dropout=config.frame_dropout if training else 0.0,
    )
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=0,
        pin_memory=True,
        drop_last=training,
    )


def make_criterion(config, device):
    """Recall-biased BCE with label smoothing."""
    pos_weight = torch.tensor([config.pos_weight], device=device)
    raw_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    eps = config.label_smoothing

    def criterion(logits, labels, mask):
        """
        logits: (B, T)
        labels: (B, T)
        mask:   (B, T) True = padding
        """
        if eps > 0:
            smoothed = labels * (1 - eps) + 0.5 * eps
        else:
            smoothed = labels
        loss = raw_criterion(logits, smoothed)
        # Zero out padding
        valid = (~mask).float()
        loss = (loss * valid).sum() / valid.sum().clamp(min=1)
        return loss

    return criterion


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    """Run evaluation, return loss and per-frame predictions (only on real frames)."""
    model.eval()
    total_loss = 0.0
    n_total = 0
    all_labels = []
    all_scores = []

    for batch in loader:
        tokens, mask, pos, fn, labels = [x.to(device) for x in batch]
        logits = model(tokens, mask, pos, fn)
        loss = criterion(logits, labels, mask)

        valid = ~mask
        n_valid = valid.sum().item()
        total_loss += loss.item() * n_valid
        n_total += n_valid

        # Collect only non-padded frame predictions
        all_labels.append(labels[valid].cpu().numpy())
        all_scores.append(torch.sigmoid(logits[valid]).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    avg_loss = total_loss / max(n_total, 1)
    return avg_loss, all_labels, all_scores


def find_best_threshold(labels, scores, metric="recall"):
    """Find threshold optimizing for recall (oversplit > undersplit)."""
    best_val, best_t = 0.0, 0.3
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= t).astype(int)
        if metric == "recall":
            val = recall_score(labels, preds, zero_division=0)
        elif metric == "f1":
            val = f1_score(labels, preds, zero_division=0)
        else:
            val = recall_score(labels, preds, zero_division=0)
        if val > best_val:
            best_val, best_t = val, t
    return float(best_t)


def compute_split_detection_metrics(labels, scores, threshold):
    """
    Compute both frame-level and sample-aware metrics.
    Returns dict of metrics.
    """
    preds = (scores >= threshold).astype(int)
    binary_labels = (labels > 0).astype(int)

    metrics = {
        "threshold": threshold,
        "frame_f1": f1_score(binary_labels, preds, zero_division=0),
        "frame_precision": precision_score(binary_labels, preds, zero_division=0),
        "frame_recall": recall_score(binary_labels, preds, zero_division=0),
    }

    if len(np.unique(binary_labels)) > 1:
        metrics["frame_auc"] = roc_auc_score(binary_labels, scores)
    else:
        metrics["frame_auc"] = 0.0

    return metrics


def print_metrics(metrics, title=""):
    bar = "=" * 52
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
    print(f"  Frame AUC      : {metrics['frame_auc']:.4f}")
    print(f"  Frame F1       : {metrics['frame_f1']:.4f}")
    print(f"  Frame Precision: {metrics['frame_precision']:.4f}")
    print(f"  Frame Recall   : {metrics['frame_recall']:.4f}")
    print(f"  Threshold      : {metrics['threshold']:.3f}")
    print(bar)
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    config = SplitTransformerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading sample metadata...")
    train_samples = load_samples("train")
    valid_samples = load_samples("valid")
    test_samples  = load_samples("test")
    print(f"  Train: {len(train_samples)} samples")
    print(f"  Valid: {len(valid_samples)} samples")
    print(f"  Test:  {len(test_samples)} samples")

    print("\nBuilding datasets...")
    train_loader = make_dataloader(train_samples, config, CACHE_ROOT, training=True)
    valid_loader = make_dataloader(valid_samples, config, CACHE_ROOT, training=False)
    print(f"  Train dataset: {len(train_loader.dataset)} samples")
    print(f"  Valid dataset: {len(valid_loader.dataset)} samples")

    # Model
    model = SplitPointTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    criterion = make_criterion(config, device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    best_recall = 0.0
    patience_counter = 0

    print(f"\n{'='*72}")
    print(f"  Training ({config.max_epochs} max epochs, patience={config.patience})")
    print(f"{'='*72}")
    print(f"  {'Epoch':>5}  {'tr_loss':>8}  {'tr_auc':>7}  {'tr_rec':>7}  "
          f"{'v_loss':>7}  {'v_auc':>6}  {'v_rec':>6}  {'v_prec':>6}  {'lr':>8}")
    print(f"  {'-'*68}")

    for epoch in range(1, config.max_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        n_train = 0
        train_labels_all = []
        train_scores_all = []

        for batch in train_loader:
            tokens, mask, pos, fn, labels = [x.to(device) for x in batch]
            optimizer.zero_grad()
            logits = model(tokens, mask, pos, fn)
            loss = criterion(logits, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            valid = ~mask
            n_valid = valid.sum().item()
            train_loss += loss.item() * n_valid
            n_train += n_valid
            train_labels_all.append(labels[valid].detach().cpu().numpy())
            train_scores_all.append(torch.sigmoid(logits[valid]).detach().cpu().numpy())

        train_loss /= max(n_train, 1)
        train_labels_all = np.concatenate(train_labels_all)
        train_scores_all = np.concatenate(train_scores_all)
        tr_binary = (train_labels_all > 0).astype(int)
        train_auc = roc_auc_score(tr_binary, train_scores_all) if len(np.unique(tr_binary)) > 1 else 0.0
        train_recall = recall_score(tr_binary, (train_scores_all >= 0.3).astype(int), zero_division=0)

        # ---- Validate ----
        val_loss, val_labels, val_scores = evaluate_epoch(
            model, valid_loader, criterion, device
        )
        val_binary = (val_labels > 0).astype(int)
        val_auc = roc_auc_score(val_binary, val_scores) if len(np.unique(val_binary)) > 1 else 0.0
        val_preds = (val_scores >= 0.3).astype(int)
        val_recall = recall_score(val_binary, val_preds, zero_division=0)
        val_precision = precision_score(val_binary, val_preds, zero_division=0)

        scheduler.step(val_recall)
        lr = optimizer.param_groups[0]["lr"]

        marker = " ←" if val_recall > best_recall else ""
        print(f"  {epoch:5d}  {train_loss:8.4f}  {train_auc:7.4f}  {train_recall:7.4f}  "
              f"{val_loss:7.4f}  {val_auc:6.4f}  {val_recall:6.4f}  {val_precision:6.4f}  "
              f"{lr:8.2e}{marker}")

        # ---- Checkpoint on best recall ----
        if val_recall > best_recall:
            best_recall = val_recall
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_recall": val_recall,
                "val_auc": val_auc,
            }, SAVE_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={config.patience})")
                break

    # ---- Final evaluation ----
    print("\nLoading best checkpoint...")
    ckpt = torch.load(SAVE_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Best epoch: {ckpt['epoch']}, val_recall: {ckpt['val_recall']:.4f}")

    # Validation — find threshold for high recall
    _, val_labels, val_scores = evaluate_epoch(model, valid_loader, criterion, device)
    threshold = find_best_threshold(val_labels, val_scores, metric="recall")
    val_metrics = compute_split_detection_metrics(val_labels, val_scores, threshold)
    print_metrics(val_metrics, title="Validation Results (recall-optimized threshold)")

    # Also show F1-optimized threshold for reference
    threshold_f1 = find_best_threshold(val_labels, val_scores, metric="f1")
    val_metrics_f1 = compute_split_detection_metrics(val_labels, val_scores, threshold_f1)
    print_metrics(val_metrics_f1, title="Validation Results (F1-optimized threshold)")

    # Test evaluation
    test_loader = make_dataloader(test_samples, config, CACHE_ROOT, training=False)
    _, test_labels, test_scores = evaluate_epoch(model, test_loader, criterion, device)
    test_metrics = compute_split_detection_metrics(test_labels, test_scores, threshold)
    print_metrics(test_metrics, title="Test Results (recall-optimized threshold)")

    # Save metrics
    metrics_path = SAVE_DIR / "splitter_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))}, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")

    # Save metadata for inference
    meta = {
        "threshold": threshold,
        "threshold_f1": threshold_f1,
        "t_max": config.t_max,
        "d_model": config.d_model,
        "max_frame_value": config.max_frame_value,
        "split_label_radius": config.split_label_radius,
    }
    meta_path = SAVE_DIR / "splitter_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved   -> {meta_path}")
    print(f"Model saved  -> {SAVE_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    train()
