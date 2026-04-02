"""
Training script for the Cross-Attention Siamese Transformer.

Usage:
    python transformer/train_cross.py

Prerequisite:
    Run transformer/generate_train_data.py for train, valid, and test splits.
    (Same data as the vanilla transformer — no new data generation needed.)

Improvements over the vanilla train.py:
  - Larger model (d_model=256, 6 encoder + 3 cross-attention layers)
  - Asymmetric focal loss: stronger penalty for missed positives (gamma_pos=2)
    than for easy negatives (gamma_neg=0.5), directly addressing the
    high-threshold / low-recall calibration issue
  - Cosine LR schedule with linear warmup (more stable for larger transformers)
  - Train AUC printed each epoch for overfitting monitoring
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

from transformer.model_cross import CrossAttentionTransformer, CrossAttnConfig
from transformer.dataset import SequencePairDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\transformer_data")
CACHE_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache")
SAVE_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\transformer_cross")


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def asymmetric_focal_loss(logits, targets, pos_weight, gamma_pos=2.0, gamma_neg=0.5):
    """
    Asymmetric focal loss for binary classification.

    Positives (merge pairs) get gamma_pos — stronger focus on hard missed merges.
    Negatives (non-merge pairs) get gamma_neg — mild downweighting of easy negatives.

    pos_weight still scales the positive contribution to compensate for class
    imbalance in the overall loss magnitude.
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    probs = torch.sigmoid(logits)
    p_t   = probs * targets + (1.0 - probs) * (1.0 - targets)   # prob of correct class
    gamma = targets * gamma_pos + (1.0 - targets) * gamma_neg
    return (((1.0 - p_t) ** gamma) * bce).mean()


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_pairs(split_name: str):
    path = DATA_DIR / f"pairs_{split_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\nRun generate_train_data.py for the '{split_name}' split first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def make_dataloader(pairs, config, cache_root, training=True):
    ds = SequencePairDataset(
        pair_metadata   = pairs,
        cache_root      = cache_root,
        t_max           = config.t_max,
        max_frame_value = config.max_frame_value,
        training        = training,
        frame_dropout   = config.frame_dropout if training else 0.0,
        augment_swap    = config.augment_swap  if training else False,
    )
    return DataLoader(
        ds,
        batch_size  = config.batch_size,
        shuffle     = training,
        num_workers = 0,
        pin_memory  = True,
        drop_last   = training,
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def find_best_threshold(labels, scores):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_scores = []
    for batch in loader:
        tok_a, mask_a, pos_a, fn_a, tok_b, mask_b, pos_b, fn_b, pw, labels = [
            x.to(device) for x in batch
        ]
        logits = model(tok_a, mask_a, pos_a, fn_a,
                       tok_b, mask_b, pos_b, fn_b, pw)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_scores.append(torch.sigmoid(logits).cpu().numpy())
    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    return total_loss / max(len(all_labels), 1), all_labels, all_scores


def print_metrics(labels, scores, threshold, title=""):
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "threshold":     threshold,
        "auc_roc":       roc_auc_score(labels, scores),
        "avg_precision": average_precision_score(labels, scores),
        "f1":            f1_score(labels, preds, zero_division=0),
        "precision":     precision_score(labels, preds, zero_division=0),
        "recall":        recall_score(labels, preds, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    bar = "=" * 52
    print(f"\n{bar}\n  {title}\n{bar}")
    print(f"  AUC-ROC       : {metrics['auc_roc']:.4f}")
    print(f"  Avg Precision : {metrics['avg_precision']:.4f}")
    print(f"  F1            : {metrics['f1']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  Threshold     : {metrics['threshold']:.3f}")
    print(f"  TN / FP       : {tn:5d} / {fp:5d}")
    print(f"  FN / TP       : {fn:5d} / {tp:5d}")
    print(bar)
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    config = CrossAttnConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"d_model={config.d_model}, encoder_layers={config.num_layers}, "
          f"cross_layers={config.num_cross_layers}")

    print("Loading pair metadata...")
    train_pairs = load_pairs("train")
    valid_pairs = load_pairs("valid")
    test_pairs  = load_pairs("test")
    print(f"  Train: {len(train_pairs)} | Valid: {len(valid_pairs)} | Test: {len(test_pairs)}")

    print("\nBuilding datasets...")
    train_loader = make_dataloader(train_pairs, config, CACHE_ROOT, training=True)
    valid_loader = make_dataloader(valid_pairs, config, CACHE_ROOT, training=False)

    n_pos = sum(1 for p in train_pairs if p["label"] == 1)
    n_neg = len(train_pairs) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"Class balance: {n_pos} pos / {n_neg} neg (pos_weight={pos_weight.item():.2f})")

    model    = CrossAttentionTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Asymmetric focal loss
    GAMMA_POS, GAMMA_NEG = 2.0, 0.5
    def criterion(logits, targets):
        return asymmetric_focal_loss(logits, targets, pos_weight,
                                     gamma_pos=GAMMA_POS, gamma_neg=GAMMA_NEG)
    print(f"Loss: asymmetric focal (gamma_pos={GAMMA_POS}, gamma_neg={GAMMA_NEG})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = cosine_schedule_with_warmup(
        optimizer, warmup_epochs=config.warmup_epochs, total_epochs=config.max_epochs
    )
    print(f"LR: cosine warmup ({config.warmup_epochs} epochs) → decay over {config.max_epochs}")

    best_auc         = 0.0
    patience_counter = 0

    print(f"\n{'='*68}")
    print(f"  Training ({config.max_epochs} max epochs, patience={config.patience})")
    print(f"{'='*68}")
    print(f"  {'Epoch':>5}  {'train_loss':>10}  {'train_auc':>9}  "
          f"{'val_loss':>8}  {'val_auc':>7}  {'lr':>8}")
    print(f"  {'-'*60}")

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss   = 0.0
        n_train      = 0
        train_labels = []
        train_scores = []

        for batch in train_loader:
            tok_a, mask_a, pos_a, fn_a, tok_b, mask_b, pos_b, fn_b, pw, labels = [
                x.to(device) for x in batch
            ]
            optimizer.zero_grad()
            logits = model(tok_a, mask_a, pos_a, fn_a,
                           tok_b, mask_b, pos_b, fn_b, pw)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            n_train    += labels.size(0)
            train_labels.append(labels.detach().cpu().numpy())
            train_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

        train_loss /= max(n_train, 1)
        train_labels = np.concatenate(train_labels)
        train_scores = np.concatenate(train_scores)
        train_auc    = roc_auc_score(train_labels, train_scores)

        val_loss, val_labels, val_scores = evaluate_epoch(
            model, valid_loader, criterion, device
        )
        val_auc = roc_auc_score(val_labels, val_scores)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        marker = " ←" if val_auc > best_auc else ""
        print(f"  {epoch:5d}  {train_loss:10.4f}  {train_auc:9.4f}  "
              f"{val_loss:8.4f}  {val_auc:7.4f}  {lr:8.2e}{marker}")

        if val_auc > best_auc:
            best_auc         = val_auc
            patience_counter = 0
            torch.save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "config":           config,
                "val_auc":          val_auc,
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
    print(f"  Best epoch: {ckpt['epoch']}, val_auc: {ckpt['val_auc']:.4f}")

    _, val_labels, val_scores = evaluate_epoch(model, valid_loader, criterion, device)
    threshold = find_best_threshold(val_labels, val_scores)
    print_metrics(val_labels, val_scores, threshold, title="Validation Results")

    test_loader = make_dataloader(test_pairs, config, CACHE_ROOT, training=False)
    _, test_labels, test_scores = evaluate_epoch(model, test_loader, criterion, device)
    test_metrics = print_metrics(test_labels, test_scores, threshold, title="Test Results")

    with open(SAVE_DIR / "metrics.json", "w") as f:
        json.dump({k: v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))}, f, indent=2)

    meta = {
        "threshold":        threshold,
        "t_max":            config.t_max,
        "d_model":          config.d_model,
        "max_frame_value":  config.max_frame_value,
        "num_cross_layers": config.num_cross_layers,
    }
    with open(SAVE_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved to {SAVE_DIR}")


if __name__ == "__main__":
    train()
