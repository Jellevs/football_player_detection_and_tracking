"""
Training script for the Temporal Bin Transformer tracklet merger.

Key training improvements over the original transformer:
  1. Asymmetric focal loss — aggressively down-weights easy negatives (gamma=2.0)
     while being gentler on positives (gamma=1.0), since false negatives
     (missed merges) are more harmful than false positives in tracking.
  2. Cosine annealing with linear warmup — more stable than ReduceLROnPlateau,
     which can get stuck at local minima with small models.
  3. Mixed precision (AMP) — faster training with compact 12-token sequences.
  4. Gradient accumulation option for effective larger batch sizes.

Usage:
    python train.py

Prerequisite:
    Run generate_train_data.py for train, valid, and test splits first.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast as _autocast, GradScaler   # PyTorch >= 2.4
    def autocast(enabled=False):
        return _autocast(device_type="cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler           # PyTorch < 2.4
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

from experiments.tracklet_merger.transformer.config import TemporalBinConfig
from experiments.tracklet_merger.transformer.model import TemporalBinTransformer
from experiments.tracklet_merger.transformer.dataset import TemporalBinDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\temporal_bin_transformer")
CACHE_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
SAVE_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\temporal_bin_transformer")


# ---------------------------------------------------------------------------
# Asymmetric Focal Loss
# ---------------------------------------------------------------------------

class AsymmetricFocalLoss(nn.Module):
    """
    Focal loss with separate gamma for positives and negatives.

    Standard focal loss: FL(p) = -alpha * (1-p)^gamma * log(p)

    Asymmetric version allows:
      - gamma_pos < gamma_neg: be less aggressive on positive down-weighting
        (we want to catch all true merges) while strongly down-weighting
        trivially easy negatives (jersey mismatch, different teams, etc.)

    Also supports label smoothing: targets are smoothed before loss computation.
    """

    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,) raw logits
        targets: (B,) binary labels
        """
        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits)
        # Clamp for numerical stability
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)

        # Positive: -target * (1-p)^gamma_pos * log(p)
        pos_loss = -targets * (1 - probs).pow(self.gamma_pos) * probs.log()
        # Negative: -(1-target) * p^gamma_neg * log(1-p)
        neg_loss = -(1 - targets) * probs.pow(self.gamma_neg) * (1 - probs).log()

        loss = pos_loss + neg_loss
        return loss.mean()


# ---------------------------------------------------------------------------
# Cosine annealing with linear warmup
# ---------------------------------------------------------------------------

class CosineAnnealingWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing schedule with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.max_epochs - self.warmup_epochs, 1
            )
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pairs(split_name: str):
    path = DATA_DIR / f"pairs_{split_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\nRun generate_train_data.py for '{split_name}' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def make_dataloader(pairs, config, cache_root, training=True):
    ds = TemporalBinDataset(
        pair_metadata=pairs,
        cache_root=cache_root,
        n_bins=config.n_temporal_bins,
        boundary_k=config.boundary_k,
        max_frame_value=config.max_frame_value,
        training=training,
        frame_dropout=config.frame_dropout if training else 0.0,
        augment_swap=config.augment_swap if training else False,
        noise_std_reid=config.noise_std_reid if training else 0.0,
        noise_std_siglip=config.noise_std_siglip if training else 0.0,
        modality_mask_prob=config.modality_mask_prob if training else 0.0,
        temporal_crop_prob=config.temporal_crop_prob if training else 0.0,
    )
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=0,  # pickle caches not fork-safe
        pin_memory=True,
        drop_last=training,
    )


def find_best_threshold(labels, scores):
    """Find probability threshold maximizing F1."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


@torch.no_grad()
def evaluate_epoch(model, loader, criterion, device, use_amp=False):
    """Run one evaluation pass."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_scores = []

    for batch in loader:
        bins_a, bins_b, pw, labels = [x.to(device) for x in batch]

        with autocast(enabled=use_amp):
            logits = model(bins_a, bins_b, pw)
            loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        all_labels.append(labels.cpu().numpy())
        all_scores.append(torch.sigmoid(logits.float()).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)
    avg_loss = total_loss / max(len(all_labels), 1)
    return avg_loss, all_labels, all_scores


def print_metrics(labels, scores, threshold, title=""):
    preds = (scores >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "threshold": threshold,
        "auc_roc": roc_auc_score(labels, scores),
        "avg_precision": average_precision_score(labels, scores),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    bar = "=" * 52
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)
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
    config = TemporalBinConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}  |  AMP: {use_amp}")

    # Load data
    print("Loading pair metadata...")
    train_pairs = load_pairs("train")
    valid_pairs = load_pairs("valid")
    test_pairs  = load_pairs("test")

    n_real = sum(1 for p in train_pairs if not p.get("is_synthetic", False))
    n_syn  = sum(1 for p in train_pairs if p.get("is_synthetic", False))
    print(f"  Train: {len(train_pairs)} pairs ({n_real} real + {n_syn} synthetic)")
    print(f"  Valid: {len(valid_pairs)} pairs")
    print(f"  Test:  {len(test_pairs)} pairs")

    print("\nBuilding datasets...")
    train_loader = make_dataloader(train_pairs, config, CACHE_ROOT, training=True)
    valid_loader = make_dataloader(valid_pairs, config, CACHE_ROOT, training=False)
    print(f"  Train dataset: {len(train_loader.dataset)} pairs")
    print(f"  Valid dataset: {len(valid_loader.dataset)} pairs")

    # Class balance info
    n_pos = sum(1 for p in train_pairs if p["label"] == 1)
    n_neg = len(train_pairs) - n_pos
    print(f"\nClass balance: {n_pos} pos / {n_neg} neg  ({100*n_pos/max(len(train_pairs),1):.1f}% pos)")

    # Model
    model = TemporalBinTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss
    criterion = AsymmetricFocalLoss(
        gamma_pos=config.focal_gamma_pos,
        gamma_neg=config.focal_gamma_neg,
        label_smoothing=config.label_smoothing,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingWarmup(
        optimizer,
        warmup_epochs=config.warmup_epochs,
        max_epochs=config.max_epochs,
        min_lr=config.min_lr,
    )
    scaler = GradScaler(enabled=use_amp)

    # Training
    best_auc = 0.0
    patience_counter = 0

    log_path = SAVE_DIR / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_auc,val_loss,val_auc,lr\n")

    print(f"\n{'='*68}")
    print(f"  Training ({config.max_epochs} max epochs, patience={config.patience})")
    print(f"  Focal loss: gamma_pos={config.focal_gamma_pos}, gamma_neg={config.focal_gamma_neg}")
    print(f"  Cosine annealing: warmup={config.warmup_epochs} epochs")
    print(f"{'='*68}")
    print(f"  {'Epoch':>5}  {'train_loss':>10}  {'train_auc':>9}  "
          f"{'val_loss':>8}  {'val_auc':>7}  {'lr':>8}")
    print(f"  {'-'*60}")

    for epoch in range(1, config.max_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        n_train = 0
        train_labels = []
        train_scores = []

        for batch in train_loader:
            bins_a, bins_b, pw, labels = [x.to(device) for x in batch]

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                logits = model(bins_a, bins_b, pw)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * labels.size(0)
            n_train += labels.size(0)
            train_labels.append(labels.detach().cpu().numpy())
            train_scores.append(torch.sigmoid(logits.float().detach()).cpu().numpy())

        train_loss /= max(n_train, 1)
        train_labels = np.concatenate(train_labels)
        train_scores = np.concatenate(train_scores)
        train_auc = roc_auc_score(train_labels, train_scores)

        # ---- Validate ----
        val_loss, val_labels, val_scores = evaluate_epoch(
            model, valid_loader, criterion, device, use_amp
        )
        val_auc = roc_auc_score(val_labels, val_scores)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        marker = " <-" if val_auc > best_auc else ""
        print(f"  {epoch:5d}  {train_loss:10.4f}  {train_auc:9.4f}  "
              f"{val_loss:8.4f}  {val_auc:7.4f}  {lr:8.2e}{marker}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss},{train_auc},{val_loss},{val_auc},{lr}\n")

        # ---- Checkpoint ----
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
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
    print(f"  Best epoch: {ckpt['epoch']}, val_auc: {ckpt['val_auc']:.4f}")

    # Validation threshold selection
    _, val_labels, val_scores = evaluate_epoch(
        model, valid_loader, criterion, device, use_amp
    )
    threshold = find_best_threshold(val_labels, val_scores)
    print_metrics(val_labels, val_scores, threshold, title="Validation Results")

    # Test evaluation
    test_loader = make_dataloader(test_pairs, config, CACHE_ROOT, training=False)
    _, test_labels, test_scores = evaluate_epoch(
        model, test_loader, criterion, device, use_amp
    )
    test_metrics = print_metrics(test_labels, test_scores, threshold, title="Test Results")

    # Save metrics
    metrics_path = SAVE_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))}, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")

    # Save metadata for inference
    meta = {
        "threshold": threshold,
        "n_temporal_bins": config.n_temporal_bins,
        "boundary_k": config.boundary_k,
        "d_model": config.d_model,
        "max_frame_value": config.max_frame_value,
        "n_params": n_params,
        "best_epoch": ckpt["epoch"],
        "best_val_auc": ckpt["val_auc"],
    }
    meta_path = SAVE_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved   -> {meta_path}")
    print(f"Model saved  -> {SAVE_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    train()
