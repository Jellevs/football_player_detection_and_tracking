"""
Training script for the Pairwise MLP baseline.

Uses only the 12-dim pairwise features (no per-frame data, no pickle caches).
Trains in seconds since there's no sequence processing.

Usage:
    python -m experiments.tracklet_merger.transformers.pairwise_mlp.train

Available data strategies:
    "xgb_matched"   -> train_data/siamese_cls_xgb_matched/
    "simple_synth"  -> train_data/siamese_cls_simple_synth/
    "real_synth"    -> train_data/siamese_cls_real_synth/
    "real_only"     -> train_data/siamese_cls_real_only/
"""

import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

import sys
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from experiments.tracklet_merger.transformers.pairwise_mlp.model import PairwiseMLP


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_STRATEGY = "simple_synth"

_TRAIN_DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\train_data")

DATA_DIR       = _TRAIN_DATA_ROOT / f"siamese_cls_{DATA_STRATEGY}"
FIXED_EVAL_DIR = _TRAIN_DATA_ROOT / "fixed_eval"
SAVE_DIR       = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\pairwise_mlp\output") / f"pairwise_mlp_{DATA_STRATEGY}"

# Training hyperparameters
LR = 1e-3
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 200
PATIENCE = 20
LABEL_SMOOTHING = 0.05
DROPOUT = 0.3
PAIRWISE_DIM = 12

SEED = 42


# ---------------------------------------------------------------------------
# Lightweight dataset (pairwise features only)
# ---------------------------------------------------------------------------

class PairwiseDataset(Dataset):
    """Dataset that only loads pairwise features and labels — no pickle caches."""

    def __init__(self, pair_metadata, training=True):
        self.pairwise = []
        self.labels = []
        self.weights = []

        for p in pair_metadata:
            self.pairwise.append(p["pairwise"].copy())
            self.labels.append(p["label"])
            self.weights.append(p.get("weight", 1.0))

        self.pairwise = np.stack(self.pairwise).astype(np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)
        self.weights = np.array(self.weights, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.pairwise[idx]),
            torch.tensor(self.labels[idx]),
            torch.tensor(self.weights[idx]),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pairs(split_name: str):
    if split_name in ("valid", "test") and FIXED_EVAL_DIR.exists():
        path = FIXED_EVAL_DIR / f"pairs_{split_name}.pkl"
        print(f"  Loading FIXED {split_name} from {path}")
    else:
        path = DATA_DIR / f"pairs_{split_name}.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def find_best_threshold(labels, scores):
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (scores >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_scores = [], []

    for pw, labels, weights in loader:
        pw, labels = pw.to(device), labels.to(device)
        logits = model(pw)
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
# Training
# ---------------------------------------------------------------------------

def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading pair metadata...")
    train_pairs = load_pairs("train")
    valid_pairs = load_pairs("valid")
    test_pairs = load_pairs("test")
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Valid: {len(valid_pairs)} pairs")
    print(f"  Test:  {len(test_pairs)} pairs")

    print("\nBuilding datasets (pairwise features only — no pickle caches needed)...")
    train_ds = PairwiseDataset(train_pairs, training=True)
    valid_ds = PairwiseDataset(valid_pairs, training=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train dataset: {len(train_ds)} pairs")
    print(f"  Valid dataset: {len(valid_ds)} pairs")

    # Class balance
    n_pos = sum(1 for p in train_pairs if p["label"] == 1)
    n_neg = len(train_pairs) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"\nClass balance: {n_pos} pos / {n_neg} neg (pos_weight={pos_weight.item():.2f})")

    # Model
    model = PairwiseMLP(pairwise_dim=PAIRWISE_DIM, dropout=DROPOUT).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss
    eps = LABEL_SMOOTHING

    def criterion(logits, targets, sample_weights=None):
        if eps > 0:
            smoothed = targets * (1 - eps) + 0.5 * eps
        else:
            smoothed = targets
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, smoothed, pos_weight=pos_weight, reduction="none",
        )
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss.mean()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    # Training loop
    best_auc = 0.0
    patience_counter = 0

    log_path = SAVE_DIR / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_auc,val_loss,val_auc,lr\n")

    print(f"\n{'='*68}")
    print(f"  Training ({MAX_EPOCHS} max epochs, patience={PATIENCE})")
    print(f"{'='*68}")
    print(f"  {'Epoch':>5}  {'train_loss':>10}  {'train_auc':>9}  "
          f"{'val_loss':>8}  {'val_auc':>7}  {'lr':>8}")
    print(f"  {'-'*60}")

    for epoch in range(1, MAX_EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_train = 0
        train_labels, train_scores = [], []

        for pw, labels, weights in train_loader:
            pw, labels, weights = pw.to(device), labels.to(device), weights.to(device)
            optimizer.zero_grad()
            logits = model(pw)
            loss = criterion(logits, labels, weights)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            n_train += labels.size(0)
            train_labels.append(labels.detach().cpu().numpy())
            train_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

        train_loss /= max(n_train, 1)
        train_labels = np.concatenate(train_labels)
        train_scores = np.concatenate(train_scores)
        train_auc = roc_auc_score(train_labels, train_scores)

        # Validate
        val_loss, val_labels, val_scores = evaluate(model, valid_loader, criterion, device)
        val_auc = roc_auc_score(val_labels, val_scores)

        scheduler.step(val_auc)
        lr = optimizer.param_groups[0]["lr"]

        marker = " ←" if val_auc > best_auc else ""
        print(f"  {epoch:5d}  {train_loss:10.4f}  {train_auc:9.4f}  "
              f"{val_loss:8.4f}  {val_auc:7.4f}  {lr:8.2e}{marker}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss},{train_auc},{val_loss},{val_auc},{lr}\n")

        # Checkpoint
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
            }, SAVE_DIR / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # Final evaluation
    print("\nLoading best checkpoint...")
    ckpt = torch.load(SAVE_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Best epoch: {ckpt['epoch']}, val_auc: {ckpt['val_auc']:.4f}")

    # Validation
    _, val_labels, val_scores = evaluate(model, valid_loader, criterion, device)
    threshold = find_best_threshold(val_labels, val_scores)
    print_metrics(val_labels, val_scores, threshold, title="Validation Results")

    # Test
    test_ds = PairwiseDataset(test_pairs, training=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    _, test_labels, test_scores = evaluate(model, test_loader, criterion, device)
    test_metrics = print_metrics(test_labels, test_scores, threshold, title="Test Results")

    # Save
    metrics_path = SAVE_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))}, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")


if __name__ == "__main__":
    train()
