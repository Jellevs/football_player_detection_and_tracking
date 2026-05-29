"""
Training script for the Siamese CLS Transformer WITHOUT pairwise features.

Ablation study: tests whether the transformer's attention mechanism can learn
to extract useful information from raw frame sequences without the handcrafted
pairwise features in the classification head.

Usage:
    python -m experiments.tracklet_merger.transformers.siamese_cls.train_no_pairwise
"""

import json
import pickle
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
)

import sys
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from experiments.tracklet_merger.transformers.siamese_cls.config import TransformerMergerConfig
from experiments.tracklet_merger.transformers.siamese_cls.model import SiameseCLSTransformer
from experiments.tracklet_merger.transformers.siamese_cls.dataset import SequencePairDataset


# ---------------------------------------------------------------------------
# Configuration for this ablation
# ---------------------------------------------------------------------------
DATA_STRATEGY = "simple_synth"
RUN_NAME = "no_pairwise_simply_synth"

_TRAIN_DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\train_data")

DATA_DIR       = _TRAIN_DATA_ROOT / f"siamese_cls_{DATA_STRATEGY}"
FIXED_EVAL_DIR = _TRAIN_DATA_ROOT / "fixed_eval"
CACHE_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
SAVE_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\siamese_cls\output") / "siamese_cls_no_pairwise_simple_synth"
RUNS_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\siamese_cls\runs")


# ---------------------------------------------------------------------------
# Import shared helpers from the main train script
# ---------------------------------------------------------------------------
from experiments.tracklet_merger.transformers.siamese_cls.train import (
    load_pairs, make_dataloader, find_best_threshold, evaluate_epoch,
    print_metrics, save_run,
)

# Override DATA_DIR and FIXED_EVAL_DIR used by load_pairs
import experiments.tracklet_merger.transformers.siamese_cls.train as _train_module
_train_module.DATA_DIR = DATA_DIR
_train_module.FIXED_EVAL_DIR = FIXED_EVAL_DIR
_train_module.SAVE_DIR = SAVE_DIR
_train_module.RUNS_DIR = RUNS_DIR
_train_module.DATA_STRATEGY = DATA_STRATEGY
_train_module.RUN_NAME = RUN_NAME


# ---------------------------------------------------------------------------
# Training loop (identical to main train.py except config change)
# ---------------------------------------------------------------------------

def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # KEY CHANGE: disable pairwise features
    config = TransformerMergerConfig(use_pairwise_features=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"*** ABLATION: use_pairwise_features = False ***")

    # Load data
    print("Loading pair metadata...")
    train_pairs = load_pairs("train", load_hard_synth=True)
    valid_pairs = load_pairs("valid", load_hard_synth=False)
    test_pairs = load_pairs("test", load_hard_synth=False)
    print(f"  Train: {len(train_pairs)} pairs")
    print(f"  Valid: {len(valid_pairs)} pairs")
    print(f"  Test:  {len(test_pairs)} pairs")

    print("\nBuilding datasets (loading pickle caches)...")
    train_loader = make_dataloader(train_pairs, config, CACHE_ROOT, training=True)
    valid_loader = make_dataloader(valid_pairs, config, CACHE_ROOT, training=False)
    print(f"  Train dataset: {len(train_loader.dataset)} pairs")
    print(f"  Valid dataset: {len(valid_loader.dataset)} pairs")

    # Class balance
    n_pos = sum(1 for p in train_pairs if p["label"] == 1)
    n_neg = len(train_pairs) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"\nClass balance: {n_pos} pos / {n_neg} neg (pos_weight={pos_weight.item():.2f})")

    # Model
    model = SiameseCLSTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss with label smoothing and per-sample weighting
    eps = config.label_smoothing

    def criterion(logits, targets, sample_weights=None):
        if eps > 0:
            smoothed = targets * (1 - eps) + 0.5 * eps
        else:
            smoothed = targets
        loss_per_sample = nn.functional.binary_cross_entropy_with_logits(
            logits, smoothed, pos_weight=pos_weight, reduction="none",
        )
        if sample_weights is not None:
            loss_per_sample = loss_per_sample * sample_weights
        return loss_per_sample.mean()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    # Training
    best_auc = 0.0
    patience_counter = 0

    # CSV training log
    log_path = SAVE_DIR / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_auc,val_loss,val_auc,lr\n")

    print(f"\n{'='*68}")
    print(f"  Training ({config.max_epochs} max epochs, patience={config.patience})")
    print(f"{'='*68}")
    print(f"  {'Epoch':>5}  {'train_loss':>10}  {'train_auc':>9}  "
          f"{'val_loss':>8}  {'val_auc':>7}  {'lr':>8}")
    print(f"  {'-'*60}")

    for epoch in range(1, config.max_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss   = 0.0
        n_train      = 0
        train_labels = []
        train_scores = []

        for batch in train_loader:
            tok_a, mask_a, pos_a, fn_a, tok_b, mask_b, pos_b, fn_b, pw, labels, sample_w = [
                x.to(device) for x in batch
            ]
            optimizer.zero_grad()
            logits = model(tok_a, mask_a, pos_a, fn_a,
                           tok_b, mask_b, pos_b, fn_b, pw)
            loss = criterion(logits, labels, sample_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            n_train    += labels.size(0)
            train_labels.append(labels.detach().cpu().numpy())
            train_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

        train_loss  /= max(n_train, 1)
        train_labels = np.concatenate(train_labels)
        train_scores = np.concatenate(train_scores)
        train_auc    = roc_auc_score(train_labels, train_scores)

        # ---- Validate ----
        val_loss, val_labels, val_scores = evaluate_epoch(
            model, valid_loader, criterion, device
        )
        val_auc = roc_auc_score(val_labels, val_scores)

        scheduler.step(val_auc)
        lr = optimizer.param_groups[0]["lr"]

        marker = " ←" if val_auc > best_auc else ""
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
    _, val_labels, val_scores = evaluate_epoch(model, valid_loader, criterion, device)
    threshold = find_best_threshold(val_labels, val_scores)
    val_metrics = print_metrics(val_labels, val_scores, threshold, title="Validation Results (NO PAIRWISE)")

    # Test evaluation
    test_loader = make_dataloader(test_pairs, config, CACHE_ROOT, training=False)
    _, test_labels, test_scores = evaluate_epoch(model, test_loader, criterion, device)
    test_metrics = print_metrics(test_labels, test_scores, threshold, title="Test Results (NO PAIRWISE)")

    # Save test metrics
    metrics_path = SAVE_DIR / "transformer_merger_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({k: v for k, v in test_metrics.items()
                   if isinstance(v, (int, float))}, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")

    # Save metadata for inference
    meta = {
        "threshold": threshold,
        "t_max": config.t_max,
        "d_model": config.d_model,
        "max_frame_value": config.max_frame_value,
        "use_pairwise_features": False,
    }
    meta_path = SAVE_DIR / "transformer_merger_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved   -> {meta_path}")
    print(f"Model saved  -> {SAVE_DIR / 'best_model.pt'}")

    # Save run
    save_run(
        config=config,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        best_epoch=ckpt["epoch"],
        training_log_path=log_path,
    )


if __name__ == "__main__":
    train()
