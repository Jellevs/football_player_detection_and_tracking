"""
Training script for the Siamese CLS Transformer tracklet merger.

Usage:
    1. Set DATA_STRATEGY below to choose the training data.
    2. Optionally set RUN_NAME to label this experiment.
    3. From the project root, run:
       python -m experiments.tracklet_merger.transformers.siamese_cls.train

Prerequisite:
    Run the corresponding generate_train_data_*.py for train, valid, and test
    splits first.

Available data strategies:
    "xgb_matched"   -> train_data/siamese_cls_xgb_matched/
    "simple_synth"  -> train_data/siamese_cls_simple_synth/
    "real_synth"    -> train_data/siamese_cls_real_synth/
    "real_only"     -> train_data/siamese_cls_real_only/

Every run is automatically saved to runs/ with full config, metrics, and
training log for later comparison.
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
# Data strategy selector
# ---------------------------------------------------------------------------
DATA_STRATEGY = "simple_synth"   # change to "simple_synth" or "xgb_matched"
RUN_NAME = ""                  # optional: label this run (e.g. "4layers_dropout03")

_TRAIN_DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\train_data")

DATA_DIR       = _TRAIN_DATA_ROOT / f"siamese_cls_{DATA_STRATEGY}"
FIXED_EVAL_DIR = _TRAIN_DATA_ROOT / "fixed_eval"
CACHE_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
SAVE_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\siamese_cls\output") / f"siamese_cls_{DATA_STRATEGY}"
RUNS_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\siamese_cls\runs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pairs(split_name: str, load_hard_synth: bool = True):
    # Val/test always from fixed eval dir; train from strategy-specific dir
    if split_name in ("valid", "test") and FIXED_EVAL_DIR.exists():
        path = FIXED_EVAL_DIR / f"pairs_{split_name}.pkl"
        print(f"  Loading FIXED {split_name} from {path}")
    else:
        path = DATA_DIR / f"pairs_{split_name}.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\nRun generate_fixed_eval_data.py or generate_train_data.py first."
        )
    with open(path, "rb") as f:
        pairs = pickle.load(f)

    # Also load hard synthetic pairs if available (only for training)
    if load_hard_synth and split_name == "train":
        hard_synth_path = DATA_DIR / f"hard_synth_pairs_{split_name}.pkl"
        if hard_synth_path.exists():
            with open(hard_synth_path, "rb") as f:
                hard_pairs = pickle.load(f)
            print(f"  Loaded {len(hard_pairs)} hard synthetic pairs for {split_name}")
            pairs.extend(hard_pairs)

    return pairs


def make_dataloader(pairs, config, cache_root, training=True):
    ds = SequencePairDataset(
        pair_metadata=pairs,
        cache_root=cache_root,
        t_max=config.t_max,
        max_frame_value=config.max_frame_value,
        training=training,
        frame_dropout=config.frame_dropout if training else 0.0,
        augment_swap=config.augment_swap if training else False,
    )
    return DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=0,       # pickle objects not fork-safe
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
def evaluate_epoch(model, loader, criterion, device):
    """Run one evaluation epoch, return loss and per-sample predictions."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_scores = []

    for batch in loader:
        tok_a, mask_a, pos_a, fn_a, tok_b, mask_b, pos_b, fn_b, pw, labels, sample_w = [
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
    avg_loss = total_loss / max(len(all_labels), 1)
    return avg_loss, all_labels, all_scores


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
# Run saving
# ---------------------------------------------------------------------------

def save_run(config, val_metrics, test_metrics, best_epoch, training_log_path):
    """Save a complete record of this training run for later comparison."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Auto-generate descriptive name from config
    auto_name = (
        f"{DATA_STRATEGY}"
        f"_L{config.num_layers}"
        f"_d{config.d_model}"
        f"_ff{config.dim_feedforward}"
        f"_h{config.nhead}"
        f"_do{config.transformer_dropout}"
        f"_fdo{config.frame_dropout}"
        f"_lr{config.lr}"
        f"_wd{config.weight_decay}"
        f"_bs{config.batch_size}"
        f"_ls{config.label_smoothing}"
        f"_sd{config.stochastic_depth}"
    )
    run_name = f"{RUN_NAME}_{auto_name}" if RUN_NAME else auto_name
    run_id = f"{timestamp}_{run_name}"
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save full config
    config_dict = asdict(config)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2, default=str)

    # Save metrics summary
    summary = {
        "run_id": run_id,
        "run_name": run_name,
        "data_strategy": DATA_STRATEGY,
        "timestamp": timestamp,
        "best_epoch": best_epoch,
        "n_params": sum(p.numel() for p in SiameseCLSTransformer(config).parameters()),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Copy training log CSV
    if training_log_path.exists():
        shutil.copy2(training_log_path, run_dir / "training_log.csv")

    # Copy best model
    best_model_path = SAVE_DIR / "best_model.pt"
    if best_model_path.exists():
        shutil.copy2(best_model_path, run_dir / "best_model.pt")

    print(f"\n  Run saved -> {run_dir}")

    # Also append to a master CSV for quick comparison
    master_csv = RUNS_DIR / "all_runs.csv"
    header_needed = not master_csv.exists()
    with open(master_csv, "a") as f:
        if header_needed:
            f.write("run_id,data_strategy,run_name,best_epoch,n_layers,d_model,"
                    "dim_ff,dropout,frame_dropout,lr,weight_decay,"
                    "val_auc,val_f1,val_ap,test_auc,test_f1,test_ap,timestamp\n")
        f.write(f"{run_id},{DATA_STRATEGY},{run_name},{best_epoch},"
                f"{config.num_layers},{config.d_model},{config.dim_feedforward},"
                f"{config.transformer_dropout},{config.frame_dropout},"
                f"{config.lr},{config.weight_decay},"
                f"{val_metrics.get('auc_roc', 0):.4f},"
                f"{val_metrics.get('f1', 0):.4f},"
                f"{val_metrics.get('avg_precision', 0):.4f},"
                f"{test_metrics.get('auc_roc', 0):.4f},"
                f"{test_metrics.get('f1', 0):.4f},"
                f"{test_metrics.get('avg_precision', 0):.4f},"
                f"{timestamp}\n")

    return run_dir


# ---------------------------------------------------------------------------
# Training loop
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

    config = TransformerMergerConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data (separate sequence-level splits)
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

    # Validation threshold selection (clean sequence-level split)
    _, val_labels, val_scores = evaluate_epoch(model, valid_loader, criterion, device)
    threshold = find_best_threshold(val_labels, val_scores)
    val_metrics = print_metrics(val_labels, val_scores, threshold, title="Validation Results")

    # Test evaluation (held-out test split)
    test_loader = make_dataloader(test_pairs, config, CACHE_ROOT, training=False)
    _, test_labels, test_scores = evaluate_epoch(model, test_loader, criterion, device)
    test_metrics = print_metrics(test_labels, test_scores, threshold, title="Test Results")

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
    }
    meta_path = SAVE_DIR / "transformer_merger_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved   -> {meta_path}")
    print(f"Model saved  -> {SAVE_DIR / 'best_model.pt'}")

    # Save this run to the runs directory for comparison
    save_run(
        config=config,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        best_epoch=ckpt["epoch"],
        training_log_path=log_path,
    )


if __name__ == "__main__":
    train()
