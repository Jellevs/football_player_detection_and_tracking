"""
Model size ablation for the Siamese CLS Transformer.

Trains the Siamese CLS architecture across a range of model sizes
to investigate whether the performance bottleneck is model capacity
or training data volume.

All runs use the best data strategy (simple_synth) and identical
training hyperparameters. Only d_model, dim_feedforward, and nhead
are varied.

Usage:
    python -m experiments.tracklet_merger.transformers.siamese_cls.train_size_ablation

Results are saved to the runs/ directory and appended to all_runs.csv.
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
    f1_score, confusion_matrix,
)

import sys
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from experiments.tracklet_merger.transformers.siamese_cls.config import TransformerMergerConfig
from experiments.tracklet_merger.transformers.siamese_cls.model import SiameseCLSTransformer
from experiments.tracklet_merger.transformers.siamese_cls.dataset import SequencePairDataset


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_STRATEGY = "simple_synth"

_TRAIN_DATA_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\train_data")
DATA_DIR       = _TRAIN_DATA_ROOT / f"siamese_cls_{DATA_STRATEGY}"
FIXED_EVAL_DIR = _TRAIN_DATA_ROOT / "fixed_eval"
CACHE_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
RUNS_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\experiments\tracklet_merger\transformers\siamese_cls\runs")

# Model sizes to ablate: (d_model, dim_feedforward, nhead)
# nhead must divide d_model evenly
SIZE_CONFIGS = [
    # Label         d_model  dim_ff  nhead  num_layers
    # ("tiny",            32,     64,     2,     4),
    # ("small",           48,     96,     4,     4),
    ("base",            80,    128,     4,     4),   # current best
    # ("medium",         128,    256,     4,     4),   # already tested
    # ("large",          192,    384,     4,     4),
    # ("xlarge",         256,    512,     4,     4),
]


# ---------------------------------------------------------------------------
# Helpers (same as train.py)
# ---------------------------------------------------------------------------

def load_pairs(split_name: str, load_hard_synth: bool = True):
    if split_name in ("valid", "test") and FIXED_EVAL_DIR.exists():
        path = FIXED_EVAL_DIR / f"pairs_{split_name}.pkl"
    else:
        path = DATA_DIR / f"pairs_{split_name}.pkl"

    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    with open(path, "rb") as f:
        pairs = pickle.load(f)

    if load_hard_synth and split_name == "train":
        hard_synth_path = DATA_DIR / f"hard_synth_pairs_{split_name}.pkl"
        if hard_synth_path.exists():
            with open(hard_synth_path, "rb") as f:
                hard_pairs = pickle.load(f)
            print(f"  Loaded {len(hard_pairs)} hard synthetic pairs")
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
        num_workers=0,
        pin_memory=True,
        drop_last=training,
    )


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
    all_labels, all_scores = [], []

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


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def train_single(size_label, d_model, dim_ff, nhead, num_layers):
    """Train one model size configuration and return metrics."""

    SEED = 42
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override config for this size
    config = TransformerMergerConfig()
    config.d_model = d_model
    config.dim_feedforward = dim_ff
    config.nhead = nhead
    config.num_layers = num_layers

    # Save directory for this run
    save_dir = RUNS_DIR / f"ablation_{size_label}_d{d_model}_ff{dim_ff}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*68}")
    print(f"  SIZE ABLATION: {size_label} (d_model={d_model}, ff={dim_ff}, nhead={nhead})")
    print(f"{'='*68}")

    # Load data
    train_pairs = load_pairs("train", load_hard_synth=True)
    valid_pairs = load_pairs("valid", load_hard_synth=False)
    test_pairs  = load_pairs("test", load_hard_synth=False)
    print(f"  Train: {len(train_pairs)} | Valid: {len(valid_pairs)} | Test: {len(test_pairs)}")

    train_loader = make_dataloader(train_pairs, config, CACHE_ROOT, training=True)
    valid_loader = make_dataloader(valid_pairs, config, CACHE_ROOT, training=False)

    # Class balance
    n_pos = sum(1 for p in train_pairs if p["label"] == 1)
    n_neg = len(train_pairs) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    # Model
    model = SiameseCLSTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    # Loss
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
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8
    )

    # Training loop
    best_auc = 0.0
    patience_counter = 0

    log_path = save_dir / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_auc,val_loss,val_auc,lr\n")

    print(f"  {'Epoch':>5}  {'train_loss':>10}  {'train_auc':>9}  "
          f"{'val_loss':>8}  {'val_auc':>7}  {'lr':>8}")
    print(f"  {'-'*60}")

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        train_loss, n_train = 0.0, 0
        train_labels, train_scores = [], []

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
            n_train += labels.size(0)
            train_labels.append(labels.detach().cpu().numpy())
            train_scores.append(torch.sigmoid(logits).detach().cpu().numpy())

        train_loss /= max(n_train, 1)
        train_labels = np.concatenate(train_labels)
        train_scores = np.concatenate(train_scores)
        train_auc = roc_auc_score(train_labels, train_scores)

        val_loss, val_labels, val_scores = evaluate_epoch(
            model, valid_loader, criterion, device
        )
        val_auc = roc_auc_score(val_labels, val_scores)
        scheduler.step(val_auc)
        lr = optimizer.param_groups[0]["lr"]

        marker = " *" if val_auc > best_auc else ""
        print(f"  {epoch:5d}  {train_loss:10.4f}  {train_auc:9.4f}  "
              f"{val_loss:8.4f}  {val_auc:7.4f}  {lr:8.2e}{marker}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss},{train_auc},{val_loss},{val_auc},{lr}\n")

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_auc": val_auc,
            }, save_dir / "best_model.pt")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # Final evaluation
    ckpt = torch.load(save_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    best_epoch = ckpt["epoch"]

    _, val_labels, val_scores = evaluate_epoch(model, valid_loader, criterion, device)
    threshold = find_best_threshold(val_labels, val_scores)
    val_auc = roc_auc_score(val_labels, val_scores)
    val_ap = average_precision_score(val_labels, val_scores)

    test_loader = make_dataloader(test_pairs, config, CACHE_ROOT, training=False)
    _, test_labels, test_scores = evaluate_epoch(model, test_loader, criterion, device)
    test_auc = roc_auc_score(test_labels, test_scores)
    test_ap = average_precision_score(test_labels, test_scores)
    test_preds = (test_scores >= threshold).astype(int)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(test_labels, test_preds).ravel()
    test_precision = tp / max(tp + fp, 1)
    test_recall = tp / max(tp + fn, 1)

    print(f"\n  {size_label}: val_auc={val_auc:.4f} test_auc={test_auc:.4f} "
          f"test_ap={test_ap:.4f} test_f1={test_f1:.4f} "
          f"prec={test_precision:.4f} rec={test_recall:.4f} params={n_params:,}")

    # Save metrics
    result = {
        "size_label": size_label,
        "d_model": d_model,
        "dim_feedforward": dim_ff,
        "nhead": nhead,
        "num_layers": num_layers,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "threshold": threshold,
        "val_auc": val_auc,
        "val_ap": val_ap,
        "test_auc": test_auc,
        "test_ap": test_ap,
        "test_f1": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    with open(save_dir / "ablation_metrics.json", "w") as f:
        json.dump(result, f, indent=2)

    # Append to master CSV
    master_csv = RUNS_DIR / "all_runs.csv"
    header_needed = not master_csv.exists()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"ablation_{size_label}_d{d_model}_ff{dim_ff}"
    with open(master_csv, "a") as f:
        if header_needed:
            f.write("run_id,data_strategy,run_name,best_epoch,n_layers,d_model,"
                    "dim_ff,dropout,frame_dropout,lr,weight_decay,"
                    "val_auc,val_f1,val_ap,test_auc,test_f1,test_ap,timestamp\n")
        f.write(f"{timestamp}_{run_name},{DATA_STRATEGY},{run_name},{best_epoch},"
                f"{config.num_layers},{config.d_model},{config.dim_feedforward},"
                f"{config.transformer_dropout},{config.frame_dropout},"
                f"{config.lr},{config.weight_decay},"
                f"{val_auc:.4f},{0:.4f},{val_ap:.4f},"
                f"{test_auc:.4f},{test_f1:.4f},{test_ap:.4f},{timestamp}\n")

    return result


# ---------------------------------------------------------------------------
# Main: run all sizes
# ---------------------------------------------------------------------------

def main():
    print("=" * 68)
    print("  MODEL SIZE ABLATION STUDY")
    print("  Testing whether capacity or data is the bottleneck")
    print("=" * 68)

    results = []
    for label, d_model, dim_ff, nhead, n_layers in SIZE_CONFIGS:
        result = train_single(label, d_model, dim_ff, nhead, n_layers)
        results.append(result)

    # Summary table
    print("\n" + "=" * 80)
    print("  ABLATION SUMMARY")
    print("=" * 80)
    print(f"  {'Label':<10} {'d_model':>7} {'ff':>5} {'Params':>10} "
          f"{'Val AUC':>8} {'Test AUC':>9} {'Test AP':>8} {'Epoch':>6}")
    print(f"  {'-'*72}")
    for r in results:
        print(f"  {r['size_label']:<10} {r['d_model']:>7} {r['dim_feedforward']:>5} "
              f"{r['n_params']:>10,} {r['val_auc']:>8.4f} {r['test_auc']:>9.4f} "
              f"{r['test_ap']:>8.4f} {r['best_epoch']:>6}")

    # Save summary
    summary_path = RUNS_DIR / "size_ablation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Summary saved -> {summary_path}")


if __name__ == "__main__":
    main()
