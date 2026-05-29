"""
Train the Frame-level Cross-Attention Pair Transformer.

Uses the same training pairs as the temporal bin transformer (10K+ pairs with
synthetic positives) but processes per-frame features instead of temporal bins.

The key innovation: self-attention across frames of BOTH tracklets, allowing
the model to compare specific frames (e.g., matching jersey views). This is
something XGBoost and the temporal bin transformer fundamentally cannot do.

Usage:
    python experiments/tracklet_merger/frame_transformer/train.py
"""

import sys
import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.tracklet_merger.frame_transformer.config import FrameTransformerConfig
from experiments.tracklet_merger.frame_transformer.model import FramePairTransformer
from experiments.tracklet_merger.frame_transformer.dataset import FramePairDataset, collate_fn


# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\temporal_bin_transformer")
CACHE_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
SAVE_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\frame_transformer")


# ── Training hyperparameters ──────────────────────────────────────────────────
MAX_EPOCHS     = 100
PATIENCE       = 20
BATCH_SIZE     = 32          # smaller than temporal bin (variable-length sequences)
LR             = 2e-4
MIN_LR         = 1e-6
WEIGHT_DECAY   = 5e-3
WARMUP_EPOCHS  = 5
GRAD_CLIP      = 1.0

# Focal loss
FOCAL_GAMMA_POS = 1.0
FOCAL_GAMMA_NEG = 2.0
LABEL_SMOOTHING = 0.05


# ── Asymmetric Focal Loss (proven effective for temporal bin transformer) ─────

class AsymmetricFocalLoss(nn.Module):
    """Focal loss with separate gamma for positives and negatives."""

    def __init__(self, gamma_pos=1.0, gamma_neg=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)

        pos_loss = -targets * (1 - probs).pow(self.gamma_pos) * probs.log()
        neg_loss = -(1 - targets) * probs.pow(self.gamma_neg) * (1 - probs).log()

        return (pos_loss + neg_loss).mean()


# ── Cosine annealing with linear warmup ───────────────────────────────────────

class CosineAnnealingWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(self.warmup_epochs, 1)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(
                self.max_epochs - self.warmup_epochs, 1
            )
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine
                for base_lr in self.base_lrs
            ]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pairs(split_name: str):
    path = DATA_DIR / f"pairs_{split_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing: {path}\nRun the data generation script for '{split_name}' first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def make_dataloader(pairs, config, training=True):
    ds = FramePairDataset(
        pair_metadata=pairs,
        cache_root=CACHE_ROOT,
        max_frames=config.max_frames_per_tracklet,
        min_frames=config.min_frames,
        training=training,
        frame_dropout=config.frame_dropout if training else 0.0,
        augment_swap=config.augment_swap if training else False,
        noise_std_reid=config.noise_std_reid if training else 0.0,
        noise_std_siglip=config.noise_std_siglip if training else 0.0,
    )
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=training,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True,
    )


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_logits = []
    all_labels = []

    for batch in loader:
        feats_a, frames_a, mask_a, feats_b, frames_b, mask_b, pw, labels = [
            x.to(device) for x in batch
        ]

        with autocast(enabled=use_amp):
            logits = model(feats_a, frames_a, mask_a, feats_b, frames_b, mask_b, pw)
            loss = criterion(logits, labels)

        total_loss += loss.item()
        n_batches += 1
        all_logits.extend(logits.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    logits_np = np.array(all_logits)
    labels_np = np.array(all_labels)
    auc = roc_auc_score(labels_np, logits_np)

    probs = 1.0 / (1.0 + np.exp(-logits_np))
    preds = (probs > 0.5).astype(int)
    f1 = f1_score(labels_np.astype(int), preds)

    return avg_loss, auc, f1


# ── Training loop ─────────────────────────────────────────────────────────────

def train():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    config = FrameTransformerConfig()
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
    train_loader = make_dataloader(train_pairs, config, training=True)
    valid_loader = make_dataloader(valid_pairs, config, training=False)
    print(f"  Train dataset: {len(train_loader.dataset)} pairs")
    print(f"  Valid dataset: {len(valid_loader.dataset)} pairs")

    # Class balance
    n_pos = sum(1 for p in train_pairs if p["label"] == 1)
    n_neg = len(train_pairs) - n_pos
    print(f"\nClass balance: {n_pos} pos / {n_neg} neg "
          f"({100*n_pos/max(len(train_pairs),1):.1f}% pos)")

    # Model
    model = FramePairTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")
    print(f"Config: d_model={config.d_model}, n_layers={config.n_layers}, "
          f"n_heads={config.n_heads}, max_frames={config.max_frames_per_tracklet}")

    # Loss
    criterion = AsymmetricFocalLoss(
        gamma_pos=FOCAL_GAMMA_POS,
        gamma_neg=FOCAL_GAMMA_NEG,
        label_smoothing=LABEL_SMOOTHING,
    )

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingWarmup(
        optimizer, warmup_epochs=WARMUP_EPOCHS, max_epochs=MAX_EPOCHS, min_lr=MIN_LR
    )
    scaler = GradScaler(enabled=use_amp)

    # Training
    best_auc = 0.0
    patience_counter = 0

    log_path = SAVE_DIR / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_auc,val_loss,val_auc,val_f1,lr\n")

    print(f"\n{'='*70}")
    print(f"  Training ({MAX_EPOCHS} max epochs, patience={PATIENCE})")
    print(f"  Batch size: {BATCH_SIZE}, LR: {LR}, Weight decay: {WEIGHT_DECAY}")
    print(f"  Focal loss: gamma_pos={FOCAL_GAMMA_POS}, gamma_neg={FOCAL_GAMMA_NEG}")
    print(f"{'='*70}")
    print(f"  {'Epoch':>5}  {'t_loss':>7}  {'t_auc':>6}  "
          f"{'v_loss':>7}  {'v_auc':>6}  {'v_f1':>5}  {'lr':>8}")
    print(f"  {'-'*58}")

    for epoch in range(1, MAX_EPOCHS + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        n_batches = 0
        train_logits = []
        train_labels = []

        for batch in train_loader:
            feats_a, frames_a, mask_a, feats_b, frames_b, mask_b, pw, labels = [
                x.to(device) for x in batch
            ]

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                logits = model(feats_a, frames_a, mask_a, feats_b, frames_b, mask_b, pw)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1
            train_logits.extend(logits.detach().cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        scheduler.step()

        avg_train_loss = train_loss / max(n_batches, 1)
        train_auc = roc_auc_score(train_labels, train_logits)

        # ---- Validate ----
        val_loss, val_auc, val_f1 = evaluate(
            model, valid_loader, criterion, device, use_amp
        )

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  {epoch:5d}  {avg_train_loss:7.4f}  {train_auc:.4f}  "
              f"{val_loss:7.4f}  {val_auc:.4f}  {val_f1:.3f}  {current_lr:.1e}")

        # Log
        with open(log_path, "a") as f:
            f.write(f"{epoch},{avg_train_loss:.6f},{train_auc:.6f},"
                    f"{val_loss:.6f},{val_auc:.6f},{val_f1:.6f},{current_lr:.8f}\n")

        # ---- Early stopping on val AUC ----
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_auc": val_auc,
                "val_f1": val_f1,
            }, SAVE_DIR / "best_model.pt")
            print(f"         -> New best! val_auc={val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (best val_auc={best_auc:.4f})")
                break

    # ── Final evaluation on test set ──────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  Loading best model for test evaluation...")
    ckpt = torch.load(SAVE_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Best epoch: {ckpt['epoch']}, val_auc: {ckpt['val_auc']:.4f}")

    test_loader = make_dataloader(test_pairs, config, training=False)
    test_loss, test_auc, test_f1 = evaluate(
        model, test_loader, criterion, device, use_amp
    )
    print(f"\n  Test results:")
    print(f"    AUC:  {test_auc:.4f}")
    print(f"    F1:   {test_f1:.4f}")
    print(f"    Loss: {test_loss:.4f}")

    # Save metrics
    metrics = {
        "best_epoch": ckpt["epoch"],
        "val_auc": float(ckpt["val_auc"]),
        "val_f1": float(ckpt.get("val_f1", 0)),
        "test_auc": float(test_auc),
        "test_f1": float(test_f1),
        "test_loss": float(test_loss),
        "n_params": n_params,
    }
    with open(SAVE_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save meta for the merger
    meta = {
        "max_frames": config.max_frames_per_tracklet,
        "min_frames": config.min_frames,
        "val_auc": float(ckpt["val_auc"]),
        "test_auc": float(test_auc),
    }
    with open(SAVE_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n  Model saved -> {SAVE_DIR / 'best_model.pt'}")
    print(f"  Meta saved  -> {SAVE_DIR / 'meta.json'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    train()
