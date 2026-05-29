"""
Training script for the Late Cross-Attention Transformer merger.

Key improvements over the temporal bin transformer training (Plan Sections 5-6):

C1. HOTA-based model selection:
    - Every K epochs, run clustering proxy evaluation (V-measure) on
      held-out validation sequences
    - Checkpoint and early-stop on clustering quality, not AUC
    - AUC kept only as a diagnostic sanity metric

C2. Auxiliary losses:
    - Supervised contrastive loss on per-tracklet CLS embeddings
      (pull same-identity tracklets together, push different apart)
    - Boundary-hardness weighting: hard-mined pairs get higher loss weight

C4. Weight EMA:
    - Maintain exponential moving average of model weights
    - Use EMA weights for validation/inference for smoother predictions

Phase 0 from the roadmap: the instrumentation phase that adds HOTA-aware
evaluation to the training loop.

Usage:
    python -m experiments.tracklet_merger.new_transformer.train

Prerequisite:
    Run generate_train_data.py for train, valid, and test splits first.
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

import copy
import json
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from torch.amp import autocast as _autocast, GradScaler
    def autocast(enabled=False):
        return _autocast(device_type="cuda", enabled=enabled)
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, confusion_matrix,
    v_measure_score,
)
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from experiments.tracklet_merger.new_transformer.config import NewTransformerConfig
from experiments.tracklet_merger.new_transformer.model import (
    LateCrossAttentionTransformer, EMAModel,
)
from experiments.tracklet_merger.new_transformer.dataset import NewTransformerDataset

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\new_transformer")
CACHE_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
SAVE_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\new_transformer")


# ---------------------------------------------------------------------------
# Asymmetric Focal Loss with sample weighting
# ---------------------------------------------------------------------------

class WeightedAsymmetricFocalLoss(nn.Module):
    """
    Asymmetric focal loss with per-sample weighting for hard negatives (C2).
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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)

        pos_loss = -targets * (1 - probs).pow(self.gamma_pos) * probs.log()
        neg_loss = -(1 - targets) * probs.pow(self.gamma_neg) * (1 - probs).log()

        loss = pos_loss + neg_loss

        if weights is not None:
            loss = loss * weights

        return loss.mean()


# ---------------------------------------------------------------------------
# Supervised Contrastive Loss (C2)
# ---------------------------------------------------------------------------

class SupConLoss(nn.Module):
    """
    Supervised contrastive loss on CLS embeddings.

    Pulls same-identity tracklet embeddings together and pushes different
    identities apart in the embedding space.  This shapes the distance
    geometry that linkage-based clustering consumes, improving the partition
    even where pairwise labels are ambiguous.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        cls_a: torch.Tensor,    # (B, d)
        cls_b: torch.Tensor,    # (B, d)
        labels: torch.Tensor,   # (B,) 0 or 1
    ) -> torch.Tensor:
        """
        Compute contrastive loss over the batch.
        Positive pairs (label=1) should be pulled together;
        all other pairs serve as negatives.
        """
        B = cls_a.size(0)
        if B < 2:
            return torch.tensor(0.0, device=cls_a.device)

        # Stack all embeddings: [a_0, b_0, a_1, b_1, ...]
        # For pairs with label=1, (a_i, b_i) are positives
        # For pairs with label=0, they are negatives to each other
        embeddings = torch.cat([cls_a, cls_b], dim=0)  # (2B, d)
        embeddings = F.normalize(embeddings, dim=1)

        # Similarity matrix
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature  # (2B, 2B)

        # Mask: for each anchor, which others are positive?
        # (a_i, b_i) are positive if label_i == 1
        positive_mask = torch.zeros(2 * B, 2 * B, dtype=torch.bool, device=cls_a.device)
        for i in range(B):
            if labels[i] > 0.5:  # positive pair
                positive_mask[i, B + i] = True
                positive_mask[B + i, i] = True

        # Self-mask
        self_mask = torch.eye(2 * B, dtype=torch.bool, device=cls_a.device)

        # If no positive pairs in batch, return 0
        if not positive_mask.any():
            return torch.tensor(0.0, device=cls_a.device)

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        # Denominator: sum over all non-self entries
        exp_sim = torch.exp(sim)
        exp_sim = exp_sim * (~self_mask).float()
        denom = exp_sim.sum(dim=1, keepdim=True) + 1e-8

        # Log probability
        log_prob = sim - torch.log(denom)

        # Average over positive pairs only
        mask_sum = positive_mask.float().sum(dim=1)
        valid = mask_sum > 0
        if not valid.any():
            return torch.tensor(0.0, device=cls_a.device)

        mean_log_prob = (log_prob * positive_mask.float()).sum(dim=1)
        mean_log_prob = mean_log_prob[valid] / mask_sum[valid]

        return -mean_log_prob.mean()


# ---------------------------------------------------------------------------
# Cosine annealing with linear warmup
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Clustering proxy evaluation (C1)
# ---------------------------------------------------------------------------

def compute_clustering_proxy(
    model: nn.Module,
    valid_pairs: list,
    dataset: NewTransformerDataset,
    device: torch.device,
    merge_threshold: float = 0.5,
    use_amp: bool = False,
) -> dict:
    """
    Compute V-measure as a fast clustering proxy for HOTA.

    For each sequence in the validation set:
    1. Collect all pairs and their predicted probabilities
    2. Build a distance matrix
    3. Run agglomerative clustering
    4. Compare predicted clusters to GT identity labels

    Returns dict with vmeasure, homogeneity, completeness scores.
    """
    model.eval()

    # Group pairs by sequence
    seq_pairs = {}
    for i, p in enumerate(dataset.pairs):
        sid = p["seq_id"]
        seq_pairs.setdefault(sid, []).append(i)

    all_vmeasures = []

    for sid, pair_indices in seq_pairs.items():
        if len(pair_indices) < 3:
            continue

        # Collect unique tracklet identifiers and GT labels
        tracklet_to_idx = {}
        tracklet_gt = {}

        for pi in pair_indices:
            p = dataset.pairs[pi]
            gt_a = p.get("gt_id_a", -1)
            gt_b = p.get("gt_id_b", -1)

            if p["is_synthetic"]:
                continue

            key_a = p.get("key_a")
            key_b = p.get("key_b")
            if key_a is None or key_b is None:
                continue

            if key_a not in tracklet_to_idx:
                tracklet_to_idx[key_a] = len(tracklet_to_idx)
                tracklet_gt[tracklet_to_idx[key_a]] = gt_a
            if key_b not in tracklet_to_idx:
                tracklet_to_idx[key_b] = len(tracklet_to_idx)
                tracklet_gt[tracklet_to_idx[key_b]] = gt_b

        n_tracklets = len(tracklet_to_idx)
        if n_tracklets < 3:
            continue

        # Build distance matrix
        dist_matrix = np.ones((n_tracklets, n_tracklets))
        np.fill_diagonal(dist_matrix, 0.0)

        for pi in pair_indices:
            p = dataset.pairs[pi]
            if p["is_synthetic"]:
                continue
            key_a = p.get("key_a")
            key_b = p.get("key_b")
            if key_a is None or key_b is None:
                continue
            if key_a not in tracklet_to_idx or key_b not in tracklet_to_idx:
                continue

            # Get prediction
            sample = dataset[pi]
            bins_a, bins_b, pw, label, weight, seq_id_t = sample

            with torch.no_grad():
                bins_a_t = bins_a.unsqueeze(0).to(device)
                bins_b_t = bins_b.unsqueeze(0).to(device)
                pw_t = pw.unsqueeze(0).to(device)

                with autocast(enabled=use_amp):
                    logit = model(bins_a_t, bins_b_t, pw_t)
                prob = torch.sigmoid(logit.float()).item()

            idx_a = tracklet_to_idx[key_a]
            idx_b = tracklet_to_idx[key_b]
            d = 1.0 - prob
            dist_matrix[idx_a, idx_b] = d
            dist_matrix[idx_b, idx_a] = d

        # Cluster
        try:
            condensed = squareform(dist_matrix)
            Z = linkage(condensed, method="average")
            pred_labels = fcluster(Z, t=merge_threshold, criterion="distance")
        except Exception:
            continue

        # GT labels
        gt_labels = np.array([tracklet_gt.get(i, -1) for i in range(n_tracklets)])

        # Filter out unknowns
        valid = gt_labels >= 0
        if valid.sum() < 3:
            continue

        vm = v_measure_score(gt_labels[valid], pred_labels[valid])
        all_vmeasures.append(vm)

    if not all_vmeasures:
        return {"vmeasure": 0.0, "n_sequences": 0}

    return {
        "vmeasure": float(np.mean(all_vmeasures)),
        "vmeasure_std": float(np.std(all_vmeasures)),
        "n_sequences": len(all_vmeasures),
    }


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
    ds = NewTransformerDataset(
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
        num_workers=0,
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
        bins_a, bins_b, pw, labels, weights, seq_ids = [x.to(device) for x in batch]

        with autocast(enabled=use_amp):
            logits = model(bins_a, bins_b, pw)
            loss = criterion(logits, labels, weights)

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
    config = NewTransformerConfig()
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
    n_hard = sum(1 for p in train_pairs if p.get("weight", 1.0) > 1.0)
    print(f"  Train: {len(train_pairs)} pairs ({n_real} real + {n_syn} synthetic, {n_hard} hard negatives)")
    print(f"  Valid: {len(valid_pairs)} pairs")
    print(f"  Test:  {len(test_pairs)} pairs")

    print("\nBuilding datasets...")
    train_loader = make_dataloader(train_pairs, config, CACHE_ROOT, training=True)
    valid_loader = make_dataloader(valid_pairs, config, CACHE_ROOT, training=False)
    valid_dataset = valid_loader.dataset
    print(f"  Train dataset: {len(train_loader.dataset)} pairs")
    print(f"  Valid dataset: {len(valid_loader.dataset)} pairs")

    # Class balance
    n_pos = sum(1 for p in train_pairs if p["label"] == 1)
    n_neg = len(train_pairs) - n_pos
    print(f"\nClass balance: {n_pos} pos / {n_neg} neg  ({100*n_pos/max(len(train_pairs),1):.1f}% pos)")

    # Model
    model = LateCrossAttentionTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # EMA (C4)
    ema = EMAModel(model, decay=config.ema_decay) if config.use_ema else None

    # Losses
    criterion = WeightedAsymmetricFocalLoss(
        gamma_pos=config.focal_gamma_pos,
        gamma_neg=config.focal_gamma_neg,
        label_smoothing=config.label_smoothing,
    )
    contrastive_loss_fn = SupConLoss(temperature=config.contrastive_temperature)

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

    # Tracking
    best_proxy = -1.0   # clustering proxy (V-measure or AUC as fallback)
    best_auc = 0.0
    patience_counter = 0
    selection_metric = "vmeasure" if config.use_vmeasure_proxy else "auc"

    log_path = SAVE_DIR / "training_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,train_loss,train_auc,val_loss,val_auc,vmeasure,lr,selection_metric\n")

    print(f"\n{'='*72}")
    print(f"  Training ({config.max_epochs} max epochs, patience={config.patience})")
    print(f"  Focal loss: gamma_pos={config.focal_gamma_pos}, gamma_neg={config.focal_gamma_neg}")
    print(f"  Contrastive weight: {config.contrastive_weight}")
    print(f"  Selection metric: {selection_metric}")
    print(f"  HOTA proxy eval interval: {config.hota_eval_interval} epochs")
    print(f"  EMA: {config.use_ema} (decay={config.ema_decay})")
    print(f"{'='*72}")
    print(f"  {'Epoch':>5}  {'train_loss':>10}  {'train_auc':>9}  "
          f"{'val_loss':>8}  {'val_auc':>7}  {'vmeasure':>8}  {'lr':>8}")
    print(f"  {'-'*66}")

    for epoch in range(1, config.max_epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        n_train = 0
        train_labels = []
        train_scores = []

        for batch in train_loader:  
            bins_a, bins_b, pw, labels, weights, seq_ids = [x.to(device) for x in batch]

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                # Single forward pass returns logits + CLS embeddings
                if config.contrastive_weight > 0:
                    logits, cls_a, cls_b = model(
                        bins_a, bins_b, pw, return_cls=True,
                    )
                else:
                    logits = model(bins_a, bins_b, pw)

                focal_loss = criterion(logits, labels, weights)

                # Supervised contrastive auxiliary loss (C2)
                if config.contrastive_weight > 0:
                    con_loss = contrastive_loss_fn(cls_a, cls_b, labels)
                    loss = focal_loss + config.contrastive_weight * con_loss
                else:
                    loss = focal_loss

            # Skip batch if loss is non-finite (prevents one bad sample from
            # poisoning the whole epoch and the model weights).
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # EMA update (C4)
            if ema is not None:
                ema.update(model)

            train_loss += focal_loss.item() * labels.size(0)
            n_train += labels.size(0)
            train_labels.append(labels.detach().cpu().numpy())
            train_scores.append(torch.sigmoid(logits.float().detach()).cpu().numpy())

        train_loss /= max(n_train, 1)
        if train_labels:
            train_labels = np.concatenate(train_labels)
            train_scores = np.concatenate(train_scores)
            train_auc = roc_auc_score(train_labels, train_scores)
        else:
            train_labels = np.array([])
            train_scores = np.array([])
            train_auc = 0.0
            print(f"  [WARN] epoch {epoch}: all batches were non-finite — skipped")

        # ---- Validate ----
        # Swap in EMA weights for validation
        if ema is not None:
            ema.apply_shadow(model)

        val_loss, val_labels, val_scores = evaluate_epoch(
            model, valid_loader, criterion, device, use_amp
        )
        val_auc = roc_auc_score(val_labels, val_scores)

        # ---- Clustering proxy evaluation (C1) ----
        vmeasure = -1.0
        if (config.hota_eval_interval > 0
                and epoch % config.hota_eval_interval == 0
                and config.use_vmeasure_proxy):
            proxy_result = compute_clustering_proxy(
                model, valid_pairs, valid_dataset, device,
                merge_threshold=0.5,
                use_amp=use_amp,
            )
            vmeasure = proxy_result.get("vmeasure", 0.0)

        # Restore original weights after validation
        if ema is not None:
            ema.restore(model)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        # ---- Model selection (C1) ----
        # When using vmeasure, ONLY checkpoint on epochs where it was
        # computed.  On other epochs keep patience counting but never
        # compare val_auc against the vmeasure best (they are different
        # scales, which is exactly the proxy-mixing the plan warns about).
        if selection_metric == "vmeasure":
            if vmeasure >= 0:
                current_score = vmeasure
                improved = current_score > best_proxy
            else:
                # No vmeasure this epoch — skip checkpointing,
                # but DO count toward patience.
                current_score = None
                improved = False
        else:
            current_score = val_auc
            improved = current_score > best_proxy

        marker = " <-" if improved else ""
        print(f"  {epoch:5d}  {train_loss:10.4f}  {train_auc:9.4f}  "
              f"{val_loss:8.4f}  {val_auc:7.4f}  {vmeasure:8.4f}  {lr:8.2e}{marker}")

        with open(log_path, "a") as f:
            f.write(f"{epoch},{train_loss},{train_auc},{val_loss},{val_auc},{vmeasure},{lr},{selection_metric}\n")

        # ---- Checkpoint ----
        if improved:
            best_proxy = current_score
            best_auc = val_auc
            patience_counter = 0

            # Save EMA weights if available, else regular weights
            if ema is not None:
                ema.apply_shadow(model)

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config,
                "val_auc": val_auc,
                "vmeasure": vmeasure,
                "selection_metric": selection_metric,
                "best_proxy": best_proxy,
            }, SAVE_DIR / "best_model.pt")

            if ema is not None:
                ema.restore(model)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(f"\n  Early stopping at epoch {epoch} (patience={config.patience})")
                break

    # ---- Final evaluation ----
    print("\nLoading best checkpoint...")
    ckpt = torch.load(SAVE_DIR / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"  Best epoch: {ckpt['epoch']}, val_auc: {ckpt['val_auc']:.4f}, "
          f"vmeasure: {ckpt.get('vmeasure', -1):.4f}")

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
        "best_vmeasure": ckpt.get("vmeasure", -1),
        "selection_metric": selection_metric,
        "use_residual_head": config.use_residual_head,
        "num_cross_attn_layers": config.num_cross_attn_layers,
        "contrastive_weight": config.contrastive_weight,
        "use_ema": config.use_ema,
    }
    meta_path = SAVE_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Meta saved   -> {meta_path}")
    print(f"Model saved  -> {SAVE_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    train()
