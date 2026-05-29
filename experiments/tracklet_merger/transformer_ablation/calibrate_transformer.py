"""
Post-hoc probability calibration for the Temporal Bin Transformer.

The transformer achieves higher AUC than XGBoost (0.9586 vs 0.9518) but
lower HOTA (87.87 vs 90.75). AUC measures ranking quality; HOTA depends
on absolute probability values because probabilities are converted to
distances (d = 1 - p) for hierarchical clustering. If the transformer's
probabilities are poorly calibrated (bunched around 0.5 instead of being
bimodal near 0 and 1), the distance matrix loses discriminative power.

This script fits three calibration methods on the validation set:
  1. Temperature scaling: p = sigmoid(logits / T)
  2. Platt scaling:       p = sigmoid(a * logits + b)
  3. Isotonic regression:  non-parametric monotonic mapping

Then prints the optimal T/a/b values so they can be plugged into the merger.

Usage:
    python experiments/tracklet_merger/transformer_ablation/calibrate_transformer.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))

import json
import pickle
import numpy as np
import torch
from pathlib import Path
from scipy.optimize import minimize_scalar, minimize
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    f1_score, precision_score, recall_score,
)

from experiments.tracklet_merger.transformer.config import TemporalBinConfig
from experiments.tracklet_merger.transformer.model import TemporalBinTransformer
from experiments.tracklet_merger.transformer.dataset import TemporalBinDataset
from torch.utils.data import DataLoader

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\temporal_bin_transformer\best_model.pt")
DATA_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\train_data\temporal_bin_transformer")
CACHE_ROOT = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\output\cache_split")
SAVE_DIR   = Path(r"C:\Users\jelle\Documents\TUEindhoven\Master\Thesis\development\tracklet_splitter_scratch\weights\temporal_bin_transformer")


def load_pairs(split_name):
    path = DATA_DIR / f"pairs_{split_name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def make_dataloader(pairs, config, training=False):
    ds = TemporalBinDataset(
        pair_metadata=pairs,
        cache_root=CACHE_ROOT,
        n_bins=config.n_temporal_bins,
        boundary_k=config.boundary_k,
        max_frame_value=config.max_frame_value,
        training=False,
        frame_dropout=0.0,
        augment_swap=False,
        noise_std_reid=0.0,
        noise_std_siglip=0.0,
        modality_mask_prob=0.0,
        temporal_crop_prob=0.0,
    )
    return DataLoader(ds, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)


@torch.no_grad()
def extract_logits(model, loader, device):
    """Extract raw logits and labels from the model."""
    model.eval()
    all_logits = []
    all_labels = []
    for batch in loader:
        bins_a, bins_b, pw, labels = [x.to(device) for x in batch]
        logits = model(bins_a, bins_b, pw)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_logits), np.concatenate(all_labels)


# ── Calibration Methods ───────────────────────────────────────────────────────

def fit_temperature(logits, labels):
    """Find temperature T that minimizes NLL on validation set."""
    def nll(T):
        T = max(T, 0.01)  # prevent division by zero
        scaled = 1.0 / (1.0 + np.exp(-logits / T))
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        return log_loss(labels, scaled)

    result = minimize_scalar(nll, bounds=(0.01, 10.0), method="bounded")
    return result.x


def fit_platt(logits, labels):
    """Find a, b that minimize NLL: p = sigmoid(a * logits + b)."""
    def nll(params):
        a, b = params
        scaled = 1.0 / (1.0 + np.exp(-(a * logits + b)))
        scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
        return log_loss(labels, scaled)

    result = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead")
    return result.x[0], result.x[1]


def fit_isotonic(logits, labels):
    """Fit isotonic regression: non-parametric monotonic calibration."""
    probs = 1.0 / (1.0 + np.exp(-logits))
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(probs, labels)
    return ir


# ── Analysis ──────────────────────────────────────────────────────────────────

def analyze_probabilities(probs, labels, name=""):
    """Print probability distribution statistics."""
    pos_probs = probs[labels == 1]
    neg_probs = probs[labels == 0]

    print(f"\n  --- {name} ---")
    print(f"  Positive pairs (n={len(pos_probs)}):")
    print(f"    Mean: {pos_probs.mean():.4f}  Median: {np.median(pos_probs):.4f}  "
          f"Std: {pos_probs.std():.4f}")
    print(f"    [0.0, 0.3): {(pos_probs < 0.3).sum():5d}  "
          f"[0.3, 0.5): {((pos_probs >= 0.3) & (pos_probs < 0.5)).sum():5d}  "
          f"[0.5, 0.7): {((pos_probs >= 0.5) & (pos_probs < 0.7)).sum():5d}  "
          f"[0.7, 1.0]: {(pos_probs >= 0.7).sum():5d}")
    print(f"  Negative pairs (n={len(neg_probs)}):")
    print(f"    Mean: {neg_probs.mean():.4f}  Median: {np.median(neg_probs):.4f}  "
          f"Std: {neg_probs.std():.4f}")
    print(f"    [0.0, 0.3): {(neg_probs < 0.3).sum():5d}  "
          f"[0.3, 0.5): {((neg_probs >= 0.3) & (neg_probs < 0.5)).sum():5d}  "
          f"[0.5, 0.7): {((neg_probs >= 0.5) & (neg_probs < 0.7)).sum():5d}  "
          f"[0.7, 1.0]: {(neg_probs >= 0.7).sum():5d}")

    auc = roc_auc_score(labels, probs)
    brier = brier_score_loss(labels, probs)
    nll = log_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7))
    print(f"  AUC: {auc:.4f}  Brier: {brier:.4f}  NLL: {nll:.4f}")

    # Separation: gap between mean positive and mean negative probability
    separation = pos_probs.mean() - neg_probs.mean()
    print(f"  Separation (mean_pos - mean_neg): {separation:.4f}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    config = ckpt["config"]
    model = TemporalBinTransformer(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from epoch {ckpt['epoch']}, val_auc={ckpt['val_auc']:.4f}")

    # Extract logits on validation set
    print("\nExtracting validation logits...")
    valid_pairs = load_pairs("valid")
    valid_loader = make_dataloader(valid_pairs, config)
    logits, labels = extract_logits(model, valid_loader, device)
    print(f"  {len(logits)} pairs, {int(labels.sum())} positive, {int((1-labels).sum())} negative")

    # ── Uncalibrated ──
    probs_raw = 1.0 / (1.0 + np.exp(-logits))
    print("\n" + "=" * 60)
    print("UNCALIBRATED")
    print("=" * 60)
    analyze_probabilities(probs_raw, labels, "Raw sigmoid")

    # ── Temperature scaling ──
    T = fit_temperature(logits, labels)
    probs_temp = 1.0 / (1.0 + np.exp(-logits / T))
    print("\n" + "=" * 60)
    print(f"TEMPERATURE SCALING  (T = {T:.4f})")
    print("=" * 60)
    analyze_probabilities(probs_temp, labels, f"Temperature T={T:.4f}")

    # ── Platt scaling ──
    a, b = fit_platt(logits, labels)
    probs_platt = 1.0 / (1.0 + np.exp(-(a * logits + b)))
    print("\n" + "=" * 60)
    print(f"PLATT SCALING  (a = {a:.4f}, b = {b:.4f})")
    print("=" * 60)
    analyze_probabilities(probs_platt, labels, f"Platt a={a:.4f}, b={b:.4f}")

    # ── Isotonic regression ──
    ir = fit_isotonic(logits, labels)
    probs_iso = ir.predict(probs_raw)
    print("\n" + "=" * 60)
    print("ISOTONIC REGRESSION")
    print("=" * 60)
    analyze_probabilities(probs_iso, labels, "Isotonic")

    # ── Summary comparison ──
    print("\n" + "=" * 60)
    print("SUMMARY: Calibration Method Comparison")
    print("=" * 60)
    methods = {
        "Raw":         probs_raw,
        f"Temp(T={T:.3f})": probs_temp,
        f"Platt(a={a:.3f},b={b:.3f})": probs_platt,
        "Isotonic":    probs_iso,
    }
    print(f"  {'Method':<28} {'AUC':>6} {'Brier':>7} {'NLL':>7} {'Separation':>11}")
    print(f"  {'-'*60}")
    for name, probs in methods.items():
        auc = roc_auc_score(labels, probs)
        brier = brier_score_loss(labels, probs)
        nll = log_loss(labels, np.clip(probs, 1e-7, 1 - 1e-7))
        pos_probs = probs[labels == 1]
        neg_probs = probs[labels == 0]
        sep = pos_probs.mean() - neg_probs.mean()
        print(f"  {name:<28} {auc:.4f} {brier:7.4f} {nll:7.4f} {sep:11.4f}")

    # ── Save calibration parameters ──
    calib = {
        "temperature": float(T),
        "platt_a": float(a),
        "platt_b": float(b),
        "raw_auc": float(roc_auc_score(labels, probs_raw)),
        "raw_brier": float(brier_score_loss(labels, probs_raw)),
        "temp_brier": float(brier_score_loss(labels, probs_temp)),
        "platt_brier": float(brier_score_loss(labels, probs_platt)),
    }
    calib_path = SAVE_DIR / "calibration.json"
    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nCalibration params saved to: {calib_path}")

    # ── Instructions ──
    print("\n" + "=" * 60)
    print("HOW TO USE IN MERGER")
    print("=" * 60)
    print(f"""
In merger.py, change _batch_classify:

  # Temperature scaling (T={T:.4f}):
  logits = self.model.classifier(cls_a, cls_b, pw)
  probs = torch.sigmoid(logits / {T:.4f}).cpu().numpy()

  # OR Platt scaling (a={a:.4f}, b={b:.4f}):
  logits = self.model.classifier(cls_a, cls_b, pw)
  probs = torch.sigmoid({a:.4f} * logits + {b:.4f}).cpu().numpy()

Then sweep merge thresholds in main.py as usual.
""")


if __name__ == "__main__":
    main()
