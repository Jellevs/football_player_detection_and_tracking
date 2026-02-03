import argparse
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from .config import MergeClassifierConfig
from .model import SiameseTrackletMerger
from .dataset import (
    TrackletPairDataset,
    FeatureNormalizer,
    load_data,
    split_by_sequence,
)
from .evaluate import evaluate_model, find_optimal_threshold


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_samples = 0

    for frag_a, frag_b, pairwise, labels in dataloader:
        frag_a = frag_a.to(device)
        frag_b = frag_b.to(device)
        pairwise = pairwise.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frag_a, frag_b, pairwise)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        n_samples += len(labels)

    return total_loss / n_samples


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    n_samples = 0
    all_labels = []
    all_scores = []

    for frag_a, frag_b, pairwise, labels in dataloader:
        frag_a = frag_a.to(device)
        frag_b = frag_b.to(device)
        pairwise = pairwise.to(device)
        labels = labels.to(device)

        logits = model(frag_a, frag_b, pairwise)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        n_samples += len(labels)

        all_labels.append(labels.cpu().numpy())
        all_scores.append(torch.sigmoid(logits).cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_scores = np.concatenate(all_scores)

    avg_loss = total_loss / n_samples
    try:
        auc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auc = 0.0

    return avg_loss, auc


def train(config: MergeClassifierConfig, device: str = "cuda"):
    # Load data
    df, a_cols, b_cols, pw_cols, seq_labels = load_data(config.training_data_dir)

    # Update config dimensions from actual data
    config.fragment_input_dim = len(a_cols)
    config.pairwise_input_dim = len(pw_cols)

    # Split by sequence
    train_df, val_df, test_df = split_by_sequence(
        df, seq_labels, config.val_ratio, config.test_ratio
    )

    # Fit normalizer on training data only
    normalizer = FeatureNormalizer()
    normalizer.fit(train_df[a_cols + b_cols + pw_cols])

    # Create datasets
    train_dataset = TrackletPairDataset(
        train_df, a_cols, b_cols, pw_cols,
        normalizer=normalizer, augment_swap=config.augment_swap,
    )
    train_dataset.training_mode = True

    val_dataset = TrackletPairDataset(
        val_df, a_cols, b_cols, pw_cols,
        normalizer=normalizer, augment_swap=False,
    )
    val_dataset.training_mode = False

    test_dataset = TrackletPairDataset(
        test_df, a_cols, b_cols, pw_cols,
        normalizer=normalizer, augment_swap=False,
    )
    test_dataset.training_mode = False

    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0
    )

    # Compute class weight
    n_pos = train_df["label"].sum()
    n_neg = len(train_df) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"Class weight: pos_weight={pos_weight.item():.2f} (neg/pos = {n_neg}/{int(n_pos)})")

    # Model, loss, optimizer
    model = SiameseTrackletMerger(config).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.lr_factor, patience=config.lr_patience
    )

    # Print model summary
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {n_params:,} trainable parameters")
    print(f"Fragment input dim: {config.fragment_input_dim}")
    print(f"Pairwise input dim: {config.pairwise_input_dim}")
    print(f"Encoder: {config.fragment_input_dim} -> {' -> '.join(map(str, config.encoder_hidden_dims))}")
    print(f"Classifier: {config.encoder_hidden_dims[-1] * 4 + config.pairwise_input_dim} -> {' -> '.join(map(str, config.classifier_hidden_dims))} -> 1")
    print()

    # Training loop
    save_dir = Path(config.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(1, config.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)
        scheduler.step(val_auc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"lr={current_lr:.1e}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "normalizer": normalizer.state_dict(),
                    "best_val_auc": best_val_auc,
                    "epoch": epoch,
                    "a_columns": a_cols,
                    "b_columns": b_cols,
                    "pairwise_columns": pw_cols,
                },
                save_dir / "best_model.pt",
            )
            print(f"  -> New best model saved (AUC={best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={config.early_stopping_patience})")
                break

    # Load best model and evaluate on test set
    print(f"\n{'='*60}")
    print("Loading best model and evaluating on test set...")
    print(f"{'='*60}")

    checkpoint = torch.load(save_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Find optimal threshold on validation set
    val_loss, val_auc = validate(model, val_loader, criterion, device)
    val_labels, val_scores = get_predictions(model, val_loader, device)
    optimal_threshold = find_optimal_threshold(val_labels, val_scores)
    print(f"Optimal threshold (from val set): {optimal_threshold:.3f}")

    # Save threshold with checkpoint
    checkpoint["optimal_threshold"] = optimal_threshold
    torch.save(checkpoint, save_dir / "best_model.pt")

    # Evaluate on test set
    test_labels, test_scores = get_predictions(model, test_loader, device)
    test_metrics = evaluate_model(test_labels, test_scores, threshold=optimal_threshold)

    print(f"\nTest Results:")
    for k, v in test_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")

    # Save test metrics
    with open(save_dir / "test_metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float, np.floating))}, f, indent=2)

    return model, test_metrics


@torch.no_grad()
def get_predictions(model, dataloader, device):
    model.eval()
    all_labels = []
    all_scores = []

    for frag_a, frag_b, pairwise, labels in dataloader:
        frag_a = frag_a.to(device)
        frag_b = frag_b.to(device)
        pairwise = pairwise.to(device)

        logits = model(frag_a, frag_b, pairwise)
        scores = torch.sigmoid(logits).cpu().numpy()

        all_labels.append(labels.numpy())
        all_scores.append(scores)

    return np.concatenate(all_labels), np.concatenate(all_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Siamese Tracklet Merger")
    parser.add_argument("--data_dir", type=str, default="output/training_data",
                        help="Directory containing *_pairs.csv files")
    parser.add_argument("--save_dir", type=str, default="weights/merge_classifier",
                        help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = MergeClassifierConfig(
        training_data_dir=Path(args.data_dir),
        model_save_dir=Path(args.save_dir),
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print(f"Device: {args.device}")
    print(f"Data dir: {config.training_data_dir}")
    print(f"Save dir: {config.model_save_dir}")
    print()

    train(config, device=args.device)
