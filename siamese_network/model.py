import torch
import torch.nn as nn


class FragmentEncoder(nn.Module):
    """Shared encoder that maps fragment features to a compact embedding."""

    def __init__(self, input_dim=41, hidden_dims=(64, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        # Remove dropout after last layer
        layers = layers[:-1]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SiameseTrackletMerger(nn.Module):
    """
    Siamese MLP for tracklet merge classification.

    Takes two fragment feature vectors and pairwise features,
    outputs a logit indicating merge probability.
    """

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            from .config import MergeClassifierConfig
            config = MergeClassifierConfig()

        self.encoder = FragmentEncoder(
            input_dim=config.fragment_input_dim,
            hidden_dims=config.encoder_hidden_dims,
            dropout=config.encoder_dropout,
        )

        # Combination: [emb_a, emb_b, |emb_a - emb_b|, emb_a * emb_b] + pairwise
        encoder_out_dim = config.encoder_hidden_dims[-1]
        combined_dim = encoder_out_dim * 4 + config.pairwise_input_dim

        # Classification head
        layers = []
        prev_dim = combined_dim
        dropouts = [config.classifier_dropout, config.encoder_dropout]
        for i, dim in enumerate(config.classifier_hidden_dims):
            drop = dropouts[i] if i < len(dropouts) else config.encoder_dropout
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(drop),
            ])
            prev_dim = dim
        # Remove dropout after last hidden layer
        layers = layers[:-1]
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        self.classifier = nn.Sequential(*layers)

    def forward(self, fragment_a, fragment_b, pairwise):
        emb_a = self.encoder(fragment_a)
        emb_b = self.encoder(fragment_b)

        combined = torch.cat([
            emb_a,
            emb_b,
            torch.abs(emb_a - emb_b),
            emb_a * emb_b,
        ], dim=-1)

        merged = torch.cat([combined, pairwise], dim=-1)
        logits = self.classifier(merged)
        return logits.squeeze(-1)
