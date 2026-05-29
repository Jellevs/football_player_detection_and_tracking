# Chat Summary: Siamese CLS No-Pairwise Ablation

## Context
Working on master's thesis about tracklet splitting/connecting for multi-object tracking. The transformer connector experiments section (`transformer_experiments.tex`) evaluates three transformer architectures (Siamese CLS, Cross-Attention, Hybrid) for predicting whether two tracklets should be merged.

## Key Discussion

All three transformer architectures currently receive **12 handcrafted pairwise features** in their classification head (temporal gap, endpoint distance, ReID cosine similarity, jersey match, etc.) — the same features used by the XGBoost connector. The Pairwise MLP baseline (which uses ONLY these 12 features, no frame-level attention) outperforms all transformers.

**Question raised:** Should we test the transformer without pairwise features? The current design gives the transformer "the answer sheet alongside the exam" — the classification head gets handcrafted summaries of exactly the patterns the attention layers should be learning.

**Decision:** Yes — it's scientifically more honest to present this as the natural first attempt. The narrative becomes:
1. We designed transformers to learn from raw frame sequences (no pairwise features)
2. They struggled → here's how they performed
3. We hypothesized limited data prevents learning these patterns from scratch
4. We added the 12 pairwise features as a bridge → performance improved
5. This confirms the data bottleneck

## Implementation Done

Three files modified/created in `experiments/tracklet_merger/transformers/siamese_cls/`:

### 1. `config.py` — Added flag
```python
use_pairwise_features: bool = True  # if False, classification head ignores pairwise features
```

### 2. `modules.py` — Modified `ClassificationHead`
- Reads `config.use_pairwise_features`
- When False: input_dim = `d_model * 4` (320) instead of `d_model * 4 + 12` (332)
- The pairwise tensor is still passed through forward() but not concatenated

### 3. `train_no_pairwise.py` — New training script
- Sets `TransformerMergerConfig(use_pairwise_features=False)`
- Uses `xgb_matched` data strategy (same training data, just no pairwise features in model)
- Saves to `output/siamese_cls_no_pairwise/`
- Imports helper functions from `train.py` and monkey-patches module-level paths
- Run with: `python -m experiments.tracklet_merger.transformers.siamese_cls.train_no_pairwise`

## Open Question
The script currently uses `xgb_matched` data strategy. The best Siamese CLS result in the thesis used `real-synth` (test AUC 0.867). Need to decide which data strategy to use for a fair comparison. Should probably match whatever we compare against.

## Existing Results (for reference)
- Siamese CLS + real-synth + pairwise features: Val AUC 0.884, Test AUC 0.867
- Siamese CLS + real-only + pairwise features: Val AUC 0.864, Test AUC 0.845
- Pairwise MLP (12 features only, no attention): Val AUC 0.902, Test AUC 0.885
- Cross-Attention + real-synth: Val AUC 0.901, Test AUC 0.882
