# Transformer Ablation Experiments

Record all HOTA results here after each run.

## Experiment Matrix

| # | Variant | Model | Calibration | Hard Constraints | Threshold | Val HOTA | Notes |
|---|---------|-------|-------------|-----------------|-----------|----------|-------|
| 1 | Baseline | temporal_bin_transformer | none | jersey+team ON | 0.50 | 87.427 | Original config |
| 2 | Baseline | temporal_bin_transformer | none | jersey+team ON | 0.70 | 87.87 | Best uncalibrated |
| 3 | Baseline, no HC | temporal_bin_transformer | none | jersey+team OFF | 0.70 | 87.4 | HC helps +0.47 |
| 4 | XGB matched data | transformer_xgb_matched | none | jersey+team OFF | sweep | abandoned | val AUC 0.8488, data starvation |
| 5a | Platt calibrated | temporal_bin_transformer | Platt(2.03,-1.57) | jersey+team ON | 0.50 | 86.089 | Calibration hurts |
| 5b | Platt calibrated | temporal_bin_transformer | Platt(2.03,-1.57) | jersey+team ON | 0.35 | 84.9 | Worse with restrictive threshold |
| 5c | Platt calibrated | temporal_bin_transformer | Platt(2.03,-1.57) | jersey+team ON | 0.70 | 87.09 | Still below uncalibrated |
| 5d | Platt calibrated | temporal_bin_transformer | Platt(2.03,-1.57) | jersey+team ON | 0.80 | 87.475 | Best Platt, still below 87.87 |

## Key Findings

1. **Best transformer HOTA: 87.87** (uncalibrated, HC ON, threshold 0.70) vs **XGBoost HOTA: 90.75**
2. **Calibration does not help.** Platt scaling improves validation metrics (Brier 0.102 → 0.057, separation 0.498 → 0.697) but strictly decreases end to end HOTA at every threshold tested.
3. **Hard constraints help the transformer** (+0.47 HOTA), opposite of XGBoost where they hurt.
4. **Data volume matters.** Matching XGBoost's data config (4,403 pairs) collapsed transformer val AUC from 0.9586 to 0.8488. Neural networks need more training pairs.
5. **Higher AUC does not imply higher HOTA.** Transformer AUC (0.9586) > XGBoost AUC (0.9518), yet XGBoost HOTA is 2.88 points higher. AUC measures global ranking; HOTA depends on absolute probability precision at the clustering boundary.

## How to Run Each Experiment

### Experiment 3: Baseline model, no hard constraints
No retraining needed. In main.py:
```python
transformer_merger = TemporalBinMerger(
    model_path=r"...\weights\temporal_bin_transformer\best_model.pt",
    meta_path=r"...\weights\temporal_bin_transformer\meta.json",
    merge_threshold=0.70,
    linkage_method="average",
    use_jersey_constraint=False,
    use_team_constraint=False,
    device=None)
```

### Experiment 4: XGBoost matched training data
1. Generate data (run 3 times, changing SPLITS_TO_PROCESS and SPLIT_OUTPUT_NAME):
   ```
   python experiments/tracklet_merger/transformer_ablation/generate_data_xgb_matched.py
   ```
2. Train:
   ```
   python experiments/tracklet_merger/transformer_ablation/train_xgb_matched.py
   ```
3. In main.py, point to `weights/transformer_xgb_matched/best_model.pt`

### Experiment 5: No SigLIP
1. Same data as experiment 4 (already generated)
2. Set `MASK_SIGLIP = True` in train_xgb_matched.py, then run it
3. In main.py, point to `weights/transformer_xgb_matched_no_siglip/best_model.pt`
