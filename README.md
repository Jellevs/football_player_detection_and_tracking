# ALERT: attribute-augmented learned refinement for tracking

A post-processing module for multi-object tracking in football broadcast video. It takes the
output of any tracking-by-detection tracker, cuts tracklets at the frames where the identity
changes, and reconnects the fragments that belong to the same player.

Master's thesis, Eindhoven University of Technology, 2026.

## Results

Tracking performance on the SoccerNet Player-Tracking test set. DetA is roughly constant at
93.59 across methods, so the differences come from association.

| Method | HOTA | AssA | IDF1 |
|---|---|---|---|
| Deep-EIoU (no post-processing) | 83.34 | 74.24 | 83.56 |
| Deep-EIoU + GTA | 85.91 | 78.87 | 88.78 |
| Rule-based connector | 88.61 | 83.88 | 92.33 |
| **XGBoost connector** | **89.86 ± 0.11** | **86.31 ± 0.22** | **94.31 ± 0.17** |
| Transformer connector | 87.70 ± 0.21 | 82.20 ± 0.39 | 91.22 ± 0.11 |
| Oracle connector | 92.13 | 90.71 | 97.23 |

The XGBoost connector adds 6.52 HOTA over the raw Deep-EIoU tracker and 3.95 over GTA, the
strongest published refinement baseline. The oracle connector, which connects fragments using
ground-truth identities, sits 2.27 HOTA above it and bounds what any connector can reach on
these input tracklets.

XGBoost and the transformer are reported as mean and standard deviation over 5 and 3 training
seeds. The other methods are deterministic.

## How it works

Two stages run on the tracker output.

**Split.** Four independent signals scan each tracklet for the frame where the identity changes.
STReID combines the temporal gap with ReID cosine distance. The jersey and team splitters look
for a change in the predicted attribute. The bounding-box splitter flags velocity anomalies with
a z-score. A tracklet is cut wherever any signal fires.

**Connect.** Every pair of fragments is scored by a connector, which turns the probability into a
distance `d = 1 - p` and runs average-linkage agglomerative clustering at threshold 0.80. Three
connectors are implemented: a rule-based decision tree, an XGBoost classifier over 47 pairwise
features, and a transformer. An oracle connector using ground-truth identities gives the ceiling.

Both stages consume player attributes predicted per frame:

- **Jersey number.** ViTPose-Base crops the torso, a ResNet-34 classifier filters illegible crops,
  and PARSeq reads the number. Shannon entropy over the character distribution gives the confidence.
- **Team.** SigLIP-Base patch16-224 embeds each crop, UMAP reduces to 3 dimensions with cosine
  distance, and KMeans with k=2 separates the teams. Distance to the centroid gives the confidence.

## Layout

```
main.py                     end-to-end pipeline over one sequence
settings.py                 paths, splits, sequence lists
convert_gsr_to_snp.py       builds the SNPT dataset from SoccerNet GSR

detect_and_track/           Deep-EIoU tracker with OSNet-x1.0 ReID
attributes/                 jersey number and team prediction
tracklets/
  split_tracklets.py        splitting entry point
  splitters/                STReID, unified jersey/team, bbox anomaly, trajectory, GTA
  connectors/               rule-based, XGBoost, transformer, MLP, oracle, majority vote
    models/                 transformer architectures (siamese CLS, cross-attention, hybrid)
utils/                      config, MOT output, evaluation, visualisation
```

## Dataset

SNPT (SoccerNet Player-Tracking) is derived from SoccerNet Game State Reconstruction, restricted
to player tracks and converted to MOT-Challenge format: 57 train, 58 validation, 49 test sequences.
`convert_gsr_to_snp.py` performs the conversion and the SNGS-XXX to SNPT-XXX renaming.

## Setup

```bash
uv sync
uv pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
```

PARSeq and centroids-reid are vendored under `attributes/jersey_number/`, so they need no
separate install. For evaluation, clone [SoccerNet TrackEval](https://github.com/SoccerNet/sn-trackeval)
into `sn-trackeval/`.

Place model weights under `weights/`, with the trained connectors in `weights/connectors/`.
Set the data and output roots in `settings.py`.

## Running

```bash
python main.py
```

This tracks every sequence under the configured data root, predicts attributes, splits, connects
with the XGBoost connector, writes MOT files, and calls TrackEval. The split output is cached per
sequence under `cache_split/`, so re-running with a different connector skips the expensive stages.

To use a different connector, set `CONNECTOR` in `settings.py` to one of `xgboost`,
`transformer`, `cross_attention`, `hybrid`, `pairwise_mlp`, `decision` or `oracle`. Each one's
weights and threshold are listed under `CONNECTORS` in the same file.

## Evaluation

`utils/run_evaluation.py` wraps TrackEval. With `sn-trackeval/` cloned into the repo root, it can
also be called directly:

```bash
python sn-trackeval/scripts/run_mot_challenge.py \
  --BENCHMARK SNMOT --SPLIT_TO_EVAL SNPT-test \
  --GT_FOLDER <data_root>/test \
  --TRACKERS_FOLDER <output_root>/evaluation/SNPT \
  --TRACKERS_TO_EVAL <method_name> \
  --SEQMAP_FILE <output_root>/evaluation/seqmaps/SNPT-test.txt \
  --METRICS HOTA CLEAR Identity \
  --DO_PREPROC False --USE_PARALLEL False \
  --TRACKER_SUB_FOLDER data --SKIP_SPLIT_FOL True
```

## Notes

This repository holds the final pipeline. The experiment scripts that produced the thesis tables
and figures, along with the training data generation, hyperparameter sweeps and ablations, are not
included.

Built on [Deep-EIoU](https://github.com/hsiangwei0903/Deep-EIoU),
[GTA-Link](https://github.com/sjc042/gta-link),
[SoccerNet TrackEval](https://github.com/SoccerNet/sn-trackeval),
[PARSeq](https://github.com/baudm/parseq),
[ViTPose](https://github.com/ViTAE-Transformer/ViTPose) and
[Jersey Number Detection](https://github.com/mkoshkina/jersey-number-pipeline), whose jersey
pipeline this one follows closely.
