# A Plan for a Transformer Merger that Outperforms XGBoost

This plan is grounded in the current codebase: the winning `XGBoostMerger`
(`tracklets/xgboost_merger.py`), the temporal-bin bi-encoder
(`experiments/tracklet_merger/transformer/`), the unrun cross-encoder
(`experiments/tracklet_merger/frame_transformer/`), and the findings already
recorded in `experiments/tracklet_merger/transformer_ablation/EXPERIMENTS.md`.

Current standing: **XGBoost HOTA ≈ 90.7**, **best transformer HOTA = 87.87**
(uncalibrated, hard constraints ON, merge threshold 0.70).

---

## 1. Why the transformer loses today (root cause)

The most important fact is the one your own ablation log already captured:

> Transformer AUC (0.9586) **>** XGBoost AUC (0.9518), yet XGBoost HOTA is **2.88
> points higher**. AUC measures global ranking; HOTA depends on absolute
> probability precision at the clustering boundary.

Everything below follows from taking that observation seriously. The merger is
**not** a binary classifier — it is a clustering system. Both mergers convert a
pairwise probability to a distance `1 − p`, build a full distance matrix, and run
**average-linkage agglomerative clustering** at a fixed distance threshold
(`merger.py` / `xgboost_merger.py`, identical logic). HOTA is determined by
whether the *global per-sequence partition* is correct, not by whether pairs are
ranked well on a pooled, downsampled, synthetic-augmented validation set.

Five concrete mechanisms are hurting the transformer:

**1.1 Objective mismatch (the central problem).** Training optimizes per-pair
asymmetric focal loss and checkpoints on `val_auc` (`train.py`,
`if val_auc > best_auc`). HOTA is never in the loop. AUC and HOTA are proven to
diverge here, so you are selecting the wrong model.

**1.2 Average-linkage is fragile to a single overconfident edge.** Average
linkage chains: if `p(A,B)` and `p(B,C)` are high but A and C are different
players, the cluster can still absorb all three, producing a catastrophic
identity merge that destroys the HOTA association score (AssA). Global ranking
quality (AUC) does not protect against this; only *locally correct, well-separated*
affinities do. The transformer's smoother, less spiky probability surface is more
prone to this than XGBoost's sharp tree splits.

**1.3 Calibration at the boundary, not globally.** The best threshold being 0.70
(i.e. merge when `p > 0.30`) signals the transformer's positive probabilities are
compressed toward the middle — under-confident on true merges. Post-hoc Platt
scaling improved Brier/separation but **hurt** HOTA at every threshold, because
global calibration does not fix the *per-sequence, boundary-specific* geometry
that clustering needs.

**1.4 The bi-encoder fights XGBoost on its own turf — and loses.** The temporal-bin
model encodes each tracklet independently into one 96-d CLS vector
(`encode_tracklet`), then compares with an MLP. That is the *same*
compress-then-compare strategy as XGBoost, but XGBoost is handed the single most
discriminative signal directly and losslessly: `pairwise_reid_cosine_sim` and
`pairwise_siglip_cosine_sim` (top feature importances). The transformer must
re-derive an equivalent from a lossy `512 → 48 → bin-average → 96-d` path, adding
noise to a signal XGBoost already has perfectly. It has *less* useful inductive
bias, not more.

**1.5 Synthetic positives are too easy and shift the score distribution.**
`generate_train_data.py` builds synthetic positives by splitting a pure tracklet
at a random interior point. Those two halves have near-identical appearance, a
near-zero temporal gap, the same jersey/team, and the same spatial location at the
cut. But the *real* positives at inference are fragments produced by the
**splitter**, which cuts precisely where appearance/identity is ambiguous — so
real positives have appearance drift, real temporal gaps, and occlusion. The model
learns "positive = looks identical and adjacent," is over-optimistic, and at
inference the genuinely hard real positives look more like negatives → score
compression → wrong boundary. Even capped at 2× real positives, these pollute
calibration.

**1.6 Data volume.** Your Experiment 4 showed that shrinking to XGBoost's data
config (4,403 pairs) collapsed transformer AUC from 0.9586 to 0.8488. The neural
net is data-hungry where XGBoost is sample-efficient on tabular features. The fix
is *more, better* data — not less.

---

## 2. Strategy

Three pillars, each attacking the root cause directly:

- **A — Data and labels that match the inference clustering distribution.**
- **B — Give the transformer XGBoost's strong signal *plus* a capability XGBoost
  cannot have (cross-tracklet interaction).**
- **C — Train, calibrate, and select against the clustering objective, not AUC.**

The guiding principle: *stop optimizing a proxy (pairwise AUC) and start
optimizing the thing that produces HOTA (the partition).*

---

## 3. Pillar A — Data generation redesign

**A1. Make synthetic positives look like real splitter output.** Instead of random
interior cuts, generate synthetic positives that mimic what the merger actually
sees:

- Cut at the frames of **largest ReID drift** within a pure tracklet (where the
  splitter would cut), so fragments carry realistic appearance gaps.
- Inject a **temporal gap** by deleting a random run of frames (e.g. 5–60) at the
  cut, simulating the occlusion/miss interval real merges must bridge.
- Sample fragment lengths to match the **empirical post-split length
  distribution** (many short fragments), not 50/50 halves.
- Verify success quantitatively: the score histogram of *real* positives and
  *synthetic* positives should overlap. If synthetics sit far to the right, they
  are still too easy.

**A2. Hard-negative mining (highest-value data change).** The merges that wreck
HOTA are false links between *same-team, plausibly-adjacent, appearance-similar
different players*. Today negatives are randomly downsampled to 3:1. Instead, mine
and oversample negatives that are hard: same predicted team, small temporal gap,
small spatial distance, **high** reid/siglip cosine similarity, but different GT
id. Hold an explicit hard:easy ratio. This teaches exactly the boundary the
clustering relies on and directly suppresses chaining (1.2).

**A3. Keep inference-distribution matching, isolate the variable.** The transformer
already trains on post-split tracklets (good — keep it). First, run a clean A/B:
train the *current* model but selected by HOTA (Pillar C) on the current data;
this tells you how much of the gap is selection vs data vs model before you change
three things at once.

**A4. Increase volume.** Generate more real-like positives by (a) enumerating all
GT-consistent non-overlapping tracklet pairs across the full train split, (b)
producing multiple post-split fragmentations using a few splitter configs, and (c)
sub-segmenting long GT-consistent groups. Aim well above the current pair count —
1.6 shows the model wants data.

**A5. Cleaner labels.** Purity 0.80 is reasonable. Tighten to ~0.9 for *positive*
eligibility (clean merges only) but route confidently-different impure tracklets
into the *hard-negative* pool rather than dropping them, instead of the current
"unknown → label 0" which injects noise into easy negatives.

---

## 4. Pillar B — Model

**B1. Make the model "XGBoost features + a learned correction" (low risk).** The
classifier head already receives the 34-d extended pairwise vector. Guarantee it
cannot underperform the proven signal: structure the head so that with the learned
(CLS-derived) terms zeroed it reproduces a logistic-regression-on-pairwise
baseline, and let the transformer learn only the residual. The model then *starts*
at the linear-baseline operating point and can only add value — it can never throw
away the cosine-similarity signal the way it implicitly can today.

**B2. Ensemble with XGBoost (near-guaranteed win, do this first).** Blend
transformer and XGBoost probabilities in **logit space**, then sweep the blend
weight and merge threshold against HOTA. Because XGBoost is already strong and the
transformer makes *different* errors, the ensemble almost certainly beats either
alone. This is the lowest-risk route to "a transformer in the loop beats XGBoost,"
and it is a safe fallback if the standalone model never crosses 90.7.

**B3. Late cross-attention bi-encoder (the recommended architecture).** This is the
sweet spot between the current bi-encoder and the expensive frame_transformer:

1. Encode each tracklet **once** into its compact token set (reuse the 11
   temporal-bin/boundary/stats tokens), and cache them — preserving the O(N)
   encode-once efficiency of the current merger.
2. For each candidate pair, run **2 cross-attention layers** over the concatenated
   ~22 tokens + a pair-CLS, so A's tokens attend to B's tokens and vice versa.

This adds genuine cross-tracklet interaction — the model can match *the one clear
jersey frame in A to the one clear frame in B* — which XGBoost (mean cosine sim
only) fundamentally cannot do. Because the per-pair sequence is ~23 tokens, the
O(N²) pair cost is trivial. It strictly dominates the current bi-encoder's
information at near-identical cost.

**B4. Stop asking the transformer to re-learn appearance similarity.** Keep the
exact cosine-sim signal as a privileged input (B1) and let the attention layers
specialize in temporal/contextual reasoning (drift patterns, gap plausibility,
boundary frames) — the part XGBoost's hand-crafted aggregates handle poorly.

**B5. Full frame-level cross-encoder — only if B3 plateaus.** The unrun
`frame_transformer` concatenates `[CLS] + frames_A + frames_B` with full
self-attention and per-frame cross-attention — the maximal version of B3. Before
running it, fix three things: (i) inference cost is O(N²_pairs × frames²) with no
encode-once caching — restrict attention to boundary/keyframe subsets or it will
be far slower than XGBoost; (ii) port the temporal-bin model's loss, augmentation,
and calibration improvements (the frame_transformer config lacks them); (iii)
select by HOTA. Treat this as the high-ceiling, high-cost option, not the default.

---

## 5. Pillar C — Training, calibration, clustering-aware selection

**C1. Select by HOTA (or a clustering proxy), not AUC — the single highest-leverage
change.** Add an in-loop evaluation that every K epochs runs the *actual* merge
(`merger.merge`) on a few held-out validation sequences and computes HOTA, or a
fast surrogate (V-measure / per-sequence clustering accuracy vs GT partition).
Checkpoint and early-stop on that. This aligns model selection with the real
objective and, on its own, may recover a meaningful slice of the gap.

**C2. Structure-aware loss, beyond per-pair focal.**

- **Supervised contrastive / triplet** on tracklet CLS embeddings within a
  sequence: pull same-identity tracklets together, push different apart. This
  shapes the distance geometry that linkage consumes, improving clustering even
  where pairwise labels are ambiguous.
- **Boundary-hardness weighting**: reweight the existing asymmetric focal loss
  toward the hard-mined pairs from A2.
- **Transitivity penalty (stretch)**: within a per-sequence minibatch, penalize
  `p(A,B)·p(B,C)·(1−p(A,C))`-style violations to discourage the chaining failures
  average linkage exploits (1.2).
- **Listwise / graph objective (stretch, highest ceiling)**: sample a whole
  sequence, form the predicted affinity matrix, and apply a differentiable
  clustering relaxation or a partition loss vs the GT grouping — training directly
  on the decision structure.

**C3. Calibration that helps clustering (global Platt failed for a reason).**

- **Per-sequence affinity normalization**: rank-normalize or z-score the distance
  matrix *within each sequence* before `fcluster`. This targets the exact
  "absolute precision at the boundary" problem and is robust to scene-to-scene
  scale shifts that a single global threshold cannot handle.
- **Sweep threshold *and* linkage against HOTA** (you already have sweep infra in
  `experiments/xgboost/` — replicate it for the transformer). Try `complete` and
  `ward` linkage, which resist chaining, against `average`.
- Apply temperature scaling **only** if it improves HOTA in that sweep, and always
  **re-sweep the threshold afterward** — scaling with the old fixed threshold is
  precisely what failed in Experiment 5.

**C4. Train/inference parity.** The training set applies heavy augmentation
(frame dropout, noise, modality mask, temporal crop) while inference encodes clean
features once — a distribution shift in the CLS. Consider light test-time
augmentation (encode 2–3 augmented views, average the CLS) and/or weight EMA for a
smoother, better-calibrated final model.

---

## 6. Phased roadmap (ROI-ranked)

**Phase 0 — Instrumentation (prerequisite).** Add in-loop HOTA/clustering-proxy
eval and HOTA-based checkpointing (C1). Replicate the threshold/linkage sweep for
the transformer (C3). Re-evaluate the *current* model selected by HOTA to get an
honest baseline (A3).

**Phase 1 — Ensemble (near-guaranteed win).** Blend transformer ⊕ XGBoost in logit
space; sweep blend + threshold by HOTA (B2). Establishes "transformer-in-the-loop
> XGBoost" immediately and gives a fallback.

**Phase 2 — Data (high ROI).** Hard-negative mining (A2) + realistic synthetic
positives (A1) + more volume (A4) + cleaner labels (A5). Retrain bi-encoder with
HOTA selection; re-sweep.

**Phase 3 — Model (the genuine edge).** Late cross-attention bi-encoder (B3) +
privileged cosine signal (B1, B4) + supervised-contrastive auxiliary loss (C2).
This is the architecture most likely to truly exceed XGBoost.

**Phase 4 — Clustering-aware training (high ceiling, stretch).** Per-sequence
affinity normalization (C3) and/or graph/listwise clustering loss (C2).

**Phase 5 — Full frame cross-encoder (only if Phase 3 plateaus and cost allows).**
Fix efficiency + port improvements + HOTA selection for `frame_transformer` (B5).

---

## 7. Success criteria, ablations, and pitfalls

**Headline metric:** end-to-end HOTA on the held-out split, target **> 90.7**.
Report HOTA, not AUC; keep AUC only as a sanity diagnostic.

**Decompose HOTA into DetA vs AssA.** The merger should move **AssA** (association);
also track IDSW and fragmentation. If AssA rises while IDSW stays low, the merger
is doing its job without over-merging.

**Ablation ladder (each isolates one change):** HOTA-selection alone → +hard
negatives → +realistic synthetics → +privileged cosine head → +cross-attention →
+contrastive loss → +per-sequence normalization → ensemble. This produces a clean
contribution story for the thesis.

**Pitfalls to watch:**

- Do not trust val AUC for model selection — it is anti-correlated with HOTA here.
- One overconfident false edge under average linkage = catastrophic merge; hard
  negatives + linkage choice + per-sequence normalization mitigate it.
- Synthetic positives quietly dominate calibration; keep them realistic, capped,
  and verify their score distribution overlaps real positives.
- Guard inference cost: prefer the late-cross-attention hybrid over the full
  frame_transformer; keep encode-once caching.
- Keep the sequence-disjoint train/valid/test split (already the case via separate
  SoccerNet folders) to avoid leakage.

---

## 8. One-paragraph summary

The transformer is losing not because it is weaker, but because it is optimized
and selected for the wrong objective. It has higher AUC yet lower HOTA because the
merger is a clustering system whose quality depends on locally correct, well-
calibrated, well-separated affinities — not global ranking. The fastest path to
beating XGBoost is to (1) select and tune by HOTA instead of AUC, (2) ensemble with
XGBoost as an immediate safety win, (3) feed the model XGBoost's privileged cosine
signal and have it learn only the residual, (4) train on hard-mined negatives and
realistic synthetic positives that match the inference distribution, and (5) add a
cheap late cross-attention stage so the model gains a capability XGBoost cannot
have: directly comparing frames across the two tracklets.
