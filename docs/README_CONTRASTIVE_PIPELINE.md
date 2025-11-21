# Contrastive Band-Subset Pipeline Report

This document summarizes the end-to-end workflow you now have for analysing hyperspectral anomalies via contrastive Integrated Gradients (IG), selecting discriminative spectral bands, validating those selections, and visualizing the resulting anomaly signals.

The pipeline is implemented entirely inside `src/` and produces all artifacts under `src/analysis_results/` unless otherwise noted.

---

## 1. Baseline Model & Dataset

- **Dataset**: Gulfport hyperspectral cube (`data/gulfport.mat`) loaded via `HSIDataset`.
- **Baseline model checkpoint**: `artifacts/models/final_wacv/gulfport_full_simple_fusion_1dec_fixed/best_model_auc.pt`.
- **Purpose**: Serves as the shared model for IG attribution, band selection, and all downstream validations.

---

## 2. Background Band Importance (Normal Regions)

**Script**: `src/integrate_gradient.py` (not modified here, provided in repo)

**Outputs** (`src/band_analysis_results/`):
- `aggregated_normalized_importance.npy`: average per-band IG importance over background patches.
- `sampled_background_indices.npy`: indices of background patches used to estimate the baseline spectrum.

These outputs feed directly into the contrastive analysis described next.

---

## 3. Contrastive Integrated Gradients (Anomaly vs Background)

**Script**: `src/contrastive_validation.py`

**Key steps**:
1. Load the baseline model and Gulfport dataset.
2. Sample anomaly patches with GT support.
3. Run IG relative to the background baseline, normalizing by per-band standard deviation.
4. Compute `delta = anomaly_mean - background_mean` to measure discriminativeness.

**Outputs**:
- `src/band_analysis_results/anomaly_per_patch_normalized_importance.npy`
- `src/band_analysis_results/anomaly_aggregated_normalized_importance.npy`
- `src/band_analysis_results/contrastive_delta.npy`
- `src/analysis_results/contrastive_delta_rankings.csv`: per-band Δ, anomaly score, background score.
- `src/analysis_results/contrastive_delta.png`: bar plot with top-|Δ| annotations.
- `src/analysis_results/contrastive_validation.json`: structured summary (top bands, counts, file refs).

Use `contrastive_delta_rankings.csv` as the canonical source for all later subsets.

---

## 4. Band Subset Dataset Generation & Training

**Script**: `src/band_subset_expermentation.py`

**Features**:
- Samples one or more `top_k` values and builds reduced hyperspectral datasets that keep only the selected bands (top-Δ anomaly-focused or bottom-Δ background-focused).
- Trains fresh `AnomalyDetectionModel` instances on each subset.
- Logs AUC results to `src/analysis_results/band_subset_auc_log.csv`.

**Usage**:
```
python src/band_subset_expermentation.py \
  --top-k-values 10,20 \
  --epochs 40 \
  --batch-size 128
```

Outputs include per-run checkpoints under `artifacts/models/current/<dataset>_anomaly_subset_k*_...` and `artifacts/models/current/<dataset>_background_subset_k*_...` plus log entries in `band_subset_auc_log.csv`.

---

## 5. Visualizing Band Subsets

**Script**: `src/visualize_band_subsets.py`

**Purpose**: Render mean intensity and pseudo-RGB views after sub-band selection.

**Command example**:
```
python src/visualize_band_subsets.py \
  --subset-type anomaly_positive \
  --top-k 10
```

**Artifacts** (`src/analysis_results/`):
- `gulfport_anomaly_positive_k10_10bands_mean_intensity.png`
- `gulfport_anomaly_positive_k10_10bands_pseudo_rgb.png`
- Equivalent files for other subsets (top_abs, background_negative, manual lists).

---

## 6. Model-Based Anomaly Heatmaps (Baseline Model)

**Script**: `src/visualize_anomaly_heatmap.py`

**Workflow**:
1. Load the original full-spectrum checkpoint (default).
2. Mask the input cube to keep only the desired bands.
3. Reconstruct the entire image with the baseline model.
4. Compute per-pixel reconstruction error (anomaly heatmap) and ROC AUC vs. ground truth.

**Example**:
```
python src/visualize_anomaly_heatmap.py \
  --subset-type anomaly_positive \
  --top-k 10 \
  --output src/analysis_results/gulfport_anomaly_k10_heatmap.png
```

**Outputs**:
- Heatmap figures such as `gulfport_anomaly_k10_heatmap.png`, `gulfport_anomaly_k20_heatmap.png`.
- Console logs showing the ROC AUC for each visualization (e.g., `[INFO] ROC AUC ...`).

Use these to demonstrate that the top-Δ bands alone still highlight the anomaly regions when passed through the baseline model.

---

## 7. AUC Sweep Across Band Counts (+ Heatmaps)

**Script**: `src/eval_band_subset_auc.py`

**Purpose**: Quantitatively compare reconstruction-error AUC for:
- Full 191-band spectrum.
- Top-Δ subsets for multiple `k`.
- Bottom-Δ subsets (sanity check).

**Command**:
```
python src/eval_band_subset_auc.py \
  --top-k-values 10,20,30
```

**Outputs**:
- `src/analysis_results/band_subset_auc_summary.csv`: tabular summary of AUC per subset.
- `src/analysis_results/band_subset_auc_plot.png`: “AUC vs. # Bands” chart with full-spectrum reference.
- `src/analysis_results/band_subset_heatmaps/*.png`: normalized reconstruction-error maps for each subset (e.g., `top_delta_k10.png`, `bottom_delta_k20.png`, `full_191bands.png`).

These results form the core empirical validation that top-Δ bands retain detection power while bottom-Δ bands collapse.

---

## 8. Putting It All Together (Recommended Report Flow)

1. **Dataset & Model**: Describe Gulfport and the baseline model architecture/checkpoint.
2. **Background IG**: Note how background bands were profiled to build the baseline spectrum.
3. **Contrastive IG**: Present `contrastive_delta.png/csv`, highlighting the top anomaly-sensitive bands.
4. **Visual Evidence**:
   - Mean/pseudo-RGB images from `visualize_band_subsets.py`.
   - Reconstruction-error heatmaps from `visualize_anomaly_heatmap.py` and `band_subset_heatmaps/`.
5. **Quantitative Validation**:
   - AUC logs from `band_subset_expermentation.py` (new models).
   - Full-spectrum vs. top/bottom `k` AUC sweep from `eval_band_subset_auc.py`.
   - Emphasize that top-Δ subsets stay near baseline AUC, while bottom-Δ subsets degrade sharply.
6. **Deliverables**: Reference all saved files (CSV, PNGs, JSON) for reproducibility.

---

## Quick Reference of Key Outputs

| Stage | File(s) |
| --- | --- |
| Background IG | `src/band_analysis_results/aggregated_normalized_importance.npy`, `sampled_background_indices.npy` |
| Contrastive Δ | `src/analysis_results/contrastive_delta_rankings.csv`, `contrastive_delta.png`, `contrastive_validation.json` |
| Subset Training Log | `src/analysis_results/band_subset_auc_log.csv` |
| Subset Visuals | `src/analysis_results/gulfport_anomaly_positive_k*_mean_intensity.png`, `*_pseudo_rgb.png` |
| Heatmaps (baseline) | `src/analysis_results/gulfport_anomaly_k*_heatmap.png` |
| AUC Sweep Summary | `src/analysis_results/band_subset_auc_summary.csv`, `band_subset_auc_plot.png`, `band_subset_heatmaps/*.png` |

---

With these scripts and artifacts, you can document the entire progression: baseline model → IG-based band discovery → visual inspection → quantitative validation. Each stage has reproducible CLI commands and saved outputs for inclusion in papers or supplementary material.
