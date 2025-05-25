import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


file_list = [
    "Cri",
    "San_Diego",  # 'abu-beach-1', 'abu-urban-1', 'aviris_1', 'aviris_2',
]

base_dir = "../Results"
cmp_dir = os.path.join(base_dir, "Crioutput")
os.makedirs(cmp_dir, exist_ok=True)

# --- Main Processing Loop ---
for ds in file_list:
    print(f"Processing: {ds}")

    mat_path = os.path.join(base_dir, f"{ds}", "residuals_best.mat")
    if not os.path.isfile(mat_path):
        print(f"  SKIPPED: Missing file {mat_path}")
        continue

    # Load residual map and ground truth
    data = sio.loadmat(mat_path)
    print(data.keys())
    residual = data["residual_map"]  
    print(residual.shape)
    gt_flat = data["gt_mask"].ravel().astype(int)
    print(gt_flat.shape)
    original = data["original"]
    print(original.shape)
    scores = residual.ravel().astype(np.float32)
    fpr, tpr, _ = roc_curve(gt_flat, scores)
    auc = roc_auc_score(gt_flat, scores)

    # --- Plot ROC Curve ---
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f" ({auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.title(f"ROC Curve — {ds}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize="small")
    plt.grid(True)
    plt.tight_layout()
    out_roc = os.path.join(cmp_dir, f"{ds}_roc.png")
    plt.savefig(out_roc, dpi=200)
    plt.close()
    print(f"  Saved ROC: {out_roc}")

    # --- Boxplots: Background vs Anomaly Scores ---
    bg_scores = scores[gt_flat == 0]
    an_scores = scores[gt_flat == 1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    axes[0].boxplot(bg_scores, showfliers=False, patch_artist=True)
    axes[0].set_title("Background Scores")
    axes[0].set_xticks([])

    axes[1].boxplot(an_scores, showfliers=False, patch_artist=True)
    axes[1].set_title("Anomaly Scores")
    axes[1].set_xticks([])

    fig.suptitle(f"Score Distribution — {ds}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_box = os.path.join(cmp_dir, f"{ds}_boxplot.png")
    fig.savefig(out_box, dpi=200)
    plt.close()
    print(f"  Saved Boxplot: {out_box}")

    # --- Heatmap ---
    heat_img = residual
    plt.figure(figsize=(6, 5))
    plt.imshow(heat_img, cmap="jet", vmin=0, vmax=heat_img.max())
    plt.colorbar(label="Anomaly Score")
    plt.title(f"Anomaly Heatmap — {ds}")
    plt.axis("off")
    plt.tight_layout()
    heat_out = os.path.join(cmp_dir, f"{ds}_heatmap.png")
    plt.savefig(heat_out, dpi=200)
    plt.close()
    print(f"  Saved Heatmap: {heat_out}")

print("All analysis completed.")
