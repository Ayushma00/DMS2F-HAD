#!/usr/bin/env python3
"""
Quick visualization script for ablation study results
Run this after your experiments complete to generate essential plots
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def quick_visualize_experiment(experiment_path):
    """Quick visualization for a single experiment"""
    
    residuals_file = os.path.join(experiment_path, "residuals_best.mat")
    if not os.path.exists(residuals_file):
        return None
    
    experiment_name = os.path.basename(experiment_path)
    
    try:
        # Load data
        data = sio.loadmat(residuals_file)
        residual_map = data["residual_map"]
        gt_mask = data["gt_mask"]
        
        # Calculate AUC
        gt_flat = gt_mask.ravel().astype(int)
        scores = residual_map.ravel().astype(np.float32)
        
        if len(np.unique(gt_flat)) <= 1:
            return None
        
        fpr, tpr, _ = roc_curve(gt_flat, scores)
        auc = roc_auc_score(gt_flat, scores)
        
        # Create visualizations directory
        viz_dir = os.path.join(experiment_path, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. ROC Curve
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC - {experiment_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{experiment_name}_roc.png"), dpi=200, bbox_inches='tight')
        plt.close()
        
        # 2. Anomaly Heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(residual_map, cmap='hot')
        plt.colorbar(label='Anomaly Score')
        plt.title(f'Heatmap - {experiment_name}\nAUC: {auc:.4f}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{experiment_name}_heatmap.png"), dpi=200, bbox_inches='tight')
        plt.close()
        
        # 3. Box Plot
        bg_scores = scores[gt_flat == 0]
        an_scores = scores[gt_flat == 1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1.boxplot(bg_scores, showfliers=False, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax1.set_title('Background Scores')
        ax1.set_ylabel('Anomaly Score')
        
        ax2.boxplot(an_scores, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7))
        ax2.set_title('Anomaly Scores')
        ax2.set_ylabel('Anomaly Score')
        
        fig.suptitle(f'Score Distribution - {experiment_name}\nAUC: {auc:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{experiment_name}_boxplot.png"), dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"✅ {experiment_name}: AUC = {auc:.4f}")
        return auc
        
    except Exception as e:
        print(f"❌ Error processing {experiment_name}: {e}")
        return None

def main():
    """Main function"""
    
    print("=== QUICK VISUALIZATION GENERATOR ===")
    
    # Find all experiment directories
    results_dir = "Results"
    experiment_dirs = []
    
    if os.path.exists(results_dir):
        for root, dirs, files in os.walk(results_dir):
            for dir_name in dirs:
                if dir_name.endswith('_fixed'):
                    experiment_path = os.path.join(root, dir_name)
                    experiment_dirs.append(experiment_path)
    
    if not experiment_dirs:
        print("No experiment directories found!")
        return
    
    print(f"Found {len(experiment_dirs)} experiments")
    
    # Process each experiment
    results = []
    for experiment_path in sorted(experiment_dirs):
        auc = quick_visualize_experiment(experiment_path)
        if auc is not None:
            experiment_name = os.path.basename(experiment_path)
            results.append((experiment_name, auc))
    
    # Print summary
    if results:
        print(f"\n=== SUMMARY ===")
        print(f"Successfully processed: {len(results)} experiments")
        
        # Sort by AUC
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 TOP 5 RESULTS:")
        for i, (name, auc) in enumerate(results[:5]):
            print(f"  {i+1}. {name}: AUC = {auc:.4f}")
        
        print(f"\n📁 Visualizations saved in each experiment's 'visualizations' folder")
        print(f"   - ROC curves")
        print(f"   - Anomaly heatmaps") 
        print(f"   - Score distribution box plots")

if __name__ == "__main__":
    main()
