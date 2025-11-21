#!/usr/bin/env python3
"""
Standalone script to visualize fusion results from existing residual.mat files
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns

def visualize_fusion_results(dataset_name, fusion_types):
    """Visualize results for a specific dataset across fusion types"""
    
    print(f"Visualizing fusion results for {dataset_name}...")
    
    # Create output directory
    output_dir = f"Results/{dataset_name}_fusion_visualization"
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    # Load results for each fusion type
    for fusion_type in fusion_types:
        mat_path = f"Results/{dataset_name}_{fusion_type}_fusion_2dec_fixed/residuals_best.mat"
        if os.path.isfile(mat_path):
            print(f"  Loading {fusion_type} fusion results...")
            
            # Load residual map and ground truth
            data = sio.loadmat(mat_path)
            residual = data["residual_map"]  
            gt_flat = data["gt_mask"].ravel().astype(int)
            original = data["original"]
            scores = residual.ravel().astype(np.float32)
            
            # Calculate ROC and AUC
            if len(np.unique(gt_flat)) > 1:
                fpr, tpr, _ = roc_curve(gt_flat, scores)
                auc = roc_auc_score(gt_flat, scores)
                
                results[fusion_type] = {
                    'residual_map': residual,
                    'gt_mask': data["gt_mask"],
                    'original': original,
                    'scores': scores,
                    'fpr': fpr,
                    'tpr': tpr,
                    'auc': auc
                }
                
                print(f"    {fusion_type} fusion - AUC: {auc:.4f}")
            else:
                print(f"    {fusion_type} fusion - No valid ground truth")
        else:
            print(f"  SKIPPED: Missing file {mat_path}")
    
    if not results:
        print(f"No valid results found for {dataset_name}")
        return
    
    # Generate comprehensive visualizations
    generate_comprehensive_plots(dataset_name, results, output_dir)
    
    print(f"Visualization completed for {dataset_name}")

def generate_comprehensive_plots(dataset_name, results, output_dir):
    """Generate comprehensive visualization plots"""
    
    fusion_types = list(results.keys())
    n_types = len(fusion_types)
    colors = plt.cm.viridis(np.linspace(0, 1, n_types))
    
    # 1. ROC Curves Comparison
    plt.figure(figsize=(10, 8))
    for i, fusion_type in enumerate(fusion_types):
        result = results[fusion_type]
        plt.plot(result['fpr'], result['tpr'], 
                color=colors[i], linewidth=2, 
                label=f'{fusion_type.title()} (AUC = {result["auc"]:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves Comparison - {dataset_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/{dataset_name}_roc_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Anomaly Maps Side-by-Side
    fig, axes = plt.subplots(2, n_types + 1, figsize=(4*(n_types+1), 8))
    fig.suptitle(f'Anomaly Detection Results - {dataset_name}', fontsize=16)
    
    # Show ground truth
    first_fusion = fusion_types[0]
    gt_mask = results[first_fusion]['gt_mask']
    axes[0, 0].imshow(gt_mask, cmap='gray')
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    # Show anomaly maps for each fusion type
    for i, fusion_type in enumerate(fusion_types):
        result = results[fusion_type]
        residual_map = result['residual_map']
        
        # Original anomaly map
        im1 = axes[0, i+1].imshow(residual_map, cmap='hot')
        axes[0, i+1].set_title(f'{fusion_type.title()}\nAnomaly Map')
        axes[0, i+1].axis('off')
        plt.colorbar(im1, ax=axes[0, i+1], fraction=0.046)
        
        # Thresholded map (top 5% as anomalies)
        threshold = np.percentile(residual_map, 95)
        binary_map = residual_map > threshold
        axes[1, i+1].imshow(binary_map, cmap='RdYlBu_r')
        axes[1, i+1].set_title(f'Thresholded\n(95th percentile)')
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_anomaly_maps.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Score Distribution Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Anomaly Score Analysis - {dataset_name}', fontsize=16)
    
    # Plot 1: Score distributions
    ax1 = axes[0, 0]
    for i, fusion_type in enumerate(fusion_types):
        result = results[fusion_type]
        scores = result['scores']
        gt_flat = result['gt_mask'].ravel().astype(int)
        
        # Separate background and anomaly scores
        bg_scores = scores[gt_flat == 0]
        an_scores = scores[gt_flat == 1]
        
        # Plot histograms
        ax1.hist(bg_scores, bins=50, alpha=0.5, label=f'{fusion_type.title()} - Background', density=True)
        ax1.hist(an_scores, bins=50, alpha=0.5, label=f'{fusion_type.title()} - Anomaly', density=True)
    
    ax1.set_xlabel('Anomaly Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Score Distributions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: AUC comparison
    ax2 = axes[0, 1]
    aucs = [results[ft]['auc'] for ft in fusion_types]
    bars = ax2.bar(fusion_types, aucs, color=colors, alpha=0.7)
    ax2.set_ylabel('AUC Score')
    ax2.set_title('AUC Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{auc:.4f}', ha='center', va='bottom')
    
    # Plot 3: Box plots
    ax3 = axes[1, 0]
    box_data = []
    box_labels = []
    for fusion_type in fusion_types:
        result = results[fusion_type]
        scores = result['scores']
        gt_flat = result['gt_mask'].ravel().astype(int)
        
        bg_scores = scores[gt_flat == 0]
        an_scores = scores[gt_flat == 1]
        
        box_data.extend([bg_scores, an_scores])
        box_labels.extend([f'{fusion_type.title()}\nBackground', f'{fusion_type.title()}\nAnomaly'])
    
    bp = ax3.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors_extended = plt.cm.viridis(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors_extended):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_ylabel('Anomaly Score')
    ax3.set_title('Score Distributions')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary
    ax4 = axes[1, 1]
    stats_data = []
    for fusion_type in fusion_types:
        result = results[fusion_type]
        scores = result['scores']
        stats_data.append({
            'fusion_type': fusion_type.title(),
            'mean': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores)
        })
    
    if stats_data:
        fusion_names = [s['fusion_type'] for s in stats_data]
        means = [s['mean'] for s in stats_data]
        stds = [s['std'] for s in stats_data]
        
        bars = ax4.bar(fusion_names, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
        ax4.set_ylabel('Mean Score')
        ax4.set_title('Score Statistics')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{dataset_name}_score_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Performance Summary Table
    create_performance_summary(dataset_name, results, output_dir)

def create_performance_summary(dataset_name, results, output_dir):
    """Create a performance summary table"""
    
    # Create summary DataFrame
    summary_data = []
    for fusion_type, result in results.items():
        scores = result['scores']
        gt_flat = result['gt_mask'].ravel().astype(int)
        
        # Separate background and anomaly scores
        bg_scores = scores[gt_flat == 0]
        an_scores = scores[gt_flat == 1]
        
        summary_data.append({
            'Fusion_Type': fusion_type.title(),
            'AUC': result['auc'],
            'Mean_Score': np.mean(scores),
            'Std_Score': np.std(scores),
            'Min_Score': np.min(scores),
            'Max_Score': np.max(scores),
            'Background_Mean': np.mean(bg_scores),
            'Background_Std': np.std(bg_scores),
            'Anomaly_Mean': np.mean(an_scores),
            'Anomaly_Std': np.std(an_scores),
            'Score_Range': np.max(scores) - np.min(scores),
            'Background_Count': len(bg_scores),
            'Anomaly_Count': len(an_scores)
        })
    
    df = pd.DataFrame(summary_data)
    
    # Save to Excel
    excel_path = f"{output_dir}/{dataset_name}_performance_summary.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Summary', index=False)
        
        # Create detailed score analysis
        detailed_data = []
        for fusion_type, result in results.items():
            scores = result['scores']
            gt_flat = result['gt_mask'].ravel().astype(int)
            
            for i, (score, label) in enumerate(zip(scores, gt_flat)):
                detailed_data.append({
                    'Fusion_Type': fusion_type.title(),
                    'Score': score,
                    'Label': 'Anomaly' if label == 1 else 'Background',
                    'Sample_ID': i
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_df.to_excel(writer, sheet_name='Detailed_Scores', index=False)
    
    print(f"Performance summary saved to: {excel_path}")
    
    # Print summary to console
    print(f"\n=== Performance Summary for {dataset_name} ===")
    print(df.to_string(index=False))

def main():
    """Main function to visualize all datasets"""
    
    # Define fusion types and datasets
    fusion_types = ["simple", "advanced", "cross_attention", "hierarchical"]
    datasets = ["aviris_2", "cat-island"]  # Add your datasets here
    
    print("=== FUSION RESULTS VISUALIZATION ===")
    
    for dataset in datasets:
        print(f"\nProcessing {dataset}...")
        visualize_fusion_results(dataset, fusion_types)
    
    print("\n=== VISUALIZATION COMPLETED ===")
    print("Check the following directories for results:")
    for dataset in datasets:
        print(f"  - Results/{dataset}_fusion_visualization/")

if __name__ == "__main__":
    main()