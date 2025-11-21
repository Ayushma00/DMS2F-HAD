#!/usr/bin/env python3
"""
Comprehensive visualization generator for all ablation study results
Generates anomaly heatmaps, ROC curves, and box plots for each dataset and fusion mode
"""

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import glob

def generate_visualizations_for_experiment(experiment_path, experiment_name):
    """Generate all visualizations for a single experiment"""
    
    # Check if residuals file exists
    residuals_file = os.path.join(experiment_path, "residuals_best.mat")
    if not os.path.exists(residuals_file):
        print(f"  SKIPPED: No residuals file found for {experiment_name}")
        return None
    
    print(f"  Processing: {experiment_name}")
    
    try:
        # Load data
        data = sio.loadmat(residuals_file)
        residual_map = data["residual_map"]
        gt_mask = data["gt_mask"]
        original = data.get("original", None)
        
        # Calculate AUC
        gt_flat = gt_mask.ravel().astype(int)
        scores = residual_map.ravel().astype(np.float32)
        
        if len(np.unique(gt_flat)) <= 1:
            print(f"    WARNING: No valid ground truth for {experiment_name}")
            return None
        
        fpr, tpr, _ = roc_curve(gt_flat, scores)
        auc = roc_auc_score(gt_flat, scores)
        
        # Create visualizations directory
        viz_dir = os.path.join(experiment_path, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # 1. ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {experiment_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{experiment_name}_roc.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Anomaly Heatmap
        plt.figure(figsize=(10, 8))
        im = plt.imshow(residual_map, cmap='hot', aspect='auto')
        plt.colorbar(im, label='Anomaly Score')
        plt.title(f'Anomaly Heatmap - {experiment_name}\nAUC: {auc:.4f}')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{experiment_name}_heatmap.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Box Plot
        bg_scores = scores[gt_flat == 0]
        an_scores = scores[gt_flat == 1]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Background scores
        ax1.boxplot(bg_scores, showfliers=False, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax1.set_title(f'Background Scores\nMean: {np.mean(bg_scores):.4f}, Std: {np.std(bg_scores):.4f}')
        ax1.set_ylabel('Anomaly Score')
        ax1.grid(True, alpha=0.3)
        
        # Anomaly scores
        ax2.boxplot(an_scores, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor='lightcoral', alpha=0.7))
        ax2.set_title(f'Anomaly Scores\nMean: {np.mean(an_scores):.4f}, Std: {np.std(an_scores):.4f}')
        ax2.set_ylabel('Anomaly Score')
        ax2.grid(True, alpha=0.3)
        
        fig.suptitle(f'Score Distribution - {experiment_name}\nAUC: {auc:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{experiment_name}_boxplot.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Score Distribution Histogram
        plt.figure(figsize=(10, 6))
        plt.hist(bg_scores, bins=50, alpha=0.6, label='Background', density=True, color='blue')
        plt.hist(an_scores, bins=50, alpha=0.6, label='Anomaly', density=True, color='red')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Density')
        plt.title(f'Score Distribution - {experiment_name}\nAUC: {auc:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{experiment_name}_histogram.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Ground Truth vs Predicted Comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Ground truth
        ax1.imshow(gt_mask, cmap='gray')
        ax1.set_title('Ground Truth')
        ax1.axis('off')
        
        # Thresholded prediction (top 5% as anomalies)
        threshold = np.percentile(residual_map, 95)
        binary_pred = residual_map > threshold
        ax2.imshow(binary_pred, cmap='RdYlBu_r')
        ax2.set_title(f'Predicted (threshold: {threshold:.4f})')
        ax2.axis('off')
        
        fig.suptitle(f'Ground Truth vs Prediction - {experiment_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"{experiment_name}_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✅ Generated 5 visualizations for {experiment_name} (AUC: {auc:.4f})")
        
        return {
            'experiment': experiment_name,
            'auc': auc,
            'background_mean': np.mean(bg_scores),
            'background_std': np.std(bg_scores),
            'anomaly_mean': np.mean(an_scores),
            'anomaly_std': np.std(an_scores),
            'score_range': np.max(scores) - np.min(scores),
            'background_count': len(bg_scores),
            'anomaly_count': len(an_scores)
        }
        
    except Exception as e:
        print(f"    ERROR processing {experiment_name}: {e}")
        return None

def find_all_experiments():
    """Find all experiment directories in Results folder"""
    
    results_dir = "Results"
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return []
    
    # Find all directories that end with '_fixed'
    experiment_dirs = []
    for root, dirs, files in os.walk(results_dir):
        for dir_name in dirs:
            if dir_name.endswith('_fixed'):
                experiment_path = os.path.join(root, dir_name)
                experiment_dirs.append(experiment_path)
    
    return sorted(experiment_dirs)

def generate_comprehensive_summary(all_results):
    """Generate comprehensive summary of all results"""
    
    if not all_results:
        print("No results to summarize!")
        return
    
    # Create summary DataFrame
    df = pd.DataFrame(all_results)
    
    # Save summary to Excel
    summary_file = "Results/comprehensive_visualization_summary.xlsx"
    with pd.ExcelWriter(summary_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Create summary statistics
        summary_stats = df.groupby('experiment').agg({
            'auc': ['mean', 'std', 'max', 'min'],
            'background_mean': 'mean',
            'anomaly_mean': 'mean',
            'score_range': 'mean'
        }).round(4)
        summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
        
        # Best performing experiments
        best_experiments = df.nlargest(10, 'auc')[['experiment', 'auc', 'background_mean', 'anomaly_mean']]
        best_experiments.to_excel(writer, sheet_name='Top_10_Experiments', index=False)
    
    print(f"\n📊 Comprehensive summary saved to: {summary_file}")
    
    # Generate overall comparison plots
    generate_overall_comparison_plots(df)

def generate_overall_comparison_plots(df):
    """Generate overall comparison plots for all experiments"""
    
    if df.empty:
        return
    
    # Create overall comparison directory
    comparison_dir = "Results/overall_comparison"
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 1. AUC Comparison across all experiments
    plt.figure(figsize=(15, 8))
    experiments = df['experiment'].values
    aucs = df['auc'].values
    
    # Sort by AUC
    sorted_indices = np.argsort(aucs)[::-1]
    sorted_experiments = experiments[sorted_indices]
    sorted_aucs = aucs[sorted_indices]
    
    bars = plt.bar(range(len(sorted_aucs)), sorted_aucs, alpha=0.7)
    plt.xlabel('Experiments')
    plt.ylabel('AUC Score')
    plt.title('AUC Comparison Across All Experiments')
    plt.xticks(range(len(sorted_experiments)), sorted_experiments, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, auc) in enumerate(zip(bars, sorted_aucs)):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "overall_auc_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Score separation analysis
    plt.figure(figsize=(12, 8))
    
    # Calculate separation metric (difference between anomaly and background means)
    separation = df['anomaly_mean'] - df['background_mean']
    
    plt.scatter(df['auc'], separation, alpha=0.7, s=50)
    plt.xlabel('AUC Score')
    plt.ylabel('Score Separation (Anomaly - Background Mean)')
    plt.title('AUC vs Score Separation')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(df['auc'], separation, 1)
    p = np.poly1d(z)
    plt.plot(df['auc'], p(df['auc']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "auc_vs_separation.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap of results by dataset and mode
    # Extract dataset and mode information from experiment names
    dataset_mode_data = []
    for exp_name in df['experiment']:
        parts = exp_name.split('_')
        if len(parts) >= 3:
            dataset = parts[0]
            mode = parts[1]
            if 'fusion' in exp_name:
                fusion_type = parts[3] if len(parts) > 3 else 'unknown'
                mode = f"{mode}_{fusion_type}"
            dataset_mode_data.append({'dataset': dataset, 'mode': mode})
        else:
            dataset_mode_data.append({'dataset': 'unknown', 'mode': 'unknown'})
    
    # Create pivot table
    df_with_info = df.copy()
    df_with_info['dataset'] = [d['dataset'] for d in dataset_mode_data]
    df_with_info['mode'] = [d['mode'] for d in dataset_mode_data]
    
    # Handle duplicate entries by taking the best AUC for each dataset-mode combination
    df_with_info = df_with_info.groupby(['dataset', 'mode'])['auc'].max().reset_index()
    
    pivot_df = df_with_info.pivot(index='dataset', columns='mode', values='auc')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='viridis', cbar_kws={'label': 'AUC Score'})
    plt.title('AUC Heatmap: Dataset vs Mode')
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, "dataset_mode_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Overall comparison plots saved to: {comparison_dir}")

def main():
    """Main function to generate all visualizations"""
    
    print("=== GENERATING COMPREHENSIVE VISUALIZATIONS ===")
    print("This will create anomaly heatmaps, ROC curves, and box plots for all experiments")
    
    # Find all experiment directories
    experiment_dirs = find_all_experiments()
    
    if not experiment_dirs:
        print("No experiment directories found!")
        return
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    
    # Process each experiment
    all_results = []
    successful_count = 0
    
    for experiment_path in experiment_dirs:
        experiment_name = os.path.basename(experiment_path)
        result = generate_visualizations_for_experiment(experiment_path, experiment_name)
        if result:
            all_results.append(result)
            successful_count += 1
    
    print(f"\n=== VISUALIZATION GENERATION COMPLETED ===")
    print(f"Successfully processed: {successful_count}/{len(experiment_dirs)} experiments")
    
    # Generate comprehensive summary
    if all_results:
        generate_comprehensive_summary(all_results)
        
        # Print top 5 results
        df = pd.DataFrame(all_results)
        top_5 = df.nlargest(5, 'auc')
        print(f"\n🏆 TOP 5 PERFORMING EXPERIMENTS:")
        for _, row in top_5.iterrows():
            print(f"  {row['experiment']}: AUC = {row['auc']:.4f}")
    
    print(f"\n📁 All visualizations saved in individual experiment folders")
    print(f"📊 Summary files saved in Results/ directory")

if __name__ == "__main__":
    main()
