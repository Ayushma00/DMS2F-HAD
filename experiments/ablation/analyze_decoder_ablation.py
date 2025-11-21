#!/usr/bin/env python3
"""
Analysis script for decoder ablation study results.
Provides comprehensive analysis and visualization of AUC results.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.auc_tracker import AUCTracker

def analyze_decoder_ablation_results(results_dir="Results"):
    """
    Comprehensive analysis of decoder ablation study results.
    """
    tracker = AUCTracker(results_dir=results_dir)
    
    # Check if results exist
    if not os.path.exists(tracker.excel_path):
        print(f"No results file found at {tracker.excel_path}")
        print("Please run the training first to generate results.")
        return
    
    print("=" * 60)
    print("DECODER ABLATION STUDY ANALYSIS")
    print("=" * 60)
    
    # Load and display basic summary
    df = tracker.get_results_summary()
    tracker.compare_decoder_configurations()
    
    # Create detailed visualizations
    create_decoder_ablation_plots(df, results_dir)
    
    # Generate detailed analysis report
    generate_analysis_report(df, results_dir)
    
    print(f"\nResults saved to: {tracker.excel_path}")
    print(f"Plots saved to: {os.path.join(results_dir, 'plots/')}")

def create_decoder_ablation_plots(df, results_dir):
    """Create comprehensive plots for decoder ablation analysis"""
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Box plot: AUC by number of decoders
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x='Num_Decoders', y='AUC')
    plt.title('AUC Distribution by Number of Decoder Blocks', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Decoder Blocks', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'auc_by_decoders_boxplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Box plot: AUC by mode and decoders
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Mode', y='AUC', hue='Num_Decoders')
    plt.title('AUC Distribution by Mode and Number of Decoders', fontsize=14, fontweight='bold')
    plt.xlabel('Mode', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.legend(title='Number of Decoders', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'auc_by_mode_decoders.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Heatmap: Dataset vs Configuration
    df_copy = df.copy()
    df_copy['Configuration'] = df_copy['Mode'] + '_' + df_copy['Num_Decoders'].astype(str) + 'dec'
    pivot_table = df_copy.pivot_table(values='AUC', index='Dataset', columns='Configuration', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd', fmt='.3f', 
                cbar_kws={'label': 'AUC Score'})
    plt.title('AUC Heatmap: Dataset vs Configuration', fontsize=14, fontweight='bold')
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Dataset', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'auc_heatmap_dataset_config.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Line plot: Mean AUC by decoders for each mode
    mean_auc = df.groupby(['Mode', 'Num_Decoders'])['AUC'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    for mode in mean_auc['Mode'].unique():
        mode_data = mean_auc[mean_auc['Mode'] == mode]
        plt.plot(mode_data['Num_Decoders'], mode_data['AUC'], 
                marker='o', linewidth=2, markersize=8, label=mode)
    
    plt.title('Mean AUC vs Number of Decoder Blocks', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Decoder Blocks', fontsize=12)
    plt.ylabel('Mean AUC Score', fontsize=12)
    plt.legend(title='Mode')
    plt.grid(True, alpha=0.3)
    plt.xticks([1, 2, 3])
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mean_auc_by_decoders_line.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Bar plot: Best AUC for each dataset
    best_results = df.loc[df.groupby('Dataset')['AUC'].idxmax()]
    best_results = best_results.sort_values('AUC', ascending=True)
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, len(best_results)))
    bars = plt.barh(range(len(best_results)), best_results['AUC'], color=colors)
    
    # Add configuration labels on bars
    for i, (idx, row) in enumerate(best_results.iterrows()):
        config = f"{row['Mode']}_{row['Num_Decoders']}dec"
        plt.text(row['AUC'] + 0.005, i, f"{config} ({row['AUC']:.3f})", 
                va='center', fontsize=9)
    
    plt.yticks(range(len(best_results)), best_results['Dataset'])
    plt.xlabel('Best AUC Score', fontsize=12)
    plt.ylabel('Dataset', fontsize=12)
    plt.title('Best AUC Score for Each Dataset', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'best_auc_by_dataset.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Statistical comparison plot
    decoder_stats = df.groupby('Num_Decoders')['AUC'].agg(['mean', 'std', 'count']).reset_index()
    
    plt.figure(figsize=(10, 6))
    x = decoder_stats['Num_Decoders']
    y = decoder_stats['mean']
    yerr = decoder_stats['std']
    
    plt.errorbar(x, y, yerr=yerr, marker='o', capsize=5, capthick=2, linewidth=2, markersize=8)
    plt.title('Mean AUC ± Standard Deviation by Number of Decoders', fontsize=14, fontweight='bold')
    plt.xlabel('Number of Decoder Blocks', fontsize=12)
    plt.ylabel('Mean AUC Score ± Std', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks([1, 2, 3])
    
    # Add text annotations
    for i, row in decoder_stats.iterrows():
        plt.text(row['Num_Decoders'], row['mean'] + row['std'] + 0.01, 
                f"n={row['count']}", ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'mean_auc_with_std.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created 6 plots in {plots_dir}/")

def generate_analysis_report(df, results_dir):
    """Generate a detailed text report of the analysis"""
    report_path = os.path.join(results_dir, "decoder_ablation_analysis_report.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DECODER ABLATION STUDY ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Basic statistics
        f.write("BASIC STATISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total experiments: {len(df)}\n")
        f.write(f"Datasets: {df['Dataset'].nunique()} ({', '.join(sorted(df['Dataset'].unique()))})\n")
        f.write(f"Modes: {df['Mode'].nunique()} ({', '.join(sorted(df['Mode'].unique()))})\n")
        f.write(f"Decoder configurations: {', '.join(map(str, sorted(df['Num_Decoders'].unique())))}\n")
        f.write(f"Overall mean AUC: {df['AUC'].mean():.4f} ± {df['AUC'].std():.4f}\n")
        f.write(f"Overall AUC range: {df['AUC'].min():.4f} - {df['AUC'].max():.4f}\n\n")
        
        # Best overall result
        best_idx = df['AUC'].idxmax()
        best_row = df.loc[best_idx]
        f.write("BEST OVERALL RESULT\n")
        f.write("-"*40 + "\n")
        f.write(f"Dataset: {best_row['Dataset']}\n")
        f.write(f"Mode: {best_row['Mode']}\n")
        f.write(f"Decoders: {best_row['Num_Decoders']}\n")
        f.write(f"AUC: {best_row['AUC']:.4f}\n")
        f.write(f"Experiment ID: {best_row['Experiment_ID']}\n\n")
        
        # Analysis by number of decoders
        f.write("ANALYSIS BY NUMBER OF DECODERS\n")
        f.write("-"*40 + "\n")
        decoder_stats = df.groupby('Num_Decoders')['AUC'].agg(['count', 'mean', 'std', 'min', 'max'])
        for num_dec, stats in decoder_stats.iterrows():
            f.write(f"\n{num_dec} Decoder(s):\n")
            f.write(f"  Experiments: {stats['count']}\n")
            f.write(f"  Mean AUC: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n")
        
        # Analysis by mode
        f.write("\n\nANALYSIS BY MODE\n")
        f.write("-"*40 + "\n")
        mode_stats = df.groupby('Mode')['AUC'].agg(['count', 'mean', 'std', 'min', 'max'])
        for mode, stats in mode_stats.iterrows():
            f.write(f"\n{mode.capitalize()} Mode:\n")
            f.write(f"  Experiments: {stats['count']}\n")
            f.write(f"  Mean AUC: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
            f.write(f"  Range: {stats['min']:.4f} - {stats['max']:.4f}\n")
        
        # Analysis by mode and decoder combination
        f.write("\n\nANALYSIS BY MODE AND DECODER COMBINATION\n")
        f.write("-"*40 + "\n")
        combo_stats = df.groupby(['Mode', 'Num_Decoders'])['AUC'].agg(['count', 'mean', 'std'])
        for (mode, num_dec), stats in combo_stats.iterrows():
            f.write(f"\n{mode.capitalize()} with {num_dec} decoder(s):\n")
            f.write(f"  Experiments: {stats['count']}\n")
            f.write(f"  Mean AUC: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        # Best result for each dataset
        f.write("\n\nBEST RESULT FOR EACH DATASET\n")
        f.write("-"*40 + "\n")
        best_by_dataset = df.loc[df.groupby('Dataset')['AUC'].idxmax()]
        for _, row in best_by_dataset.sort_values('AUC', ascending=False).iterrows():
            f.write(f"{row['Dataset']}: {row['AUC']:.4f} ({row['Mode']}_{row['Num_Decoders']}dec)\n")
        
        # Statistical tests and insights
        f.write("\n\nSTATISTICAL INSIGHTS\n")
        f.write("-"*40 + "\n")
        
        # Check if more decoders generally improve performance
        decoder_means = df.groupby('Num_Decoders')['AUC'].mean()
        if decoder_means[3] > decoder_means[2] > decoder_means[1]:
            f.write("✓ More decoder blocks consistently improve performance\n")
        elif decoder_means[2] > decoder_means[1] and decoder_means[2] > decoder_means[3]:
            f.write("→ 2 decoder blocks appear optimal (diminishing returns at 3)\n")
        else:
            f.write("→ Decoder count effect varies by dataset/mode\n")
        
        # Best mode overall
        best_mode = df.groupby('Mode')['AUC'].mean().idxmax()
        f.write(f"✓ Best performing mode overall: {best_mode}\n")
        
        # Consistency analysis
        mode_std = df.groupby('Mode')['AUC'].std()
        most_consistent = mode_std.idxmin()
        f.write(f"✓ Most consistent mode: {most_consistent} (lowest std: {mode_std[most_consistent]:.4f})\n")
    
    print(f"Generated detailed analysis report: {report_path}")

if __name__ == "__main__":
    analyze_decoder_ablation_results()
