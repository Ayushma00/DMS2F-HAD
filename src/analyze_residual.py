import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
import seaborn as sns

def analyze_residuals_for_fusion_comparison(dataset_name, fusion_types):
    """Analyze and compare residual maps across different fusion types"""
    
    print(f"Analyzing fusion comparison for {dataset_name}...")
    
    # Create comparison directory
    cmp_dir = f"Results/{dataset_name}_fusion_comparison"
    os.makedirs(cmp_dir, exist_ok=True)
    
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
    
    # Generate comprehensive comparison plots
    generate_fusion_comparison_plots(dataset_name, results, cmp_dir)
    
    # Save comparison results to Excel
    save_comparison_results(dataset_name, results, cmp_dir)
    
    print(f"Fusion comparison analysis completed for {dataset_name}")

def generate_fusion_comparison_plots(dataset_name, results, cmp_dir):
    """Generate comprehensive comparison plots"""
    
    fusion_types = list(results.keys())
    n_types = len(fusion_types)
    
    if n_types == 0:
        return
    
    # 1. ROC Curves Comparison
    plt.figure(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0, 1, n_types))
    
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
    plt.savefig(f"{cmp_dir}/{dataset_name}_roc_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Anomaly Score Distributions
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
    
    # Plot 3: Anomaly maps comparison
    ax3 = axes[1, 0]
    # Show first fusion type's anomaly map
    first_fusion = fusion_types[0]
    residual_map = results[first_fusion]['residual_map']
    im = ax3.imshow(residual_map, cmap='hot')
    ax3.set_title(f'Anomaly Map - {first_fusion.title()}')
    ax3.axis('off')
    plt.colorbar(im, ax=ax3, fraction=0.046)
    
    # Plot 4: Ground truth
    ax4 = axes[1, 1]
    gt_mask = results[first_fusion]['gt_mask']
    ax4.imshow(gt_mask, cmap='gray')
    ax4.set_title('Ground Truth')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{cmp_dir}/{dataset_name}_score_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Anomaly maps side-by-side comparison
    fig, axes = plt.subplots(2, n_types, figsize=(4*n_types, 8))
    fig.suptitle(f'Anomaly Maps Comparison - {dataset_name}', fontsize=16)
    
    for i, fusion_type in enumerate(fusion_types):
        result = results[fusion_type]
        residual_map = result['residual_map']
        gt_mask = result['gt_mask']
        
        # Original anomaly map
        if n_types == 1:
            ax1 = axes[0]
            ax2 = axes[1]
        else:
            ax1 = axes[0, i]
            ax2 = axes[1, i]
        
        im1 = ax1.imshow(residual_map, cmap='hot')
        ax1.set_title(f'{fusion_type.title()}\nAnomaly Map')
        ax1.axis('off')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        # Thresholded map
        threshold = np.percentile(residual_map, 95)  # Top 5% as anomalies
        binary_map = residual_map > threshold
        ax2.imshow(binary_map, cmap='RdYlBu_r')
        ax2.set_title(f'Thresholded\n(95th percentile)')
        ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{cmp_dir}/{dataset_name}_anomaly_maps_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Statistical comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Statistical Comparison - {dataset_name}', fontsize=16)
    
    # Box plots of scores
    ax1 = axes[0, 0]
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
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
    colors_extended = plt.cm.viridis(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors_extended):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Anomaly Score')
    ax1.set_title('Score Distributions')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # AUC comparison with error bars (if multiple samples)
    ax2 = axes[0, 1]
    aucs = [results[ft]['auc'] for ft in fusion_types]
    bars = ax2.bar(fusion_types, aucs, color=colors, alpha=0.7)
    ax2.set_ylabel('AUC Score')
    ax2.set_title('AUC Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Score statistics
    ax3 = axes[1, 0]
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
        
        bars = ax3.bar(fusion_names, means, yerr=stds, capsize=5, alpha=0.7, color=colors)
        ax3.set_ylabel('Mean Score')
        ax3.set_title('Score Statistics')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
    
    # Score range comparison
    ax4 = axes[1, 1]
    if stats_data:
        mins = [s['min'] for s in stats_data]
        maxs = [s['max'] for s in stats_data]
        
        x_pos = np.arange(len(fusion_names))
        ax4.bar(x_pos - 0.2, mins, 0.4, label='Min', alpha=0.7, color='lightblue')
        ax4.bar(x_pos + 0.2, maxs, 0.4, label='Max', alpha=0.7, color='lightcoral')
        
        ax4.set_xlabel('Fusion Type')
        ax4.set_ylabel('Score Value')
        ax4.set_title('Score Range Comparison')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(fusion_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{cmp_dir}/{dataset_name}_statistical_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def save_comparison_results(dataset_name, results, cmp_dir):
    """Save comparison results to Excel"""
    
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
    excel_path = f"{cmp_dir}/{dataset_name}_fusion_comparison_results.xlsx"
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
    
    print(f"Comparison results saved to: {excel_path}")

def compare_gate_distributions(dataset_name, fusion_types):
    """Compare gate distributions across fusion types"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    results_dir = f"Results/{dataset_name}_fusion_comparison"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load gate statistics for each fusion type
    gate_stats = {}
    for fusion_type in fusion_types:
        stats_file = f"Results/{dataset_name}_{fusion_type}_fusion_2dec_fixed/gate_statistics.txt"
        if os.path.exists(stats_file):
            gate_stats[fusion_type] = load_gate_statistics(stats_file)
    
    if not gate_stats:
        print(f"No gate statistics found for {dataset_name}")
        return
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Gate Distribution Comparison - {dataset_name}', fontsize=16)
    
    # Plot 1: Mean gate values over epochs
    ax1 = axes[0, 0]
    for fusion_type, stats in gate_stats.items():
        epochs = list(stats.keys())
        means = [stats[ep]['mean'] for ep in epochs]
        ax1.plot(epochs, means, label=fusion_type.title(), linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Gate Value')
    ax1.set_title('Gate Value Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final gate distributions
    ax2 = axes[0, 1]
    final_means = []
    fusion_labels = []
    for fusion_type, stats in gate_stats.items():
        if stats:
            final_epoch = max(stats.keys())
            final_means.append(stats[final_epoch]['mean'])
            fusion_labels.append(fusion_type.title())
    
    bars = ax2.bar(fusion_labels, final_means, alpha=0.7)
    ax2.set_ylabel('Final Mean Gate Value')
    ax2.set_title('Final Gate Values by Fusion Type')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, final_means):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom')
    
    # Plot 3: Standard deviation evolution
    ax3 = axes[1, 0]
    for fusion_type, stats in gate_stats.items():
        epochs = list(stats.keys())
        stds = [stats[ep]['std'] for ep in epochs]
        ax3.plot(epochs, stds, label=fusion_type.title(), linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gate Value Std Dev')
    ax3.set_title('Gate Stability Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Box plot of final distributions
    ax4 = axes[1, 1]
    final_stats = []
    for fusion_type, stats in gate_stats.items():
        if stats:
            final_epoch = max(stats.keys())
            final_stats.append({
                'fusion_type': fusion_type.title(),
                'mean': stats[final_epoch]['mean'],
                'std': stats[final_epoch]['std'],
                'min': stats[final_epoch]['min'],
                'max': stats[final_epoch]['max']
            })
    
    if final_stats:
        fusion_names = [s['fusion_type'] for s in final_stats]
        means = [s['mean'] for s in final_stats]
        stds = [s['std'] for s in final_stats]
        
        # Create box plot data (simulated from mean and std)
        box_data = []
        for i, (mean, std) in enumerate(zip(means, stds)):
            # Generate sample data based on mean and std
            sample_data = np.random.normal(mean, std, 1000)
            box_data.append(sample_data)
        
        bp = ax4.boxplot(box_data, labels=fusion_names, patch_artist=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(box_data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax4.set_ylabel('Gate Values')
        ax4.set_title('Final Gate Distribution Comparison')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/{dataset_name}_gate_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gate comparison saved to: {results_dir}/{dataset_name}_gate_comparison.png")

def load_gate_statistics(stats_file):
    """Load gate statistics from text file"""
    stats = {}
    try:
        with open(stats_file, 'r') as f:
            lines = f.readlines()
        
        # Parse the statistics
        for line in lines:
            if line.strip() and not line.startswith('Epoch') and not line.startswith('-') and not line.startswith('Gate'):
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    try:
                        epoch = int(parts[0])
                        stats[epoch] = {
                            'count': int(parts[1]),
                            'mean': float(parts[2]),
                            'std': float(parts[3]),
                            'median': float(parts[4]),
                            'min': float(parts[5]),
                            'max': float(parts[6])
                        }
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"Error loading gate statistics from {stats_file}: {e}")
    
    return stats

# Original analyze_residual function (keep for backward compatibility)
def analyze_residuals_original():
    """Original analyze_residual function"""
    file_list = [
        "Cri", "San_Diego", 'abu-beach-1', 'abu-urban-1', 'aviris_1', 'aviris_2',
        'los-angeles-2','cat-island', 'gulfport','HYDICE_urban',
        'texas-goast','pavia', 'abu-beach-3','abu-urban-2','abu-urban-3',
        'abu-urban-4','abu-urban-5', 'Salians_syn'
    ]

    base_dir = "../Results"
    cmp_dir = os.path.join(base_dir, "output")
    os.makedirs(cmp_dir, exist_ok=True)

    for ds in file_list:
        print(f"Processing: {ds}")

        mat_path = os.path.join(base_dir, f"{ds}", "residuals_best.mat")
        if not os.path.isfile(mat_path):
            print(f"  SKIPPED: Missing file {mat_path}")
            continue

        # Load residual map and ground truth
        data = sio.loadmat(mat_path)
        residual = data["residual_map"]  
        gt_flat = data["gt_mask"].ravel().astype(int)
        original = data["original"]
        scores = residual.ravel().astype(np.float32)
        fpr, tpr, _ = roc_curve(gt_flat, scores)
        auc = roc_auc_score(gt_flat, scores)
        
        # Save AUC to Excel file
        excel_path = os.path.join(cmp_dir, "auc_results.xlsx")
        
        if os.path.exists(excel_path):
            df = pd.read_excel(excel_path)
        else:
            df = pd.DataFrame(columns=['Dataset', 'AUC'])
        
        if ds in df['Dataset'].values:
            df.loc[df['Dataset'] == ds, 'AUC'] = auc
        else:
            new_row = pd.DataFrame({'Dataset': [ds], 'AUC': [auc]})
            df = pd.concat([df, new_row], ignore_index=True)
        
        df.to_excel(excel_path, index=False)

        # Generate plots (ROC, boxplot, heatmap)
        generate_single_dataset_plots(ds, fpr, tpr, auc, scores, gt_flat, residual, cmp_dir)

    print("All analysis completed.")

def generate_single_dataset_plots(ds, fpr, tpr, auc, scores, gt_flat, residual, cmp_dir):
    """Generate plots for a single dataset"""
    
    # ROC Curve
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

    # Boxplots
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

    # Heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(residual, cmap="jet", vmin=0, vmax=residual.max())
    plt.colorbar(label="Anomaly Score")
    plt.title(f"Anomaly Heatmap — {ds}")
    plt.axis("off")
    plt.tight_layout()
    heat_out = os.path.join(cmp_dir, f"{ds}_heatmap.png")
    plt.savefig(heat_out, dpi=200)
    plt.close()

if __name__ == "__main__":
    # Run original analysis
    analyze_residuals_original()
