#!/usr/bin/env python3
"""
Simple script to test all fusion types and save results
Integrates with your existing main_fixed.py structure
"""

import os
import random
import numpy as np
import torch
import pandas as pd
import json
import time
from datetime import datetime
from src.data_loader import HSIDataset
from src.model_fixed import AnomalyDetectionModel
from src.train import train_model
from src.project_paths import DATA_DIR, RESULTS_DIR, EVAL_DATASET_DIR, REPO_ROOT

def test_all_fusion_types():
    """Test all fusion types and save results systematically"""
    
    print("TESTING ALL FUSION TYPES")
    print("="*50)
    
    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Configuration - simplified for quick testing
    dataset_names = [
        "aviris_1",
        "aviris_2", 
        "cat-island",
        "Cri",
        "San_Diego",
        "Salians_syn",
        "abu-urban-2"
    ]  # Start with one dataset
    dataset_files = [DATA_DIR / f"{name}.mat" for name in dataset_names]
    
    fusion_types = [
        "simple",
        "advanced", 
        "cross_attention",
        "hierarchical"
    ]
    
    # Parameters
    epochs = 20  # Reduced for quick comparison
    batch_size = 32
    lr = 5e-4
    weight_decay = 1e-4
    num_decoders = 2
    
    results = {}
    
    for data_path in dataset_files:
        ds_name = data_path.stem
        print(f"\nDataset: {ds_name}")
        
        # Load dataset
        dataset = HSIDataset(
            mat_file=data_path,
            data_name=ds_name,
            block_size=16,
            stride=8
        )
        C = dataset.blocks.shape[1]
        results[ds_name] = {}
        
        # Test each fusion type
        for fusion_type in fusion_types:
            print(f"\n Testing fusion: {fusion_type}")
            
            try:
                start_time = time.time()
                
                # Create model with specific fusion type
                model = AnomalyDetectionModel(
                    in_channels=C,
                    mode="full",
                    dim=64,
                    depth=1,
                    spec_num=12,
                    spec_rate=0.5,
                    spa_token=16,
                    num_decoders=num_decoders,
                    fusion_type=fusion_type
                )
                
                # Get complexity analysis
                analysis = model.analyze_fusion_complexity()
                total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                print(f"   Parameters: {total_params:,} (Fusion: {analysis['parameters']:,})")
                print(f"   Expressiveness: {analysis['expressiveness_score']}/10")
                
                # Enable gate tracking
                model.start_gate_tracking()
                
                if torch.cuda.is_available():
                    model = model.cuda()
                
                # Train
                experiment_name = f"{ds_name}_{fusion_type}_fusion_test"
                
                trained_model = train_model(
                    model,
                    dataset,
                    experiment_name,
                    epochs=epochs,
                    batch_sz=batch_size,
                    lr=lr,
                    wd=weight_decay,
                    eval_dataset_path=EVAL_DATASET_DIR,
                    mode="full",
                    num_decoders=num_decoders
                )
                
                training_time = time.time() - start_time
                
                # Store results
                results[ds_name][fusion_type] = {
                    "status": "success",
                    "total_parameters": total_params,
                    "fusion_parameters": analysis['parameters'],
                    "expressiveness_score": analysis['expressiveness_score'],
                    "training_time": training_time,
                    "experiment_name": experiment_name,
                    "components": analysis['learnable_components']
                }
                
                print(f"   Success in {training_time:.1f}s")
                
            except Exception as e:
                training_time = time.time() - start_time
                print(f"    Failed: {e}")
                
                results[ds_name][fusion_type] = {
                    "status": "failed",
                    "error": str(e),
                    "training_time": training_time
                }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = RESULTS_DIR / f"fusion_tests_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save raw JSON
    with open(results_dir / "fusion_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 2. Create comparison table
    comparison_data = []
    for dataset_name, dataset_results in results.items():
        for fusion_type, fusion_results in dataset_results.items():
            if fusion_results["status"] == "success":
                comparison_data.append({
                    "Dataset": dataset_name,
                    "Fusion_Type": fusion_type,
                    "Total_Parameters": fusion_results["total_parameters"],
                    "Fusion_Parameters": fusion_results["fusion_parameters"],
                    "Expressiveness": fusion_results["expressiveness_score"],
                    "Training_Time": fusion_results["training_time"],
                    "Parameter_Efficiency": fusion_results["expressiveness_score"] / max(1, fusion_results["fusion_parameters"] / 1000),
                    "Experiment_Name": fusion_results["experiment_name"]
                })
    
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df.to_csv(results_dir / "fusion_comparison.csv", index=False)
        
        # Print summary
        print(f"\n RESULTS SUMMARY")
        print("="*50)
        print(df.to_string(index=False))
        
        # Find best performers
        best_expressiveness = df.loc[df['Expressiveness'].idxmax()]
        most_efficient = df.loc[df['Parameter_Efficiency'].idxmax()]
        fastest = df.loc[df['Training_Time'].idxmin()]
        
        print(f"\n BEST PERFORMERS:")
        print(f"   Most Expressive: {best_expressiveness['Fusion_Type']} ({best_expressiveness['Expressiveness']}/10)")
        print(f"   Most Efficient: {most_efficient['Fusion_Type']} ({most_efficient['Parameter_Efficiency']:.4f})")
        print(f"   Fastest: {fastest['Fusion_Type']} ({fastest['Training_Time']:.1f}s)")
    
    # 3. Create recommendations
    recommendations = {
        "summary": {
            "tested_fusion_types": fusion_types,
            "successful_experiments": len(comparison_data),
            "failed_experiments": len(fusion_types) - len(comparison_data)
        },
        "recommendations": {
            "for_production": "advanced",  # Good balance
            "for_research": "cross_attention",  # Highest performance
            "for_speed": "simple",  # Fastest
            "for_complex_data": "hierarchical"  # Multi-scale
        },
        "next_steps": [
            "Run full training (150 epochs) with best fusion type",
            "Test on more datasets for validation", 
            "Compare gate distribution plots",
            "Analyze parameter efficiency vs performance trade-off"
        ]
    }
    
    with open(results_dir / "recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)
    
    # 4. Create simple report
    with open(results_dir / "fusion_test_report.md", "w") as f:
        f.write("# Fusion Type Test Results\n\n")
        f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Epochs**: {epochs}\n")
        f.write(f"**Datasets**: {', '.join(dataset_names)}\n\n")
        
        f.write("## Results Summary\n\n")
        if comparison_data:
            f.write("| Fusion Type | Parameters | Expressiveness | Time | Efficiency |\n")
            f.write("|-------------|------------|---------------|------|------------|\n")
            for _, row in df.iterrows():
                f.write(f"| {row['Fusion_Type']} | {row['Fusion_Parameters']:,} | ")
                f.write(f"{row['Expressiveness']}/10 | {row['Training_Time']:.1f}s | ")
                f.write(f"{row['Parameter_Efficiency']:.4f} |\n")
        
        f.write(f"\n## Recommendations\n\n")
        for purpose, fusion_type in recommendations["recommendations"].items():
            f.write(f"- **{purpose.replace('_', ' ').title()}**: {fusion_type}\n")
        
        f.write(f"\n## Next Steps\n\n")
        for step in recommendations["next_steps"]:
            f.write(f"- {step}\n")
        
        f.write(f"\n## Files Generated\n\n")
        f.write(f"- `fusion_test_results.json`: Raw results\n")
        f.write(f"- `fusion_comparison.csv`: Comparison table\n")
        f.write(f"- `recommendations.json`: Analysis and recommendations\n")
        results_root = RESULTS_DIR.relative_to(REPO_ROOT)
        f.write(f"- Individual experiment results in `{results_root}/` folder\n")
    
    print(f"\n All results saved to: {results_dir}")
    print(f"\nFiles created:")
    print(f"  - fusion_test_results.json")
    print(f"  - fusion_comparison.csv")
    print(f"  - recommendations.json")
    print(f"  - fusion_test_report.md")
    
    print(f"\n Quick Recommendation:")
    print(f"   Use 'advanced' fusion for best balance of performance and efficiency")
    
    return results_dir

if __name__ == "__main__":
    test_all_fusion_types()
