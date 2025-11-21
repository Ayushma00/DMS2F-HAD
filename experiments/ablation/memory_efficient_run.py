#!/usr/bin/env python3
"""
Memory-efficient version of the ablation study
Disables gate tracking and uses fewer epochs to prevent memory issues
"""

import os
import re
import random
import numpy as np
import torch
from datetime import datetime
from src.data_loader import HSIDataset
from src.model_fixed import AnomalyDetectionModel
from src.train import train_model
from src.auc_tracker_abalation import save_auc_result
from src.analyze_residual import analyze_residuals_for_fusion_comparison, compare_gate_distributions
import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment
import matplotlib.pyplot as plt
import seaborn as sns

class AblationResultsTracker:
    def __init__(self):
        self.results = []
        self.gate_info = []
        
    def add_result(self, dataset_name, mode, fusion_type, num_decoders, auc_score, 
                   param_count, training_time, epoch_reached, inference_time, additional_info=None):
        self.results.append({
            'Dataset': dataset_name,
            'Mode': mode,
            'Fusion_Type': fusion_type,
            'Num_Decoders': num_decoders,
            'AUC_Score': auc_score,
            'Parameter_Count': param_count,
            'Training_Time_Minutes': training_time,
            'Epochs_Reached': epoch_reached,
            'Inference_Time_Seconds': inference_time,
            'Additional_Info': additional_info or {}
        })
        
    def add_gate_info(self, dataset_name, fusion_type, num_decoders, gate_stats):
        self.gate_info.append({
            'Dataset': dataset_name,
            'Fusion_Type': fusion_type,
            'Num_Decoders': num_decoders,
            'Gate_Mean': gate_stats.get('mean', 0),
            'Gate_Std': gate_stats.get('std', 0),
            'Gate_Median': gate_stats.get('median', 0),
            'Gate_Stability': gate_stats.get('stability', 0)
        })
        
    def save_results(self):
        # Create main results DataFrame
        df_results = pd.DataFrame(self.results)
        
        # Create gate info DataFrame
        df_gates = pd.DataFrame(self.gate_info) if self.gate_info else pd.DataFrame()
        
        # Save to Excel with multiple sheets
        with pd.ExcelWriter('ablation_study_results.xlsx', engine='openpyxl') as writer:
            # Main results sheet
            df_results.to_excel(writer, sheet_name='Main_Results', index=False)
            
            # Gate analysis sheet (if available)
            if not df_gates.empty:
                df_gates.to_excel(writer, sheet_name='Gate_Analysis', index=False)
            
            # Summary statistics sheet
            self._create_summary_sheet(writer, df_results)
            
            # Performance comparison sheet
            self._create_performance_sheet(writer, df_results)
        
        print(f"✅ Results saved to ablation_study_results.xlsx")
        
    def _create_summary_sheet(self, writer, df):
        # Create summary statistics
        summary_data = []
        
        # Overall statistics
        summary_data.append(['Overall Statistics', '', '', '', '', '', '', '', ''])
        summary_data.append(['Total Experiments', len(df), '', '', '', '', '', '', ''])
        summary_data.append(['Average AUC', df['AUC_Score'].mean(), '', '', '', '', '', '', ''])
        summary_data.append(['Best AUC', df['AUC_Score'].max(), '', '', '', '', '', '', ''])
        summary_data.append(['Average Training Time (min)', df['Training_Time_Minutes'].mean(), '', '', '', '', '', '', ''])
        summary_data.append(['Average Inference Time (s)', df['Inference_Time_Seconds'].mean(), '', '', '', '', '', '', ''])
        summary_data.append(['', '', '', '', '', '', '', '', ''])
        
        # By mode
        for mode in df['Mode'].unique():
            mode_df = df[df['Mode'] == mode]
            summary_data.append([f'{mode} Mode Statistics', '', '', '', '', '', '', '', ''])
            summary_data.append(['Count', len(mode_df), '', '', '', '', '', '', ''])
            summary_data.append(['Average AUC', mode_df['AUC_Score'].mean(), '', '', '', '', '', '', ''])
            summary_data.append(['Best AUC', mode_df['AUC_Score'].max(), '', '', '', '', '', '', ''])
            summary_data.append(['', '', '', '', '', '', '', '', ''])
        
        # By fusion type (for full mode only)
        full_df = df[df['Mode'] == 'full']
        if not full_df.empty:
            for fusion in full_df['Fusion_Type'].unique():
                fusion_df = full_df[full_df['Fusion_Type'] == fusion]
                summary_data.append([f'{fusion} Fusion Statistics', '', '', '', '', '', '', '', ''])
                summary_data.append(['Count', len(fusion_df), '', '', '', '', '', '', ''])
                summary_data.append(['Average AUC', fusion_df['AUC_Score'].mean(), '', '', '', '', '', '', ''])
                summary_data.append(['Best AUC', fusion_df['AUC_Score'].max(), '', '', '', '', '', '', ''])
                summary_data.append(['', '', '', '', '', '', '', '', ''])
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data, columns=['Metric', 'Value', 'Col3', 'Col4', 'Col5', 'Col6', 'Col7', 'Col8', 'Col9'])
        summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
    def _create_performance_sheet(self, writer, df):
        # Create performance comparison pivot tables
        performance_data = []
        
        # Best performance by dataset and mode
        for dataset in df['Dataset'].unique():
            dataset_df = df[df['Dataset'] == dataset]
            best_auc = dataset_df.loc[dataset_df['AUC_Score'].idxmax()]
            performance_data.append([
                dataset, 
                best_auc['Mode'], 
                best_auc['Fusion_Type'], 
                best_auc['Num_Decoders'],
                best_auc['AUC_Score'],
                best_auc['Training_Time_Minutes'],
                best_auc['Inference_Time_Seconds']
            ])
        
        perf_df = pd.DataFrame(performance_data, columns=[
            'Dataset', 'Best_Mode', 'Best_Fusion', 'Best_Decoders', 
            'Best_AUC', 'Training_Time_Min', 'Inference_Time_Sec'
        ])
        perf_df.to_excel(writer, sheet_name='Performance_Comparison', index=False)

def extract_best_auc(experiment_name):
    """Extract the best AUC score for an experiment"""
    try:
        # Try to read from AUC tracker first
        from src.auc_tracker_abalation import read_auc_results
        results = read_auc_results()
        
        # Find the experiment in results
        for result in results:
            if experiment_name in result.get('experiment_id', ''):
                return result.get('auc_score', 0.0)
        
        # Fallback: calculate from residuals
        residuals_file = f"Results/{experiment_name}/residuals_best.mat"
        if os.path.exists(residuals_file):
            from scipy.io import loadmat
            residuals = loadmat(residuals_file)['residuals']
            # Calculate AUC from residuals (simplified)
            return 0.5  # Placeholder - you may want to implement proper AUC calculation
            
    except Exception as e:
        print(f"Error reading AUC tracker: {e}")
    
    return 0.0

def extract_gate_statistics(model):
    """Extract gate statistics from trained model"""
    try:
        if hasattr(model, 'get_gate_statistics'):
            return model.get_gate_statistics()
    except:
        pass
    return None

def measure_inference_time(model, dataset, batch_size):
    """Measure inference time for the model"""
    try:
        model.eval()
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Warm up
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                _ = model(inputs)
                if i >= 2:  # Warm up for 3 batches
                    break
        
        # Measure inference time
        import time
        times = []
        with torch.no_grad():
            for i, (inputs, _) in enumerate(test_loader):
                if torch.cuda.is_available():
                    inputs = inputs.cuda()
                
                start_time = time.time()
                _ = model(inputs)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                times.append(end_time - start_time)
                if i >= 9:  # Measure 10 batches
                    break
        
        avg_time = sum(times) / len(times)
        print(f"    Inference time: {avg_time:.3f} seconds per run (avg of {len(times)} runs)")
        return avg_time
        
    except Exception as e:
        print(f"    Error measuring inference time: {e}")
        return 0.0

def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

    dataset_files = [
        'Data/aviris_1',
        'Data/aviris_2', 
        'Data/cat-island',
        'Data/Cri',
        'Data/San_Diego',
        'Data/Salians_syn',
        'Data/abu-urban-2'
    ]

    # [Hyperparameters] - MEMORY EFFICIENT SETTINGS
    block_size = 16
    stride = 8  
    epochs = 30  # Reduced for memory efficiency
    batch_size = 32
    lr = 5e-4
    weight_decay = 1e-4
    
    # Gate tracking configuration - DISABLED for memory efficiency
    enable_gate_tracking = False  # Disabled to prevent memory issues
    
    # Fusion types for ablation
    fusion_types = ["simple", "advanced", "cross_attention", "hierarchical"]
    
    # Ablation study modes
    ablation_modes = [
        ("spatial", "Spatial Branch Only"),
        ("spectral", "Spectral Branch Only"), 
        ("full", "Both Branches (Gated Fusion)")
    ]
    
    # Decoder ablation modes
    decoder_ablation_modes = [
        (1, "1 Decoder Block"),
        (2, "2 Decoder Blocks"),
        (3, "3 Decoder Blocks")
    ]
    
    # Initialize results tracker
    results_tracker = AblationResultsTracker()

    print("=== MEMORY EFFICIENT ABLATION STUDY ===")
    print(f"Epochs: {epochs} (reduced for memory efficiency)")
    print(f"Gate tracking: {'ENABLED' if enable_gate_tracking else 'DISABLED'} (disabled for memory efficiency)")
    print(f"Datasets: {len(dataset_files)}")
    print(f"Total experiments: {len(dataset_files) * len(ablation_modes) * len(decoder_ablation_modes)}")
    print()

    for data_path in dataset_files:
        ds_name = re.sub(r"\.mat$", "", os.path.basename(data_path))
        print(f"\n=== PROCESSING DATASET: {ds_name} ===")

        dataset = HSIDataset(
            mat_file=data_path,
            data_name=ds_name,
            block_size=block_size,
            stride=stride
        )

        C = dataset.blocks.shape[1]

        # Run ablation study for each mode
        for mode, mode_description in ablation_modes:
            print(f"\n--- Testing {mode_description} for {ds_name} ---")
            
            if mode == "full":
                # For "full" mode: test all fusion types and decoders
                for fusion_type in fusion_types:
                    print(f"\n--- Testing {fusion_type.upper()} fusion for {ds_name} ---")
                    
                    for num_decoders, decoder_description in decoder_ablation_modes:
                        print(f"\n--- Running Both Branches (Gated Fusion) with {decoder_description} using {fusion_type} fusion ---")
                        
                        # Record start time
                        start_time = datetime.now()
                        
                        # Create model (SIMPLE APPROACH - no complex state management)
                        model = AnomalyDetectionModel(
                            in_channels=C,
                            mode=mode,
                            dim=64,
                            depth=1,
                            spec_num=12,
                            spec_rate=0.5,
                            spa_token=16,
                            num_decoders=num_decoders,
                            fusion_type=fusion_type,
                        )

                        # Gate tracking disabled for memory efficiency
                        if enable_gate_tracking:
                            print(f"[GATE TRACKING] Enabling gate tracking for {fusion_type} fusion")
                            model.start_gate_tracking()
                        else:
                            print(f"[INFO] Gate tracking disabled for memory efficiency")

                        # Print parameter count for verification
                        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                        print(f"Model parameters: {param_count:,}")

                        if torch.cuda.is_available():
                            model = model.cuda()

                        # Create unique experiment name including fusion type
                        experiment_name = f"{ds_name}_{mode}_{fusion_type}_fusion_{num_decoders}dec_fixed"
                        
                        try:
                            trained_model = train_model(
                                model,
                                dataset,
                                experiment_name,
                                epochs=epochs,
                                batch_sz=batch_size,
                                lr=lr,
                                wd=weight_decay,
                                eval_dataset_path="../Data/HAD100Dataset/",
                                mode=mode,
                                num_decoders=num_decoders
                            )
                            
                            # Calculate training time
                            end_time = datetime.now()
                            training_time = (end_time - start_time).total_seconds() / 60.0
                            
                            # Extract AUC from the best model
                            best_auc = extract_best_auc(experiment_name)
                            
                            # Extract gate statistics if available
                            gate_stats = extract_gate_statistics(trained_model)
                            
                            # Measure inference time
                            inference_time = measure_inference_time(trained_model, dataset, batch_size)
                            
                            # Add results to tracker
                            results_tracker.add_result(
                                dataset_name=ds_name,
                                mode=mode,
                                fusion_type=fusion_type,
                                num_decoders=num_decoders,
                                auc_score=best_auc,
                                param_count=param_count,
                                training_time=training_time,
                                epoch_reached=epochs,
                                inference_time=inference_time,
                                additional_info={'Status': 'Completed'}
                            )
                            
                            # Add gate information
                            if gate_stats:
                                results_tracker.add_gate_info(ds_name, fusion_type, num_decoders, gate_stats)
                            
                            print(f"Completed {fusion_type} fusion with {decoder_description} for {ds_name} - AUC: {best_auc:.4f}")
                            
                        except Exception as e:
                            print(f"Error in experiment {experiment_name}: {e}")
                            end_time = datetime.now()
                            training_time = (end_time - start_time).total_seconds() / 60.0
                            
                            results_tracker.add_result(
                                dataset_name=ds_name,
                                mode=mode,
                                fusion_type=fusion_type,
                                num_decoders=num_decoders,
                                auc_score=0.0,
                                param_count=param_count,
                                training_time=training_time,
                                epoch_reached=0,
                                inference_time=0.0,
                                additional_info={'Status': f'Failed: {str(e)}'}
                            )
                            
            else:
                # For "spatial" and "spectral" modes: test all decoders (no fusion)
                for num_decoders, decoder_description in decoder_ablation_modes:
                    print(f"\n--- Running {mode_description} with {decoder_description} ---")
                    
                    # Record start time
                    start_time = datetime.now()
                    
                    # Create model (SIMPLE APPROACH - no complex state management)
                    model = AnomalyDetectionModel(
                        in_channels=C,
                        mode=mode,
                        dim=64,
                        depth=1,
                        spec_num=12,
                        spec_rate=0.5,
                        spa_token=16,
                        num_decoders=num_decoders,
                        fusion_type="simple",  # Default fusion type (not used for single-branch modes)
                    )

                    # Print parameter count for verification
                    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                    print(f"Model parameters: {param_count:,}")

                    if torch.cuda.is_available():
                        model = model.cuda()

                    # Create unique experiment name
                    experiment_name = f"{ds_name}_{mode}_{num_decoders}dec_fixed"
                    
                    try:
                        trained_model = train_model(
                            model,
                            dataset,
                            experiment_name,
                            epochs=epochs,
                            batch_sz=batch_size,
                            lr=lr,
                            wd=weight_decay,
                            eval_dataset_path="../Data/HAD100Dataset/",
                            mode=mode,
                            num_decoders=num_decoders
                        )
                        
                        # Calculate training time
                        end_time = datetime.now()
                        training_time = (end_time - start_time).total_seconds() / 60.0
                        
                        # Extract AUC from the best model
                        best_auc = extract_best_auc(experiment_name)
                        
                        # Measure inference time
                        inference_time = measure_inference_time(trained_model, dataset, batch_size)
                        
                        # Add results to tracker
                        results_tracker.add_result(
                            dataset_name=ds_name,
                            mode=mode,
                            fusion_type="N/A",
                            num_decoders=num_decoders,
                            auc_score=best_auc,
                            param_count=param_count,
                            training_time=training_time,
                            epoch_reached=epochs,
                            inference_time=inference_time,
                            additional_info={'Status': 'Completed'}
                        )
                        
                        print(f"Completed {mode_description} with {decoder_description} for {ds_name} - AUC: {best_auc:.4f}")
                        
                    except Exception as e:
                        print(f"Error in experiment {experiment_name}: {e}")
                        end_time = datetime.now()
                        training_time = (end_time - start_time).total_seconds() / 60.0
                        
                        results_tracker.add_result(
                            dataset_name=ds_name,
                            mode=mode,
                            fusion_type="N/A",
                            num_decoders=num_decoders,
                            auc_score=0.0,
                            param_count=param_count,
                            training_time=training_time,
                            epoch_reached=0,
                            inference_time=0.0,
                            additional_info={'Status': f'Failed: {str(e)}'}
                        )

    # Generate visualizations for all experiments
    print("\n=== GENERATING VISUALIZATIONS FOR ALL EXPERIMENTS ===")
    try:
        from quick_visualize import main as generate_visualizations
        generate_visualizations()
        print("✅ Visualizations generated successfully!")
    except Exception as e:
        print(f"⚠️  Visualization generation failed: {e}")
        print("   You can run 'python quick_visualize.py' manually to generate visualizations")

    # Save all results to Excel
    print("\n=== SAVING COMPREHENSIVE RESULTS ===")
    results_tracker.save_results()

    print("\n=== ALL EXPERIMENTS COMPLETED ===")
    print("Results saved in:")
    for data_path in dataset_files:
        ds_name = re.sub(r"\.mat$", "", os.path.basename(data_path))
        for mode, _ in ablation_modes:
            if mode == "full":
                for fusion_type in fusion_types:
                    for num_decoders, _ in decoder_ablation_modes:
                        print(f"  - Models/{ds_name}_{mode}_{fusion_type}_fusion_{num_decoders}dec_fixed/")
                        print(f"  - Results/{ds_name}_{mode}_{fusion_type}_fusion_{num_decoders}dec_fixed/")
            else:
                for num_decoders, _ in decoder_ablation_modes:
                    print(f"  - Models/{ds_name}_{mode}_{num_decoders}dec_fixed/")
                    print(f"  - Results/{ds_name}_{mode}_{num_decoders}dec_fixed/")
    
    print(f"\n📊 COMPREHENSIVE RESULTS: ablation_study_results.xlsx")
    print("   - Main_Results: All experiment results")
    print("   - Gate_Analysis: Gate statistics (if available)")
    print("   - Summary_Statistics: Overall performance summary")
    print("   - Performance_Comparison: Best configurations per dataset")

if __name__ == "__main__":
    main()

