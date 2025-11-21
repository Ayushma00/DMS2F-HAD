import os
import re
import random
import numpy as np
import torch
from src.data_loader import HSIDataset
from src.model_fixed import AnomalyDetectionModel  # Use the fixed model
from src.train import train_model


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

    # [Hyperparameters]
    block_size = 16
    stride = 8  
    epochs = 150
    batch_size = 32
    lr = 5e-4
    weight_decay = 1e-4
    
    # Gate tracking configuration
    enable_gate_tracking = True  # Set to False to disable gate tracking for faster training
    
    # Fusion configuration - IMPROVED FUSION MECHANISMS
    fusion_type = "advanced"  # Options: "simple", "advanced", "cross_attention", "hierarchical"

    # Ablation study modes - FIXED VERSION
    ablation_modes = [
        # ("spatial", "Spatial Branch Only"),
        # ("spectral", "Spectral Branch Only"), 
        ("full", "Both Branches (Gated Fusion)")
    ]
    
    # Decoder ablation modes
    decoder_ablation_modes = [
        # (1, "1 Decoder Block"),
        (2, "2 Decoder Blocks"),
        # (3, "3 Decoder Blocks")
    ]
    
    # Store results for comparison
    ablation_results = {}

    for data_path in dataset_files:
        ds_name = re.sub(r"\.mat$", "", os.path.basename(data_path))
        print(f"\n\n=== PROCESSING DATASET: {ds_name} ===")

        dataset = HSIDataset(
            mat_file=data_path,
            data_name = ds_name,
            block_size=block_size,
            stride=stride
        )

        C = dataset.blocks.shape[1]
        ablation_results[ds_name] = {}

        # Run ablation study for each branch mode and decoder configuration
        for mode, mode_description in ablation_modes:
            for num_decoders, decoder_description in decoder_ablation_modes:
                print(f"\n--- Running {mode_description} with {decoder_description} for {ds_name} ---")
                
                # FIXED: Use the corrected model with decoder ablation and improved fusion
                model = AnomalyDetectionModel(
                    in_channels=C,
                    mode=mode,  # This now properly controls which branches are computed
                    dim=64,
                    depth=1,
                    spec_num=12,
                    spec_rate=0.5,
                    spa_token=16,
                    num_decoders=num_decoders,  # Add decoder ablation parameter
                    fusion_type=fusion_type,  # Improved fusion mechanism
                )

                # Enable gate tracking for 'full' mode experiments
                if mode == "full" and enable_gate_tracking:
                    print(f"[GATE TRACKING] Enabling gate tracking for {mode_description}")
                    model.start_gate_tracking()
                elif mode == "full" and not enable_gate_tracking:
                    print(f"[INFO] Gate tracking disabled for faster training")

                # Print parameter count for verification
                param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
                print(f"Model parameters: {param_count:,}")

                if torch.cuda.is_available():
                    model = model.cuda()

                # Create unique dataset name for each mode and decoder configuration
                experiment_name = f"{ds_name}_{mode}_{num_decoders}dec_fixed"  # Include decoder count
                
                trained_model = train_model(
                    model,
                    dataset,
                    experiment_name,
                    epochs=epochs,
                    batch_sz=batch_size,
                    lr=lr,
                    wd=weight_decay,
                    eval_dataset_path="../Data/HAD100Dataset/",
                    mode=mode,  # Pass mode for AUC tracking
                    num_decoders=num_decoders  # Pass decoder count for AUC tracking
                )
                
                print(f"Completed {mode_description} with {decoder_description} for {ds_name}")

    print("\n=== ABLATION STUDY COMPLETED ===")
    print("Results saved in separate folders for each mode and decoder configuration:")
    for data_path in dataset_files:
        ds_name = re.sub(r"\.mat$", "", os.path.basename(data_path))
        for mode, _ in ablation_modes:
            for num_decoders, _ in decoder_ablation_modes:
                print(f"  - Models/{ds_name}_{mode}_{num_decoders}dec_fixed/")
                print(f"  - Results/{ds_name}_{mode}_{num_decoders}dec_fixed/")
                
                # Show gate tracking results if enabled
                if mode == "full" and enable_gate_tracking:
                    results_dir = f"Results/{ds_name}_{mode}_{num_decoders}dec_fixed"
                    print(f"    Gate Analysis Files:")
                    print(f"      - {results_dir}/gate_distribution_analysis.png")
                    print(f"      - {results_dir}/gate_distribution_early.png")
                    print(f"      - {results_dir}/gate_distribution_late.png")
                    print(f"      - {results_dir}/gate_statistics.txt")
                    print(f"      - {results_dir}/intermediate_gates/ (every 30 epochs)")
    
    if enable_gate_tracking:
        print(f"\n🎨 GATE TRACKING ENABLED: Look for KDE histogram plots in Results folders!")
        print("   These plots show how gate values evolve across all 150 epochs.")
    else:
        print(f"\n⚡ GATE TRACKING DISABLED: Set enable_gate_tracking=True to generate plots.")
    
    print(f"\n🔧 FUSION MECHANISM: Using '{fusion_type}' fusion")
    if fusion_type == "simple":
        print("   ⚠️  Warning: Simple fusion has limited expressiveness")
        print("   💡 Consider upgrading to 'advanced' for better performance")
    elif fusion_type == "advanced":
        print("   ✅ Recommended: Good balance of performance and efficiency")
    elif fusion_type == "cross_attention":
        print("   🚀 High performance but computationally expensive")
    elif fusion_type == "hierarchical":
        print("   🎯 Multi-scale processing for complex anomalies")


if __name__ == "__main__":
    main()
