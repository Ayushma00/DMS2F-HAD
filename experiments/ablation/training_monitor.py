import os
import re
import random
import numpy as np
import torch
from src.data_loader import HSIDataset
from src.model import AnomalyDetectionModel
from src.train import train_model


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

    dataset_files = [
       

        # 'Data/los-angeles-2',
        
        'Data/aviris_1',
        'Data/aviris_2',
        'Data/cat-island',
        'Data/Cri',
        # 'Data/gulfport',
        # 'Data/HYDICE_urban',
        'Data/San_Diego',
        'Data/Salians_syn','Data/abu-urban-2'

        # 'Data/texas-goast','Data/pavia','Data/los-angeles-1',
        # 'Data/abu-beach-3','Data/abu-urban-2','Data/abu-urban-3',
        # 'Data/abu-urban-4','Data/abu-urban-5',
        

    ]

    # [Hyperparameters]
    block_size = 16
    stride = 8  
    epochs = 150  # More epochs
    batch_size = 32  # Smaller batch for better gradients
    lr = 5e-4  # Lower learning rate
    weight_decay = 1e-4

    # Ablation study modes
    ablation_modes = [
        ("spatial", "Spatial Branch Only"),
        ("spectral", "Spectral Branch Only"), 
        ("full", "Both Branches (Gated Fusion)")
    ]

    '''
    block_size = 16
        stride = 8  
        epochs = 150  # More epochs
        batch_size = 32  # Smaller batch for better gradients
        lr = 5e-4  # Lower learning rate
        weight_decay = 1e-4
        AVIRIS 2, CAT ISLAND, GULF PORT, 
    '''

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

        # Run ablation study for each mode
        for mode, mode_description in ablation_modes:
            print(f"\n--- Running {mode_description} for {ds_name} ---")
            
            model = AnomalyDetectionModel(
                in_channels=C,
                mode=mode,  # This is the key change for ablation
                dim=64,
                depth=1,
                spec_num=12,
                spec_rate=0.5,
                spa_token=16,
            )

            if torch.cuda.is_available():
                model = model.cuda()

            # Create unique dataset name for each mode
            experiment_name = f"{ds_name}_{mode}"
            
            trained_model = train_model(
                model,
                dataset,
                experiment_name,  # Different name for each ablation
                epochs=epochs,
                batch_sz=batch_size,
                lr=lr,
                wd=weight_decay,
                eval_dataset_path="../Data/HAD100Dataset/"  
            )
            
            print(f"Completed {mode_description} for {ds_name}")

    print("\n=== ABLATION STUDY COMPLETED ===")
    print("Results saved in separate folders for each mode:")
    for data_path in dataset_files:
        ds_name = re.sub(r"\.mat$", "", os.path.basename(data_path))
        for mode, _ in ablation_modes:
            print(f"  - Models/{ds_name}_{mode}/")
            print(f"  - Results/{ds_name}_{mode}/")


if __name__ == "__main__":
    main()
