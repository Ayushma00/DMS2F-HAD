import random
import numpy as np
import torch
from src.data_loader import HSIDataset
from src.model import AnomalyDetectionModel
from src.train import train_model
from src.project_paths import (
    DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    EVAL_DATASET_DIR,
    REPO_ROOT,
)


def main():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = True

    dataset_names = [
        # 'los-angeles-2',
        "aviris_1",
        "aviris_2",
        "cat-island",
        "Cri",
        # 'gulfport',
        # 'HYDICE_urban',
        "San_Diego",
        "Salians_syn",
        "abu-urban-2",
        # 'texas-goast','pavia','los-angeles-1',
        # 'abu-beach-3','abu-urban-3',
        # 'abu-urban-4','abu-urban-5',
    ]

    dataset_files = [DATA_DIR / f"{name}.mat" for name in dataset_names]

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
        ds_name = data_path.stem
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
                eval_dataset_path=EVAL_DATASET_DIR,  
            )
            
            print(f"Completed {mode_description} for {ds_name}")

    print("\n=== ABLATION STUDY COMPLETED ===")
    print("Results saved in separate folders for each mode:")
    models_root = MODELS_DIR.relative_to(REPO_ROOT)
    results_root = RESULTS_DIR.relative_to(REPO_ROOT)
    for data_path in dataset_files:
        ds_name = data_path.stem
        for mode, _ in ablation_modes:
            print(f"  - {models_root}/{ds_name}_{mode}/")
            print(f"  - {results_root}/{ds_name}_{mode}/")


if __name__ == "__main__":
    main()
