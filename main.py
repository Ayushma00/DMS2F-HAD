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
       
        "Data/Cri.mat",
        "Data/San_Diego.mat"
    ]

    # [Hyperparameters]
    block_size = 16
    stride = 8
    epochs = 100
    batch_size = 64
    lr = 1e-3
    weight_decay = 6.938599279960116e-05

    for data_path in dataset_files:
        ds_name = re.sub(r"\.mat$", "", os.path.basename(data_path))
        print(f"\n\n=== PROCESSING DATASET: {ds_name} ===")

        dataset = HSIDataset(
            mat_file=data_path,
            block_size=block_size,
            stride=stride
        )

        C = dataset.blocks.shape[1]

        model = AnomalyDetectionModel(
            in_channels=C,
            mode="full",
            dim=64,
            depth=1,
            spec_num=12,
            spec_rate=0.5,
            spa_token=16,
        )

        if torch.cuda.is_available():
            model = model.cuda()

        trained_model = train_model(
            model,
            dataset,
            f"{ds_name}",
            epochs=epochs,
            batch_sz=batch_size,
            lr=lr,
            wd=weight_decay,
            eval_dataset_path="../Data/HAD100Dataset/"  
        )


if __name__ == "__main__":
    main()
