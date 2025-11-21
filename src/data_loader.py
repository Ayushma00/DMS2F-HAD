import torch
from torch.utils.data import Dataset
import numpy as np
import logging
from scipy.io import loadmat
import h5py
from pathlib import Path
from src.mask import Mask
from src.block_utils import block_embedding
from src.project_paths import DATA_DIR, REPO_ROOT


def _resolve_data_path(mat_file: str) -> Path:
    """Resolve legacy Data/ paths to the new data directory."""
    mat_path = Path(mat_file)
    parts = mat_path.parts
    if parts:
        if parts[0] == "Data":
            mat_path = DATA_DIR.joinpath(*parts[1:])
        elif parts[0] in {".", "./"} and len(parts) > 1 and parts[1] == "Data":
            mat_path = DATA_DIR.joinpath(*parts[2:])
        elif len(parts) > 1 and parts[0] == ".." and parts[1] == "Data":
            mat_path = DATA_DIR.joinpath(*parts[2:])
        elif not mat_path.is_absolute():
            candidate = (REPO_ROOT / mat_path).resolve()
            if candidate.exists():
                mat_path = candidate
    if mat_path.suffix == "":
        candidate = mat_path.with_suffix(".mat")
        if candidate.exists():
            mat_path = candidate
    if not mat_path.exists():
        fallback = DATA_DIR / mat_path.name
        if fallback.suffix == "":
            fallback = fallback.with_suffix(".mat")
        if fallback.exists():
            mat_path = fallback
    return mat_path
logging.basicConfig(level=logging.INFO)


class HSIDataset(Dataset):
    """
    Hyperspectral image dataset

    Args:
        mat_file: Path to the mat file
        data_name: Name of the dataset
        block_size: Size of the block
        stride: Stride of the block
    """
    def __init__(self, mat_file,data_name, block_size=16, stride=8):
        if isinstance(mat_file, Path):
            mat_file = str(mat_file)
        if isinstance(mat_file, str):
            resolved_path = _resolve_data_path(mat_file)
            mat_file = str(resolved_path)
            if data_name == "Salians_syn":
                mat = loadmat(mat_file)
                self.image = mat["hsi"]
                map_file = loadmat(str(DATA_DIR / "Salians_gt.mat"))
                self.gt_mask =map_file["hsi_gt"]
            else:
                mat = loadmat(mat_file)
                self.image = mat["data"]
                self.gt_mask = mat["map"] if "map" in mat else None

            print(f"Loaded data shape: {self.image.shape}")
            # Ensure bands are in the last dimension
            if self.image.shape[0] == 30:  # If bands are in first dimension
                self.image = np.transpose(self.image, (1, 2, 0))
                print(f"Transposed data shape: {self.image.shape}")
        else:
            mat = mat_file
            self.image = mat["data"] if "data" in mat else mat.get("hsi", None)
            self.gt_mask = mat["map"] if "map" in mat else None

        self.image = self.image.astype(np.float32)
        data_min = self.image.min()
        data_max = self.image.max()
        if data_max > data_min:
            self.image = (self.image - data_min) / (data_max - data_min)
        else:
            self.image = np.zeros_like(self.image, dtype=np.float32)

        blocks, (H_pad, W_pad), positions = block_embedding(
            self.image, block_size, stride
        )
        self.blocks = blocks  

        self.block_size = block_size
        self.stride = stride
        self.positions = positions  

        self.padded_shape = (H_pad, W_pad)
        self.H = self.image.shape[0]  
        self.W = self.image.shape[1]  
        self.N = blocks.shape[0]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # In our self-supervised setting, the input is also the reconstruction target.
        return self.blocks[idx], self.blocks[idx]


class HSIMaskedDataset(Dataset):
    """
    Hyperspectral image dataset with masked blocks

    Args:
        mat_file: Path to the mat file
        block_size: Size of the block
        stride: Stride of the block
    """
    def __init__(self, mat_file, block_size=16, stride=8):
        if isinstance(mat_file, Path):
            mat_file = str(mat_file)
        if isinstance(mat_file, str):
            resolved_path = _resolve_data_path(mat_file)
            mat_file = str(resolved_path)
            if Path(mat_file).stem == "Salians_syn":
                mat = loadmat(mat_file)
                self.image = mat["hsi"]
            else:
                mat = loadmat(mat_file)
                self.image = mat["data"]
                self.gt_mask = mat["map"] if "map" in mat else None

        else:
            mat = mat_file

        # Normalize image to [0,1]
        self.image = self.image.astype(np.float32)
        data_min = self.image.min()
        data_max = self.image.max()
        if data_max > data_min:
            self.image = (self.image - data_min) / (data_max - data_min)
        else:
            self.image = np.zeros_like(self.image, dtype=np.float32)
        blocks, (H_pad, W_pad), positions = block_embedding(
            self.image, block_size, stride
        )
        self.blocks = torch.from_numpy(blocks.numpy()).float()
        self.N, self.C, self.bs, _ = self.blocks.shape
        logging.info(f"Blocks shape: {self.blocks.shape}")

        self.block_size = block_size
        self.stride = stride
        self.positions = positions  
        self.padded_shape = (H_pad, W_pad)
        self.H = self.image.shape[0] 
        self.W = self.image.shape[1]  
        self.mask_generator = Mask(
            w=block_size, h=block_size, resize=block_size, sub_w_num=4, sub_h_num=4
        )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        block = self.blocks[idx]
        mask = self.mask_generator(n=1)[0]
        mask = torch.from_numpy(mask).float()
        if mask.shape != block.shape[1:]:
            mask = mask.unsqueeze(0)
        masked_block = block * mask
        return masked_block, block
