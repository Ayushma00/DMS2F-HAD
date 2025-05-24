import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat
import h5py
from src.mask import Mask
from src.block_utils import block_embedding


class HSIDataset(Dataset):

    def __init__(self, mat_file, block_size=16, stride=8):
        if isinstance(mat_file, str):
            mat = loadmat(mat_file)
            self.image = mat["data"]
            self.gt_mask = mat["map"] if "map" in mat else None

        else:
            mat = mat_file

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

    def __init__(self, mat_file, block_size=16, stride=8):
        if isinstance(mat_file, str):
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
        print(
            self.blocks.shape
        ) 
        self.block_size = block_size
        self.stride = stride
        self.positions = positions  
        self.padded_shape = (H_pad, W_pad)
        self.H = self.image.shape[0] 
        self.W = self.image.shape[1]  
        self.N = blocks.shape[0]
        self.mask_generator = Mask(
            w=block_size, h=block_size, resize=block_size, sub_w_num=4, sub_h_num=4
        )

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        block = self.blocks[idx]
        mask = self.mask_generator(n=1)[0]
        mask = torch.from_numpy(mask).float()
        masked_block = block * mask.unsqueeze(0)
        return masked_block, block
