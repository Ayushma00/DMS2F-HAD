import numpy as np
import torch


def block_search(image_shape, block_size, stride, pad=True):
    """
    Computes top-left coordinates for overlapping blocks to cover an image.

    Parameters
    ----------
    image_shape : tuple
        Shape of the image as (height, width).
    block_size : int
        Size of the square block (block_size x block_size).
    stride : int
        Stride for moving the block.
    pad : bool
        Whether to pad the image to ensure full coverage.

    Returns
    -------
    positions : list of tuples
        List of (i, j) coordinates for the top-left corner of each block.
    padded_shape : tuple
        Shape of the padded image as (height, width).
    """
    H, W = image_shape[:2]
    pad_h = pad_w = 0
    if pad:
        if (H - block_size) % stride != 0:
            pad_h = stride - ((H - block_size) % stride)
        if (W - block_size) % stride != 0:
            pad_w = stride - ((W - block_size) % stride)
    H_pad = H + pad_h
    W_pad = W + pad_w
    positions = []
    for i in range(0, H_pad - block_size + 1, stride):
        for j in range(0, W_pad - block_size + 1, stride):
            positions.append((i, j))
    return positions, (H_pad, W_pad)


def block_embedding(image, block_size, stride):
    """
    Extracts overlapping blocks from a 3D image (H x W x C).

    Parameters
    ----------
    image : np.ndarray
        Input image array of shape (H, W, C).
    block_size : int
        Size of the square block.
    stride : int
        Stride between blocks.

    Returns
    -------
    blocks : torch.Tensor
        Tensor of shape (N, C, block_size, block_size).
    padded_shape : tuple
        Shape of the padded image (H_pad, W_pad).
    positions : list
        Top-left corner coordinates of blocks.
    """
    if image.ndim == 3:
        H, W = image.shape[0], image.shape[1]
        positions, (H_pad, W_pad) = block_search((H, W), block_size, stride, pad=True)
        pad_h = H_pad - H
        pad_w = W_pad - W
        if pad_h > 0 or pad_w > 0:
            image_array = np.pad(
                image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant"
            )
        else:
            image_array = image
        blocks = []
        for i, j in positions:
            block = image_array[i : i + block_size, j : j + block_size, :]
            blocks.append(block)
        blocks = np.stack(blocks, axis=0)  # shape: (N, block_size, block_size, C)
        blocks = torch.from_numpy(blocks.astype(np.float32)).permute(
            0, 3, 1, 2
        )  # (N, C, block_size, block_size)
        return blocks, (H_pad, W_pad), positions
    else:
        raise ValueError("Unsupported image format for block embedding.")


def block_fold(blocks, image_shape, block_size, stride, positions):
    """
    Reconstructs an image from overlapping blocks by averaging overlapping areas.

    Parameters
    ----------
    blocks : torch.Tensor
        Tensor of shape (N, C, block_size, block_size).
    original_shape : tuple
        Shape of the original image (H, W).
    block_size : int
        Size of each block.
    stride : int
        Stride used during block extraction.
    positions : list
        Top-left coordinates of the blocks.

    Returns
    -------
    image : torch.Tensor
        Reconstructed image of shape (C, H, W).
    """
    H_orig, W_orig = image_shape
    H_pad = max(i for i, j in positions) + block_size
    W_pad = max(j for i, j in positions) + block_size
    C = blocks.shape[1]
    output = torch.zeros((C, H_pad, W_pad), dtype=blocks.dtype)
    count = torch.zeros((H_pad, W_pad), dtype=blocks.dtype)
    for block, (i, j) in zip(blocks, positions):
        output[:, i : i + block_size, j : j + block_size] += block
        count[i : i + block_size, j : j + block_size] += 1
    count[count == 0] = 1  # avoid division by zero
    output = output / count.unsqueeze(0)
    output = output[:, :H_orig, :W_orig]
    return output


def img2mask(img):
    """
    Convert a 3D anomaly map into a 2D normalized mask.
    Parameters
    ----------
    img : numpy.ndarray or torch.Tensor
        Either shape (bands, H, W) or (1, bands, H, W).

    Returns
    -------
    mask : numpy.ndarray
        Float array of shape (H, W) with values in [0,1], where high values
        correspond to large per‑pixel anomaly scores.
    """
    if torch.is_tensor(img):
        img = img.detach().cpu().numpy()
    if img.ndim == 4 and img.shape[0] == 1:
        img = img[0]
    mask = np.linalg.norm(img, axis=0)
    mn, mx = mask.min(), mask.max()
    if mx > mn:
        mask = (mask - mn) / (mx - mn)
    else:
        mask = np.zeros_like(mask)

    return mask
