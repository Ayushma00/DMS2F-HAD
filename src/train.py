import os
import time
import numpy as np
import scipy.io as sio
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.block_utils import block_fold
from skimage.transform import resize
# from datasets.HADDatasets import HADTestDataset

def save_residuals(recon_full, orig_full, gt_mask, out_path):
    sio.savemat(out_path, {
        'residual_map': recon_full,
        'gt_mask': gt_mask,
        'original': orig_full
    })

def evaluate_model(model, dataset, device, batch_sz):
    model.eval()
    torch.set_grad_enabled(False)
    val_loader = DataLoader(dataset, batch_size=batch_sz, shuffle=False, num_workers=4, pin_memory=True)
    val_aucs, residual_maps, original_maps, gt_maps, image_map = [], [], [], [], []
    skip_count = 0

    for masked_in, target in val_loader:
        masked_in = masked_in.to(device)
        out, _ = model(masked_in)

        for recon, gt, mask in zip(out.cpu(), target, masked_in.cpu()):
            recon_np = recon.numpy()
            original_np = mask.numpy()
            diff = (recon_np - original_np)

            gt_np = gt.numpy()[0] if gt.ndim == 3 else gt.numpy()
            gt_resized = resize(gt_np, recon_np.shape[1:], order=0, preserve_range=True, anti_aliasing=False)
            err_map = np.linalg.norm(diff, axis=0)
            residual_maps.append(err_map)
            original_maps.append(gt_resized)

            original_vis = np.mean(original_np, axis=0)
            image_map.append(original_vis)

            gt_flat = gt_resized.astype(int).ravel()
            if np.unique(gt_flat).size < 2:
                skip_count += 1  
                continue
            auc = roc_auc_score(gt_flat, err_map.ravel())
            val_aucs.append(auc)

    if skip_count > 0:
        print(f"[INFO] Skipped {skip_count} images with uniform ground truth.")  

    val_auc = np.mean(val_aucs) if val_aucs else 0.0
    return val_auc, residual_maps, image_map, [np.array(o > 0, dtype=int) for o in original_maps]


def train_model(model, dataset, dataset_name, epochs, batch_sz, lr, wd, eval_dataset_path="../Data/HAD100Dataset/"):
    device = next(model.parameters()).device
    train_loader = DataLoader(dataset, batch_size=batch_sz, shuffle=True, num_workers=4, pin_memory=True)
    mse_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scaler = GradScaler()

    best_auc = 0.0
    best_auc_path = None
    test_ds = DataLoader(dataset, batch_size=batch_sz, shuffle=False, num_workers=4, pin_memory=True)

    model_dir = os.path.join("Models", dataset_name)
    os.makedirs(model_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss, mse_loss_total, l1_loss_total = 0.0, 0.0, 0.0
        start_time = time.time()

        for masked_inputs, targets in train_loader:
            masked_inputs = masked_inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            with autocast():
                recon, _ = model(masked_inputs)
                mse_loss = mse_crit(recon, targets)
                l1_loss = l1_crit(recon, targets)
                loss = mse_loss + 0.1 * l1_loss

            if loss.requires_grad:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item() * masked_inputs.size(0)
            mse_loss_total += mse_loss.item() * masked_inputs.size(0)
            l1_loss_total += l1_loss.item() * masked_inputs.size(0)

        epoch_time = time.time() - start_time
        num_samples = len(dataset)
        print(f"Epoch {epoch} — Train Loss: {running_loss / num_samples:.4f} | MSE: {mse_loss_total / num_samples:.4f} | L1: {l1_loss_total / num_samples:.4f} | Time: {epoch_time:.2f}s")

        # [Refactored] Evaluation
        if hasattr(dataset, 'blocks'):
            val_loader = DataLoader(dataset.blocks, batch_size=batch_sz, shuffle=False, num_workers=4, pin_memory=True)
            recs = []
            for blocks in val_loader:
                out, _ = model(blocks.to(device))
                recs.append(out.cpu())
            recs = torch.cat(recs, dim=0)
            recon_full = block_fold(recs, (dataset.H, dataset.W), dataset.block_size, dataset.stride, dataset.positions)
            orig_full = torch.tensor(dataset.image.astype(np.float32)).permute(2, 0, 1)
            diff = (recon_full - orig_full).numpy()
            err_map = np.linalg.norm(diff, axis=0).ravel()
            gt_flat = dataset.gt_mask.ravel().astype(int)
            val_auc = roc_auc_score(gt_flat, err_map)
            save_gt = dataset.gt_mask
        else:
            val_auc, recon_full, orig_full, save_gt = evaluate_model(model, test_ds, device, batch_sz)  # [Refactored]

        
        if val_auc > best_auc:
            best_auc = val_auc
            best_auc_path = os.path.join(model_dir, "best_model_auc.pt")
            torch.save(model.state_dict(), best_auc_path)

            out_mat = os.path.join("Results", dataset_name, "residuals_best.mat")
            os.makedirs(os.path.dirname(out_mat), exist_ok=True)
            save_residuals(recon_full, orig_full, save_gt, out_mat)
            print(f" --New best AUC={best_auc:.4f}, saved model and residuals.")


        last_model_path = os.path.join(model_dir, "last_model.pt")
        torch.save(model.state_dict(), last_model_path)

    return model
