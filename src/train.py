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
from src.auc_tracker_abalation import save_auc_result
import csv
from datetime import datetime
from pathlib import Path
from src.project_paths import MODELS_DIR, RESULTS_DIR, EVAL_DATASET_DIR
# from datasets.HADDatasets import HADTestDataset

def save_residuals(recon_full, orig_full, gt_mask, out_path):
    sio.savemat(str(out_path), {
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
    gt_masks = [np.array(o > 0, dtype=np.uint8) for o in original_maps]
    return val_auc, residual_maps, image_map, gt_masks


def train_model(model, dataset, dataset_name, epochs, batch_sz, lr, wd, eval_dataset_path=None, mode=None, num_decoders=None):
    if eval_dataset_path is None:
        eval_dataset_path = EVAL_DATASET_DIR
    device = next(model.parameters()).device
    train_loader = DataLoader(dataset, batch_size=batch_sz, shuffle=True, num_workers=4, pin_memory=True)
    mse_crit = nn.MSELoss()
    l1_crit = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    # scaler = GradScaler()

    best_auc = 0.0
    best_auc_path = None
    test_ds = DataLoader(dataset, batch_size=batch_sz, shuffle=False, num_workers=4, pin_memory=True)

    model_dir = Path(MODELS_DIR) / dataset_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Gate tracking disabled for simplicity
    
    model.train()
    
    # Ensure all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[TRAIN] Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
    
    if trainable_params == 0:
        raise ValueError("No trainable parameters found! Model cannot train.")
    
    for epoch in range(epochs):
        
        running_loss, mse_loss_total, l1_loss_total = 0.0, 0.0, 0.0
        start_time = time.time()

        for masked_inputs, targets in train_loader:
            masked_inputs = masked_inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            try:
                # with autocast():
                recon, _ = model(masked_inputs)
                mse_loss = mse_crit(recon, targets)
                l1_loss = l1_crit(recon, targets)
                loss = mse_loss + 0.1 * l1_loss

                # Debug: Check model parameters and gradients
                if epoch == 0:  # Only print once
                    print(f"[DEBUG] Model parameters requiring grad: {sum(p.requires_grad for p in model.parameters())}")
                    print(f"[DEBUG] Total parameters: {sum(p.numel() for p in model.parameters())}")
                    print(f"[DEBUG] Output requires grad: {recon.requires_grad}")
                    print(f"[DEBUG] Loss requires grad: {loss.requires_grad}")

                # Check if loss requires gradients
                if not loss.requires_grad:
                    print(f"[WARNING] Loss does not require gradients. Loss value: {loss.item()}")
                    print(f"[ERROR] Model parameters are not properly configured for training!")
                    continue

                # if loss.requires_grad:
                #     scaler.scale(loss).backward()
                #     scaler.step(optimizer)
                #     scaler.update()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * masked_inputs.size(0)
                mse_loss_total += mse_loss.item() * masked_inputs.size(0)
                l1_loss_total += l1_loss.item() * masked_inputs.size(0)
                
            except Exception as e:
                print(f"[ERROR] Training step failed: {e}")
                print(f"Input shape: {masked_inputs.shape}, Target shape: {targets.shape}")
                continue

        epoch_time = time.time() - start_time
        num_samples = len(dataset)
        print(f"Epoch {epoch} — Train Loss: {running_loss / num_samples:.4f} | MSE: {mse_loss_total / num_samples:.4f} | L1: {l1_loss_total / num_samples:.4f} | Time: {epoch_time:.2f}s")

        
        if hasattr(dataset, 'blocks'):
            print("[INFO] Performing block-based validation on dataset.")
            with torch.no_grad():
                val_loader = DataLoader(dataset.blocks, batch_size=batch_sz, shuffle=False, num_workers=4, pin_memory=True)
                recs = []
                val_time = time.time()
                for blocks in val_loader:
                    out, _ = model(blocks.to(device))
                    recs.append(out.cpu())
                val_time_end = time.time()
                print("validation time taken::",val_time_end-val_time)
                recs = torch.cat(recs, dim=0)
                recon_full = block_fold(recs, (dataset.H, dataset.W), dataset.block_size, dataset.stride, dataset.positions)
                orig_full = torch.tensor(dataset.image.astype(np.float32)).permute(2, 0, 1)
                diff = recon_full - orig_full
                err_map = (diff**2).sum(dim=0).numpy().ravel()
                res_map = np.linalg.norm(diff, axis=0)
                # err_map = np.linalg.norm(diff, axis=0).ravel()
                gt_flat = dataset.gt_mask.ravel().astype(int)
                val_auc = roc_auc_score(gt_flat, res_map.ravel())
                save_gt = dataset.gt_mask
        else:
            val_auc, recon_full, orig_full, save_gt = evaluate_model(model, test_ds, device, batch_sz)  

        
        if val_auc > best_auc:
            best_auc = val_auc
            best_auc_path = model_dir / "best_model_auc.pt"
            torch.save(model.state_dict(), best_auc_path)

            out_mat = Path(RESULTS_DIR) / dataset_name / "residuals_best.mat"
            out_mat.parent.mkdir(parents=True, exist_ok=True)
            save_residuals(res_map, orig_full, save_gt, out_mat)
            print(f" --New best AUC={best_auc:.4f}, saved model and residuals.")
            
            # Save AUC result to Excel tracker if mode and num_decoders are provided
            if mode is not None and num_decoders is not None:
                # Extract clean dataset name and fusion type
                clean_dataset_name = dataset_name
                fusion_type = "N/A"
                
                # Handle fusion type extraction for full mode
                if mode == "full":
                    # Extract fusion type from experiment name
                    if "_simple_fusion_" in dataset_name:
                        fusion_type = "simple"
                    elif "_addition_fusion_" in dataset_name:
                        fusion_type = "addition"
                    elif "_concat_conv_fusion_" in dataset_name:
                        fusion_type = "concat_conv"
                    elif "_advanced_fusion_" in dataset_name:
                        fusion_type = "advanced"
                    elif "_cross_attention_fusion_" in dataset_name:
                        fusion_type = "cross_attention"
                    elif "_hierarchical_fusion_" in dataset_name:
                        fusion_type = "hierarchical"
                
                # Remove suffixes to get clean dataset name
                for suffix in [f"_{mode}_1dec_fixed", f"_{mode}_2dec_fixed", f"_{mode}_3dec_fixed", 
                              f"_{mode}_fixed", "_fixed"]:
                    if clean_dataset_name.endswith(suffix):
                        clean_dataset_name = clean_dataset_name[:-len(suffix)]
                        break
                
                # Remove fusion type suffix if present
                for fusion_suffix in ["_simple_fusion", "_addition_fusion", "_concat_conv_fusion"]:
                    if clean_dataset_name.endswith(fusion_suffix):
                        clean_dataset_name = clean_dataset_name[:-len(fusion_suffix)]
                        break
                
                additional_info = {
                    'Epoch': epoch + 1,
                    'Final_AUC': best_auc,
                    'Model_Path': best_auc_path,
                    'Fusion_Type': fusion_type
                }
                save_auc_result(clean_dataset_name, mode, num_decoders, best_auc, 
                              results_dir="Results", additional_info=additional_info)
        
        # Save the last model only every 5 epochs to reduce disk usage
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            last_model_path = os.path.join(model_dir, "last_model.pt")
            torch.save(model.state_dict(), last_model_path)

    # Simplified - removed gate tracking and inference time logging

    return model
