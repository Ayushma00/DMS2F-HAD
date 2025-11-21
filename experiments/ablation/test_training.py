#!/usr/bin/env python3
"""
Simple test script to verify training works
"""
import os
import torch
import numpy as np
from src.data_loader import HSIDataset
from src.model_fixed import AnomalyDetectionModel
from src.train import train_model

def test_training():
    """Test training with a simple configuration"""
    
    # Use a small dataset for testing
    dataset_path = 'Data/aviris_1'
    ds_name = 'aviris_1'
    
    print(f"Testing training with dataset: {dataset_path}")
    
    # Create dataset
    dataset = HSIDataset(
        mat_file=dataset_path,
        data_name=ds_name,
        block_size=16,
        stride=8
    )
    
    C = dataset.blocks.shape[1]
    print(f"Dataset channels: {C}")
    
    # Test different modes
    test_configs = [
        ("spatial", 1, "simple"),
        ("spectral", 1, "simple"),
        ("full", 1, "simple"),
    ]
    
    for mode, num_decoders, fusion_type in test_configs:
        print(f"\n=== Testing {mode} mode with {num_decoders} decoder(s) ===")
        
        # Create model
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
        
        # Enable gate tracking for full mode
        if mode == "full":
            model.start_gate_tracking()
        
        # Print parameter count
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {param_count:,}")
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("Model moved to CUDA")
        
        # Test forward pass
        try:
            test_batch = torch.randn(2, C, 16, 16)
            if torch.cuda.is_available():
                test_batch = test_batch.cuda()
            
            with torch.no_grad():
                output, fused = model(test_batch)
                print(f"Forward pass successful: output shape {output.shape}")
                
                # Test that output requires gradients
                test_loss = torch.nn.functional.mse_loss(output, test_batch)
                print(f"Test loss: {test_loss.item():.6f}")
                print(f"Loss requires grad: {test_loss.requires_grad}")
                
        except Exception as e:
            print(f"Forward pass failed: {e}")
            continue
        
        # Test training with few epochs
        experiment_name = f"test_{ds_name}_{mode}_{fusion_type}_{num_decoders}dec"
        
        try:
            trained_model = train_model(
                model,
                dataset,
                experiment_name,
                epochs=5,  # Just 5 epochs for testing
                batch_sz=8,  # Small batch size
                lr=5e-4,
                wd=1e-4,
                eval_dataset_path="../Data/HAD100Dataset/",
                mode=mode,
                num_decoders=num_decoders
            )
            print(f"Training completed successfully for {mode} mode!")
            
        except Exception as e:
            print(f"Training failed for {mode} mode: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_training()
