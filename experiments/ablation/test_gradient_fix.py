#!/usr/bin/env python3
"""
Test script to verify gradient computation is working
"""
import torch
import numpy as np
from src.model_fixed import AnomalyDetectionModel

def test_gradient_computation():
    """Test that gradients are properly computed"""
    
    print("Testing gradient computation...")
    
    # Create a simple model
    model = AnomalyDetectionModel(
        in_channels=224,  # AVIRIS has 224 channels
        mode="full",
        dim=64,
        depth=1,
        spec_num=12,
        spec_rate=0.5,
        spa_token=16,
        num_decoders=1,
        fusion_type="simple",
    )
    
    # Move to CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
        print("Model moved to CUDA")
    
    # Check parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create test input
    batch_size = 2
    channels = 224
    height, width = 16, 16
    
    test_input = torch.randn(batch_size, channels, height, width)
    if torch.cuda.is_available():
        test_input = test_input.cuda()
    
    print(f"Test input shape: {test_input.shape}")
    print(f"Input requires grad: {test_input.requires_grad}")
    
    # Test forward pass
    model.train()
    output, fused = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output requires grad: {output.requires_grad}")
    
    # Test loss computation
    target = torch.randn_like(output)
    if torch.cuda.is_available():
        target = target.cuda()
    
    loss = torch.nn.functional.mse_loss(output, target)
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss requires grad: {loss.requires_grad}")
    
    # Test backward pass
    if loss.requires_grad:
        loss.backward()
        print("✅ Backward pass successful!")
        
        # Check gradients
        grad_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
                print(f"  {name}: grad norm = {param.grad.norm().item():.6f}")
        
        grad_norm = grad_norm ** 0.5
        print(f"Total gradient norm: {grad_norm:.6f}")
        
        if grad_norm > 0:
            print("✅ Gradients are being computed correctly!")
            return True
        else:
            print("❌ Gradients are zero!")
            return False
    else:
        print("❌ Loss does not require gradients!")
        return False

if __name__ == "__main__":
    success = test_gradient_computation()
    if success:
        print("\n🎉 Gradient computation test PASSED!")
    else:
        print("\n💥 Gradient computation test FAILED!")

