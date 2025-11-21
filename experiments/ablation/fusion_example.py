#!/usr/bin/env python3
"""
Example demonstrating the three simple fusion methods in the model.

This script shows how to use the simplified fusion methods:
1. "simple" - Original gated fusion
2. "addition" - Element-wise addition
3. "concat_conv" - Concatenation + 1x1 Conv
"""

import torch
from src.model_fixed import AnomalyDetectionModel

def demonstrate_fusion_methods():
    """Demonstrate the three fusion methods"""
    
    print("=== SIMPLIFIED FUSION METHODS DEMONSTRATION ===\n")
    
    # Model parameters
    in_channels = 100  # Example hyperspectral channels
    dim = 64
    
    # Create sample input
    batch_size = 2
    height, width = 16, 16
    x = torch.randn(batch_size, in_channels, height, width)
    
    fusion_types = ["simple", "addition", "concat_conv"]
    
    for fusion_type in fusion_types:
        print(f"--- {fusion_type.upper()} FUSION ---")
        
        # Create model with specific fusion type
        model = AnomalyDetectionModel(
            in_channels=in_channels,
            mode="full",  # Use both spatial and spectral branches
            dim=dim,
            depth=1,
            spec_num=12,
            spec_rate=0.5,
            spa_token=16,
            num_decoders=1,
            fusion_type=fusion_type
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output, features = model(x)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Feature shape: {features.shape}")
        
        # Analyze fusion complexity
        analysis = model.analyze_fusion_complexity()
        print(f"  Fusion parameters: {analysis['parameters']:,}")
        print(f"  Expressiveness score: {analysis['expressiveness_score']}/3")
        print(f"  Components: {', '.join(analysis['learnable_components'])}")
        print()

def show_fusion_recommendations():
    """Show recommendations for each fusion type"""
    
    print("=== FUSION METHOD RECOMMENDATIONS ===\n")
    
    recommendations = AnomalyDetectionModel.get_fusion_recommendations()
    
    for fusion_type, info in recommendations.items():
        print(f"--- {fusion_type.upper()} FUSION ---")
        print(f"  Pros: {', '.join(info['pros'])}")
        print(f"  Cons: {', '.join(info['cons'])}")
        print(f"  Use case: {info['use_case']}")
        print()

def compare_fusion_methods():
    """Compare the three fusion methods"""
    
    print("=== FUSION METHOD COMPARISON ===\n")
    
    in_channels = 100
    fusion_types = ["simple", "addition", "concat_conv"]
    results = []
    
    for fusion_type in fusion_types:
        model = AnomalyDetectionModel(
            in_channels=in_channels,
            mode="full",
            dim=64,
            fusion_type=fusion_type,
            num_decoders=1
        )
        
        total_params = sum(p.numel() for p in model.parameters())
        analysis = model.analyze_fusion_complexity()
        
        results.append({
            'type': fusion_type,
            'total_params': total_params,
            'fusion_params': analysis['parameters'],
            'expressiveness': analysis['expressiveness_score']
        })
    
    # Print comparison table
    print(f"{'Fusion Type':<12} {'Total Params':<12} {'Fusion Params':<13} {'Expressiveness':<13}")
    print("-" * 50)
    for result in results:
        print(f"{result['type']:<12} {result['total_params']:<12,} {result['fusion_params']:<13,} {result['expressiveness']:<13}")
    
    print(f"\nRecommendations:")
    print("- Use 'addition' for fastest training and inference")
    print("- Use 'simple' for learnable spatial-spectral weighting")
    print("- Use 'concat_conv' to preserve all feature information")

if __name__ == "__main__":
    demonstrate_fusion_methods()
    show_fusion_recommendations() 
    compare_fusion_methods()

