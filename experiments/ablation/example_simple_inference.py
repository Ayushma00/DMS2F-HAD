#!/usr/bin/env python3
"""
Simple example showing the inference time tracking CSV output.

This script demonstrates what the CSV file will contain after running your training experiments.
"""

import pandas as pd
import os

def show_csv_example():
    """Show example of what the inference times CSV will contain"""
    
    print("=== SIMPLE INFERENCE TIME TRACKING ===\n")
    
    # Check if CSV file exists
    csv_file = "Results/inference_times.csv"
    if os.path.exists(csv_file):
        print("✅ Found inference times CSV file!")
        df = pd.read_csv(csv_file)
        print(f"\nData from {csv_file}:")
        print(df.to_string(index=False))
        
        print(f"\nSummary:")
        print(f"Total experiments: {len(df)}")
        print(f"Average epoch time: {df['avg_epoch_time_seconds'].mean():.3f} seconds")
        print(f"Fastest dataset: {df.loc[df['avg_epoch_time_seconds'].idxmin(), 'dataset_name']}")
        print(f"Slowest dataset: {df.loc[df['avg_epoch_time_seconds'].idxmax(), 'dataset_name']}")
        
    else:
        print("❌ No CSV file found yet.")
        print("Run your training experiments first to generate the CSV file.")
        
        print("\nExample of what the CSV will contain:")
        example_data = {
            'timestamp': ['2024-01-15T10:30:00', '2024-01-15T10:45:00'],
            'dataset_name': ['aviris_1', 'cat-island'],
            'mode': ['full', 'full'],
            'fusion_type': ['simple', 'simple'],
            'num_decoders': [1, 1],
            'avg_epoch_time_seconds': [15.234, 12.456],
            'total_epochs': [120, 120],
            'total_training_time_seconds': [1828.08, 1494.72]
        }
        df_example = pd.DataFrame(example_data)
        print(df_example.to_string(index=False))

def explain_tracking():
    """Explain how the simple tracking works"""
    
    print("\n=== HOW IT WORKS ===")
    print("1. During training, each epoch time is measured")
    print("2. After training completes, average epoch time is calculated")
    print("3. Data is automatically saved to Results/inference_times.csv")
    print("4. CSV contains: dataset, mode, fusion type, decoders, timing info")
    
    print("\n=== CSV COLUMNS ===")
    print("- timestamp: When the experiment was completed")
    print("- dataset_name: Name of the dataset")
    print("- mode: Model mode (spatial/spectral/full)")
    print("- fusion_type: Type of fusion used")
    print("- num_decoders: Number of decoder blocks")
    print("- avg_epoch_time_seconds: Average time per epoch")
    print("- total_epochs: Total number of epochs trained")
    print("- total_training_time_seconds: Total training time")

if __name__ == "__main__":
    show_csv_example()
    explain_tracking()







