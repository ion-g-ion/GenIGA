#!/usr/bin/env python3
"""
Example usage of the new DDPMTrainer class for IGA geometry generation.

This script demonstrates:
1. Basic training setup
2. Model training with automatic snapshot management
3. Loading from snapshots
4. Inference/sampling
5. Custom configuration options
"""

import torch
import numpy as np
from ddpm_trainer import DDPMTrainer


def basic_training_example():
    """Basic training example with default settings."""
    print("=== Basic Training Example ===")
    
    # Create trainer
    trainer = DDPMTrainer(
        N=[32, 32],  # Grid dimensions
        d=2,         # Spatial dimension
        num_time_steps=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Setup datasets
    trainer.setup_datasets(
        dataset_path_train="data/generated_data_helmholtz2d_0.h5",
        dataset_path_test="data/generated_data_helmholtz2d_0.h5",
        batch_size=32  # Smaller batch for demo
    )
    
    # Setup model
    trainer.setup_model(lr=1e-4, ema_decay=0.9999)
    
    # Train with automatic snapshot management
    trainer.train(
        num_epochs=10,
        save_interval=5,
        snapshot_name="basic_example"
    )
    
    print(f"Training completed! Results saved to: {trainer.snapshot_dir}")
    return trainer


def advanced_training_example():
    """Advanced training with custom model configuration."""
    print("=== Advanced Training Example ===")
    
    # Custom model configuration
    model_config = {
        'Channels': [32, 64, 128, 256, 256, 192],
        'Attentions': [False, True, False, False, False, True],
        'Upscales': [False, False, False, True, True, True],
        'num_groups': 4,
        'dropout_prob': 0.1,
        'num_heads': 8,
        'input_channels': 3,  # d + 1 (geometry + solution)
        'output_channels': 1,
        'use_scalar_conditioning': True,
        'conditioning_dim': 1
    }
    
    # Custom conditioning configuration
    conditioning_config = {
        'mode': 'film',  # or 'channel'
        'normalize': True
    }
    
    # Create trainer with custom config
    trainer = DDPMTrainer(
        N=[64, 64],  # Larger grid
        d=2,
        num_time_steps=1000,
        model_config=model_config,
        conditioning_config=conditioning_config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Setup datasets
    trainer.setup_datasets(
        dataset_path_train="data/generated_data_helmholtz2d_0.h5",
        dataset_path_test="data/generated_data_helmholtz2d_0.h5",
        batch_size=16  # Smaller batch for larger grid
    )
    
    # Setup model with different learning rate
    trainer.setup_model(lr=5e-5, ema_decay=0.999)
    
    # Train
    trainer.train(
        num_epochs=15,
        save_interval=3,
        snapshot_name="advanced_example"
    )
    
    return trainer


def loading_and_inference_example():
    """Example of loading from snapshot and performing inference."""
    print("=== Loading and Inference Example ===")
    
    try:
        # Load trainer from snapshot
        trainer = DDPMTrainer.load_from_snapshot(
            "snapshots/basic_example",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        print("Successfully loaded trainer from snapshot!")
        print(f"Model has {sum(p.numel() for p in trainer.model.parameters()):,} parameters")
        print(f"Trained for {trainer.current_epoch} epochs")
        
        # Perform inference
        print("Performing inference...")
        
        # Create dummy geometry coefficients and noise
        batch_size = 1
        geo_coeffs = torch.randn(batch_size, trainer.d, *trainer.N)
        noise = torch.randn(batch_size, 1, *trainer.N)
        
        # Apply scaling (in practice, you'd get this from your dataset)
        if trainer.scaling_params is not None:
            geo_coeffs = (geo_coeffs - trainer.scaling_params['geo_mean']) / trainer.scaling_params['geo_std']
        
        # Set conditioning parameter
        if trainer.has_wave_numbers:
            # Use a test wave number
            wave_number = torch.tensor([10.0])  # Example wave number
            scalar_param = (wave_number - trainer.scaling_params['wave_number_mean']) / trainer.scaling_params['wave_number_std']
        else:
            scalar_param = torch.tensor([0.0])
        
        # Sample solution
        solution = trainer.sample(geo_coeffs, noise, scalar_param=scalar_param)
        
        # Denormalize solution
        if trainer.scaling_params is not None:
            solution = solution * trainer.scaling_params['targets_std'] + trainer.scaling_params['targets_mean']
        
        print(f"Generated solution shape: {solution.shape}")
        print(f"Solution range: [{solution.min():.4f}, {solution.max():.4f}]")
        
    except Exception as e:
        print(f"Could not load snapshot: {e}")
        print("Make sure you've run the basic training example first!")


def checkpoint_management_example():
    """Example of checkpoint management and resuming training."""
    print("=== Checkpoint Management Example ===")
    
    # Create trainer
    trainer = DDPMTrainer(
        N=[32, 32],
        d=2,
        num_time_steps=500,  # Fewer steps for demo
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Setup datasets
    trainer.setup_datasets(
        dataset_path_train="data/generated_data_helmholtz2d_0.h5",
        dataset_path_test="data/generated_data_helmholtz2d_0.h5",
        batch_size=32
    )
    
    # Setup model
    trainer.setup_model(lr=1e-4)
    
    # Train for a few epochs
    print("Training for 5 epochs...")
    trainer.train(
        num_epochs=5,
        save_interval=2,
        snapshot_name="checkpoint_example"
    )
    
    # Save current state
    trainer.save("checkpoint_example_manual_save")
    
    # Simulate loading and resuming training
    print("Loading and resuming training...")
    resumed_trainer = DDPMTrainer.load_from_snapshot("snapshots/checkpoint_example_manual_save")
    
    # Resume training for more epochs
    # Note: You'd need to setup datasets again for resumed training
    print(f"Resumed from epoch {resumed_trainer.current_epoch}")


def custom_snapshot_organization_example():
    """Example showing how snapshots are organized."""
    print("=== Snapshot Organization Example ===")
    
    import os
    
    # Create trainer and train briefly
    trainer = DDPMTrainer(N=[16, 16], d=2, num_time_steps=100)
    trainer.setup_datasets(
        dataset_path_train="data/generated_data_helmholtz2d_0.h5",
        dataset_path_test="data/generated_data_helmholtz2d_0.h5",
        batch_size=32
    )
    trainer.setup_model()
    
    # Train and save
    trainer.train(num_epochs=2, snapshot_name="organization_example")
    
    # Show directory structure
    snapshot_dir = "snapshots/organization_example"
    if os.path.exists(snapshot_dir):
        print(f"\nSnapshot directory structure for '{snapshot_dir}':")
        for root, dirs, files in os.walk(snapshot_dir):
            level = root.replace(snapshot_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")


def main():
    """Run all examples."""
    print("DDPMTrainer Examples")
    print("=" * 50)
    
    # Check if data exists
    import os
    if not os.path.exists("data/generated_data_helmholtz2d_0.h5"):
        print("Warning: Dataset file not found. Some examples may fail.")
        print("Please ensure 'data/generated_data_helmholtz2d_0.h5' exists.")
        return
    
    try:
        # Run examples
        basic_training_example()
        print()
        
        advanced_training_example()
        print()
        
        loading_and_inference_example()
        print()
        
        checkpoint_management_example()
        print()
        
        custom_snapshot_organization_example()
        
    except Exception as e:
        print(f"Example failed with error: {e}")
        print("This might be due to missing data files or GPU availability.")
    
    print("\nAll examples completed!")
    print("Check the 'snapshots' directory to see the organized results.")


if __name__ == "__main__":
    main()
