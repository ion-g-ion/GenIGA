# DDPM Trainer for IGA Geometry Generation

This document describes the new `DDPMTrainer` class that provides a clean, organized interface for training DDPM models on IGA geometry data.

## Overview

The `DDPMTrainer` class encapsulates the entire training pipeline and provides:

- **Clean API**: Simple methods for training, sampling, and saving
- **Organized Snapshots**: Automatic organization of models, configurations, and visualizations
- **Checkpoint Management**: Comprehensive saving/loading of training state
- **Physics Integration**: Built-in physics-based evaluation and visualization
- **Dual Equation Support**: Handles both Poisson and Helmholtz equations
- **Backward Compatibility**: Works with existing code through legacy functions

## Quick Start

### Basic Usage

```python
from ddpm_trainer import DDPMTrainer

# Create trainer
trainer = DDPMTrainer(
    N=[32, 32],  # Grid dimensions
    d=2,         # Spatial dimension
    num_time_steps=1000
)

# Setup datasets and model
trainer.setup_datasets(
    dataset_path_train="data/train.h5",
    dataset_path_test="data/test.h5",
    batch_size=64
)
trainer.setup_model(lr=1e-4)

# Train with automatic snapshot management
trainer.train(
    num_epochs=50,
    save_interval=10,
    snapshot_name="my_experiment"
)
```

### Loading and Inference

```python
# Load from snapshot
trainer = DDPMTrainer.load_from_snapshot("snapshots/my_experiment")

# Generate samples
geo_coeffs = ...  # Your geometry coefficients (scaled)
noise = torch.randn(1, 1, 32, 32)
scalar_param = torch.tensor([5.0])  # Wave number for Helmholtz

solution = trainer.sample(geo_coeffs, noise, scalar_param=scalar_param)
```

## Snapshot Organization

The trainer automatically creates organized snapshots with the following structure:

```
snapshots/
└── experiment_name_YYYYMMDD_HHMMSS/
    ├── config.json                    # Training configuration
    ├── scaling_params.pkl             # Data scaling parameters
    ├── training_history.png           # Loss and LR plots
    ├── checkpoints/
    │   ├── checkpoint_epoch_5.pt      # Periodic checkpoints
    │   ├── checkpoint_epoch_10.pt
    │   └── final_checkpoint.pt        # Final model state
    └── visualizations/
        ├── u_ref_helmholtz_epoch_5.png    # Reference solutions
        ├── error_helmholtz_epoch_5.png    # Error visualizations
        └── iga_solution_helmholtz_epoch_5.png  # IGA field plots
```

## Class Methods

### Core Methods

#### `__init__(N, d=2, num_time_steps=1000, model_config=None, conditioning_config=None, device="cuda")`
Initialize the trainer with grid dimensions and configuration.

#### `setup_datasets(dataset_path_train, dataset_path_test, batch_size=64)`
Setup training and test datasets with automatic scaling computation.

#### `setup_model(lr=1e-4, ema_decay=0.9999)`
Initialize model, optimizer, and training components.

#### `train(num_epochs=15, save_interval=5, snapshot_name=None)`
Train the model with automatic checkpointing and evaluation.

#### `sample(geo_coeffs, z, scalar_param=None)`
Generate solutions using the trained model.

### Checkpoint Management

#### `save_checkpoint(epoch, is_final=False)`
Save model checkpoint and training state.

#### `load_checkpoint(checkpoint_path)`
Load model from checkpoint file.

#### `save(snapshot_name)`
Save complete snapshot with all components.

#### `load_from_snapshot(snapshot_path, device="cuda")` (class method)
Load complete trainer from snapshot directory.

## Configuration Options

### Model Configuration

```python
model_config = {
    'Channels': [32, 64, 128, 256, 256, 192],
    'Attentions': [False, True, False, False, False, True],
    'Upscales': [False, False, False, True, True, True],
    'num_groups': 4,
    'dropout_prob': 0.1,
    'num_heads': 8,
    'input_channels': 3,  # d + 1
    'output_channels': 1,
    'use_scalar_conditioning': True,
    'conditioning_dim': 1
}
```

### Conditioning Configuration

```python
conditioning_config = {
    'mode': 'film',      # 'film' or 'channel'
    'normalize': True    # Whether to normalize conditioning parameters
}
```

## Features

### Automatic Equation Detection
The trainer automatically detects whether the dataset contains wave numbers (Helmholtz equation) or not (Poisson equation) and configures conditioning appropriately.

### Physics-Based Evaluation
During training, the trainer:
- Computes physics residuals using sparse matrix operations
- Solves reference solutions for comparison
- Generates comprehensive visualizations
- Reports quantitative error metrics

### Scaling and Normalization
- Automatic computation of dataset statistics
- Proper scaling of geometry coefficients and targets
- Normalization of conditioning parameters (wave numbers)
- Automatic denormalization during inference

### Training Monitoring
- Real-time loss tracking
- Learning rate scheduling
- Exponential moving average (EMA) for stable inference
- Comprehensive training history logging

## Backward Compatibility

The original `train()` function is preserved for backward compatibility:

```python
from diff_iga import train

# This still works exactly as before
trainer = train(
    N=[32, 32],
    num_epochs=50,
    dataset_path_train="data/train.h5",
    dataset_path_test="data/test.h5",
    snapshot_name="legacy_experiment"
)
```

## File Structure

- `ddpm_trainer.py` - Main trainer class
- `diff_iga.py` - Refactored original script with backward compatibility
- `example_usage.py` - Comprehensive usage examples
- `TRAINER_README.md` - This documentation

## Examples

See `example_usage.py` for comprehensive examples including:
- Basic training setup
- Advanced configuration
- Loading and inference
- Checkpoint management
- Snapshot organization

## Migration Guide

### From Old Code
```python
# Old way
train([32, 32], num_epochs=50, ...)

# New way (recommended)
trainer = DDPMTrainer(N=[32, 32])
trainer.setup_datasets(...)
trainer.setup_model(...)
trainer.train(num_epochs=50, ...)
```

### Benefits of Migration
- Better organization of results
- More flexible configuration
- Easier model reuse and sharing
- Comprehensive logging and visualization
- Simplified inference workflow

## Best Practices

1. **Use descriptive snapshot names** that include experiment details
2. **Set appropriate save intervals** to balance storage and recovery needs
3. **Monitor training through visualizations** in the snapshots directory
4. **Use the snapshot system** for model sharing and reproducibility
5. **Leverage the physics evaluation** to validate model performance

## Dependencies

The trainer requires the same dependencies as the original code:
- PyTorch
- NumPy
- SciPy
- matplotlib
- tqdm
- timm (for EMA)
- h5py
- pyiga

## Troubleshooting

### Common Issues

1. **"Model not initialized" error**: Call `setup_model()` before training
2. **"Datasets not setup" error**: Call `setup_datasets()` before training
3. **GPU memory issues**: Reduce batch size or grid dimensions
4. **Missing data files**: Ensure dataset paths are correct

### Performance Tips

1. Use appropriate batch sizes for your GPU memory
2. Enable mixed precision training for larger models
3. Use multiple workers for data loading
4. Monitor GPU utilization during training

## Future Enhancements

Planned improvements include:
- Mixed precision training support
- Distributed training capabilities
- Advanced visualization options
- Integration with experiment tracking tools
- Support for additional PDE types
