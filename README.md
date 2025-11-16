# GenIGA

Generative modeling for Isogeometric Analysis (IGA) using Diffusion Models.

## Overview

GenIGA investigates whether diffusion models can generate physical PDE solutions conditioned on IGA geometry deformations. The diffusion model learns to generate solutions for PDEs (Poisson and Helmholtz) given a specific geometry parametrization.

## Key Features

- **Single-Patch Diffusion Model** (`SinglePatchDiffIGA`): Train DDPM models on IGA geometry data
- **Physics Validation**: Built-in functions to verify solution physicality via residual computation
- **Flexible Conditioning**: Support for scalar conditioning (e.g., wave numbers)
- **Multiple PDE Types**: Poisson and Helmholtz equations supported
- **Efficient Storage**: Datasets store only solutions and geometries (matrices computed on-the-fly)

## Project Structure

```
geniga/
├── physics/
│   └── validation.py       # Physicality validation functions
├── geometry/
│   ├── datasets.py          # Dataset loaders
│   └── random.py            # Random geometry generation
└── nn/
    └── diff.py              # UNET and DDPM scheduler

ddpm_trainer.py              # SinglePatchDiffIGA trainer class
generate_dataset.py          # Dataset generation script
```

## Quick Start

### Training a Model

```python
from geniga import SinglePatchDiffIGA

# Initialize trainer
trainer = SinglePatchDiffIGA(
    N=[32, 32],              # Grid size
    d=2,                     # Spatial dimension
    num_time_steps=1000      # DDPM timesteps
)

# Setup datasets
trainer.setup_datasets(
    dataset_path_train="experiments/datasets/dataset_poisson_small_train.h5",
    dataset_path_test="experiments/datasets/dataset_poisson_small_test.h5",
    batch_size=64
)

# Initialize model
trainer.initialize_model()

# Train
trainer.train(epochs=100)
```

### Validating Solution Physicality

```python
from geniga.physics.validation import compute_poisson_residual, compute_helmholtz_residual

# For Poisson equation
residual = compute_poisson_residual(
    solution=solution_tensor,
    geo_coeffs=geometry_coeffs,
    spline_space=spline_space
)

# For Helmholtz equation
residual = compute_helmholtz_residual(
    solution=solution_tensor,
    geo_coeffs=geometry_coeffs,
    wave_number=k,
    spline_space=spline_space
)

# Check physicality
is_physical = residual < 1e-6
```

## Documentation

- [`TRAINER_README.md`](TRAINER_README.md) - Detailed trainer documentation
- [`PHYSICALITY_FUNCTIONS.md`](PHYSICALITY_FUNCTIONS.md) - Physics validation guide

## Requirements

- PyTorch
- PyIGA
- NumPy, SciPy
- h5py
- tqdm
- matplotlib

## Background

This repository explores using diffusion models to generate physically valid PDE solutions conditioned on geometry. The key insight is that by training on geometry-solution pairs, the model can learn the physics-geometry relationship and generate new solutions for novel geometries.
