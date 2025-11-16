"""GenIGA: Generative IGA using Diffusion Models."""

# Physics validation functions
from .physics.validation import (
    compute_poisson_residual,
    compute_helmholtz_residual
)

# Geometry generation functions
from .geometry.random import (
    generate_random_geom,
    generate_random_bcs
)

# Dataset
from .geometry.datasets import SinglePatchIGADataset

# Neural network models
from .nn.diff import UNET, DDPM_Scheduler

# DDPM Trainer
from .ddpm_trainer import SinglePatchDiffIGA

__all__ = [
    # Physics
    'compute_poisson_residual',
    'compute_helmholtz_residual',
    # Geometry
    'generate_random_geom', 
    'generate_random_bcs',
    'SinglePatchIGADataset',
    # Neural networks
    'UNET',
    'DDPM_Scheduler',
    # Trainer
    'SinglePatchDiffIGA'
]

__version__ = '0.1.0'

