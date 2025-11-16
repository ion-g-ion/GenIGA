# DDPM Trainer Class for IGA Geometry Generation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from timm.utils import ModelEmaV3
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pickle
import h5py
import yaml
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from pyiga import bspline, assemble, geometry, vis

from .nn.diff import UNET, DDPM_Scheduler
from .geometry.datasets import SinglePatchIGADataset
from .physics.validation import compute_poisson_residual, compute_helmholtz_residual


class SinglePatchDiffIGA:
    """
    DDPM Trainer for IGA geometry generation supporting both Poisson and Helmholtz equations.
    
    This class encapsulates the entire training pipeline including:
    - Model initialization and configuration
    - Training loop with proper scaling and conditioning
    - Sampling/inference capabilities
    - Comprehensive saving system with snapshots
    - Physics-based evaluation and visualization
    - Physics loss during training (optional)
    """
    
    def __init__(self, 
                 N: List[int],
                 d: int = 2,
                 num_time_steps: int = 1000,
                 model_config: Optional[Dict[str, Any]] = None,
                 conditioning_config: Optional[Dict[str, Any]] = None,
                 equation_type: Optional[str] = None,
                 physics_loss_config: Optional[Dict[str, Any]] = None,
                 scalar_constants_config: Optional[Dict[str, str]] = None,
                 device: str = "cuda"):
        """
        Initialize the DDPM Trainer.
        
        Args:
            N: Grid dimensions [Nx, Ny]
            d: Spatial dimension (default: 2)
            num_time_steps: Number of diffusion time steps
            model_config: Configuration for UNET model
            conditioning_config: Configuration for conditioning
            equation_type: Type of PDE equation ("poisson", "helmholtz", or None for no physics evaluation)
            physics_loss_config: Configuration for physics loss during training
                {
                    'method': 'full_denoising' | 'per_step' | None,
                    'weight': float,  # Overall weight for physics loss
                    'exponent': float,  # For per_step weighting (default: 2.0)
                    'min_weight': float  # Minimum weight at t=0 for per_step (default: 0.0)
                }
            scalar_constants_config: Maps constant names used in training to dataset field names
                Example: {'wave_number': 'wave_numbers', 'viscosity': 'viscosity'}
            device: Device to run on
        """
        self.N = N
        self.d = d
        self.num_time_steps = num_time_steps
        self.device = device
        self.equation_type = equation_type  # "poisson", "helmholtz", or None
        
        # Physics loss configuration
        self.physics_loss_config = physics_loss_config
        
        # Scalar constants configuration
        self.scalar_constants_config = scalar_constants_config
        self.scalar_constants = {}  # Will store which constants are available
        self.scalar_constants_field_map = {}  # Maps internal names to dataset field names
        
        # Default model configuration
        self.model_config = model_config or {
            'input_channels': d + 1,  # geometry + solution channels
            'output_channels': 1,
            'use_scalar_conditioning': True,
            'conditioning_dim': 1
        }
        
        # Default conditioning configuration
        self.conditioning_config = conditioning_config or {
            'mode': 'film',  # 'film' or 'channel'
            'normalize': True
        }
        
        # Initialize components
        self.scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        self.ema = None
        
        # Dataset and scaling parameters
        self.train_dataset = None
        self.test_dataset = None
        self.scaling_params = None
        
        # For backward compatibility
        self.has_wave_numbers = False
        
        # IGA geometry setup
        self.deg = 2
        self.spline_space = tuple(bspline.make_knots(self.deg, 0.0, 1.0, n-self.deg) for n in N)
        self.iga_geo = geometry.tensor_product(
            geometry.BSplineFunc(self.spline_space[0], np.linspace(0, 1, self.spline_space[0].numdofs)),
            geometry.BSplineFunc(self.spline_space[1], np.linspace(0, 1, self.spline_space[1].numdofs))
        )
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        
    def setup_datasets(self, dataset_path_train: str, dataset_path_test: str, batch_size: int = 64, in_memory: bool = True):
        """Setup training and test datasets."""
        print("Setting up datasets...")
        
        self.train_dataset = SinglePatchIGADataset(d=self.d, h5_files=[dataset_path_train], in_memory=in_memory)
        self.test_dataset = SinglePatchIGADataset(
            d=self.d, h5_files=[dataset_path_test], return_bc_data=True, in_memory=in_memory
        )
        
        # Auto-detect scalar constants from dataset if not configured
        if self.scalar_constants_config is None:
            # Backward compatibility: auto-detect wave_numbers
            if self.train_dataset.has_wave_numbers:
                self.scalar_constants_config = {'wave_number': 'wave_numbers'}
                print("Auto-detected wave_numbers in dataset")
        
        # Build field name mapping and check availability
        if self.scalar_constants_config:
            for const_name, field_name in self.scalar_constants_config.items():
                # Check if field exists in dataset
                # For now, we check has_wave_numbers as example - this should be generalized
                if field_name == 'wave_numbers' and self.train_dataset.has_wave_numbers:
                    self.scalar_constants[const_name] = True
                    self.scalar_constants_field_map[const_name] = field_name
                else:
                    # For future: check dataset for arbitrary field names
                    # For now, assume it exists if configured
                    self.scalar_constants[const_name] = True
                    self.scalar_constants_field_map[const_name] = field_name
        
        # Backward compatibility: set has_wave_numbers
        self.has_wave_numbers = self.scalar_constants.get('wave_number', False)
        
        # Auto-detect equation type if not specified and wave_numbers present
        if self.equation_type is None and self.has_wave_numbers:
            self.equation_type = 'helmholtz'
            print("Auto-detected equation type: Helmholtz (from wave_numbers)")
        elif self.equation_type is None:
            self.equation_type = 'poisson'
            print("Auto-detected equation type: Poisson")
        elif self.equation_type:
            print(f"Using configured equation type: {self.equation_type}")
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, 
            drop_last=True, num_workers=4
        )
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=1, shuffle=True, 
            drop_last=True, num_workers=4
        )
        
        # Compute scaling statistics
        self._compute_scaling_statistics()
        
    def _compute_scaling_statistics(self):
        """Compute scaling statistics from training data."""
        print("Computing scaling statistics...")
        
        geo_coeffs_all = []
        targets_all = []
        scalar_constants_data = {name: [] for name in self.scalar_constants.keys()}
        
        for batch_data in tqdm(self.train_loader, desc="Computing statistics"):
            # Handle variable batch structure based on constants
            if len(batch_data) == 2:
                geo_coeffs, targets = batch_data
            else:
                geo_coeffs, targets = batch_data[0], batch_data[1]
                # Extract scalar constants (assuming they come after targets)
                idx = 2
                for const_name in self.scalar_constants.keys():
                    if idx < len(batch_data):
                        scalar_constants_data[const_name].append(batch_data[idx])
                        idx += 1
                
            geo_coeffs = geo_coeffs.reshape([-1, self.d] + self.N)
            targets = targets.reshape([-1, 1] + self.N)
            geo_coeffs_all.append(geo_coeffs)
            targets_all.append(targets)
        
        geo_coeffs_all = torch.cat(geo_coeffs_all, dim=0)
        targets_all = torch.cat(targets_all, dim=0)
        
        # Compute statistics
        geo_mean = geo_coeffs_all.mean(dim=0, keepdim=True)
        geo_std = geo_coeffs_all.std(dim=0, keepdim=True) + 1e-8
        targets_mean = targets_all.mean()
        targets_std = targets_all.std() + 1e-8
        
        # Scalar constants statistics
        scalar_constants_stats = {}
        for const_name, data_list in scalar_constants_data.items():
            if data_list:
                data_all = torch.cat(data_list, dim=0)
                mean = data_all.mean()
                std = data_all.std() + 1e-8
                scalar_constants_stats[const_name] = {
                    'mean': mean.to(self.device),
                    'std': std.to(self.device)
                }
                print(f"{const_name} - Mean: {mean:.4f}, Std: {std:.4f}")
                print(f"{const_name} range: [{data_all.min():.2f}, {data_all.max():.2f}]")
        
        print(f"Geometry coefficients - Mean: {geo_mean.mean():.4f}, Std: {geo_std.mean():.4f}")
        print(f"Targets - Mean: {targets_mean:.4f}, Std: {targets_std:.4f}")
        
        # Store scaling parameters
        self.scaling_params = {
            'geo_mean': geo_mean.to(self.device),
            'geo_std': geo_std.to(self.device),
            'targets_mean': targets_mean.to(self.device),
            'targets_std': targets_std.to(self.device),
            'scalar_constants': scalar_constants_stats,
            'has_wave_numbers': self.has_wave_numbers  # For backward compatibility
        }
        
        # Backward compatibility: add wave_number_mean/std if present
        if 'wave_number' in scalar_constants_stats:
            self.scaling_params['wave_number_mean'] = scalar_constants_stats['wave_number']['mean']
            self.scaling_params['wave_number_std'] = scalar_constants_stats['wave_number']['std']
        else:
            self.scaling_params['wave_number_mean'] = None
            self.scaling_params['wave_number_std'] = None
    
    def setup_model(self, lr: float = 1e-4, ema_decay: float = 0.9999):
        """Setup model, optimizer, and training components."""
        print("Setting up model...")
        
        # Determine conditioning dimension from scalar constants
        conditioning_dim = len(self.scalar_constants) if self.scalar_constants else 1
        
        if conditioning_dim > 0:
            print(f"Using scalar conditioning with {conditioning_dim} parameter(s)")
        else:
            print("No scalar conditioning")
        
        self.model_config.update({
            'conditioning_dim': conditioning_dim,
            'time_steps': self.num_time_steps
        })
        
        # Initialize model
        self.model = UNET(**self.model_config).to(self.device)
        self.model.set_conditioning_mode(self.conditioning_config['mode'])
        
        # Setup optimizer and scheduler
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-5)
        self.ema = ModelEmaV3(self.model, decay=ema_decay)
        
        print(f"Model initialized with {sum(p.numel() for p in self.model.parameters()):,} parameters")
    
    def _get_scalar_params_from_batch(self, batch_data) -> Dict[str, torch.Tensor]:
        """Extract scalar parameters from batch data."""
        scalar_params = {}
        
        if not self.scalar_constants:
            return scalar_params
        
        # Extract based on batch structure
        idx = 2  # After geo_coeffs and targets
        for const_name in self.scalar_constants.keys():
            if idx < len(batch_data):
                scalar_params[const_name] = batch_data[idx].to(self.device)
                idx += 1
        
        return scalar_params
    
    def _normalize_scalar_params(self, scalar_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Normalize scalar parameters and combine for model input."""
        if not scalar_params:
            return torch.zeros(1).to(self.device)
        
        normalized = []
        for const_name, value in scalar_params.items():
            stats = self.scaling_params.get('scalar_constants', {}).get(const_name)
            if stats:
                normalized.append((value - stats['mean']) / stats['std'])
            else:
                normalized.append(torch.zeros_like(value))
        
        # For now, concatenate into single vector (model expects single scalar_param)
        # In future, could modify UNET to accept dict
        if normalized:
            # Average normalized scalars for now (since model expects single value per sample)
            # First get batch size
            batch_size = normalized[0].shape[0] if len(normalized[0].shape) > 0 else 1
            if len(normalized) == 1:
                return normalized[0]
            else:
                return torch.stack(normalized, dim=0).mean(dim=0)  # Average across scalars
        else:
            # Fallback: return zeros with batch size from first scalar param
            batch_size = list(scalar_params.values())[0].shape[0] if scalar_params else 1
            return torch.zeros(batch_size).to(self.device)
    
    def _compute_physics_loss(self, solution: torch.Tensor, geo_coeffs: torch.Tensor,
                             scalar_params: Dict[str, torch.Tensor],
                             bc_indices: Optional[torch.Tensor] = None,
                             bc_values: Optional[torch.Tensor] = None,
                             rhs: Optional[torch.Tensor] = None,
                             timestep: Optional[int] = None) -> Optional[torch.Tensor]:
        """Compute physics loss if configured."""
        if self.physics_loss_config is None or self.physics_loss_config.get('method') is None:
            return None
        
        if self.equation_type is None:
            return None
        
        method = self.physics_loss_config['method']
        weight = self.physics_loss_config.get('weight', 1.0)
        
        # For per_step method, compute time-dependent weighting
        if method == 'per_step' and timestep is not None:
            exponent = self.physics_loss_config.get('exponent', 2.0)
            min_weight = self.physics_loss_config.get('min_weight', 0.0)
            t_norm = (self.num_time_steps - timestep) / self.num_time_steps  # Higher weight near end
            step_weight = min_weight + (1.0 - min_weight) * (t_norm ** exponent)
            weight = weight * step_weight
        elif method == 'per_step':
            # If per_step but no timestep provided, skip
            return None
        
        # Reshape tensors for residual computation
        solution_flat = solution.view(-1)
        geo_coeffs_flat = geo_coeffs.reshape([-1, self.d] + self.N)
        geo_coeffs_flat = geo_coeffs_flat.reshape(-1, geo_coeffs_flat.shape[1] * np.prod(self.N))[0]
        
        # Compute residual based on equation type
        if self.equation_type == 'helmholtz':
            wave_number = scalar_params.get('wave_number')
            if wave_number is None:
                return None
            residual_norm = compute_helmholtz_residual(
                solution=solution_flat,
                geo_coeffs=geo_coeffs_flat,
                wave_number=wave_number,
                spline_space=self.spline_space,
                rhs=rhs,
                return_norm=True,
                norm_type="2"  # L2 norm for loss
            )
        elif self.equation_type == 'poisson':
            residual_norm = compute_poisson_residual(
                solution=solution_flat,
                geo_coeffs=geo_coeffs_flat,
                spline_space=self.spline_space,
                rhs=rhs,
                return_norm=True,
                norm_type="2"  # L2 norm for loss
            )
        else:
            return None
        
        return weight * residual_norm
    
    def train(self, 
              num_epochs: int = 15, 
              save_interval: int = 5, 
              working_directory: str = "experiments",
              checkpoint_name: str = "model",
              description: str = "",
              batch_size: Optional[int] = None,
              lr: Optional[float] = None,
              ema_decay: Optional[float] = None,
              seed: Optional[int] = None,
              dataset_path_train: Optional[str] = None,
              dataset_path_test: Optional[str] = None,
              snapshot_name: Optional[str] = None):
        """
        Train the DDPM model.
        
        Args:
            num_epochs: Number of training epochs
            save_interval: Save checkpoint every N epochs
            working_directory: Directory where all files will be saved
            checkpoint_name: Prefix for all saved files (config, model, plots, etc.)
            description: Meaningful description of this training run
            batch_size: Batch size (stored in config)
            lr: Learning rate (stored in config)
            ema_decay: EMA decay rate (stored in config)
            seed: Random seed (stored in config)
            dataset_path_train: Training dataset path (stored in config)
            dataset_path_test: Test dataset path (stored in config)
            snapshot_name: (Deprecated) Use checkpoint_name instead
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        if self.scaling_params is None:
            raise ValueError("Datasets not setup. Call setup_datasets() first.")
        
        # Handle backward compatibility
        if snapshot_name is not None and checkpoint_name == "model":
            checkpoint_name = snapshot_name
            
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Working directory: {working_directory}")
        print(f"Checkpoint prefix: {checkpoint_name}")
        
        # Setup working directory
        self.working_directory = working_directory
        self.checkpoint_name = checkpoint_name
        self.description = description
        os.makedirs(self.working_directory, exist_ok=True)
        
        # Store training arguments for config
        self.training_args = {
            'num_epochs': num_epochs,
            'save_interval': save_interval,
            'batch_size': batch_size,
            'lr': lr,
            'ema_decay': ema_decay,
            'seed': seed,
            'dataset_path_train': dataset_path_train,
            'dataset_path_test': dataset_path_test,
            'description': description
        }
        
        # Save training configuration
        self._save_config()
        
        # Track validation losses
        self.validation_history = []
        
        criterion = nn.MSELoss(reduction='mean')
        
        for epoch in range(num_epochs):
            total_loss = 0
            total_physics_loss = 0
            self.model.train()
            
            # Training loop
            for idx, batch_data in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                # Extract data
                geo_coeffs = batch_data[0].to(self.device)
                targets = batch_data[1].to(self.device)
                scalar_params = self._get_scalar_params_from_batch(batch_data)
                
                batch_size = geo_coeffs.shape[0]
                geo_coeffs = geo_coeffs.reshape([-1, self.d] + self.N)
                targets = targets.reshape([-1, 1] + self.N)
                
                # Apply scaling
                geo_coeffs_scaled = (geo_coeffs - self.scaling_params['geo_mean']) / self.scaling_params['geo_std']
                targets_scaled = (targets - self.scaling_params['targets_mean']) / self.scaling_params['targets_std']
                
                # Generate noise and timesteps
                t = torch.randint(0, self.num_time_steps, (batch_size,))
                eps = torch.randn_like(targets_scaled, requires_grad=False)
                a = self.scheduler.alpha[t].view((batch_size, 1) + self.d*(1,)).to(self.device)
                noisy_targets = (torch.sqrt(a) * targets_scaled) + (torch.sqrt(1 - a) * eps)
                
                # Setup conditioning
                scalar_param_combined = self._normalize_scalar_params(scalar_params)
                
                # Forward pass
                output = self.model(torch.cat([noisy_targets, geo_coeffs_scaled], dim=1), t, scalar_param=scalar_param_combined)
                
                # Compute diffusion loss
                self.optimizer.zero_grad()
                diffusion_loss = criterion(output, eps)
                total_loss += diffusion_loss.item()
                
                # Compute physics loss if enabled
                physics_loss = None
                if self.physics_loss_config and self.physics_loss_config.get('method') is not None:
                    method = self.physics_loss_config.get('method')
                    
                    if method == 'per_step':
                        # Per-step physics loss: compute on current noisy state
                        # Get current timestep (average for batch)
                        current_t = t.float().mean().item()
                        # Denoise prediction for residual computation
                        # Estimate clean solution from current noisy state
                        alpha_t = self.scheduler.alpha[t].view((batch_size, 1) + self.d*(1,)).to(self.device)
                        pred_clean = (noisy_targets - torch.sqrt(1 - alpha_t) * output) / torch.sqrt(alpha_t)
                        # Denormalize for physics evaluation
                        pred_clean_denorm = pred_clean * self.scaling_params['targets_std'] + self.scaling_params['targets_mean']
                        geo_coeffs_denorm = geo_coeffs_scaled * self.scaling_params['geo_std'] + self.scaling_params['geo_mean']
                        
                        physics_loss = self._compute_physics_loss(
                            solution=pred_clean_denorm[0],  # Use first sample
                            geo_coeffs=geo_coeffs_denorm[0],
                            scalar_params=scalar_params,
                            timestep=int(current_t)
                        )
                    elif method == 'full_denoising':
                        # Full denoising physics loss: compute on fully denoised solution
                        # For efficiency, only compute on first sample of batch and every Nth batch
                        if idx % 10 == 0:  # Every 10th batch to reduce computation
                            with torch.enable_grad():
                                # Perform full denoising starting from current noisy state (first sample only)
                                denoised = self._denoise_with_gradients(
                                    noisy_targets[0:1], geo_coeffs_scaled[0:1], 
                                    scalar_param_combined[0:1], t[0:1]
                                )
                                # Denormalize for physics evaluation
                                denoised_denorm = denoised * self.scaling_params['targets_std'] + self.scaling_params['targets_mean']
                                geo_coeffs_denorm = geo_coeffs_scaled[0:1] * self.scaling_params['geo_std'] + self.scaling_params['geo_mean']
                                
                                # Extract scalar params for first sample
                                scalar_params_sample = {}
                                for k, v in scalar_params.items():
                                    if v is not None and len(v) > 0:
                                        scalar_params_sample[k] = v[0:1]
                                
                                physics_loss = self._compute_physics_loss(
                                    solution=denoised_denorm[0],
                                    geo_coeffs=geo_coeffs_denorm[0],
                                    scalar_params=scalar_params_sample,
                                    timestep=None
                                )
                    
                total_loss_tensor = diffusion_loss
                if physics_loss is not None:
                    total_loss_tensor = total_loss_tensor + physics_loss
                    total_physics_loss += physics_loss.item()
                
                # Backward pass
                total_loss_tensor.backward()
                self.optimizer.step()
                self.ema.update(self.model)
            
            # Compute validation loss
            val_loss = self._compute_validation_loss()
            
            # Evaluation with visualization
            if (epoch + 1) % save_interval == 0:
                residuals = self.evaluate(visualize=True, epoch=epoch + 1)
                if residuals is not None:
                    print(f"Evaluation residuals: L2 norm = {residuals:.4e}")
            
            # Update learning rate
            self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_last_lr()[0]
            avg_loss = total_loss / len(self.train_loader)
            avg_physics_loss = total_physics_loss / len(self.train_loader) if total_physics_loss > 0 else 0
            
            loss_str = f'Epoch {epoch+1} | Train Loss {avg_loss:.5f} | Val Loss {val_loss:.5f}'
            if avg_physics_loss > 0:
                loss_str += f' | Physics Loss {avg_physics_loss:.5f}'
            loss_str += f' | LR {current_lr:.2e}'
            print(loss_str)
            
            # Store training history
            history_entry = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                'val_loss': val_loss,
                'lr': current_lr
            }
            if avg_physics_loss > 0:
                history_entry['physics_loss'] = avg_physics_loss
            self.training_history.append(history_entry)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        # Final save
        self.save_checkpoint(num_epochs, is_final=True)
        
        # Generate and save training plots
        self._save_training_plots()
        
        print(f"Training completed. Results saved to {self.working_directory} with prefix '{self.checkpoint_name}'")
    
    def evaluate(self, visualize: bool = False, epoch: Optional[int] = None, 
                 num_samples_per_geometry: int = 1, max_geometries: Optional[int] = None) -> Optional[Dict[str, torch.Tensor]]:
        """
        Evaluate model over test set and optionally create visualizations.
        
        Args:
            visualize: If True, create visualization plots (only for first geometry)
            epoch: Optional epoch number for file naming (defaults to current_epoch)
            num_samples_per_geometry: Number of samples to generate per geometry (default: 1)
            max_geometries: Maximum number of geometries to evaluate (default: all)
            
        Returns:
            Dictionary with residual statistics if physics evaluation is enabled:
                - 'mean': mean residual across all samples
                - 'std': standard deviation of residuals
                - 'min': minimum residual
                - 'max': maximum residual
                - 'median': median residual
                - 'all_residuals': tensor of all individual residuals
            Returns None if physics evaluation is not enabled
        """
        if self.model is None:
            raise ValueError("Model not initialized.")
        
        if epoch is None:
            epoch = self.current_epoch if hasattr(self, 'current_epoch') else 0
        
        self.model.eval()
        
        residuals = []
        
        with torch.no_grad():
            for geom_idx, batch_data in enumerate(self.test_loader):
                # Check if we've reached max_geometries
                if max_geometries is not None and geom_idx >= max_geometries:
                    break
                
                # Unpack based on what's available
                geo_coeffs = batch_data[0].to(self.device)
                targets = batch_data[1].to(self.device)
                
                # Initialize optional components
                bc_indices = None
                bc_values = None
                rhs = None
                scalar_params = {}
                
                # Unpack remaining components
                idx_data = 2
                if self.test_dataset.has_bc_data:
                    bc_indices = batch_data[idx_data].to(self.device)
                    bc_values = batch_data[idx_data + 1].to(self.device)
                    idx_data += 2
                
                if self.test_dataset.rhs is not None:
                    rhs = batch_data[idx_data].to(self.device)
                    idx_data += 1
                
                # Extract scalar parameters (they come after bc_data and rhs)
                scalar_params = {}
                for const_name in self.scalar_constants.keys():
                    if idx_data < len(batch_data):
                        scalar_params[const_name] = batch_data[idx_data].to(self.device)
                        idx_data += 1
                
                geo_coeffs = geo_coeffs.reshape([-1, self.d] + self.N)
                targets = targets.reshape([-1, 1] + self.N)
                
                # Update IGA geometry
                self.iga_geo.coeffs = np.concatenate([
                    geo_coeffs.cpu().numpy()[0, k, ..., None] for k in range(self.d)
                ], axis=-1)
                
                # Apply scaling
                geo_coeffs_scaled = (geo_coeffs - self.scaling_params['geo_mean']) / self.scaling_params['geo_std']
                scalar_param_combined = self._normalize_scalar_params(scalar_params)
                
                # Generate multiple samples for this geometry
                for sample_idx in range(num_samples_per_geometry):
                    # Generate solution with different noise
                    noise = torch.randn_like(targets, requires_grad=False)
                    
                    solution = self.sample(geo_coeffs_scaled, noise, scalar_param=scalar_param_combined)
                    
                    # Denormalize solution
                    solution = solution * self.scaling_params['targets_std'] + self.scaling_params['targets_mean']
                    
                    # Physics evaluation (only if equation_type is set)
                    if self.equation_type is not None:
                        # Only visualize first sample of first geometry
                        should_visualize = visualize and geom_idx == 0 and sample_idx == 0
                        residual_norm = self._physics_evaluation(
                            solution, geo_coeffs, bc_indices, bc_values, rhs, 
                            scalar_params, visualize=should_visualize, epoch=epoch
                        )
                        if residual_norm is not None:
                            residuals.append(residual_norm)
                
                print(f"Evaluated geometry {geom_idx + 1}/{len(self.test_loader)} with {num_samples_per_geometry} sample(s)")
        
        if residuals:
            residuals_tensor = torch.stack(residuals)
            stats = {
                'mean': residuals_tensor.mean(),
                'std': residuals_tensor.std(),
                'min': residuals_tensor.min(),
                'max': residuals_tensor.max(),
                'median': residuals_tensor.median(),
                'all_residuals': residuals_tensor
            }
            
            print(f"\n=== Evaluation Statistics ===")
            print(f"Total samples: {len(residuals)} ({len(self.test_loader) if max_geometries is None else min(max_geometries, len(self.test_loader))} geometries × {num_samples_per_geometry} samples)")
            print(f"Mean residual: {stats['mean']:.4e}")
            print(f"Std residual:  {stats['std']:.4e}")
            print(f"Min residual:  {stats['min']:.4e}")
            print(f"Max residual:  {stats['max']:.4e}")
            print(f"Median residual: {stats['median']:.4e}")
            
            return stats
        return None
    
    def _physics_evaluation(self, solution: torch.Tensor, geo_coeffs: torch.Tensor,
                          bc_indices: Optional[torch.Tensor], bc_values: Optional[torch.Tensor],
                          rhs: Optional[torch.Tensor],
                          scalar_params: Dict[str, torch.Tensor],
                          visualize: bool = False, epoch: int = 0) -> Optional[torch.Tensor]:
        """Perform physics-based evaluation using validation functions."""
        if self.equation_type is None:
            return None
        
        # Reshape tensors for validation
        solution_flat = solution.view(-1)
        geo_coeffs_flat = geo_coeffs.reshape([-1, self.d] + self.N)
        geo_coeffs_flat = geo_coeffs_flat.reshape(-1, geo_coeffs_flat.shape[1] * np.prod(self.N))[0]
        
        # Compute residual using appropriate validation function
        if self.equation_type == 'helmholtz':
            wave_number = scalar_params.get('wave_number')
            if wave_number is None:
                return None
            residual_norm = compute_helmholtz_residual(
                solution=solution_flat,
                geo_coeffs=geo_coeffs_flat,
                wave_number=wave_number,
                spline_space=self.spline_space,
                rhs=rhs,
                return_norm=True,
                norm_type="2"  # L2 norm
            )
            equation_name = "Helmholtz"
            wave_num_str = f", k={wave_number.item():.2f}" if wave_number.numel() == 1 else ""
        else:  # poisson
            residual_norm = compute_poisson_residual(
                solution=solution_flat,
                geo_coeffs=geo_coeffs_flat,
                spline_space=self.spline_space,
                rhs=rhs,
                return_norm=True,
                norm_type="2"  # L2 norm
            )
            equation_name = "Poisson"
            wave_num_str = ""
        
        print(f"{equation_name} residual (L2 norm{wave_num_str}): {residual_norm:.4e}")
        
        # Check if solution is physical
        is_physical = residual_norm < 1e-6
        print(f"Solution is {'physical' if is_physical else 'non-physical'} (threshold: 1e-6)")
        
        # Store residual for tracking
        if not hasattr(self, 'residual_history'):
            self.residual_history = []
        self.residual_history.append(residual_norm.item())
        
        # Create visualizations if requested
        if visualize and hasattr(self, 'working_directory'):
            equation_suffix = self.equation_type
            
            # IGA solution visualization
            solution_np = solution_flat.detach().cpu().numpy()
            u_func = geometry.BSplineFunc(self.spline_space, solution_np)
            plt.figure(figsize=(10, 8))
            vis.plot_field(u_func, self.iga_geo)
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            title = f'{equation_name} Solution, Residual: {residual_norm.item():.2e}{wave_num_str}'
            plt.title(title)
            viz_filename = f'{self.checkpoint_name}_solution_{equation_suffix}_epoch_{epoch}.png'
            plt.savefig(os.path.join(self.working_directory, viz_filename))
            plt.close()
        
        return residual_norm
    
    def sample(self, geo_coeffs: torch.Tensor, z: torch.Tensor, 
               scalar_param: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the trained model using DDPM reverse process.
        
        Args:
            geo_coeffs: Geometry coefficients (already scaled)
            z: Initial noise
            scalar_param: Conditioning parameter (already normalized)
            
        Returns:
            Generated solution (in normalized space)
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        model_to_use = self.ema.module if self.ema is not None else self.model
        model_to_use.eval()
        
        B = z.shape[0]
        beta = self.scheduler.beta.to(self.device)
        alpha = self.scheduler.alpha.to(self.device)
        
        if scalar_param is None:
            scalar_param = torch.zeros(B).to(self.device)
        
        with torch.no_grad():
            for t in reversed(range(1, self.num_time_steps)):
                t_tensor = torch.full((B,), t, dtype=torch.long, device=self.device)
                temp = beta[t_tensor] / (torch.sqrt(1-alpha[t_tensor]) * torch.sqrt(1-beta[t_tensor]))
                x_in = torch.cat([z, geo_coeffs], dim=1)
                
                z = 1/torch.sqrt(1-beta[t_tensor]) * z - temp * model_to_use(x_in, t_tensor, scalar_param=scalar_param)
                
                if t > 1:  # Don't add noise at the final step
                    eps = torch.randn_like(z)
                    z = z + eps * torch.sqrt(beta[t_tensor])
            
            # Final denoising step
            t_tensor = torch.full((B,), 0, dtype=torch.long, device=self.device)
            temp = beta[t_tensor] / (torch.sqrt(1-alpha[t_tensor]) * torch.sqrt(1-beta[t_tensor]))
            x_in = torch.cat([z, geo_coeffs], dim=1)
            z = 1/torch.sqrt(1-beta[t_tensor]) * z - temp * model_to_use(x_in, t_tensor, scalar_param=scalar_param)
        
        return z
    
    def _denoise_with_gradients(self, z: torch.Tensor, geo_coeffs: torch.Tensor,
                                 scalar_param: torch.Tensor, t_start: torch.Tensor) -> torch.Tensor:
        """Perform denoising with gradients enabled (for full_denoising physics loss)."""
        B = z.shape[0]
        beta = self.scheduler.beta.to(self.device)
        alpha = self.scheduler.alpha.to(self.device)
        
        z_current = z
        for t in reversed(range(1, self.num_time_steps)):
            t_tensor = torch.full((B,), t, dtype=torch.long, device=self.device)
            temp = beta[t_tensor] / (torch.sqrt(1-alpha[t_tensor]) * torch.sqrt(1-beta[t_tensor]))
            x_in = torch.cat([z_current, geo_coeffs], dim=1)
            
            z_current = 1/torch.sqrt(1-beta[t_tensor]) * z_current - temp * self.model(x_in, t_tensor, scalar_param=scalar_param)
            
            if t > 1:
                eps = torch.randn_like(z_current)
                z_current = z_current + eps * torch.sqrt(beta[t_tensor])
        
        # Final denoising step
        t_tensor = torch.full((B,), 0, dtype=torch.long, device=self.device)
        temp = beta[t_tensor] / (torch.sqrt(1-alpha[t_tensor]) * torch.sqrt(1-beta[t_tensor]))
        x_in = torch.cat([z_current, geo_coeffs], dim=1)
        z_current = 1/torch.sqrt(1-beta[t_tensor]) * z_current - temp * self.model(x_in, t_tensor, scalar_param=scalar_param)
        
        return z_current
    
    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save model checkpoint and training state with prefix."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'scaling_params': self.scaling_params,
            'training_history': self.training_history,
            'model_config': self.model_config,
            'conditioning_config': self.conditioning_config,
            'equation_type': self.equation_type,
            'physics_loss_config': self.physics_loss_config,
            'scalar_constants_config': self.scalar_constants_config
        }
        
        if is_final:
            checkpoint_filename = f"{self.checkpoint_name}_model_final.pt"
        else:
            checkpoint_filename = f"{self.checkpoint_name}_model_epoch_{epoch}.pt"
        
        checkpoint_path = os.path.join(self.working_directory, checkpoint_filename)
        torch.save(checkpoint, checkpoint_path)
        
        print(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.model is None:
            # Recreate model from saved config
            self.model_config = checkpoint['model_config']
            self.conditioning_config = checkpoint['conditioning_config']
            self.model = UNET(**self.model_config).to(self.device)
            self.model.set_conditioning_mode(self.conditioning_config['mode'])
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler if available
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if 'ema_state_dict' in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint['ema_state_dict'])
        
        # Load scaling parameters and training history
        self.scaling_params = checkpoint.get('scaling_params')
        self.training_history = checkpoint.get('training_history', [])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.equation_type = checkpoint.get('equation_type')
        self.physics_loss_config = checkpoint.get('physics_loss_config')
        self.scalar_constants_config = checkpoint.get('scalar_constants_config')
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def _save_config(self):
        """Save training configuration with prefix."""
        config = {
            # Model architecture
            'N': self.N,
            'd': self.d,
            'num_time_steps': self.num_time_steps,
            'model_config': self.model_config,
            'conditioning_config': self.conditioning_config,
            
            # Equation and physics
            'equation_type': self.equation_type,
            'physics_loss_config': self.physics_loss_config,
            'scalar_constants_config': self.scalar_constants_config,
            'has_wave_numbers': self.has_wave_numbers,
            
            # Training arguments
            'training_args': self.training_args if hasattr(self, 'training_args') else {},
            
            # Description
            'description': self.description if hasattr(self, 'description') else '',
            
            # System info
            'timestamp': datetime.now().isoformat(),
            'device': self.device
        }
        
        # Save as JSON with prefix
        config_path = os.path.join(self.working_directory, f"{self.checkpoint_name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Configuration saved to: {config_path}")
    
    def _compute_validation_loss(self) -> float:
        """Compute validation loss on test set."""
        if self.test_loader is None:
            return 0.0
        
        self.model.eval()
        criterion = nn.MSELoss(reduction='mean')
        total_val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                # Extract data
                geo_coeffs = batch_data[0].to(self.device)
                targets = batch_data[1].to(self.device)
                scalar_params = self._get_scalar_params_from_batch(batch_data)
                
                batch_size = geo_coeffs.shape[0]
                geo_coeffs = geo_coeffs.reshape([-1, self.d] + self.N)
                targets = targets.reshape([-1, 1] + self.N)
                
                # Apply scaling
                geo_coeffs_scaled = (geo_coeffs - self.scaling_params['geo_mean']) / self.scaling_params['geo_std']
                targets_scaled = (targets - self.scaling_params['targets_mean']) / self.scaling_params['targets_std']
                
                # Generate noise and timesteps
                t = torch.randint(0, self.num_time_steps, (batch_size,))
                eps = torch.randn_like(targets_scaled, requires_grad=False)
                a = self.scheduler.alpha[t].view((batch_size, 1) + self.d*(1,)).to(self.device)
                noisy_targets = (torch.sqrt(a) * targets_scaled) + (torch.sqrt(1 - a) * eps)
                
                # Setup conditioning
                scalar_param_combined = self._normalize_scalar_params(scalar_params)
                
                # Forward pass
                output = self.model(torch.cat([noisy_targets, geo_coeffs_scaled], dim=1), t, scalar_param=scalar_param_combined)
                
                # Compute loss
                val_loss = criterion(output, eps)
                total_val_loss += val_loss.item()
                num_batches += 1
        
        self.model.train()
        return total_val_loss / max(num_batches, 1)
    
    def _save_training_plots(self):
        """Generate and save training plots using Plotly as JSON."""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            print("Warning: plotly not installed. Skipping plot generation.")
            return
        
        if not self.training_history:
            print("No training history to plot.")
            return
        
        epochs = [h['epoch'] for h in self.training_history]
        train_losses = [h['train_loss'] for h in self.training_history]
        val_losses = [h['val_loss'] for h in self.training_history]
        
        # Create training and validation loss plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=train_losses,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        fig.add_trace(go.Scatter(
            x=epochs,
            y=val_losses,
            mode='lines+markers',
            name='Validation Loss',
            line=dict(color='red', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title=f'Training and Validation Loss - {self.checkpoint_name}',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(x=0.7, y=0.95)
        )
        
        # Save as JSON
        loss_plot_path = os.path.join(self.working_directory, f"{self.checkpoint_name}_loss_plot.json")
        fig.write_json(loss_plot_path)
        print(f"Loss plot saved to: {loss_plot_path}")
        
        # If physics loss is present, create a separate plot
        if any('physics_loss' in h for h in self.training_history):
            physics_losses = [h.get('physics_loss', 0) for h in self.training_history]
            
            fig_physics = go.Figure()
            
            fig_physics.add_trace(go.Scatter(
                x=epochs,
                y=physics_losses,
                mode='lines+markers',
                name='Physics Loss',
                line=dict(color='green', width=2),
                marker=dict(size=6)
            ))
            
            fig_physics.update_layout(
                title=f'Physics Loss - {self.checkpoint_name}',
                xaxis_title='Epoch',
                yaxis_title='Physics Loss',
                hovermode='x unified',
                template='plotly_white'
            )
            
            physics_plot_path = os.path.join(self.working_directory, f"{self.checkpoint_name}_physics_loss_plot.json")
            fig_physics.write_json(physics_plot_path)
            print(f"Physics loss plot saved to: {physics_plot_path}")
        
        # Create learning rate plot
        if any('lr' in h for h in self.training_history):
            lrs = [h['lr'] for h in self.training_history]
            
            fig_lr = go.Figure()
            
            fig_lr.add_trace(go.Scatter(
                x=epochs,
                y=lrs,
                mode='lines+markers',
                name='Learning Rate',
                line=dict(color='purple', width=2),
                marker=dict(size=6)
            ))
            
            fig_lr.update_layout(
                title=f'Learning Rate Schedule - {self.checkpoint_name}',
                xaxis_title='Epoch',
                yaxis_title='Learning Rate',
                hovermode='x unified',
                template='plotly_white',
                yaxis_type='log'
            )
            
            lr_plot_path = os.path.join(self.working_directory, f"{self.checkpoint_name}_lr_plot.json")
            fig_lr.write_json(lr_plot_path)
            print(f"Learning rate plot saved to: {lr_plot_path}")
    
    def save(self, snapshot_name: str):
        """
        Save complete model snapshot with all components.
        
        Args:
            snapshot_name: Name for the snapshot directory
        """
        self.snapshot_name = snapshot_name
        self.snapshot_dir = os.path.join("snapshots", snapshot_name)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(os.path.join(self.snapshot_dir, "checkpoints"), exist_ok=True)
        
        # Save configuration
        self._save_config()
        
        # Save final checkpoint
        self.save_checkpoint(self.current_epoch, is_final=True)
        
        # Save training history plot
        if self.training_history:
            self._save_training_plots()
        
        print(f"Complete snapshot saved to {self.snapshot_dir}")
    
    
    @classmethod
    def load_from_snapshot(cls, snapshot_path: str, device: str = "cuda"):
        """
        Load trainer from a complete snapshot directory.
        
        Args:
            snapshot_path: Path to snapshot directory
            device: Device to load on
            
        Returns:
            Initialized SinglePatchDiffIGA instance
        """
        # Load configuration
        config_path = os.path.join(snapshot_path, "config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create trainer instance
        trainer = cls(
            N=config['N'],
            d=config['d'],
            num_time_steps=config['num_time_steps'],
            model_config=config['model_config'],
            conditioning_config=config['conditioning_config'],
            equation_type=config.get('equation_type'),
            physics_loss_config=config.get('physics_loss_config'),
            scalar_constants_config=config.get('scalar_constants_config'),
            device=device
        )
        
        # Load scaling parameters
        scaling_path = os.path.join(snapshot_path, "scaling_params.pkl")
        if os.path.exists(scaling_path):
            with open(scaling_path, 'rb') as f:
                scaling_params = pickle.load(f)
            # Move tensors to device
            trainer.scaling_params = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                    for k, v in scaling_params.items()}
            # Handle nested scalar_constants
            if 'scalar_constants' in trainer.scaling_params:
                for name, stats in trainer.scaling_params['scalar_constants'].items():
                    stats['mean'] = stats['mean'].to(device) if isinstance(stats['mean'], torch.Tensor) else stats['mean']
                    stats['std'] = stats['std'].to(device) if isinstance(stats['std'], torch.Tensor) else stats['std']
            
            trainer.has_wave_numbers = trainer.scaling_params.get('has_wave_numbers', False)
        
        # Load checkpoint
        checkpoint_path = os.path.join(snapshot_path, "checkpoints", "final_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
        
        trainer.snapshot_dir = snapshot_path
        print(f"Trainer loaded from snapshot: {snapshot_path}")
        
        return trainer

