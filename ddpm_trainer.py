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
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from pyiga import bspline, assemble, geometry, vis

from geniga.nn.diff import UNET, DDPM_Scheduler
from geniga.geometry.datasets import SinglePatchIGADataset


class DDPMTrainer:
    """
    DDPM Trainer for IGA geometry generation supporting both Poisson and Helmholtz equations.
    
    This class encapsulates the entire training pipeline including:
    - Model initialization and configuration
    - Training loop with proper scaling and conditioning
    - Sampling/inference capabilities
    - Comprehensive saving system with snapshots
    - Physics-based evaluation and visualization
    """
    
    def __init__(self, 
                 N: List[int],
                 d: int = 2,
                 num_time_steps: int = 1000,
                 model_config: Optional[Dict[str, Any]] = None,
                 conditioning_config: Optional[Dict[str, Any]] = None,
                 device: str = "cuda"):
        """
        Initialize the DDPM Trainer.
        
        Args:
            N: Grid dimensions [Nx, Ny]
            d: Spatial dimension (default: 2)
            num_time_steps: Number of diffusion time steps
            model_config: Configuration for UNET model
            conditioning_config: Configuration for conditioning
            device: Device to run on
        """
        self.N = N
        self.d = d
        self.num_time_steps = num_time_steps
        self.device = device
        
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
            d=self.d, h5_files=[dataset_path_test], return_matrix=True, in_memory=in_memory
        )
        
        # Check equation type
        self.has_wave_numbers = self.train_dataset.has_wave_numbers
        equation_type = 'Helmholtz' if self.has_wave_numbers else 'Poisson'
        print(f"Dataset equation type: {equation_type}")
        
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
        wave_numbers_all = []
        
        for batch_data in tqdm(self.train_loader, desc="Computing statistics"):
            if self.has_wave_numbers:
                geo_coeffs, targets, wave_numbers = batch_data
                wave_numbers_all.append(wave_numbers)
            else:
                geo_coeffs, targets = batch_data
                
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
        
        # Wave number statistics
        wave_number_mean, wave_number_std = None, None
        if self.has_wave_numbers:
            wave_numbers_all = torch.cat(wave_numbers_all, dim=0)
            wave_number_mean = wave_numbers_all.mean()
            wave_number_std = wave_numbers_all.std() + 1e-8
            print(f"Wave numbers - Mean: {wave_number_mean:.4f}, Std: {wave_number_std:.4f}")
            print(f"Wave number range: [{wave_numbers_all.min():.2f}, {wave_numbers_all.max():.2f}]")
        
        print(f"Geometry coefficients - Mean: {geo_mean.mean():.4f}, Std: {geo_std.mean():.4f}")
        print(f"Targets - Mean: {targets_mean:.4f}, Std: {targets_std:.4f}")
        
        # Store scaling parameters
        self.scaling_params = {
            'geo_mean': geo_mean.to(self.device),
            'geo_std': geo_std.to(self.device),
            'targets_mean': targets_mean.to(self.device),
            'targets_std': targets_std.to(self.device),
            'wave_number_mean': wave_number_mean.to(self.device) if wave_number_mean is not None else None,
            'wave_number_std': wave_number_std.to(self.device) if wave_number_std is not None else None,
            'has_wave_numbers': self.has_wave_numbers
        }
    
    def setup_model(self, lr: float = 1e-4, ema_decay: float = 0.9999):
        """Setup model, optimizer, and training components."""
        print("Setting up model...")
        
        # Update model config based on equation type
        if self.has_wave_numbers:
            conditioning_dim = 1  # Single wave number for Helmholtz
            print("Using scalar conditioning for Helmholtz equation (wave number)")
        else:
            conditioning_dim = 1  # Can be adjusted for other PDE parameters
            print("Using scalar conditioning for Poisson equation")
        
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
    
    def train(self, num_epochs: int = 15, save_interval: int = 5, snapshot_name: Optional[str] = None):
        """
        Train the DDPM model.
        
        Args:
            num_epochs: Number of training epochs
            save_interval: Save checkpoint every N epochs
            snapshot_name: Name for the snapshot folder
        """
        if self.model is None:
            raise ValueError("Model not initialized. Call setup_model() first.")
        if self.scaling_params is None:
            raise ValueError("Datasets not setup. Call setup_datasets() first.")
        
        print(f"Starting training for {num_epochs} epochs...")
        
        # Setup snapshot directory
        if snapshot_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            equation_type = "helmholtz" if self.has_wave_numbers else "poisson"
            snapshot_name = f"ddpm_{equation_type}_{timestamp}"
        
        self.snapshot_dir = os.path.join("snapshots", snapshot_name)
        os.makedirs(self.snapshot_dir, exist_ok=True)
        os.makedirs(os.path.join(self.snapshot_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.snapshot_dir, "visualizations"), exist_ok=True)
        
        # Save training configuration
        self._save_config()
        
        criterion = nn.MSELoss(reduction='mean')
        
        for epoch in range(num_epochs):
            total_loss = 0
            self.model.train()
            
            # Training loop
            for idx, batch_data in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
                if self.has_wave_numbers:
                    geo_coeffs, targets, wave_numbers = batch_data
                    geo_coeffs, targets, wave_numbers = geo_coeffs.to(self.device), targets.to(self.device), wave_numbers.to(self.device)
                else:
                    geo_coeffs, targets = batch_data
                    geo_coeffs, targets = geo_coeffs.to(self.device), targets.to(self.device)
                
                batch_size = geo_coeffs.shape[0]
                geo_coeffs = geo_coeffs.reshape([-1, self.d] + self.N)
                targets = targets.reshape([-1, 1] + self.N)
                
                # Apply scaling
                geo_coeffs = (geo_coeffs - self.scaling_params['geo_mean']) / self.scaling_params['geo_std']
                targets = (targets - self.scaling_params['targets_mean']) / self.scaling_params['targets_std']
                
                # Generate noise and timesteps
                t = torch.randint(0, self.num_time_steps, (batch_size,))
                eps = torch.randn_like(targets, requires_grad=False)
                a = self.scheduler.alpha[t].view((batch_size, 1) + self.d*(1,)).to(self.device)
                noisy_targets = (torch.sqrt(a) * targets) + (torch.sqrt(1 - a) * eps)
                
                # Setup conditioning
                if self.has_wave_numbers:
                    scalar_param = (wave_numbers - self.scaling_params['wave_number_mean']) / self.scaling_params['wave_number_std']
                else:
                    scalar_param = torch.zeros(batch_size).to(self.device)
                
                # Forward pass
                output = self.model(torch.cat([noisy_targets, geo_coeffs], dim=1), t, scalar_param=scalar_param)
                
                # Compute loss and backprop
                self.optimizer.zero_grad()
                loss = criterion(output, eps)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                self.ema.update(self.model)
            
            # Evaluation
            if (epoch + 1) % save_interval == 0:
                self._evaluate_and_visualize(epoch)
            
            # Update learning rate
            self.lr_scheduler.step()
            current_lr = self.lr_scheduler.get_last_lr()[0]
            avg_loss = total_loss / len(self.train_loader)
            
            print(f'Epoch {epoch+1} | Loss {avg_loss:.5f} | LR {current_lr:.2e}')
            
            # Store training history
            self.training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'lr': current_lr
            })
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1)
        
        # Final save
        self.save_checkpoint(num_epochs, is_final=True)
        print(f"Training completed. Results saved to {self.snapshot_dir}")
    
    def _evaluate_and_visualize(self, epoch: int):
        """Evaluate model and create visualizations."""
        self.model.eval()
        
        with torch.no_grad():
            for idx, batch_data in enumerate(self.test_loader):
                if self.has_wave_numbers:
                    geo_coeffs, targets, indptr, indices, values, rhs, wave_number = batch_data
                    geo_coeffs, targets, wave_number = geo_coeffs.to(self.device), targets.to(self.device), wave_number.to(self.device)
                else:
                    geo_coeffs, targets, indptr, indices, values, rhs = batch_data
                    geo_coeffs, targets = geo_coeffs.to(self.device), targets.to(self.device)
                
                indptr, indices, values, rhs = indptr.to(self.device), indices.to(self.device), values.to(self.device), rhs.to(self.device)
                
                geo_coeffs = geo_coeffs.reshape([-1, self.d] + self.N)
                targets = targets.reshape([-1, 1] + self.N)
                
                # Update IGA geometry
                self.iga_geo.coeffs = np.concatenate([
                    geo_coeffs.cpu().numpy()[0, k, ..., None] for k in range(self.d)
                ], axis=-1)
                
                # Apply scaling
                geo_coeffs_scaled = (geo_coeffs - self.scaling_params['geo_mean']) / self.scaling_params['geo_std']
                
                # Generate solution
                noise = torch.randn_like(targets, requires_grad=False)
                
                if self.has_wave_numbers:
                    inference_scalar_param = (wave_number - self.scaling_params['wave_number_mean']) / self.scaling_params['wave_number_std']
                    print(f"Using wave number: {wave_number.item():.2f} (normalized: {inference_scalar_param.item():.2f})")
                else:
                    inference_scalar_param = torch.tensor(0.0).to(self.device)
                
                solution = self.sample(geo_coeffs_scaled, noise, scalar_param=inference_scalar_param)
                
                # Denormalize solution
                solution = solution * self.scaling_params['targets_std'] + self.scaling_params['targets_mean']
                
                # Physics evaluation
                self._physics_evaluation(solution, indptr, indices, values, rhs, epoch, wave_number if self.has_wave_numbers else None)
                
                break  # Only evaluate one sample
    
    def _physics_evaluation(self, solution: torch.Tensor, indptr: torch.Tensor, 
                          indices: torch.Tensor, values: torch.Tensor, rhs: torch.Tensor,
                          epoch: int, wave_number: Optional[torch.Tensor] = None):
        """Perform physics-based evaluation and create visualizations."""
        # Convert to numpy for SciPy operations
        n_unknowns = indptr.numel() - 1
        indptr_np = indptr.long().cpu().numpy().flatten()
        indices_np = indices.long().cpu().numpy().flatten()
        values_np = values.cpu().numpy().flatten()
        
        A_csr = sp.csr_matrix((values_np, indices_np, indptr_np), shape=(n_unknowns, n_unknowns))
        
        uc_pred_np = solution.view(-1).detach().cpu().numpy()
        rhs_np = rhs.view(-1).detach().cpu().numpy()
        
        # Compute residual
        res_np = A_csr.dot(uc_pred_np) - rhs_np
        
        # Create masks
        bd_mask_nd = np.ones(tuple(self.N), dtype=np.float64)
        bd_mask_nd[(slice(1, -1),) * len(self.N)] = 0
        bd_mask_vec = bd_mask_nd.reshape(-1)
        int_mask_vec = 1.0 - bd_mask_vec
        
        int_idx = np.where(int_mask_vec == 1.0)[0]
        residual_norm = np.max(np.abs(res_np[int_idx]))
        print(f"Interior residual (∞-norm): {residual_norm:.4e}")
        
        # Solve reference solution
        Pin = sp.diags(int_mask_vec, 0, shape=(n_unknowns, n_unknowns), format="csr")
        Pbd = sp.diags(bd_mask_vec, 0, shape=(n_unknowns, n_unknowns), format="csr")
        g_vec = bd_mask_vec * uc_pred_np
        sys_mat = Pin @ A_csr @ Pin + Pbd
        rhs_ref = Pin.dot(rhs_np - A_csr.dot(g_vec)) + g_vec
        u_ref = splinalg.spsolve(sys_mat, rhs_ref)
        
        # Compute error
        err_2 = np.linalg.norm(u_ref - uc_pred_np) / np.linalg.norm(u_ref)
        print(f"Solution error: {err_2:.4e}")
        
        # Create visualizations
        equation_suffix = "helmholtz" if self.has_wave_numbers else "poisson"
        viz_dir = os.path.join(self.snapshot_dir, "visualizations")
        
        # Reference solution
        plt.figure(figsize=(8, 6))
        plt.imshow(u_ref.reshape(self.N), origin='lower')
        plt.colorbar()
        plt.title('Reference Solution')
        plt.savefig(os.path.join(viz_dir, f'u_ref_{equation_suffix}_epoch_{epoch}.png'))
        plt.close()
        
        # Error plot
        plt.figure(figsize=(8, 6))
        error_field = (u_ref.reshape(self.N) - uc_pred_np.reshape(self.N)) / np.linalg.norm(u_ref.reshape(self.N), ord=np.inf)
        plt.imshow(error_field, origin='lower')
        plt.colorbar()
        plt.title('Normalized Error')
        plt.savefig(os.path.join(viz_dir, f'error_{equation_suffix}_epoch_{epoch}.png'))
        plt.close()
        
        # IGA solution visualization
        u_func = geometry.BSplineFunc(self.spline_space, uc_pred_np)
        plt.figure(figsize=(10, 8))
        vis.plot_field(u_func, self.iga_geo)
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        if self.has_wave_numbers and wave_number is not None:
            plt.title(f'Helmholtz Solution (k={wave_number.item():.2f})')
        else:
            plt.title('Poisson Solution')
        plt.savefig(os.path.join(viz_dir, f'iga_solution_{equation_suffix}_epoch_{epoch}.png'))
        plt.close()
    
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
    
    def save_checkpoint(self, epoch: int, is_final: bool = False):
        """Save model checkpoint and training state."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'ema_state_dict': self.ema.state_dict(),
            'scaling_params': self.scaling_params,
            'training_history': self.training_history,
            'model_config': self.model_config,
            'conditioning_config': self.conditioning_config
        }
        
        checkpoint_name = "final_checkpoint.pt" if is_final else f"checkpoint_epoch_{epoch}.pt"
        checkpoint_path = os.path.join(self.snapshot_dir, "checkpoints", checkpoint_name)
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
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def _save_config(self):
        """Save training configuration to snapshot directory."""
        config = {
            'N': self.N,
            'd': self.d,
            'num_time_steps': self.num_time_steps,
            'model_config': self.model_config,
            'conditioning_config': self.conditioning_config,
            'has_wave_numbers': self.has_wave_numbers,
            'equation_type': 'Helmholtz' if self.has_wave_numbers else 'Poisson',
            'timestamp': datetime.now().isoformat(),
            'device': self.device
        }
        
        # Save as JSON
        config_path = os.path.join(self.snapshot_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Also save scaling parameters separately
        if self.scaling_params is not None:
            scaling_path = os.path.join(self.snapshot_dir, "scaling_params.pkl")
            # Convert tensors to CPU for pickling
            scaling_params_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                for k, v in self.scaling_params.items()}
            with open(scaling_path, 'wb') as f:
                pickle.dump(scaling_params_cpu, f)
        
        print(f"Configuration saved to {self.snapshot_dir}")
    
    def save(self, snapshot_name: str):
        """
        Save complete model snapshot with all components.
        
        Args:
            snapshot_name: Name for the snapshot directory
        """
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
    
    def _save_training_plots(self):
        """Save training history plots."""
        epochs = [h['epoch'] for h in self.training_history]
        losses = [h['loss'] for h in self.training_history]
        lrs = [h['lr'] for h in self.training_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Loss plot
        ax1.plot(epochs, losses)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        # Learning rate plot
        ax2.plot(epochs, lrs)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.snapshot_dir, "training_history.png"))
        plt.close()
    
    @classmethod
    def load_from_snapshot(cls, snapshot_path: str, device: str = "cuda"):
        """
        Load trainer from a complete snapshot directory.
        
        Args:
            snapshot_path: Path to snapshot directory
            device: Device to load on
            
        Returns:
            Initialized DDPMTrainer instance
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
            trainer.has_wave_numbers = trainer.scaling_params.get('has_wave_numbers', False)
        
        # Load checkpoint
        checkpoint_path = os.path.join(snapshot_path, "checkpoints", "final_checkpoint.pt")
        if os.path.exists(checkpoint_path):
            trainer.load_checkpoint(checkpoint_path)
        
        trainer.snapshot_dir = snapshot_path
        print(f"Trainer loaded from snapshot: {snapshot_path}")
        
        return trainer
