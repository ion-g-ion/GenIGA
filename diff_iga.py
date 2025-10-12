# Refactored DDPM IGA Training Script
# Now uses DDPMTrainer class for better organization and maintainability

from ddpm_trainer import DDPMTrainer
import torch
import numpy as np
from typing import List, Optional


def train(N: List[int],
          d: int = 2,
          batch_size: int = 64,
          num_time_steps: int = 1000,
          num_epochs: int = 15,
          seed: int = -1,
          ema_decay: float = 0.9999,
          lr: float = 1e-4,
          checkpoint_path: Optional[str] = None,
          dataset_path_train: str = "",
          dataset_path_test: str = "",
          equation_type: str = "poisson",
          snapshot_name: Optional[str] = None):
    """
    Train DDPM model using the new DDPMTrainer class.
    
    This function provides backward compatibility with the original interface
    while leveraging the new organized trainer class.
    """
    
    # Set seed if provided
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Initialize trainer
    trainer = DDPMTrainer(
        N=N,
        d=d,
        num_time_steps=num_time_steps,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Setup datasets
    trainer.setup_datasets(
        dataset_path_train=dataset_path_train,
        dataset_path_test=dataset_path_test,
        batch_size=batch_size
    )
    
    # Setup model
    trainer.setup_model(lr=lr, ema_decay=ema_decay)
    
    # Load checkpoint if provided
    if checkpoint_path is not None:
        trainer.load_checkpoint(checkpoint_path)
    
    # Train the model
    trainer.train(
        num_epochs=num_epochs,
        save_interval=5,
        snapshot_name=snapshot_name
    )
    
    return trainer


def sample_from_model(
    geo_coeffs: torch.Tensor,
    z: torch.Tensor,
    model = None,  # Deprecated, kept for compatibility
    ema = None,  # Deprecated, kept for compatibility
    num_time_steps: int = 1000,
    device: str = "cuda",
    scalar_param = 1.0,
    trainer: Optional[DDPMTrainer] = None
):
    """
    Legacy sampling function for backward compatibility.
    
    DEPRECATED: Use DDPMTrainer.sample() method instead.
    
    This function is kept for backward compatibility but now delegates
    to the DDPMTrainer class method.
    """
    if trainer is None:
        raise ValueError("trainer parameter is required. Use DDPMTrainer.sample() directly instead.")
    
    return trainer.sample(geo_coeffs, z, scalar_param)


def main():
    """
    Main function demonstrating the new DDPMTrainer usage.
    """
    # Example 1: Using the legacy train function (backward compatible)
    print("Training Helmholtz equation model...")
    trainer = train(
        N=[32, 32], 
        checkpoint_path=None, 
        lr=1e-4, 
        num_epochs=20,  # Reduced for demo
        num_time_steps=1000,
        dataset_path_train="data/generated_data_helmholtz2d_0.h5", 
        dataset_path_test="data/generated_data_helmholtz2d_0.h5",
        equation_type="helmholtz",
        snapshot_name="helmholtz_demo"
    )
    
    # Example 2: Direct trainer usage (recommended for new code)
    print("\nAlternatively, using DDPMTrainer directly:")
    
    # Create trainer
    direct_trainer = DDPMTrainer(
        N=[32, 32],
        d=2,
        num_time_steps=1000,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Setup and train
    direct_trainer.setup_datasets(
        dataset_path_train="data/generated_data_helmholtz2d_0.h5",
        dataset_path_test="data/generated_data_helmholtz2d_0.h5",
        batch_size=64
    )
    direct_trainer.setup_model(lr=1e-4)
    
    # Uncomment to train:
    # direct_trainer.train(num_epochs=20, snapshot_name="direct_helmholtz_demo")
    
    # Example 3: Loading from snapshot
    print("\nExample of loading from snapshot:")
    print("loaded_trainer = DDPMTrainer.load_from_snapshot('snapshots/helmholtz_demo')")
    
    print("\nTraining completed! Check the 'snapshots' directory for organized results.")


if __name__ == '__main__':
    main()
