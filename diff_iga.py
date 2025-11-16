# Refactored DDPM IGA Training Script
# Now uses SinglePatchDiffIGA class for better organization and maintainability

from geniga import SinglePatchDiffIGA
import torch
import numpy as np
import plotly.graph_objects as go
import os
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
          working_directory: str = "experiments",
          checkpoint_name: str = "model",
          description: str = "",
          physics_loss_config: Optional[dict] = None):
    """
    Train DDPM model using the new DDPMTrainer class.
    
    This function provides backward compatibility with the original interface
    while leveraging the new organized trainer class.
    
    Args:
        N: Grid dimensions
        d: Spatial dimension
        batch_size: Batch size for training
        num_time_steps: Number of diffusion timesteps
        num_epochs: Number of training epochs
        seed: Random seed (-1 for no seed)
        ema_decay: EMA decay rate
        lr: Learning rate
        checkpoint_path: Path to checkpoint to resume from
        dataset_path_train: Path to training dataset
        dataset_path_test: Path to test dataset
        equation_type: Type of equation ("poisson" or "helmholtz")
        working_directory: Directory where all experiment files will be saved
        checkpoint_name: Prefix for all saved files (config, model, plots, etc.)
        description: Meaningful description of this training run
        physics_loss_config: Configuration for physics loss
    """
    
    # Set seed if provided
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Initialize trainer
    trainer = SinglePatchDiffIGA(
        N=N,
        d=d,
        num_time_steps=num_time_steps,
        equation_type=equation_type,
        physics_loss_config=physics_loss_config,
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
    
    # Train the model with new prefix-based saving
    trainer.train(
        num_epochs=num_epochs,
        save_interval=5,
        working_directory=working_directory,
        checkpoint_name=checkpoint_name,
        description=description,
        batch_size=batch_size,
        lr=lr,
        ema_decay=ema_decay,
        seed=seed,
        dataset_path_train=dataset_path_train,
        dataset_path_test=dataset_path_test
    )
    
    # Evaluate on entire test set with 100 samples per geometry
    print("\n" + "=" * 80)
    print("Performing final evaluation on test set")
    print("Generating 100 samples per test geometry...")
    print("=" * 80)
    
    stats = trainer.evaluate(
        visualize=False,
        num_samples_per_geometry=100,
        max_geometries=None  # Evaluate all test geometries
    )
    
    # Create histogram of residual norms
    if stats is not None:
        residuals = stats['all_residuals'].cpu().numpy()
        
        # Print statistics to terminal
        print(f"\n{'='*60}")
        print(f"Final Evaluation Statistics:")
        print(f"{'='*60}")
        print(f"Total samples: {len(residuals)}")
        print(f"Mean residual:   {stats['mean']:.4e}")
        print(f"Std residual:    {stats['std']:.4e}")
        print(f"Median residual: {stats['median']:.4e}")
        print(f"Min residual:    {stats['min']:.4e}")
        print(f"Max residual:    {stats['max']:.4e}")
        print(f"{'='*60}\n")
        
        # Create Plotly histogram
        fig = go.Figure(data=[go.Histogram(
            x=residuals,
            nbinsx=50,
            marker=dict(
                color='rgba(0, 123, 255, 0.7)',
                line=dict(color='black', width=1)
            )
        )])
        
        fig.update_layout(
            title=f'Distribution of Physics Residuals<br><sub>{len(residuals)} samples ({equation_type} equation)</sub>',
            xaxis_title='Residual Norm (L2)',
            yaxis_title='Frequency',
            showlegend=False,
            template='plotly_white',
            width=900,
            height=600
        )
        
        # Save histogram
        os.makedirs(working_directory, exist_ok=True)
        histogram_path = os.path.join(working_directory, f"{checkpoint_name}_residual_histogram.html")
        fig.write_html(histogram_path)
        
        print(f"Residual histogram saved to: {histogram_path}")
    else:
        print("\nWarning: Physics evaluation not available (equation_type may not be set)")
    
    return trainer


def sample_from_model(
    geo_coeffs: torch.Tensor,
    z: torch.Tensor,
    model = None,  # Deprecated, kept for compatibility
    ema = None,  # Deprecated, kept for compatibility
    num_time_steps: int = 1000,
    device: str = "cuda",
    scalar_param = 1.0,
    trainer: Optional[SinglePatchDiffIGA] = None
):
    """
    Legacy sampling function for backward compatibility.
    
    DEPRECATED: Use DDPMTrainer.sample() method instead.
    
    This function is kept for backward compatibility but now delegates
    to the DDPMTrainer class method.
    """
    if trainer is None:
        raise ValueError("trainer parameter is required. Use SinglePatchDiffIGA.sample() directly instead.")
    
    return trainer.sample(geo_coeffs, z, scalar_param)


def main():
    """
    Main function to train different models:
    1. poisson_small without physics loss
    2. poisson_small with physics loss
    3. helmholtz without physics loss
    4. helmholtz with physics loss
    """
    # Common training parameters
    N = [32, 32]
    num_epochs = 20
    num_time_steps = 1000
    lr = 1e-4
    
    # Physics loss configuration
    physics_loss_config = {
        'method': 'full_denoising',
        'weight': 1.0
    }
    
    # 1. Train poisson_small without physics loss
    print("=" * 80)
    print("Training poisson_small WITHOUT physics loss")
    print("=" * 80)
    trainer_poisson_no_physics = train(
        N=N,
        checkpoint_path=None,
        lr=lr,
        num_epochs=num_epochs,
        num_time_steps=num_time_steps,
        dataset_path_train="experiments/datasets/dataset_poisson_small_train.h5",
        dataset_path_test="experiments/datasets/dataset_poisson_small_test.h5",
        equation_type="poisson",
        working_directory="experiments",
        checkpoint_name="poisson_small_no_physics",
        description="Poisson equation training without physics loss. Baseline model.",
        physics_loss_config=None
    )
    
    # 2. Train poisson_small with physics loss
    print("\n" + "=" * 80)
    print("Training poisson_small WITH physics loss")
    print("=" * 80)
    trainer_poisson_with_physics = train(
        N=N,
        checkpoint_path=None,
        lr=lr,
        num_epochs=num_epochs,
        num_time_steps=num_time_steps,
        dataset_path_train="experiments/datasets/dataset_poisson_small_train.h5",
        dataset_path_test="experiments/datasets/dataset_poisson_small_test.h5",
        equation_type="poisson",
        working_directory="experiments",
        checkpoint_name="poisson_small_with_physics",
        description="Poisson equation training with full denoising physics loss.",
        physics_loss_config=physics_loss_config
    )
    
    # 3. Train helmholtz without physics loss
    print("\n" + "=" * 80)
    print("Training helmholtz WITHOUT physics loss")
    print("=" * 80)
    trainer_helmholtz_no_physics = train(
        N=N,
        checkpoint_path=None,
        lr=lr,
        num_epochs=num_epochs,
        num_time_steps=num_time_steps,
        dataset_path_train="experiments/datasets/dataset_2d_helmholtz_train.h5",
        dataset_path_test="experiments/datasets/dataset_2d_helmholtz_test.h5",
        equation_type="helmholtz",
        working_directory="experiments",
        checkpoint_name="helmholtz_no_physics",
        description="Helmholtz equation training without physics loss. Baseline model.",
        physics_loss_config=None
    )
    
    # 4. Train helmholtz with physics loss
    print("\n" + "=" * 80)
    print("Training helmholtz WITH physics loss")
    print("=" * 80)
    trainer_helmholtz_with_physics = train(
        N=N,
        checkpoint_path=None,
        lr=lr,
        num_epochs=num_epochs,
        num_time_steps=num_time_steps,
        dataset_path_train="experiments/datasets/dataset_2d_helmholtz_train.h5",
        dataset_path_test="experiments/datasets/dataset_2d_helmholtz_test.h5",
        equation_type="helmholtz",
        working_directory="experiments",
        checkpoint_name="helmholtz_with_physics",
        description="Helmholtz equation training with full denoising physics loss.",
        physics_loss_config=physics_loss_config
    )
    
    print("\n" + "=" * 80)
    print("All training completed! Check the 'experiments' directory for organized results.")
    print("=" * 80)


if __name__ == '__main__':
    main()
