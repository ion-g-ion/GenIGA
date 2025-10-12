#!/usr/bin/env python3
"""
Example script demonstrating scalar and vector conditioning with the improved UNET.

This example shows how to:
1. Use scalar conditioning (single physical parameter)
2. Use vector conditioning (multiple physical parameters)
3. Switch between FiLM and channel concatenation conditioning modes
4. Helmholtz equation specific conditioning (wave number)
"""

import torch
import numpy as np
from geniga.nn.diff import UNET, DDPM_Scheduler

def example_helmholtz_conditioning():
    """Example using Helmholtz equation wave number conditioning"""
    print("=== Helmholtz Equation Wave Number Conditioning ===")
    
    # Initialize model for Helmholtz equation
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=1,  # Single wave number parameter
        input_channels=3,    # geometry (2) + solution (1)
        Channels=[32, 64, 128, 256, 256, 192]
    ).cuda()
    
    # Set to FiLM conditioning (works well for scalar wave number)
    model.set_conditioning_mode('film')
    
    # Example input data
    batch_size = 4
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).cuda()  # [geometry + solution channels]
    t = torch.randint(0, 1000, (batch_size,)).cuda()
    
    # Helmholtz wave number examples:
    # Wave numbers typically range from 2.0 to 16.0 in the dataset
    # These should be normalized for conditioning: k_normalized = (k - k_mean) / k_std
    
    # Example normalization parameters (from dataset statistics)
    k_mean, k_std = 9.0, 4.0  # Approximate values for k ∈ [2, 16]
    
    # 1. Single wave number for all batch items
    k_raw = 5.0  # Raw wave number
    k_normalized = (k_raw - k_mean) / k_std  # Normalize for conditioning
    output1 = model(x, t, scalar_param=k_normalized)
    print(f"Wave number conditioning (single): k={k_raw:.2f} (normalized: {k_normalized:.3f})")
    
    # 2. Different wave numbers for each batch item
    k_raw_batch = torch.tensor([3.0, 7.0, 12.0, 15.0])  # Raw wave numbers
    k_normalized_batch = (k_raw_batch - k_mean) / k_std  # Normalize
    output2 = model(x, t, scalar_param=k_normalized_batch.cuda())
    print(f"Wave number conditioning (per batch):")
    for i, (k_raw, k_norm) in enumerate(zip(k_raw_batch, k_normalized_batch)):
        print(f"  Batch {i}: k={k_raw:.2f} (normalized: {k_norm:.3f})")
    
    print(f"Output shape: {output1.shape}")
    print("Note: Higher wave numbers lead to more oscillatory solutions")
    print("      Lower wave numbers lead to smoother solutions")
    print()

def example_scalar_conditioning():
    """Example using scalar conditioning (single parameter)"""
    print("=== Scalar Conditioning Example ===")
    
    # Initialize model with scalar conditioning
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=1,  # Single scalar parameter
        input_channels=3,    # geometry (2) + solution (1)
        Channels=[32, 64, 128, 256, 256, 192]
    ).cuda()
    
    # Set to FiLM conditioning (recommended for flexibility)
    model.set_conditioning_mode('film')
    
    # Example input data
    batch_size = 4
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).cuda()  # [geometry + solution channels]
    t = torch.randint(0, 1000, (batch_size,)).cuda()
    
    # Different ways to specify scalar conditioning:
    
    # 1. Single scalar for all batch items
    scalar_param = 1.5
    output1 = model(x, t, scalar_param=scalar_param)
    print(f"Scalar conditioning (single value): {scalar_param}")
    
    # 2. Different scalar for each batch item
    scalar_param = torch.tensor([1.0, 1.5, 2.0, 0.5]).cuda()
    output2 = model(x, t, scalar_param=scalar_param)
    print(f"Scalar conditioning (per batch): {scalar_param}")
    
    print(f"Output shape: {output1.shape}")
    print()

def example_vector_conditioning():
    """Example using vector conditioning (multiple parameters)"""
    print("=== Vector Conditioning Example ===")
    
    # Initialize model with vector conditioning
    conditioning_dim = 4  # e.g., [stiffness, damping, forcing, boundary_strength]
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=conditioning_dim,
        input_channels=3,
        Channels=[32, 64, 128, 256, 256, 192]
    ).cuda()
    
    # FiLM conditioning works better for vector inputs
    model.set_conditioning_mode('film')
    
    # Example input data
    batch_size = 4
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).cuda()
    t = torch.randint(0, 1000, (batch_size,)).cuda()
    
    # Different ways to specify vector conditioning:
    
    # 1. Same vector for all batch items
    vector_param = torch.tensor([1.0, 0.5, 1.5, 0.8]).cuda()  # [stiffness, damping, forcing, boundary]
    output1 = model(x, t, scalar_param=vector_param)
    print(f"Vector conditioning (same for all): {vector_param}")
    print(f"  Input shape: {vector_param.shape} -> Interpreted as ({conditioning_dim},)")
    
    # 2. Different vector for each batch item
    vector_param = torch.tensor([
        [1.0, 0.5, 1.5, 0.8],  # batch item 0
        [2.0, 0.3, 1.0, 1.2],  # batch item 1
        [0.5, 0.8, 2.0, 0.6],  # batch item 2
        [1.5, 0.4, 1.8, 1.0],  # batch item 3
    ]).cuda()
    output2 = model(x, t, scalar_param=vector_param)
    print(f"Vector conditioning (per batch):")
    print(f"  Shape: {vector_param.shape} -> Interpreted as ({batch_size}, {conditioning_dim})")
    print(f"  Values: {vector_param}")
    
    print(f"Output shape: {output1.shape}")
    print()

def example_conditioning_modes():
    """Example comparing FiLM vs Channel conditioning"""
    print("=== Conditioning Modes Comparison ===")
    
    # Initialize model
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=2,  # Two parameters
        input_channels=3,
        Channels=[32, 64, 128, 256, 256, 192]
    ).cuda()
    
    # Example data
    batch_size = 3
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).cuda()
    t = torch.randint(0, 1000, (batch_size,)).cuda()
    conditioning_param = torch.tensor([[1.0, 0.5], [1.5, 0.8], [2.0, 0.3]]).cuda()
    
    # Test FiLM conditioning
    model.set_conditioning_mode('film')
    output_film = model(x, t, scalar_param=conditioning_param)
    print(f"FiLM conditioning output shape: {output_film.shape}")
    
    # Test Channel conditioning
    model.set_conditioning_mode('channel')
    output_channel = model(x, t, scalar_param=conditioning_param)
    print(f"Channel conditioning output shape: {output_channel.shape}")
    
    print("Note: FiLM conditioning is generally better for vector inputs")
    print("      Channel conditioning can be simpler for scalar inputs")
    print()

def main():
    """Run all examples"""
    print("UNET Conditioning Examples\n")
    
    example_helmholtz_conditioning()
    example_scalar_conditioning()
    example_vector_conditioning()
    example_conditioning_modes()
    
    print("=== Usage Recommendations ===")
    print("1. For Helmholtz equation: Use scalar conditioning with normalized wave number")
    print("2. Use FiLM conditioning for vector inputs (multiple physical parameters)")
    print("3. Use either FiLM or channel conditioning for scalar inputs")
    print("4. FiLM conditioning is more parameter-efficient and flexible")
    print("5. Vector conditioning allows encoding multiple physical properties:")
    print("   - Material properties (stiffness, density, damping)")
    print("   - Boundary conditions (strength, type)")
    print("   - Loading conditions (amplitude, frequency)")
    print("   - Geometric parameters (aspect ratio, curvature)")
    print("6. Always normalize conditioning parameters for stable training")

if __name__ == "__main__":
    main() 