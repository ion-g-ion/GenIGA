"""
Pytest tests for UNET conditioning functionality.

Tests cover:
1. Scalar conditioning (single physical parameter)
2. Vector conditioning (multiple physical parameters)
3. FiLM vs Channel concatenation conditioning modes
4. Helmholtz equation specific conditioning (wave number)
"""

import pytest
import torch
from geniga.nn.diff import UNET


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_helmholtz_conditioning(device):
    """Test Helmholtz equation wave number conditioning"""
    # Initialize model for Helmholtz equation
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=1,  # Single wave number parameter
        input_channels=3,    # geometry (2) + solution (1)
        Channels=[32, 64, 128, 256, 256, 192]
    ).to(device)
    
    # Set to FiLM conditioning (works well for scalar wave number)
    model.set_conditioning_mode('film')
    
    # Example input data
    batch_size = 4
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).to(device)  # [geometry + solution channels]
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Example normalization parameters (from dataset statistics)
    k_mean, k_std = 9.0, 4.0  # Approximate values for k ∈ [2, 16]
    
    # 1. Single wave number for all batch items
    k_raw = 5.0  # Raw wave number
    k_normalized = (k_raw - k_mean) / k_std  # Normalize for conditioning
    output1 = model(x, t, scalar_param=k_normalized)
    assert output1.shape == (batch_size, 1, H, W)
    
    # 2. Different wave numbers for each batch item
    k_raw_batch = torch.tensor([3.0, 7.0, 12.0, 15.0])  # Raw wave numbers
    k_normalized_batch = (k_raw_batch - k_mean) / k_std  # Normalize
    output2 = model(x, t, scalar_param=k_normalized_batch.to(device))
    assert output2.shape == (batch_size, 1, H, W)
    
    # Verify outputs are different for different wave numbers
    assert not torch.allclose(output1, output2, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_scalar_conditioning(device):
    """Test scalar conditioning (single parameter)"""
    # Initialize model with scalar conditioning
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=1,  # Single scalar parameter
        input_channels=3,    # geometry (2) + solution (1)
        Channels=[32, 64, 128, 256, 256, 192]
    ).to(device)
    
    # Set to FiLM conditioning (recommended for flexibility)
    model.set_conditioning_mode('film')
    
    # Example input data
    batch_size = 4
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).to(device)  # [geometry + solution channels]
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # 1. Single scalar for all batch items
    scalar_param = 1.5
    output1 = model(x, t, scalar_param=scalar_param)
    assert output1.shape == (batch_size, 1, H, W)
    assert isinstance(output1, torch.Tensor)
    
    # 2. Different scalar for each batch item
    scalar_param = torch.tensor([1.0, 1.5, 2.0, 0.5]).to(device)
    output2 = model(x, t, scalar_param=scalar_param)
    assert output2.shape == (batch_size, 1, H, W)
    assert isinstance(output2, torch.Tensor)
    
    # Verify outputs are different for different scalar parameters
    assert not torch.allclose(output1, output2, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_vector_conditioning(device):
    """Test vector conditioning (multiple parameters)"""
    # Initialize model with vector conditioning
    conditioning_dim = 4  # e.g., [stiffness, damping, forcing, boundary_strength]
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=conditioning_dim,
        input_channels=3,
        Channels=[32, 64, 128, 256, 256, 192]
    ).to(device)
    
    # FiLM conditioning works better for vector inputs
    model.set_conditioning_mode('film')
    
    # Example input data
    batch_size = 4
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # 1. Same vector for all batch items
    vector_param = torch.tensor([1.0, 0.5, 1.5, 0.8]).to(device)  # [stiffness, damping, forcing, boundary]
    output1 = model(x, t, scalar_param=vector_param)
    assert output1.shape == (batch_size, 1, H, W)
    assert vector_param.shape == (conditioning_dim,)
    
    # 2. Different vector for each batch item
    vector_param = torch.tensor([
        [1.0, 0.5, 1.5, 0.8],  # batch item 0
        [2.0, 0.3, 1.0, 1.2],  # batch item 1
        [0.5, 0.8, 2.0, 0.6],  # batch item 2
        [1.5, 0.4, 1.8, 1.0],  # batch item 3
    ]).to(device)
    output2 = model(x, t, scalar_param=vector_param)
    assert output2.shape == (batch_size, 1, H, W)
    assert vector_param.shape == (batch_size, conditioning_dim)
    
    # Verify outputs are different for different vector parameters
    assert not torch.allclose(output1, output2, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_conditioning_modes(device):
    """Test FiLM vs Channel conditioning modes"""
    # Initialize model
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=2,  # Two parameters
        input_channels=3,
        Channels=[32, 64, 128, 256, 256, 192]
    ).to(device)
    
    # Example data
    batch_size = 3
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    conditioning_param = torch.tensor([[1.0, 0.5], [1.5, 0.8], [2.0, 0.3]]).to(device)
    
    # Test FiLM conditioning
    model.set_conditioning_mode('film')
    output_film = model(x, t, scalar_param=conditioning_param)
    assert output_film.shape == (batch_size, 1, H, W)
    
    # Test Channel conditioning
    model.set_conditioning_mode('channel')
    output_channel = model(x, t, scalar_param=conditioning_param)
    assert output_channel.shape == (batch_size, 1, H, W)
    
    # Both modes should produce valid outputs
    assert isinstance(output_film, torch.Tensor)
    assert isinstance(output_channel, torch.Tensor)
    
    # Outputs should be different (different conditioning mechanisms)
    assert not torch.allclose(output_film, output_channel, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_conditioning_mode_switching(device):
    """Test that conditioning mode can be switched dynamically"""
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=1,
        input_channels=3,
        Channels=[32, 64, 128, 256, 256, 192]
    ).to(device)
    
    batch_size = 2
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    scalar_param = torch.tensor([1.0, 1.5]).to(device)
    
    # Switch to FiLM
    model.set_conditioning_mode('film')
    output1 = model(x, t, scalar_param=scalar_param)
    
    # Switch to Channel
    model.set_conditioning_mode('channel')
    output2 = model(x, t, scalar_param=scalar_param)
    
    # Switch back to FiLM
    model.set_conditioning_mode('film')
    output3 = model(x, t, scalar_param=scalar_param)
    
    # All outputs should be valid
    assert output1.shape == (batch_size, 1, H, W)
    assert output2.shape == (batch_size, 1, H, W)
    assert output3.shape == (batch_size, 1, H, W)
    
    # Outputs 1 and 3 should be similar (same mode), but different from 2
    assert not torch.allclose(output1, output2, atol=1e-6)
    # Note: output1 and output3 might not be exactly the same due to randomness in dropout/batch norm
    # but they should be from the same mode


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_conditioning_gradient_flow(device):
    """Test that gradients flow through conditioning parameters"""
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=1,
        input_channels=3,
        Channels=[32, 64, 128, 256, 256, 192]
    ).to(device)
    
    batch_size = 2
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Create learnable conditioning parameter
    scalar_param = torch.tensor([1.0, 1.5], device=device, requires_grad=True)
    
    model.set_conditioning_mode('film')
    output = model(x, t, scalar_param=scalar_param)
    
    # Compute loss and backward
    loss = output.mean()
    loss.backward()
    
    # Check that gradients exist
    assert scalar_param.grad is not None
    assert scalar_param.grad.shape == scalar_param.shape
    assert torch.any(scalar_param.grad != 0)


@pytest.mark.parametrize("conditioning_dim", [1, 2, 4, 8])
@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_different_conditioning_dims(device, conditioning_dim):
    """Test that model works with different conditioning dimensions"""
    model = UNET(
        use_scalar_conditioning=True,
        conditioning_dim=conditioning_dim,
        input_channels=3,
        Channels=[32, 64, 128, 256, 256, 192]
    ).to(device)
    
    batch_size = 2
    H, W = 32, 32
    x = torch.randn(batch_size, 3, H, W).to(device)
    t = torch.randint(0, 1000, (batch_size,)).to(device)
    
    # Create conditioning parameter matching the dimension
    if conditioning_dim == 1:
        conditioning_param = torch.tensor([1.0, 1.5]).to(device)
    else:
        conditioning_param = torch.randn(batch_size, conditioning_dim).to(device)
    
    model.set_conditioning_mode('film')
    output = model(x, t, scalar_param=conditioning_param)
    
    assert output.shape == (batch_size, 1, H, W)
    assert isinstance(output, torch.Tensor)

