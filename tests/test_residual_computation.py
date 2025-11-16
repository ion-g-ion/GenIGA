import pytest 
import torch
import numpy as np
from geniga.physics.validation import (
    compute_poisson_residual,
    compute_helmholtz_residual
)
from geniga.geometry.random import generate_random_geom as generate_geom, generate_random_bcs as generate_bcs
from pyiga import bspline, assemble, geometry, vis
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, cho_factor
import scipy.sparse.linalg
import matplotlib.pyplot as plt


def test_poisson_residual_torch():
    """
    Test the Poisson residual computation.
    """
    # Create a dummy spline space
    epsilon_strain = 4.0
    epsilon_rotation = 4.0
    bc_scale = 2.0
    bc_sigma = 8.0
    N = (64, 65)
    deg = 2
    geo, spline_space, X_new, Y_new = generate_geom(N, deg, epsilon_strain=epsilon_strain, epsilon_rotation=epsilon_rotation)
    bcs = generate_bcs(geo, spline_space, scale=bc_scale, sigma=bc_sigma)
            
    rhs = assemble.inner_products(spline_space, lambda x, y: 0 * x, f_physical=True, geo=geo).ravel()
            
    A = assemble.stiffness(spline_space, geo)
    LS = assemble.RestrictedLinearSystem(A, rhs, bcs)

    u = scipy.sparse.linalg.spsolve(LS.A, LS.b)
    u_func = geometry.BSplineFunc(spline_space, LS.complete(u))

    u_coeffs = u_func.coeffs
    geo_coeffs = geo.coeffs 
    
    assert u_coeffs.shape == N
    assert geo_coeffs.shape == (*N, 2)

    residual = compute_poisson_residual(torch.tensor(u_coeffs), torch.tensor(geo_coeffs), spline_space, return_norm=False)
    assert residual.shape == u_coeffs.shape
    assert isinstance(residual, torch.Tensor)

    assert compute_poisson_residual(torch.tensor(u_coeffs), torch.tensor(geo_coeffs), spline_space) < 1e-8

def test_helmholtz_residual_torch():
    """
    Test the Helmholtz residual computation.
    """
    # Create a dummy spline space
    epsilon_strain = 4.0
    epsilon_rotation = 4.0
    bc_scale = 2.0
    bc_sigma = 8.0
    N = (64, 65)
    deg = 2
    geo, spline_space, X_new, Y_new = generate_geom(N, deg, epsilon_strain=epsilon_strain, epsilon_rotation=epsilon_rotation)
    bcs = generate_bcs(geo, spline_space, scale=bc_scale, sigma=bc_sigma)
            
    wave_number = np.random.uniform(2.0, 8.0)
            
    rhs = assemble.inner_products(spline_space, lambda x, y: 0 * x, f_physical=True, geo=geo).ravel()
           
    # Assemble matrices for Helmholtz equation: -∇²u + k²u = f
    A = assemble.stiffness(spline_space, geo)  # Stiffness matrix (-∇² term)
    M = assemble.mass(spline_space, geo)       # Mass matrix (k²u term)
    # Combine matrices: -A + k²M (note: stiffness matrix already has negative sign)
    combined_matrix = -A + wave_number**2 * M
    LS = assemble.RestrictedLinearSystem(combined_matrix, rhs, bcs)

    u = scipy.sparse.linalg.spsolve(LS.A, LS.b)
    u_func = geometry.BSplineFunc(spline_space, LS.complete(u))

    u_coeffs = u_func.coeffs
    geo_coeffs = geo.coeffs 
    
    assert u_coeffs.shape == N
    assert geo_coeffs.shape == (*N, 2)

    residual = compute_helmholtz_residual(torch.tensor(u_coeffs), torch.tensor(geo_coeffs), wave_number, spline_space, return_norm=False)
    assert residual.shape == u_coeffs.shape
    assert isinstance(residual, torch.Tensor)

    assert compute_helmholtz_residual(torch.tensor(u_coeffs), torch.tensor(geo_coeffs), wave_number, spline_space) < 1e-8

@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_batched(device):
    """
    Test the batched residual computation.
    """
    # Create a dummy spline space
    epsilon_strain = 4.0
    epsilon_rotation = 4.0
    bc_scale = 2.0
    bc_sigma = 8.0
    N = (64, 65)
    deg = 2
    Nb = 4
    geo, spline_space, X_new, Y_new = generate_geom(N, deg, epsilon_strain=epsilon_strain, epsilon_rotation=epsilon_rotation)
    # Create a random batch of u_coeffs
    u_coeffs = torch.randn(Nb, N[0], N[1]).to(device)
    wave_number = torch.rand(Nb).to(device) * 6.0 + 2.0  # shape (Nb,), random in [2, 8)
    geo_coeffs = torch.from_numpy(geo.coeffs).to(device)


    def single_residual(u_c, w_k):
        return compute_helmholtz_residual(u_c, geo_coeffs, w_k, spline_space, return_norm=False)

    residual = torch.vmap(single_residual)(u_coeffs, wave_number)

    assert residual.device == u_coeffs.device
    assert residual.shape == (Nb, N[0], N[1])
    assert isinstance(residual, torch.Tensor)

@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
def test_batched_gradient_through_residual(device):
    """
    Test gradient propagation through batched residual computation
    with respect to a learnable input x added to u_coeffs.
    """
    Nb = 5  # batch size for this gradient test
    epsilon_strain = 4.0
    epsilon_rotation = 4.0
    bc_scale = 2.0
    bc_sigma = 8.0
    N = (6, 7)
    deg = 2
    geo, spline_space, X_new, Y_new = generate_geom(N, deg, epsilon_strain=epsilon_strain, epsilon_rotation=epsilon_rotation)
    geo_coeffs = torch.from_numpy(geo.coeffs).to(device)
    u_coeffs = (torch.randn(Nb, N[0], N[1], device=device, requires_grad=False) / 1000.0)
    # CRITICAL: Create x directly on device with requires_grad=True to ensure it's a leaf tensor
    # Using .to(device) after creation can make it non-leaf, which prevents .grad from being populated
    x = torch.randn(Nb, N[0], N[1], device=device, requires_grad=True)
    # Random wave numbers in [2, 8)
    wave_number = torch.rand(Nb, device=device) * 6.0 + 2.0

    # Add x to u_coeffs to get perturbed coefficients
    def single_residual(u_c, w_k):
        return compute_helmholtz_residual(u_c, geo_coeffs, w_k, spline_space, return_norm=False)

    u_coeffs_perturbed = u_coeffs + x
    residuals = torch.vmap(single_residual)(u_coeffs_perturbed, wave_number)
    norm = torch.norm(residuals)
    norm.backward()
    assert residuals.device == x.device
    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert torch.any(x.grad != 0)
    
    # Test finite difference versus autograd gradient for a single element in x
    idx = (0, 3, 3)  # choose index (batch_idx, i, j)
    h = 1e-3

    # Save original value for restoration
    original_val = x[idx].item()

    # Perturb positively
    x_perturb = x.clone().detach()
    x_perturb[idx] = original_val + h
    x_perturb.requires_grad = True
    u_coeffs_perturbed_pos = u_coeffs + x_perturb
    residuals_pos = torch.vmap(single_residual)(u_coeffs_perturbed_pos, wave_number)
    norm_pos = torch.norm(residuals_pos).item()

    # Perturb negatively
    x_perturb = x.clone().detach()
    x_perturb[idx] = original_val - h
    x_perturb.requires_grad = True
    u_coeffs_perturbed_neg = u_coeffs + x_perturb
    residuals_neg = torch.vmap(single_residual)(u_coeffs_perturbed_neg, wave_number)
    norm_neg = torch.norm(residuals_neg).item()

    # Finite difference approximation
    finite_diff_grad = (norm_pos - norm_neg) / (2 * h)

    # Autograd gradient
    autograd_grad = x.grad[idx].item()

    print(f"Finite difference grad: {finite_diff_grad:.6e}, Autograd grad: {autograd_grad:.6e}")
   
    assert abs(finite_diff_grad - autograd_grad)/abs(autograd_grad) < 1e-2, f"Finite difference grad: {finite_diff_grad:.6e}, Autograd grad: {autograd_grad:.6e}, norm_pos: {norm_pos:.6e}, norm_neg: {norm_neg:.6e}"

