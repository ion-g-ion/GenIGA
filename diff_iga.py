# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange  # pip install einops
from typing import List
import random
import math
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from timm.utils import ModelEmaV3  # pip install timm
from tqdm import tqdm  # pip install tqdm
import matplotlib.pyplot as plt  # pip install matplotlib
import torch.optim as optim
import numpy as np
import os  # Add this import for directory handling
import h5py
from geniga.nn.diff import SinusoidalEmbeddings, ResBlock, Attention, UnetLayer, UNET, DDPM_Scheduler
from geniga.geometry.datasets import SinglePatchIGADataset
import scipy.sparse as sp  # sparse operations
import scipy.sparse.linalg as splinalg  # sparse linear solver
from pyiga import bspline, assemble, geometry, vis


def train(N: list[int],
          d: int = 2,
          batch_size: int = 128,
          num_time_steps: int = 1000,
          num_epochs: int = 15,
          seed: int = -1,
          ema_decay: float = 0.9999,
          lr=2e-5,
          checkpoint_path: str = None,
          dataset_path_train: str = "",
          dataset_path_test: str = ""):

    deg = 2
    spline_space = tuple(bspline.make_knots(deg, 0.0, 1.0, n-deg) for n in N)
    iga_geo = geometry.tensor_product(geometry.BSplineFunc(spline_space[0], np.linspace(
        0, 1,  spline_space[0].numdofs)), geometry.BSplineFunc(spline_space[1], np.linspace(0, 1, spline_space[1].numdofs)))

    train_dataset = SinglePatchIGADataset(d=d, h5_files=[dataset_path_train])
    test_dataset = SinglePatchIGADataset(
        d=d, h5_files=[dataset_path_test], return_matrix=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=True, drop_last=True, num_workers=4)

    # Compute scaling statistics from training data
    print("Computing scaling statistics...")
    geo_coeffs_all = []
    targets_all = []

    for geo_coeffs, targets in tqdm(train_loader, desc="Computing statistics"):
        geo_coeffs = geo_coeffs.reshape([-1, d] + N)
        targets = targets.reshape([-1, 1] + N)
        geo_coeffs_all.append(geo_coeffs)
        targets_all.append(targets)

    geo_coeffs_all = torch.cat(geo_coeffs_all, dim=0)
    targets_all = torch.cat(targets_all, dim=0)

    # Compute mean and std for standardization
    geo_mean = geo_coeffs_all.mean(dim=0, keepdim=True)
    # Add small epsilon to avoid division by zero
    geo_std = geo_coeffs_all.std(dim=0, keepdim=True) + 1e-8

    targets_mean = targets_all.mean()
    targets_std = targets_all.std() + 1e-8

    print(
        f"Geometry coefficients - Mean: {geo_mean.mean():.4f}, Std: {geo_std.mean():.4f}")
    print(f"Targets - Mean: {targets_mean:.4f}, Std: {targets_std:.4f}")

    # Store scaling parameters
    scaling_params = {
        'geo_mean': geo_mean,
        'geo_std': geo_std,
        'targets_mean': targets_mean,
        'targets_std': targets_std
    }

    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    model = UNET().cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    ema = ModelEmaV3(model, decay=ema_decay)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['weights'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # Load scaling parameters if available
        if 'scaling_params' in checkpoint:
            scaling_params = checkpoint['scaling_params']
            geo_mean = scaling_params['geo_mean'].cuda()
            geo_std = scaling_params['geo_std'].cuda()
            targets_mean = scaling_params['targets_mean']
            targets_std = scaling_params['targets_std']
            print("Loaded scaling parameters from checkpoint")

    # Move scaling parameters to GPU
    geo_mean = geo_mean.cuda()
    geo_std = geo_std.cuda()

    criterion = nn.MSELoss(reduction='mean')

    os.makedirs('checkpoints', exist_ok=True)

    for i in range(num_epochs):
        total_loss = 0

        for idx, (geo_coeffs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {i+1}/{num_epochs}")):
            geo_coeffs, targets = geo_coeffs.cuda(), targets.cuda()
            geo_coeffs = geo_coeffs.reshape([-1, d] + N)
            targets = targets.reshape([-1, 1] + N)

            # Apply scaling
            geo_coeffs = (geo_coeffs - geo_mean) / geo_std
            targets = (targets - targets_mean) / targets_std

            t = torch.randint(0, num_time_steps, (batch_size,))
            eps = torch.randn_like(targets, requires_grad=False)
            a = scheduler.alpha[t].view((batch_size, 1) + d*(1,)).cuda()
            noisy_targets = (torch.sqrt(a) * targets) + \
                (torch.sqrt(1 - a) * eps)
            output = model(torch.cat([noisy_targets, geo_coeffs], dim=1), t)
            optimizer.zero_grad()
            loss = criterion(output, eps)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            ema.update(model)

        # Test the model
        for idx, (geo_coeffs, targets, indptr, indices, values, rhs) in enumerate(test_loader):
            geo_coeffs, targets = geo_coeffs.cuda(), targets.cuda()
            indptr, indices, values, rhs = indptr.cuda(
            ), indices.cuda(), values.cuda(), rhs.cuda()

            geo_coeffs = geo_coeffs.reshape([-1, d] + N)
            # Update geo.coeffs with new coordinates
            iga_geo.coeffs = np.concatenate([geo_coeffs.cpu().numpy()[0,k,..., None] for k in range(d)], axis=-1)
            targets = targets.reshape([-1, 1] + N)

            # Apply scaling to geometry coefficients
            geo_coeffs_scaled = (geo_coeffs - geo_mean) / geo_std

            # Generate normalized noise (since the model operates in normalized space)
            noise = torch.randn_like(targets, requires_grad=False)

            solution = sample_from_model(geo_coeffs_scaled, noise, model, None, num_time_steps,
                                         device="cuda")

            # Denormalize solution
            solution = solution * targets_std + targets_mean

            # Assemble SciPy CSR matrix directly -----------------------------------
            n_unknowns = indptr.numel() - 1  # number of DOFs

            indptr_np = indptr.long().cpu().numpy().flatten()
            indices_np = indices.long().cpu().numpy().flatten()
            values_np = values.cpu().numpy().flatten()

            A_csr = sp.csr_matrix(
                (values_np, indices_np, indptr_np), shape=(n_unknowns, n_unknowns))
            # (Optional) create a PyTorch CSR if you still need it later
            A = torch.sparse_csr_tensor(
                indptr.long(), indices.long(), values, size=(n_unknowns, n_unknowns))

            rhs = rhs.view(-1, 1)  # Ensure RHS is a column vector


            # ------------------------------------------------------------------
            # Physics check: compute residual using SciPy sparse on CPU
            # ------------------------------------------------------------------
            # Move tensors to CPU and numpy --------------------------------------
            uc_pred_np = solution.view(-1).detach().cpu().numpy()
            rhs_np = rhs.view(-1).detach().cpu().numpy()

            # Residual ------------------------------------------------------------
            res_np = A_csr.dot(uc_pred_np) - rhs_np

            # Interior / boundary masks -----------------------------------------
            # 1=boundary, 0=interior
            bd_mask_nd = np.ones(tuple(N), dtype=np.float64)
            bd_mask_nd[(slice(1, -1),) * len(N)] = 0  # set interior to 0
            bd_mask_vec = bd_mask_nd.reshape(-1)      # shape (n,)
            int_mask_vec = 1.0 - bd_mask_vec           # 1 interior, 0 boundary

            int_idx = np.where(int_mask_vec == 1.0)[0]
            bd_idx = np.where(bd_mask_vec == 1.0)[0]

            residual_norm = np.max(np.abs(res_np[int_idx]))
            print(
                f"Infinity-norm of interior residual (SciPy): {residual_norm:.4e}")

            # ------------------------------------------------------------------
            # Build diagonal projectors Pin_tt and Pbd_tt -----------------------
            Pin = sp.diags(int_mask_vec, 0, shape=(
                n_unknowns, n_unknowns), format="csr")
            Pbd = sp.diags(bd_mask_vec, 0, shape=(
                n_unknowns, n_unknowns), format="csr")

            # g_tt vector: predicted boundary values -----------------------------
            g_vec = bd_mask_vec * uc_pred_np  # size n, zeros interior

            # Assemble modified stiffness matrix --------------------------------
            sys_mat = Pin @ A_csr @ Pin + Pbd

            # RHS according to formula ------------------------------------------
            rhs_ref = Pin.dot(rhs_np - A_csr.dot(g_vec)) + g_vec

            # Solve system -------------------------------------------------------
            u_ref = splinalg.spsolve(sys_mat, rhs_ref)

            # Error on interior nodes vs. ground-truth ---------------------------
            err_2 = np.linalg.norm(u_ref-uc_pred_np) / np.linalg.norm(u_ref)
            print(f"Error of solution: {err_2:.4e}")

            plt.figure()
            plt.imshow(u_ref.reshape(N))
            plt.colorbar()
            plt.savefig(f'checkpoints/u_ref_{i}.png')

            plt.figure()
            plt.imshow((u_ref.reshape(N)-uc_pred_np.reshape(N)) /
                       np.linalg.norm(u_ref.reshape(N), ord=np.inf))
            plt.colorbar()
            plt.savefig(f'checkpoints/u_ref_diff_{i}.png')
            plt.close()

            plt.figure()
            plt.imshow(targets.cpu().numpy().reshape(N))
            plt.colorbar()
            plt.savefig(f'checkpoints/targets_{i}.png')

            print(iga_geo.coeffs.shape)
            u_func = geometry.BSplineFunc(spline_space, uc_pred_np)
            
            plt.figure()
            vis.plot_field(u_func, iga_geo)
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
            #plt.figure()
            #plt.imshow(uc_pred_np.reshape(N))
            #plt.colorbar()
            plt.savefig(f'checkpoints/uc_pred_np_{i}.png')

            break

        print(f'Epoch {i+1} | Loss {total_loss / len(train_loader):.5f}')

    checkpoint = {
        'weights': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict(),
        'scaling_params': scaling_params
    }
    torch.save(checkpoint, 'checkpoints/ddpm_checkpoint.pt')


def sample_from_model(
    geo_coeffs: torch.Tensor,
    z: torch.Tensor,
    model: nn.Module,
    ema: ModelEmaV3 | None,
    num_time_steps: int,
    device: str = "cuda",
):
    """Run the reverse DDPM sampling loop.

    Args:
        geo_coeffs (torch.Tensor):
            Geometry coefficients with shape ``(B, d, *N)`` where
            ``*N`` is a sequence of ``d`` spatial dimensions. The tensor must
            already live on *device*.
        z (torch.Tensor):
            Initial Gaussian noise of shape ``(B, 1, *N)`` whose spatial part
            matches ``geo_coeffs``.
        model (nn.Module):
            Neural network that predicts the noise \(ϵ̂). Typically an instance
            of :class:`UNET`, but any network with the signature
            ``model(x, t)``, producing a tensor of shape ``(B, 1, *N)``, is
            accepted.
        ema (ModelEmaV3 | None):
            Exponential-moving-average wrapper created during training. If
            provided, the EMA weights will be used for sampling; pass *None* to
            use the raw *model* parameters.
        num_time_steps (int):
            Number of diffusion steps (*T*) used during training; determines
            the length of the reverse trajectory.
        device (str, optional):
            Target device on which sampling is executed. Defaults to ``"cuda"``.

    Returns:
        torch.Tensor: The denoised solution ``x₀`` with shape ``(B, 1, *N)`` on
        CPU memory. The solution is in normalized space and needs to be
        denormalized using the scaling parameters.
    """
    # Select which parameters to use -------------------------------------------------
    _model = model
    if ema is not None:
        _model = ema.module  # type: ignore[attr-defined]
    _model.eval()

    # ------------------------------------------------------------------
    # Validate batch shapes -------------------------------------------
    # Allow arbitrary spatial rank (*N) as long as the leading dims are
    # (B, channels, *N).  geo_coeffs carries `d` geometry channels.
    # z carries a single solution channel.
    # ------------------------------------------------------------------
    assert geo_coeffs.dim() >= 3, "geo_coeffs must be (B, d, *N) with N≥1"
    assert z.dim() == geo_coeffs.dim(), (
        "z and geo_coeffs must have the same number of dimensions"
    )
    assert geo_coeffs.shape[0] == z.shape[0], "Batch size mismatch between inputs"
    assert z.shape[1] == 1, "z is expected to have exactly one channel (solution)"

    B = z.shape[0]

    # Scheduler ---------------------------------------------------------------------
    scheduler = DDPM_Scheduler(num_time_steps=num_time_steps)
    beta = scheduler.beta.to(device)
    alpha = scheduler.alpha.to(device)

    # Prepare network input: concatenate along channel axis --------------------------
    x_in = torch.cat([z, geo_coeffs], dim=1)  # B × (1+d) × H × W

    with torch.no_grad():
        for t in reversed(range(1, num_time_steps)):
            t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
            temp = beta[t_tensor] / \
                (torch.sqrt(1-alpha[t_tensor])*torch.sqrt(1-beta[t_tensor]))
            x_in = torch.cat([z, geo_coeffs], dim=1)
            z = 1/torch.sqrt(1-beta[t_tensor])*z - \
                temp*_model(x_in.to(device), t_tensor)

            eps = torch.randn_like(z)
            z = z + eps*torch.sqrt(beta[t_tensor])

        t = 0.
        t_tensor = torch.full((B,), t, dtype=torch.long, device=device)
        temp = beta[t_tensor] / \
            (torch.sqrt(1-alpha[t_tensor])*torch.sqrt(1-beta[t_tensor]))
        z = 1/torch.sqrt(1-beta[t_tensor])*z - temp * \
            _model(x_in.to(device), t_tensor)

    return z.cpu().detach()


def main():
    train([32, 32], checkpoint_path=None, lr=2e-5, num_epochs=100,
          dataset_path_train="data/generated_data_small.h5", dataset_path_test="data/generated_data_small_test.h5")
    # inference('checkpoints/ddpm_checkpoint')


if __name__ == '__main__':
    main()
