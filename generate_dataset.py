import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2
from pyiga import bspline, assemble, geometry, vis
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, cho_factor
import scipy.sparse.linalg
import h5py
from tqdm import tqdm
import os
from geniga.geometry.random import generate_random_geom as generate_geom, generate_random_bcs as generate_bcs



def solve_pde(equation, n_samples, N, output_file, deg=2, epsilon_strain=4.0, epsilon_rotation=4.0, 
              bc_scale=2.0, bc_sigma=8.0, wave_number_range=(2.0, 16.0)):
    """
    Solve PDE and generate dataset for either Poisson or Helmholtz equation.
    
    Args:
        equation: "poisson" or "helmholtz"
        n_samples: Number of samples to generate
        N: Grid size tuple (nx, ny)
        output_file: Output HDF5 file path
        deg: B-spline degree
        epsilon_strain: Strain parameter for geometry generation
        epsilon_rotation: Rotation parameter for geometry generation
        bc_scale: Boundary condition scale
        bc_sigma: Boundary condition sigma
        wave_number_range: Tuple (min, max) for random wave number generation (Helmholtz only)
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with h5py.File(output_file, "w") as h5f:
        solutions_dataset = h5f.create_dataset("solutions", (n_samples, np.prod(N)), dtype=np.float64)
        geo_coeffs_dataset = h5f.create_dataset("geo_coeffs", (n_samples, 2 * np.prod(N)), dtype=np.float64)
        
        # Create wave number dataset for Helmholtz equation
        if equation == "helmholtz":
            wave_numbers_dataset = h5f.create_dataset("wave_numbers", (n_samples,), dtype=np.float64)

        for k in tqdm(range(n_samples), desc=f"Generating {equation} samples"):
            geo, spline_space, X_new, Y_new = generate_geom(N, deg, epsilon_strain=epsilon_strain, epsilon_rotation=epsilon_rotation)
            bcs = generate_bcs(geo, spline_space, scale=bc_scale, sigma=bc_sigma)
            
            # Generate random wave number for Helmholtz equation
            if equation == "helmholtz":
                wave_number = np.random.uniform(wave_number_range[0], wave_number_range[1])
            
            rhs = assemble.inner_products(spline_space, lambda x, y: 0 * x, f_physical=True, geo=geo).ravel()
            if equation == "poisson":
                A = assemble.stiffness(spline_space, geo)
                LS = assemble.RestrictedLinearSystem(A, rhs, bcs)
            elif equation == "helmholtz":
                # Assemble matrices for Helmholtz equation: -∇²u + k²u = f
                A = assemble.stiffness(spline_space, geo)  # Stiffness matrix (-∇² term)
                M = assemble.mass(spline_space, geo)       # Mass matrix (k²u term)
                # Combine matrices: -A + k²M (note: stiffness matrix already has negative sign)
                combined_matrix = -A + wave_number**2 * M
                LS = assemble.RestrictedLinearSystem(combined_matrix, rhs, bcs)
            else:
                raise ValueError(f"Unknown equation: {equation}")

            u = scipy.sparse.linalg.spsolve(LS.A, LS.b)
            u_func = geometry.BSplineFunc(spline_space, LS.complete(u))

            # Get the matrix to store (either A for Poisson or combined_matrix for Helmholtz)
            if equation == "poisson":
                matrix_to_store = A
            elif equation == "helmholtz":
                matrix_to_store = combined_matrix
            else:
                raise ValueError(f"Unknown equation: {equation}")

            # Create datasets dynamically in the first iteration when sizes are known
            if k == 0:
                mats_data_dataset = h5f.create_dataset("mats_data", (n_samples, matrix_to_store.data.shape[0]), dtype=np.float64)
                mats_indices_dataset = h5f.create_dataset("mats_indices", (n_samples, matrix_to_store.indices.shape[0]), dtype=np.int32)
                mats_indptr_dataset = h5f.create_dataset("mats_indptr", (n_samples, matrix_to_store.indptr.shape[0]), dtype=np.int32)
                rhs_dataset = h5f.create_dataset("rhs", (n_samples, rhs.shape[0]), dtype=np.float64)

            # Save data to HDF5
            solutions_dataset[k] = u_func.coeffs.flatten()
            mats_data_dataset[k, :matrix_to_store.data.shape[0]] = matrix_to_store.data
            mats_indices_dataset[k, :matrix_to_store.indices.shape[0]] = matrix_to_store.indices
            mats_indptr_dataset[k, :matrix_to_store.indptr.shape[0]] = matrix_to_store.indptr
            rhs_dataset[k, :rhs.shape[0]] = rhs
            geo_coeffs_dataset[k] = geo.coeffs.flatten()
            
            # Save wave number for Helmholtz equation
            if equation == "helmholtz":
                wave_numbers_dataset[k] = wave_number

        # Ensure data is written to disk
        h5f.flush()
    
    print(f"Generated {n_samples} samples for {equation} equation with grid size {N}")
    return geo, u_func


# Main function to generate different datasets
if __name__ == "__main__":
    # Generate Poisson datasets with 32x32 grid
    print("Generating Poisson datasets...")
    
    # Poisson training dataset
    solve_pde(
        equation="poisson",
        n_samples=32768,
        N=(32, 32),
        output_file="data/poisson_small/dataset_train.h5",
        deg=2,
        epsilon_strain=4.0,
        epsilon_rotation=4.0,
        bc_scale=2.0,
        bc_sigma=8.0
    )
    
    # Poisson test dataset
    solve_pde(
        equation="poisson",
        n_samples=1024,
        N=(32, 32),
        output_file="data/poisson_small/dataset_test.h5",
        deg=2,
        epsilon_strain=4.0,
        epsilon_rotation=4.0,
        bc_scale=2.0,
        bc_sigma=8.0
    )
    
    # Generate Helmholtz datasets with 128x128 grid
    print("Generating Helmholtz datasets...")
    
    # Helmholtz training dataset
    solve_pde(
        equation="helmholtz",
        n_samples=32768,
        N=(128, 128),
        output_file="data/helmholtz/dataset_train.h5",
        deg=2,
        epsilon_strain=4.0,
        epsilon_rotation=4.0,
        bc_scale=2.0,
        bc_sigma=8.0,
        wave_number_range=(2.0, 16.0)
    )
    
    # Helmholtz test dataset
    solve_pde(
        equation="helmholtz",
        n_samples=1024,
        N=(128, 128),
        output_file="data/helmholtz/dataset_test.h5",
        deg=2,
        epsilon_strain=4.0,
        epsilon_rotation=4.0,
        bc_scale=2.0,
        bc_sigma=8.0,
        wave_number_range=(2.0, 16.0)
    )
    
    print("All datasets generated successfully!")
