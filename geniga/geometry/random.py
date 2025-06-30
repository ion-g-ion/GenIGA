import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2
from pyiga import bspline, assemble, geometry, vis
from scipy.spatial.distance import cdist
from scipy.linalg import cho_solve, cho_factor
import scipy.sparse.linalg
from typing import Tuple


def generate_random_geom(
    N: Tuple[int, int] = (128, 128),
    deg: int = 2,
    epsilon_strain: float = 4.0,
    epsilon_rotation: float = 3.0
) -> Tuple[geometry.BSplineFunc, Tuple, np.ndarray, np.ndarray]:
    """
    Generates a random geometry using strain and rotation fields, then applies a random rotation.

    Args:
        N (Tuple[int, int]): Number of points in x and y directions.
        deg (int): Polynomial degree of B-spline basis.
        epsilon_strain (float): Scaling factor for strain fields.
        epsilon_rotation (float): Scaling factor for rotation fields.

    Returns:
        Tuple[geometry.BSplineFunc, Tuple, np.ndarray, np.ndarray]:
            A geometry object, the knot vectors, and the updated coordinate grids (X_new, Y_new).
    """
    Nx, Ny = N  # Unpack N into Nx and Ny
    # Step 1: Grid
    x_max = np.random.uniform(0.5, 2.5)
    y_max = np.random.uniform(0.5, 2.5)
    x = np.linspace(0, x_max, Nx)
    y = np.linspace(0, y_max, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    J = 0.0
    while J < 0.08:
        
        # Step 2: Generate symmetric (strain) and antisymmetric (rotation) fields
        h_xx = epsilon_strain * gaussian_filter(np.random.randn(Nx, Ny), sigma=5) 
        h_yy = epsilon_strain * gaussian_filter(np.random.randn(Nx, Ny), sigma=5)
        h_xy = epsilon_strain * gaussian_filter(np.random.randn(Nx, Ny), sigma=5)

        rotation_field = epsilon_rotation * gaussian_filter(np.random.randn(Nx, Ny), sigma=8)
        rotation_field = epsilon_rotation * np.random.randn() * np.exp(-((X-np.random.rand()*x.max())**2 + (Y-np.random.rand()*y.max())**2)/0.5)

        # Step 3: Build divergence terms
        div_hx = np.gradient(h_xx, x, axis=0) + np.gradient(h_xy, y, axis=1)
        div_hy = np.gradient(h_xy, x, axis=0) + np.gradient(h_yy, y, axis=1)

        div_hx += np.gradient(rotation_field, y, axis=1)
        div_hy += -np.gradient(rotation_field, x, axis=0)

        # Step 4: Fourier solve
        kx = np.fft.fftfreq(Nx, d=x[1]-x[0]) * 2*np.pi
        ky = np.fft.fftfreq(Ny, d=y[1]-y[0]) * 2*np.pi
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        Laplacian = -(KX**2 + KY**2)
        Laplacian[0, 0] = 1.0

        div_hx_hat = fft2(div_hx)
        div_hy_hat = fft2(div_hy)

        u_x_hat = div_hx_hat / Laplacian
        u_y_hat = div_hy_hat / Laplacian

        u_x = np.real(ifft2(u_x_hat))
        u_y = np.real(ifft2(u_y_hat))

        # Step 5: New grid
        X_new = X + u_x
        Y_new = Y + u_y

        # Center around the center of mass
        X_com = np.mean(X_new)
        Y_com = np.mean(Y_new)
        X_new -= X_com
        Y_new -= Y_com

        # Apply random rotation
        theta = np.random.uniform(0, 2 * np.pi)  # Random rotation angle
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                    [np.sin(theta), np.cos(theta)]])
        coords = np.stack([X_new.ravel(), Y_new.ravel()])
        rotated_coords = rotation_matrix @ coords
        X_new = rotated_coords[0].reshape(X_new.shape)
        Y_new = rotated_coords[1].reshape(Y_new.shape)

        # define discretization space
        kvs = tuple(bspline.make_knots(deg, 0.0, 1.0, n-deg) for n in N)

        geo = geometry.tensor_product(geometry.BSplineFunc(kvs[0], np.linspace(0, 1,kvs[0].numdofs)), geometry.BSplineFunc(kvs[1], np.linspace(0,1,kvs[1].numdofs)))
        # Update geo.coeffs with new coordinates
        geo.coeffs = np.stack([Y_new, X_new], axis=-1)

        J = geo.grid_jacobian([s.greville()[1:-1] for s in kvs])
        J = np.linalg.det(J).min() / np.linalg.det(J).max()
        #print("Jacobian determinant ratio:", J)
        
    return geo, kvs, X_new, Y_new

def generate_random_bcs(
    geo: geometry.BSplineFunc,
    kvs,
    scale: float = 2.0,
    sigma: float = 16.0
):
    """
    Creates Dirichlet boundary conditions for the provided geometry.

    Args:
        geo (geometry.BSplineFunc): The geometry object.
        kvs: Knot vectors used to define the geometry.
        scale (float): Scaling factor for the random boundary values.
        sigma (float): Smoothing parameter for the Gaussian filter.

    Returns:
        Any: A structure containing Dirichlet boundary condition indices and values.
    """
    # Define boundary conditions

    g = lambda x,y: 0*x # source term
    # compute Dirichlet boundary conditions
    bcs = assemble.compute_dirichlet_bcs(kvs, geo, [('left', g), ('top', g), ('right', g), ('bottom', g)])
    
    V = scale * np.random.randn() * gaussian_filter(np.random.randn(*geo.coeffs.shape[:2]), sigma=sigma).flatten()
    
    bcs[1][:] = V[bcs[0]]
    return bcs



