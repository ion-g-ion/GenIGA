"""Physics validation module for IGA solutions."""

from .validation import compute_poisson_residual, compute_helmholtz_residual

__all__ = ['compute_poisson_residual', 'compute_helmholtz_residual']

