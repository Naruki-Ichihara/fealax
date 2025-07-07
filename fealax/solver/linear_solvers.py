"""Minimal linear solver implementations for finite element systems.

This module provides only the essential linear solver functions needed for
JIT-compiled finite element analysis. All functions are JIT-compatible for
optimal performance.

Functions:
    solve: Main JIT-compiled linear solver interface
    jax_sparse_direct_solve: JAX experimental sparse direct solver  
    jax_iterative_solve: JAX iterative solver with preconditioning
"""

import jax
import jax.numpy as np
import jax.scipy.sparse.linalg
from jax.experimental.sparse import BCOO
from typing import Dict, Any

from fealax import logger
from .linear_algebra import jax_get_diagonal

from jax import config
config.update("jax_enable_x64", True)


def solve(A: BCOO, b: np.ndarray, solver_options: Dict[str, Any] = {}) -> np.ndarray:
    """JIT-compiled linear system solver for finite element problems.
    
    This function is fully JIT-compiled for maximum performance and provides
    the primary interface for solving assembled finite element systems.
    
    Args:
        A (BCOO): System matrix in JAX BCOO sparse format.
        b (np.ndarray): Right-hand side vector.
        solver_options (dict, optional): Solver configuration. Defaults to {}.
            Supported keys:
            - 'method': Solver method ('bicgstab', 'cg'). Defaults to 'bicgstab'.
            - 'precond': Enable preconditioning. Defaults to True.
            - 'tol': Linear solver tolerance. Defaults to 1e-10.
            - 'atol': Absolute tolerance. Defaults to 1e-10.
            - 'maxiter': Maximum iterations. Defaults to 10000.
            
    Returns:
        np.ndarray: Solution vector x.
        
    Example:
        >>> solver_options = {'method': 'bicgstab', 'precond': True, 'tol': 1e-8}
        >>> solution = solve(A, b, solver_options)
    """
    # Extract parameters for JIT compilation
    method = solver_options.get('method', 'bicgstab')
    use_precond = solver_options.get('precond', True)
    tol = solver_options.get('tol', 1e-10)
    atol = solver_options.get('atol', 1e-10)
    maxiter = solver_options.get('maxiter', 10000)
    
    # Setup preconditioning
    if use_precond:
        # Use existing JIT-compatible diagonal extraction
        diagonal = jax_get_diagonal(A)
        # Safe division for preconditioning
        safe_diag = np.where(np.abs(diagonal) > 1e-12, diagonal, 1.0)
        precond = lambda x: x / safe_diag
    else:
        precond = None
    
    # Initial guess
    x0 = np.zeros_like(b)
    
    # Solve using JAX iterative methods
    if method == 'cg':
        solution, info = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, M=precond, tol=tol, maxiter=maxiter
        )
    else:
        # Default to bicgstab (handles 'bicgstab' and any other string)
        solution, info = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0, M=precond, tol=tol, atol=atol, maxiter=maxiter
        )
    
    return solution




def jax_sparse_direct_solve(A: BCOO, b: np.ndarray) -> np.ndarray:
    """Solve linear system using JAX's experimental sparse direct solver.
    
    Args:
        A (BCOO): Sparse system matrix in JAX BCOO format.
        b (np.ndarray): Right-hand side vector.
        
    Returns:
        np.ndarray: Solution vector x.
        
    Note:
        This uses JAX's experimental sparse solver. For most cases, 
        use the iterative solve() function instead.
    """
    logger.debug("JAX Sparse Direct Solver - Solving linear system")
    
    try:
        # Convert BCOO to CSR format for JAX sparse solver
        sorted_idx = np.lexsort((A.indices[:, 1], A.indices[:, 0]))
        sorted_indices = A.indices[sorted_idx]
        sorted_data = A.data[sorted_idx]
        
        rows = sorted_indices[:, 0]
        cols = sorted_indices[:, 1]
        
        # Build CSR format arrays
        row_counts = np.bincount(rows, length=A.shape[0])
        indptr = np.concatenate([np.array([0]), np.cumsum(row_counts)])
        
        # Create CSR matrix and solve
        from jax.experimental.sparse import CSR
        A_csr = CSR((sorted_data, cols, indptr), shape=A.shape)
        x = jax.experimental.sparse.linalg.spsolve(A_csr.data, A_csr.indices, A_csr.indptr, b)
        
        logger.debug("JAX Sparse Direct Solver - Finished solving")
        return x
        
    except Exception as e:
        logger.warning(f"JAX sparse direct solver failed: {e}")
        # Fallback to iterative solver
        return solve(A, b, {'method': 'bicgstab', 'precond': True})


