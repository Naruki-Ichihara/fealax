"""Linear algebra utilities and matrix operations for finite element solvers."""

import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
from typing import Union, Optional


def jax_get_diagonal(A: BCOO) -> np.ndarray:
    """Extract diagonal elements from BCOO sparse matrix.
    
    This implementation is JIT-compatible, avoiding boolean indexing
    which causes issues with JAX's JIT compilation.
    
    Args:
        A (BCOO): Sparse matrix in BCOO format.
        
    Returns:
        np.ndarray: Diagonal elements.
    """
    # Check which entries are on the diagonal
    is_diagonal = A.indices[:, 0] == A.indices[:, 1]
    
    # Create output array
    diagonal = np.zeros(A.shape[0])
    
    # Use scatter operation to accumulate diagonal values
    # This handles duplicate diagonal entries correctly
    diagonal = diagonal.at[A.indices[:, 0]].add(
        np.where(is_diagonal, A.data, 0.0)
    )
    
    return diagonal


def zero_rows_jax(A: BCOO, row_indices: np.ndarray) -> BCOO:
    """Zero out specified rows AND columns in JAX BCOO matrix and set diagonal entries to 1.0.
    
    This maintains matrix symmetry for symmetric problems like elasticity.
    
    Args:
        A (BCOO): Input sparse matrix.
        row_indices (np.ndarray): Indices of rows/columns to zero out.
        
    Returns:
        BCOO: Matrix with specified rows and columns zeroed and diagonal entries set to 1.0.
    """
    # Create mask for entries not in the specified rows OR columns
    # This maintains symmetry by zeroing both rows and columns
    row_mask = ~np.isin(A.indices[:, 0], row_indices)
    col_mask = ~np.isin(A.indices[:, 1], row_indices)
    mask = row_mask & col_mask
    
    # Also remove existing diagonal entries in the rows/columns to be zeroed
    # to avoid duplicate diagonal entries
    diagonal_mask = (A.indices[:, 0] == A.indices[:, 1])
    constrained_diagonal_mask = np.isin(A.indices[:, 0], row_indices) & diagonal_mask
    mask = mask & ~constrained_diagonal_mask
    
    # Filter indices and data
    new_indices = A.indices[mask, :]
    new_data = A.data[mask]
    
    # Add diagonal entries for the constrained DOFs (exactly 1.0, no duplicates)
    diagonal_indices = np.column_stack([row_indices, row_indices])
    diagonal_data = np.ones(len(row_indices))
    
    # Combine filtered matrix with diagonal entries
    all_indices = np.vstack([new_indices, diagonal_indices])
    all_data = np.concatenate([new_data, diagonal_data])
    
    return BCOO((all_data, all_indices), shape=A.shape)


def jax_matrix_multiply(A: BCOO, B: BCOO) -> BCOO:
    """Multiply two JAX BCOO matrices.
    
    Args:
        A (BCOO): First matrix.
        B (BCOO): Second matrix.
        
    Returns:
        BCOO: Result of A @ B.
    """
    # Use JAX's native sparse matrix multiplication
    return A @ B


def array_to_jax_vec(arr: Union[np.ndarray, np.ndarray], size: Optional[int] = None) -> np.ndarray:
    """Convert a JAX or NumPy array to a JAX array.

    Args:
        arr (array-like): JAX array (DeviceArray) or NumPy array of shape (N,).
        size (int, optional): Vector size. If None, uses len(arr) as vector size.

    Returns:
        np.ndarray: JAX array with values from arr.
    """
    arr_jax = np.array(arr).flatten()  # ensure JAX, ensure 1D
    if size is not None and arr_jax.shape[0] != size:
        # Pad or truncate to desired size
        if arr_jax.shape[0] < size:
            arr_jax = np.pad(arr_jax, (0, size - arr_jax.shape[0]))
        else:
            arr_jax = arr_jax[:size]
    return arr_jax