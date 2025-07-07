"""JAX-compatible utility functions for finite element operations.

This module provides JAX-compatible alternatives to NumPy functions that have
dynamic output sizes, which are incompatible with JAX's JIT compilation.
"""

import jax
import jax.numpy as np
from typing import Tuple, Optional


def fixed_size_argwhere(condition: np.ndarray, max_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """JAX-compatible argwhere with fixed output size.
    
    This function provides a JAX-compatible alternative to np.argwhere that returns
    a fixed-size array suitable for JIT compilation. It returns both the indices
    and a mask indicating which entries are valid.
    
    Args:
        condition: Boolean array to search for True values
        max_size: Maximum number of True values expected. If None, uses the size of condition
        
    Returns:
        Tuple of:
        - indices: Array of shape (max_size, ndim) containing indices where condition is True
        - mask: Boolean array of shape (max_size,) indicating which entries in indices are valid
        
    Example:
        >>> x = np.array([True, False, True, False, True])
        >>> indices, mask = fixed_size_argwhere(x, max_size=5)
        >>> valid_indices = indices[mask]  # Will contain [0, 2, 4]
    """
    # Flatten the condition array for easier processing
    flat_condition = condition.flatten()
    ndim = len(condition.shape)
    
    if max_size is None:
        max_size = flat_condition.size
    
    # Create indices for all elements
    all_indices = np.arange(flat_condition.size)
    
    # Use where to get indices, but with a fixed size
    # We'll use a trick: sort by condition (True values first)
    sort_key = (~flat_condition).astype(np.int32)
    sorted_indices = all_indices[np.argsort(sort_key)]
    
    # Take the first max_size indices
    selected_flat_indices = sorted_indices[:max_size]
    
    # Create mask for valid entries
    mask = np.arange(max_size) < np.sum(flat_condition)
    
    # Convert flat indices back to multi-dimensional indices
    if ndim == 1:
        indices = selected_flat_indices[:, None]
    else:
        indices = np.stack(np.unravel_index(selected_flat_indices, condition.shape), axis=1)
    
    return indices, mask


def fixed_size_where(condition: np.ndarray, max_size: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """JAX-compatible where with fixed output size.
    
    Similar to fixed_size_argwhere but returns a single array of indices for 1D arrays.
    
    Args:
        condition: 1D boolean array to search for True values
        max_size: Maximum number of True values expected. If None, uses the size of condition
        
    Returns:
        Tuple of:
        - indices: Array of shape (max_size,) containing indices where condition is True
        - mask: Boolean array of shape (max_size,) indicating which entries are valid
    """
    if len(condition.shape) != 1:
        raise ValueError("fixed_size_where only supports 1D arrays")
    
    indices, mask = fixed_size_argwhere(condition, max_size)
    return indices[:, 0], mask


