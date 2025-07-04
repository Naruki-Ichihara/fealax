"""Boundary condition management for finite element problems.

This module provides classes and utilities for defining and applying boundary
conditions in finite element problems. It includes Dirichlet boundary condition
specifications and efficient enforcement methods.

Key Classes:
    DirichletBC: Dirichlet boundary condition specification
    BoundaryConditionManager: Manages boundary condition application and data

Key Functions:
    precompute_bc_data: Pre-compute boundary condition data for JIT compilation
    apply_bcs_to_assembled_system: Apply boundary conditions to assembled system

Example:
    >>> from fealax.problem.boundary_conditions import DirichletBC
    >>> bc = DirichletBC(
    ...     subdomain=lambda x: x[0] < 1e-6,  # left boundary
    ...     vec=0,  # x-component
    ...     eval=lambda x: 0.0  # zero displacement
    ... )
"""

import jax
import jax.numpy as np
import jax.flatten_util
from dataclasses import dataclass
from typing import Callable, List, Dict, Optional, Any, Tuple
from jax.experimental.sparse import BCOO


@dataclass
class DirichletBC:
    """Dirichlet boundary condition specification.

    Defines a Dirichlet (essential) boundary condition by specifying the subdomain
    where the condition applies, which vector component to constrain, and the
    prescribed value function.

    Attributes:
        subdomain (Callable): Function that defines the boundary subdomain.
            Takes a point coordinate array and returns boolean indicating
            whether the boundary condition applies at that location.
            Signature: subdomain(x: np.ndarray) -> bool
        vec (int): Vector component index to apply the boundary condition to.
            Must be in range [0, vec-1] where vec is the number of solution
            components (e.g., 0, 1, 2 for x, y, z components of displacement).
        eval (Callable): Function that evaluates the prescribed boundary value.
            Takes a point coordinate array and returns the prescribed value.
            Signature: eval(x: np.ndarray) -> float

    Example:
        >>> # Fixed displacement in x-direction on left boundary
        >>> bc = DirichletBC(
        ...     subdomain=lambda x: np.abs(x[0]) < 1e-6,  # left boundary
        ...     vec=0,  # x-component
        ...     eval=lambda x: 0.0  # zero displacement
        ... )
    """

    subdomain: Callable[[np.ndarray], bool]
    vec: int
    eval: Callable[[np.ndarray], float]


class BoundaryConditionManager:
    """Manages boundary condition application and data preprocessing.
    
    This class provides utilities for processing boundary conditions,
    pre-computing boundary condition data for efficient JIT compilation,
    and applying boundary conditions to assembled finite element systems.
    """

    @staticmethod
    def precompute_bc_data(fes: List[Any], offset: List[int]) -> Dict[str, Any]:
        """Pre-compute boundary condition data for efficient JIT compilation.
        
        This method extracts all boundary condition information from the finite element
        spaces and converts it to JAX-compatible arrays. This separation allows the
        boundary condition processing (which uses non-JIT operations like np.argwhere)
        to be done once during setup, while the actual application can be JIT-compiled.
        
        Args:
            fes (List[Any]): List of finite element spaces containing boundary condition data.
            offset (List[int]): List of DOF offsets for each finite element space.
            
        Returns:
            Dict[str, Any]: Dictionary containing:
                - 'indices': JAX array of global DOF indices with BCs applied
                - 'values': JAX array of prescribed boundary condition values  
                - 'has_bcs': Boolean indicating if any BCs are present
                - 'fe_info': List of per-finite-element BC information
                - 'num_constrained_dofs': Number of constrained DOFs
                
        Note:
            This method should be called once after problem setup. The returned
            data can then be used with JIT-compiled functions for applying BCs.
            
        Example:
            >>> bc_data = BoundaryConditionManager.precompute_bc_data(fes, offset)
            >>> if bc_data['has_bcs']:
            ...     # Use bc_data['indices'] and bc_data['values'] in JIT functions
        """
        bc_indices_list = []
        bc_values_list = []
        fe_info_list = []
        
        for fe_idx, fe in enumerate(fes):
            fe_bc_info = {
                'fe_idx': fe_idx,
                'bc_count': len(fe.node_inds_list),
                'offset': offset[fe_idx],
                'vec': fe.vec
            }
            
            for bc_idx in range(len(fe.node_inds_list)):
                # Get BC data from finite element
                node_inds = fe.node_inds_list[bc_idx]
                vec_inds = fe.vec_inds_list[bc_idx] 
                values = fe.vals_list[bc_idx]
                
                # Convert to global DOF indices
                global_dof_inds = node_inds * fe.vec + vec_inds + offset[fe_idx]
                
                # Collect data
                bc_indices_list.extend(global_dof_inds.tolist())
                bc_values_list.extend(values.tolist())
                
                fe_bc_info[f'bc_{bc_idx}'] = {
                    'node_inds': node_inds,
                    'vec_inds': vec_inds,
                    'values': values,
                    'global_dofs': global_dof_inds
                }
            
            fe_info_list.append(fe_bc_info)
        
        # Convert to JAX arrays (empty arrays if no BCs)
        bc_indices = np.array(bc_indices_list) if bc_indices_list else np.array([], dtype=np.int32)
        bc_values = np.array(bc_values_list) if bc_values_list else np.array([])
        
        return {
            'indices': bc_indices,
            'values': bc_values, 
            'has_bcs': len(bc_indices_list) > 0,
            'fe_info': fe_info_list,
            'num_constrained_dofs': len(bc_indices_list)
        }

    @staticmethod
    def apply_bcs_to_assembled_system(A: BCOO, b: np.ndarray, bc_data: Dict[str, Any]) -> Tuple[BCOO, np.ndarray]:
        """Apply boundary conditions to an assembled finite element system.
        
        This method applies Dirichlet boundary conditions to the assembled system
        matrix and right-hand side vector using the row elimination method. It uses
        the precomputed boundary condition data to avoid non-JIT-compatible operations.
        
        Args:
            A (BCOO): System matrix in JAX sparse format.
            b (np.ndarray): Right-hand side vector.
            bc_data (Dict[str, Any]): Precomputed boundary condition data from precompute_bc_data().
            
        Returns:
            Tuple[BCOO, np.ndarray]: (A_bc, b_bc) where:
                - A_bc: System matrix with BCs applied
                - b_bc: RHS vector with BCs applied
                
        Note:
            This method can be made JIT-compatible in the future by using only
            JAX operations. Currently it uses the existing BC application logic
            from the solver module.
            
        Example:
            >>> bc_data = BoundaryConditionManager.precompute_bc_data(fes, offset)
            >>> A_bc, b_bc = BoundaryConditionManager.apply_bcs_to_assembled_system(
            ...     A, b, bc_data
            ... )
        """
        if not bc_data['has_bcs']:
            # No boundary conditions to apply
            return A, b
        
        # Apply boundary conditions using existing solver infrastructure
        # This uses the row elimination method implemented in solver.py
        from fealax.solver import zero_rows_jax
        
        # Zero out rows corresponding to constrained DOFs and set diagonal to 1
        A_bc = zero_rows_jax(A, bc_data['indices'])
        
        # Set RHS values for constrained DOFs
        b_bc = b.at[bc_data['indices']].set(bc_data['values'])
        
        return A_bc, b_bc


def process_dirichlet_bcs(dirichlet_bcs: Optional[List[DirichletBC]]) -> Optional[List[List[Callable]]]:
    """Process Dirichlet boundary conditions into finite element format.
    
    Converts a list of DirichletBC objects into the internal format used by
    finite element spaces for boundary condition processing.
    
    Args:
        dirichlet_bcs (Optional[List[DirichletBC]]): List of Dirichlet boundary conditions.
        
    Returns:
        Optional[List[List[Callable]]]: Processed boundary condition information in format:
            [subdomain_functions, vec_components, eval_functions] or None if no BCs.
            
    Example:
        >>> bcs = [DirichletBC(lambda x: x[0] < 1e-6, 0, lambda x: 0.0)]
        >>> bc_info = process_dirichlet_bcs(bcs)
        >>> # bc_info = [[subdomain_func], [0], [eval_func]]
    """
    if dirichlet_bcs is None:
        return None
    
    return [
        [bc.subdomain for bc in dirichlet_bcs],
        [bc.vec for bc in dirichlet_bcs],
        [bc.eval for bc in dirichlet_bcs],
    ]


def validate_dirichlet_bcs(dirichlet_bcs: List[DirichletBC], vec_size: int) -> None:
    """Validate Dirichlet boundary conditions for consistency.
    
    Checks that all boundary condition specifications are valid and consistent
    with the problem setup.
    
    Args:
        dirichlet_bcs (List[DirichletBC]): List of boundary conditions to validate.
        vec_size (int): Number of vector components in the solution.
        
    Raises:
        ValueError: If any boundary condition is invalid or inconsistent.
        
    Example:
        >>> bcs = [DirichletBC(lambda x: x[0] < 1e-6, 0, lambda x: 0.0)]
        >>> validate_dirichlet_bcs(bcs, vec_size=3)  # OK
        >>> bcs = [DirichletBC(lambda x: x[0] < 1e-6, 3, lambda x: 0.0)]
        >>> validate_dirichlet_bcs(bcs, vec_size=3)  # Raises ValueError
    """
    for i, bc in enumerate(dirichlet_bcs):
        if not callable(bc.subdomain):
            raise ValueError(f"Boundary condition {i}: subdomain must be callable")
        
        if not isinstance(bc.vec, int):
            raise ValueError(f"Boundary condition {i}: vec must be an integer")
        
        if bc.vec < 0 or bc.vec >= vec_size:
            raise ValueError(
                f"Boundary condition {i}: vec={bc.vec} must be in range [0, {vec_size-1}]"
            )
        
        if not callable(bc.eval):
            raise ValueError(f"Boundary condition {i}: eval must be callable")