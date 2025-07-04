"""Boundary condition handling utilities for finite element solvers.

This module provides functions for enforcing Dirichlet boundary conditions
using the row elimination method. It includes both standard and JIT-compiled
versions for performance optimization.

Key Functions:
    apply_bc_vec: Apply boundary conditions to residual vector
    apply_bc: Create boundary condition-aware residual function
    assign_bc: Assign prescribed values to boundary DOFs
    copy_bc: Extract boundary values to a new vector
    get_flatten_fn: Create flattened version of solution list function
    jit_apply_bc_vec: JIT-compiled version of apply_bc_vec

The row elimination method modifies the residual and Jacobian matrix to
enforce Dirichlet boundary conditions while maintaining system symmetry.
"""

import jax
import jax.numpy as np
import jax.flatten_util
from typing import Callable, List, Any


def apply_bc_vec(res_vec: np.ndarray, dofs: np.ndarray, problem: Any, scale: float = 1.0) -> np.ndarray:
    """Apply Dirichlet boundary conditions to residual vector.
    
    Modifies the residual vector to enforce Dirichlet boundary conditions
    using the row elimination method. This function directly modifies the
    residual at constrained degrees of freedom.
    
    Args:
        res_vec (np.ndarray): Global residual vector to modify.
        dofs (np.ndarray): Current solution degrees of freedom.
        problem (Problem): Finite element problem containing boundary condition data.
        scale (float, optional): Scaling factor for boundary condition values. Defaults to 1.0.
        
    Returns:
        np.ndarray: Modified residual vector with boundary conditions applied.
        
    Note:
        This function implements the row elimination method where constrained
        DOFs are set to (current_value - prescribed_value) * scale.
    """
    res_list = problem.unflatten_fn_sol_list(res_vec)
    sol_list = problem.unflatten_fn_sol_list(dofs)

    for ind, fe in enumerate(problem.fes):
        res = res_list[ind]
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]], unique_indices=True
            )
            res = res.at[fe.node_inds_list[i], fe.vec_inds_list[i]].add(
                -fe.vals_list[i] * scale
            )

        res_list[ind] = res

    return jax.flatten_util.ravel_pytree(res_list)[0]


def apply_bc(res_fn: Callable[[np.ndarray], np.ndarray], problem: Any, scale: float = 1.0) -> Callable[[np.ndarray], np.ndarray]:
    """Create a boundary condition-aware residual function.
    
    Wraps a residual function to automatically apply Dirichlet boundary
    conditions using the row elimination method.
    
    Args:
        res_fn (Callable): Original residual function that takes DOFs and returns residual.
        problem (Problem): Finite element problem with boundary condition information.
        scale (float, optional): Scaling factor for boundary conditions. Defaults to 1.0.
        
    Returns:
        Callable: Modified residual function that enforces boundary conditions.
        
    Example:
        >>> res_fn_bc = apply_bc(problem.compute_residual, problem)
        >>> residual = res_fn_bc(dofs)
    """
    def res_fn_bc(dofs):
        """Apply Dirichlet boundary conditions"""
        res_vec = res_fn(dofs)
        return apply_bc_vec(res_vec, dofs, problem, scale)

    return res_fn_bc


def assign_bc(dofs: np.ndarray, problem: Any) -> np.ndarray:
    """Assign prescribed values to Dirichlet boundary condition DOFs.
    
    Sets the solution values at constrained degrees of freedom to their
    prescribed Dirichlet boundary condition values.
    
    Args:
        dofs (np.ndarray): Solution vector to modify.
        problem (Problem): Finite element problem with boundary condition data.
        
    Returns:
        np.ndarray: Modified solution vector with boundary conditions enforced.
    """
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(fe.vals_list[i])
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def copy_bc(dofs: np.ndarray, problem: Any) -> np.ndarray:
    """Extract boundary condition values to a new zero vector.
    
    Creates a new vector filled with zeros except at boundary condition
    locations, where it copies the values from the input DOFs.
    
    Args:
        dofs (np.ndarray): Source solution vector.
        problem (Problem): Finite element problem with boundary condition data.
        
    Returns:
        np.ndarray: New vector with only boundary DOF values copied.
    """
    new_dofs = np.zeros_like(dofs)
    sol_list = problem.unflatten_fn_sol_list(dofs)
    new_sol_list = problem.unflatten_fn_sol_list(new_dofs)

    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        new_sol = new_sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            new_sol = new_sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(
                sol[fe.node_inds_list[i], fe.vec_inds_list[i]]
            )
        new_sol_list[ind] = new_sol

    return jax.flatten_util.ravel_pytree(new_sol_list)[0]


def get_flatten_fn(fn_sol_list: Callable[[List[np.ndarray]], List[np.ndarray]], problem: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Create a flattened version of a solution list function.
    
    Converts a function that operates on solution lists to one that operates
    on flattened DOF vectors, handling the conversion automatically.
    
    Args:
        fn_sol_list (Callable): Function that takes solution list and returns values.
        problem (Problem): Finite element problem with flattening utilities.
        
    Returns:
        Callable: Function that takes flattened DOFs and returns flattened values.
    """

    def fn_dofs(dofs):
        sol_list = problem.unflatten_fn_sol_list(dofs)
        val_list = fn_sol_list(sol_list)
        return jax.flatten_util.ravel_pytree(val_list)[0]

    return fn_dofs


@jax.jit
def jit_apply_bc_vec(res_vec: np.ndarray, dofs: np.ndarray, bc_indices: np.ndarray, bc_values: np.ndarray) -> np.ndarray:
    """JIT-compatible version of apply_bc_vec using pre-extracted BC data.
    
    This function is designed for use in JIT-compiled code where the boundary
    condition indices and values have been pre-extracted from the problem instance.
    
    Args:
        res_vec (np.ndarray): Residual vector to modify.
        dofs (np.ndarray): Current solution degrees of freedom.
        bc_indices (np.ndarray): Pre-computed boundary condition DOF indices.
        bc_values (np.ndarray): Pre-computed boundary condition values.
        
    Returns:
        np.ndarray: Modified residual vector with boundary conditions applied.
        
    Note:
        This function requires boundary condition data to be pre-extracted using
        extract_solver_data() or similar utility function for JIT compatibility.
    """
    # Apply boundary conditions by setting residual at BC DOFs to (current_value - prescribed_value)
    res_updated = res_vec.at[bc_indices].set(dofs[bc_indices] - bc_values)
    return res_updated