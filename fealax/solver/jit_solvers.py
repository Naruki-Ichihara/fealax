"""JIT-compiled solver functions for high-performance finite element computations.

This module contains JIT-compiled implementations of core solver functions optimized
for GPU acceleration and maximum performance. All functions in this module are
decorated with @jax.jit to ensure optimal compilation and execution.

The module includes:
    - JIT-compiled Newton step functions with different linear solvers
    - JIT-compiled residual norm computations
    - JIT-compiled solver utilities
    - Preconditioning options for iterative solvers

Key Functions:
    jit_newton_step_bicgstab_precond: Newton step with BiCGSTAB and preconditioning
    jit_newton_step_bicgstab_no_precond: Newton step with BiCGSTAB without preconditioning
    jit_newton_step_sparse: Newton step with sparse direct solver
    jit_newton_step_full: Newton step with configurable solver parameters
    jit_residual_norm: JIT-compiled residual norm computation
    jit_newton_step: Main dispatcher for JIT Newton steps

Example:
    Using JIT-compiled Newton step:
    
    >>> from fealax.solver.jit_solvers import jit_newton_step
    >>> import jax.numpy as np
    >>> 
    >>> # After computing residual and Jacobian
    >>> solver_options = {'jax_solver': {'precond': True}}
    >>> new_dofs = jit_newton_step(dofs, A, res_vec, solver_options)
    
    Direct use of specific JIT functions:
    
    >>> # For maximum performance when solver type is known
    >>> new_dofs = jit_newton_step_bicgstab_precond(dofs, A, res_vec)
    >>> res_norm = jit_residual_norm(res_vec)

Note:
    All functions in this module are JIT-compiled and optimized for performance.
    They should be used within the context of the main solver functions for
    optimal memory management and boundary condition handling.
"""

import jax
import jax.numpy as np
from jax.experimental.sparse import BCOO
from typing import Dict, Any

from .linear_algebra import jax_get_diagonal

from jax import config
config.update("jax_enable_x64", True)


@jax.jit
def jit_newton_step_bicgstab_precond(dofs: np.ndarray, A: BCOO, res_vec: np.ndarray) -> np.ndarray:
    """JIT-compiled Newton step using BiCGSTAB solver with preconditioning.
    
    Performs a single Newton step using the BiCGSTAB iterative solver with
    Jacobi preconditioning. This is the most commonly used JIT solver function
    for general finite element problems.
    
    Args:
        dofs (np.ndarray): Current degrees of freedom vector.
        A (BCOO): System Jacobian matrix in JAX BCOO sparse format.
        res_vec (np.ndarray): Residual vector at current solution state.
        
    Returns:
        np.ndarray: Updated degrees of freedom after Newton step.
        
    Note:
        Uses Jacobi preconditioning based on the diagonal of the system matrix.
        Safe division is employed to avoid division by zero in the preconditioner.
    """
    neg_res = -res_vec
    jacobi = jax_get_diagonal(A)
    # Safe division for preconditioning with proper scaling
    jacobi_max = np.max(np.abs(jacobi))
    safe_jacobi = np.where(np.abs(jacobi) > jacobi_max * 1e-8, jacobi, np.sign(jacobi) * jacobi_max * 1e-8)
    pc = lambda x: x / safe_jacobi
    
    delta_dofs, _ = jax.scipy.sparse.linalg.bicgstab(
        A, neg_res, x0=None, M=pc, tol=1e-10, atol=1e-10, maxiter=10000
    )
    return dofs + delta_dofs


@jax.jit
def jit_newton_step_bicgstab_no_precond(dofs: np.ndarray, A: BCOO, res_vec: np.ndarray) -> np.ndarray:
    """JIT-compiled Newton step using BiCGSTAB solver without preconditioning.
    
    Performs a single Newton step using the BiCGSTAB iterative solver without
    any preconditioning. This can be useful for well-conditioned problems or
    when preconditioning overhead is not justified.
    
    Args:
        dofs (np.ndarray): Current degrees of freedom vector.
        A (BCOO): System Jacobian matrix in JAX BCOO sparse format.
        res_vec (np.ndarray): Residual vector at current solution state.
        
    Returns:
        np.ndarray: Updated degrees of freedom after Newton step.
        
    Note:
        No preconditioning is used, which may result in slower convergence
        for ill-conditioned problems but can be faster for well-conditioned systems.
    """
    neg_res = -res_vec
    
    delta_dofs, _ = jax.scipy.sparse.linalg.bicgstab(
        A, neg_res, x0=None, M=None, tol=1e-10, atol=1e-10, maxiter=10000
    )
    return dofs + delta_dofs


@jax.jit
def jit_newton_step_sparse(dofs: np.ndarray, A: BCOO, res_vec: np.ndarray) -> np.ndarray:
    """JIT-compiled Newton step using sparse direct solver.
    
    Performs a single Newton step using JAX's sparse direct solver. This is
    typically more robust than iterative methods but may be slower for large
    systems or when GPU memory is limited.
    
    Args:
        dofs (np.ndarray): Current degrees of freedom vector.
        A (BCOO): System Jacobian matrix in JAX BCOO sparse format.
        res_vec (np.ndarray): Residual vector at current solution state.
        
    Returns:
        np.ndarray: Updated degrees of freedom after Newton step.
        
    Note:
        Uses JAX's sparse direct solver which provides robust solutions but
        may have higher memory requirements than iterative methods.
    """
    neg_res = -res_vec
    from .linear_solvers import jax_sparse_direct_solve
    delta_dofs = jax_sparse_direct_solve(A, neg_res)
    return dofs + delta_dofs


@jax.jit
def jit_newton_step_full(dofs: np.ndarray, A: BCOO, res_vec: np.ndarray, use_precond: bool, 
                        tol: float, atol: float, maxiter: int) -> np.ndarray:
    """JIT-compiled full Newton step with configurable solver parameters.
    
    Performs a Newton step with fully configurable linear solver parameters.
    This function allows fine-tuning of the linear solver behavior while
    maintaining JIT compilation benefits.
    
    Args:
        dofs (np.ndarray): Current degrees of freedom vector.
        A (BCOO): System Jacobian matrix in JAX BCOO sparse format.
        res_vec (np.ndarray): Residual vector at current solution state.
        use_precond (bool): Whether to use Jacobi preconditioning.
        tol (float): Relative tolerance for linear solver.
        atol (float): Absolute tolerance for linear solver.
        maxiter (int): Maximum iterations for linear solver.
        
    Returns:
        np.ndarray: Updated degrees of freedom after Newton step.
        
    Note:
        This function provides the most flexibility for linear solver tuning
        while maintaining JIT compilation. The static arguments are handled
        by the JIT compiler for optimal performance.
    """
    neg_res = -res_vec
    
    if use_precond:
        jacobi = jax_get_diagonal(A)
        # Safe division for preconditioning
        safe_jacobi = np.where(np.abs(jacobi) > 1e-12, jacobi, 1.0)
        pc = lambda x: x / safe_jacobi
    else:
        pc = None
    
    delta_dofs, _ = jax.scipy.sparse.linalg.bicgstab(
        A, neg_res, x0=None, M=pc, tol=tol, atol=atol, maxiter=maxiter
    )
    return dofs + delta_dofs


@jax.jit 
def jit_residual_norm(res_vec: np.ndarray) -> float:
    """JIT-compiled residual norm computation.
    
    Computes the L2 norm of the residual vector using JIT compilation for
    optimal performance. This is used frequently in convergence checking.
    
    Args:
        res_vec (np.ndarray): Residual vector.
        
    Returns:
        float: L2 norm of the residual vector.
        
    Note:
        This function is JIT-compiled for maximum performance in convergence
        checking loops where residual norms are computed repeatedly.
    """
    return np.linalg.norm(res_vec)


def jit_newton_step(dofs: np.ndarray, A: BCOO, res_vec: np.ndarray, solver_options: Dict[str, Any]) -> np.ndarray:
    """Non-JIT Newton step dispatcher that calls JIT functions.
    
    This function handles solver option parsing outside JIT, then calls
    the appropriate JIT-compiled solver function. This design allows for
    flexible solver selection while maintaining JIT compilation benefits.
    
    Args:
        dofs (np.ndarray): Current degrees of freedom vector.
        A (BCOO): System Jacobian matrix in JAX BCOO sparse format.
        res_vec (np.ndarray): Residual vector at current solution state.
        solver_options (Dict[str, Any]): Solver configuration options.
        
    Returns:
        np.ndarray: Updated degrees of freedom after Newton step.
        
    Note:
        This function serves as a dispatcher that selects the appropriate
        JIT-compiled solver based on the solver options. The actual linear
        solve is performed by JIT-compiled functions for optimal performance.
        
    Example:
        >>> # Using JAX solver with preconditioning
        >>> options = {'jax_solver': {'precond': True}}
        >>> new_dofs = jit_newton_step(dofs, A, res_vec, options)
        >>> 
        >>> # Using sparse direct solver
        >>> options = {'jax_sparse_solver': {}}
        >>> new_dofs = jit_newton_step(dofs, A, res_vec, options)
    """
    # Extract solver options outside JIT
    if "jax_solver" in solver_options:
        precond = solver_options["jax_solver"].get("precond", True)
        if precond:
            return jit_newton_step_bicgstab_precond(dofs, A, res_vec)
        else:
            return jit_newton_step_bicgstab_no_precond(dofs, A, res_vec)
    elif "jax_sparse_solver" in solver_options:
        return jit_newton_step_sparse(dofs, A, res_vec)
    else:
        # Default to BiCGSTAB with preconditioning
        return jit_newton_step_bicgstab_precond(dofs, A, res_vec)


# Apply JIT compilation with static arguments for optimal performance
jit_newton_step_full = jax.jit(jit_newton_step_full, static_argnames=['use_precond', 'maxiter'])