"""Newton-Raphson solver implementations for nonlinear finite element problems.

This module provides comprehensive Newton-Raphson solution algorithms for finite element
problems, including JIT-compiled versions for optimal performance. It handles nonlinear
solving using Newton's method, line search algorithms, incremental loading, and various
Newton iteration strategies.

The module includes:
    - Main Newton-Raphson solver with multiple compilation strategies
    - Legacy solver implementations for backward compatibility
    - JIT-compiled Newton solvers for maximum performance
    - Line search algorithms for robust nonlinear convergence
    - Incremental linear solver for Newton updates
    - Matrix assembly and constraint handling utilities
    - Automatic differentiation wrappers for sensitivity analysis

Key Functions:
    newton_solve: Main Newton solver with JIT-ready implementation
    _solver: Legacy Newton solver implementation
    _jit_solver: Current JIT-compiled Newton solver implementation
    linear_incremental_solver: Solve linear system for Newton increment
    line_search: Perform line search to optimize Newton step size
    get_A: Construct system matrix with boundary condition enforcement
    
Example:
    Basic Newton-Raphson solver usage:
    
    >>> from fealax.solver.newton_solvers import newton_solve
    >>> from fealax.problem import Problem
    >>> 
    >>> # Setup problem (see problem.py documentation)
    >>> problem = Problem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    >>> 
    >>> # Solve with Newton-Raphson method
    >>> solver_options = {'tol': 1e-6, 'precond': True}
    >>> solution = newton_solve(problem, solver_options)
    
    Using JIT compilation for better performance:
    
    >>> solver_options = {'tol': 1e-8}
    >>> solution = newton_solve(problem, solver_options)

Note:
    This module requires JAX for automatic differentiation and linear algebra.
    GPU acceleration is available through JAX when configured properly.
"""

import jax
import jax.numpy as np
import jax.flatten_util
from jax.experimental.sparse import BCOO
import time
from typing import Dict, List, Optional, Callable, Union, Tuple, Any
import gc

from fealax import logger
# linear_solver import removed - using solve() instead
from .linear_algebra import jax_get_diagonal, zero_rows_jax, jax_matrix_multiply, array_to_jax_vec
from .boundary_conditions import apply_bc_vec, apply_bc, assign_bc, copy_bc, get_flatten_fn, jit_apply_bc_vec
from .jit_solvers import (
    jit_newton_step_bicgstab_precond,
    jit_newton_step_bicgstab_no_precond,
    jit_newton_step_sparse,
    jit_newton_step_full,
    jit_newton_step,
    jit_residual_norm
)
# extract_solver_data is defined below to avoid circular imports

from jax import config
config.update("jax_enable_x64", True)
CHUNK_SIZE = 100000000


def linear_incremental_solver(problem: Any, res_vec: np.ndarray, A: BCOO, dofs: np.ndarray, solver_options: Dict[str, Any]) -> np.ndarray:
    """Solve linear system for Newton-Raphson increment.
    
    Computes the Newton increment by solving the linearized system at each
    Newton iteration. Handles constraint enforcement and optional line search.
    
    Args:
        problem (Problem): Finite element problem instance.
        res_vec (np.ndarray): Current residual vector.
        A (BCOO): Jacobian matrix at current solution state.
        dofs (np.ndarray): Current solution degrees of freedom.
        solver_options (dict): Solver configuration options.
        
    Returns:
        np.ndarray: Updated solution after applying Newton increment.
        
    Note:
        The function automatically constructs appropriate initial guesses
        that satisfy boundary conditions and handles prolongation matrices
        for constrained problems.
    """
    logger.debug(f"Solving linear system...")
    b = -res_vec

    # x0 will always be correct at boundary locations
    x0_1 = assign_bc(np.zeros(problem.num_total_dofs_all_vars), problem)
    if problem.prolongation_matrix is not None:
        x0_2 = copy_bc(problem.prolongation_matrix @ dofs, problem)
        x0 = problem.prolongation_matrix.T @ (x0_1 - x0_2)
    else:
        x0_2 = copy_bc(dofs, problem)
        x0 = x0_1 - x0_2

    from .linear_solvers import solve
    inc = solve(A, b, solver_options)

    line_search_flag = (
        solver_options["line_search_flag"]
        if "line_search_flag" in solver_options
        else False
    )
    if line_search_flag:
        dofs = line_search(problem, dofs, inc)
    else:
        dofs = dofs + inc

    return dofs


def line_search(problem: Any, dofs: np.ndarray, inc: np.ndarray) -> np.ndarray:
    """Perform line search to optimize Newton step size.
    
    Implements a simple backtracking line search to find an optimal step size
    along the Newton direction. Particularly useful for finite deformation
    problems and nonlinear material behavior.
    
    Args:
        problem (Problem): Finite element problem instance.
        dofs (np.ndarray): Current solution degrees of freedom.
        inc (np.ndarray): Newton increment direction.
        
    Returns:
        np.ndarray: Updated solution with optimized step size.
        
    Note:
        Uses a simple halving strategy with a maximum of 3 iterations.
        The implementation is basic and could be enhanced with more
        sophisticated line search algorithms.
        
    Todo:
        Implement more robust line search methods for finite deformation plasticity.
    """
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)

    def res_norm_fn(alpha):
        res_vec = res_fn(dofs + alpha * inc)
        return np.linalg.norm(res_vec)

    # grad_res_norm_fn = jax.grad(res_norm_fn)
    # hess_res_norm_fn = jax.hessian(res_norm_fn)

    # tol = 1e-3
    # alpha = 1.
    # lr = 1.
    # grad_alpha = 1.
    # while np.abs(grad_alpha) > tol:
    #     grad_alpha = grad_res_norm_fn(alpha)
    #     hess_alpha = hess_res_norm_fn(alpha)
    #     alpha = alpha - 1./hess_alpha*grad_alpha
    #     print(f"alpha = {alpha}, grad_alpha = {grad_alpha}, hess_alpha = {hess_alpha}")

    alpha = 1.0
    res_norm = res_norm_fn(alpha)
    for i in range(3):
        alpha *= 0.5
        res_norm_half = res_norm_fn(alpha)
        logger.debug(f"i = {i}, res_norm = {res_norm}, res_norm_half = {res_norm_half}")
        if res_norm_half > res_norm:
            alpha *= 2.0
            break
        res_norm = res_norm_half

    return dofs + alpha * inc


def get_A(problem: Any) -> Union[BCOO, Tuple[BCOO, BCOO]]:
    """Construct JAX BCOO matrix with boundary condition enforcement.
    
    Converts the assembled sparse matrix to JAX BCOO format and applies
    boundary condition enforcement via row elimination. Handles
    prolongation matrices for constraint enforcement.
    
    Args:
        problem (Problem): Finite element problem with assembled sparse matrix.
        
    Returns:
        BCOO or Tuple[BCOO, BCOO]: System matrix. If prolongation
            matrix is present, returns both original and reduced matrices.
            
    Note:
        The function zeros out rows corresponding to Dirichlet boundary
        conditions and applies prolongation operations for multipoint constraints.
    """
    A = problem.csr_array
    logger.info(
        f"Global sparse matrix takes about {A.data.shape[0]*8*3/2**30} G memory to store."
    )

    # Collect all row indices to zero out
    rows_to_zero = []
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = np.array(
                fe.node_inds_list[i] * fe.vec
                + fe.vec_inds_list[i]
                + problem.offset[ind],
                dtype=np.int32,
            )
            rows_to_zero.extend(row_inds)
    
    # Zero out the rows
    if rows_to_zero:
        A = zero_rows_jax(A, np.array(rows_to_zero))

    # Linear multipoint constraints
    if problem.prolongation_matrix is not None:
        P = problem.prolongation_matrix
        
        # Compute A_reduced = P^T @ A @ P
        tmp = jax_matrix_multiply(A, P)
        P_T = BCOO((P.data, P.indices[:, np.array([1, 0])]), shape=(P.shape[1], P.shape[0]))
        A_reduced = jax_matrix_multiply(P_T, tmp)
        return A, A_reduced
    return A


def extract_solver_data(problem: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Extract JIT-compatible data from problem after setup.
    
    This function pre-computes all boundary condition and matrix data that
    cannot be JIT compiled, returning JIT-compatible arrays and functions.
    
    Args:
        problem: Finite element problem (after setup/BC computation)
        
    Returns:
        Tuple containing:
        - bc_indices: Flattened boundary condition DOF indices  
        - bc_values: Flattened boundary condition values
    """
    # Pre-compute all boundary condition data
    bc_indices_list = []
    bc_values_list = []
    
    for fe in problem.fes:
        for i in range(len(fe.node_inds_list)):
            # Convert to global DOF indices
            node_inds = fe.node_inds_list[i]
            vec_inds = fe.vec_inds_list[i]
            global_dof_inds = node_inds * fe.vec + vec_inds
            
            bc_indices_list.extend(global_dof_inds.tolist())
            bc_values_list.extend(fe.vals_list[i].tolist())
    
    bc_indices = np.array(bc_indices_list, dtype=np.int32)
    bc_values = np.array(bc_values_list)
    
    return bc_indices, bc_values


################################################################################
# JIT-compatible solver functions are now imported from jit_solvers module


def _jit_solver(problem: Any, solver_options: Dict[str, Any] = {}) -> List[np.ndarray]:
    """JIT-compatible solver that separates setup from core iteration.
    
    This is the current JIT solver implementation used by newton_solve().
    
    This solver pre-computes all boundary conditions and problem setup outside
    of JIT, then uses JIT-compiled functions for the Newton iteration loop.
    
    Args:
        problem: Finite element problem instance
        solver_options: Solver configuration
        
    Returns:
        Solution list
    """
    # Removed logging for vmap compatibility
    start = time.time()
    
    # Get tolerances
    rel_tol = solver_options.get("rel_tol", 1e-8)
    tol = solver_options.get("tol", 1e-6)
    
    # Initialize solution
    if "initial_guess" in solver_options:
        initial_guess = jax.lax.stop_gradient(solver_options["initial_guess"])
        dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]
    else:
        if problem.prolongation_matrix is not None:
            dofs = np.zeros(problem.prolongation_matrix.shape[1])
        else:
            dofs = np.zeros(problem.num_total_dofs_all_vars)
            # Apply boundary conditions to initial guess for better convergence
            dofs = assign_bc(dofs, problem)
    
    # Pre-extract boundary condition data (not JIT compatible)
    bc_indices, bc_values = extract_solver_data(problem)
    
    # Pre-compute initial residual and matrix (not JIT compatible)
    def compute_residual_and_matrix(dofs):
        """Non-JIT function to compute residual and matrix."""
        if problem.prolongation_matrix is not None:
            dofs_full = problem.prolongation_matrix @ dofs
        else:
            dofs_full = dofs
            
        sol_list = problem.unflatten_fn_sol_list(dofs_full)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        
        # Apply boundary conditions
        res_vec = apply_bc_vec(res_vec, dofs_full, problem)
        
        # Compute matrix
        problem.compute_csr(CHUNK_SIZE)
        A_result = get_A(problem)
        if problem.prolongation_matrix is not None:
            A, A_reduced = A_result
            res_vec = problem.prolongation_matrix.T @ res_vec
        else:
            A = A_result
            A_reduced = A
            
        return res_vec, A_reduced
    
    # Initial residual computation (not JIT)
    res_vec, A = compute_residual_and_matrix(dofs)
    res_val_initial = jit_residual_norm(res_vec)  # Keep as JAX array for vmap compatibility
    res_val = res_val_initial
    
    # Handle case where initial residual is already very small
    # Use jax.lax.cond for vmap compatibility instead of Python if
    def converged_immediately(args):
        dofs, problem = args
        if problem.prolongation_matrix is not None:
            dofs = problem.prolongation_matrix @ dofs
            
        if problem.macro_term is not None:
            dofs = dofs + problem.macro_term
            
        sol_list = problem.unflatten_fn_sol_list(dofs)
        return sol_list
    
    def continue_iteration(args):
        dofs, problem = args
        return dofs  # Will continue to iteration loop
    
    # Check if already converged (vmap compatible)
    is_converged = res_val_initial < tol
    
    # Handle early convergence (vmap compatible)
    if hasattr(res_val_initial, 'shape'):
        # We're in a JAX context (possibly vmap), handle differently
        # Don't use early return for vmap compatibility
        pass
    else:
        # Traditional path - check for immediate convergence
        res_val_float = float(res_val_initial)
        if res_val_float < tol:
            # Solution is already converged
            if problem.prolongation_matrix is not None:
                dofs = problem.prolongation_matrix @ dofs
                
            if problem.macro_term is not None:
                dofs = dofs + problem.macro_term
                
            sol_list = problem.unflatten_fn_sol_list(dofs)
            return sol_list
    
    # Newton iteration loop
    max_iter = solver_options.get("max_iter", 50)
    iteration = 0
    
    while iteration < max_iter:
        # Check convergence - use float conversion only outside vmap context
        try:
            # Try to convert to float for convergence check
            if hasattr(res_val, 'shape'):
                # In JAX context, use JAX operations for convergence check
                if hasattr(res_val_initial, 'shape') and np.any(res_val_initial > 0):
                    rel_res_val = res_val / res_val_initial
                    converged = (rel_res_val <= rel_tol) | (res_val <= tol)
                else:
                    converged = res_val <= tol
                
                # Convert to boolean for Python while loop
                # This might still cause issues in vmap context
                if float(converged) > 0.5:
                    break
            else:
                # Traditional convergence check
                res_val_float = float(res_val)
                res_val_initial_float = float(res_val_initial)
                
                if res_val_initial_float > 0:
                    rel_res_val = res_val_float / res_val_initial_float
                    converged = (rel_res_val <= rel_tol) or (res_val_float <= tol)
                else:
                    converged = res_val_float <= tol
                    
                if converged:
                    break
        except:
            # If conversion fails (vmap context), just do one more iteration
            pass
            
        # JIT-compiled Newton step
        dofs = jit_newton_step(dofs, A, res_vec, solver_options)
        
        # Re-compute residual and matrix (not JIT)
        res_vec, A = compute_residual_and_matrix(dofs)
        res_val = jit_residual_norm(res_vec)  # Keep as JAX array for vmap compatibility
        
        # Removed logging for vmap compatibility
        iteration += 1
    
    # Finalize solution
    if problem.prolongation_matrix is not None:
        dofs = problem.prolongation_matrix @ dofs
        
    if problem.macro_term is not None:
        dofs = dofs + problem.macro_term
        
    sol_list = problem.unflatten_fn_sol_list(dofs)
    
    # Removed timing and logging for vmap compatibility
    return sol_list


# Legacy _solver function removed - use newton_solve() instead


# jit_newton_step_full is now imported from jit_solvers module


# Legacy hybrid and pure JIT functions removed - use _jit_solver instead




def newton_solve(problem: Any, solver_options: Dict[str, Any] = {}) -> List[np.ndarray]:
    """JIT-compiled Newton solver for nonlinear finite element problems.
    
    This solver uses JIT compilation for optimal performance. All linear algebra
    operations are JIT-compiled while assembly remains outside JIT boundary.
    
    Args:
        problem (Problem): Finite element problem instance.
        solver_options (dict, optional): Solver configuration. Defaults to {}.
            Available options:
            - tol (float): Absolute convergence tolerance (default: 1e-6)
            - rel_tol (float): Relative convergence tolerance (default: 1e-8)
            - max_iter (int): Maximum Newton iterations (default: 20)
            - method (str): Linear solver method (default: 'bicgstab')
            - precond (bool): Enable preconditioning (default: True)
            - linear_tol (float): Linear solver tolerance (default: 1e-10)
            - linear_atol (float): Linear solver absolute tolerance (default: 1e-10)
            - linear_maxiter (int): Linear solver max iterations (default: 10000)
            
    Returns:
        List[np.ndarray]: Solution list with converged degrees of freedom.
        
    Example:
        Basic usage:
        >>> solution = newton_solve(problem, {
        ...     'tol': 1e-6, 
        ...     'precond': True
        ... })
    """
    logger.debug("Using JIT-compiled Newton solver")
    # Always use JIT solver for optimal performance
    return _jit_solver(problem, solver_options)


def _gradient_solver(problem: Any, converged_solution: List[np.ndarray], v_list: List[np.ndarray], solver_options: Dict[str, Any] = {}) -> Any:
    """Vmap-compatible gradient solver that reuses assembled system from forward pass.
    
    This solver computes gradients by reusing the Jacobian matrix that was already
    assembled during the forward solve, making it much faster and vmap-compatible
    since it skips the complex assembly process.
    
    Args:
        problem (Problem): Finite element problem (with cached assembled system).
        converged_solution (List[np.ndarray]): Solution from forward pass.
        v_list (List[np.ndarray]): Vector for vector-Jacobian product.
        solver_options (dict, optional): Solver options.
        
    Returns:
        Gradients with respect to problem parameters.
        
    Note:
        This function assumes the problem has already been solved and the system
        matrices are cached/available for reuse.
    """
    
    # Get the cached Jacobian matrix from the forward solve
    # The problem should have the assembled system cached
    A_result = get_A(problem)
    if problem.prolongation_matrix is not None:
        A, A_reduced = A_result
    else:
        A = A_result
        A_reduced = A
    
    # Convert v_list to vector form
    v_vec = jax.flatten_util.ravel_pytree(v_list)[0]
    
    if problem.prolongation_matrix is not None:
        v_vec = problem.prolongation_matrix.T @ v_vec
    
    # Create transpose matrix for adjoint solve
    from jax.experimental.sparse import BCOO
    A_reduced_T = BCOO((A_reduced.data, A_reduced.indices[:, np.array([1, 0])]), 
                       shape=(A_reduced.shape[1], A_reduced.shape[0]))
    
    # Solve adjoint system: A^T λ = v
    from .linear_solvers import solve
    adjoint_options = solver_options.copy()
    adjoint_vec = solve(A_reduced_T, v_vec, adjoint_options)
    
    if problem.prolongation_matrix is not None:
        adjoint_vec = problem.prolongation_matrix @ adjoint_vec
    
    # Compute parameter sensitivities using cached solution and adjoint vector
    def constraint_fn(params):
        """Constraint function that computes residual for given parameters."""
        # This is the key optimization: we reuse the solution structure
        # and only recompute parameter-dependent parts
        problem.set_params(params)
        
        # Recompute only the parameter-dependent residual terms
        # This should be much faster than full assembly
        res_fn = problem.compute_residual
        from .boundary_conditions import get_flatten_fn, apply_bc
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        
        # Use the converged solution to evaluate residual at solution point
        dofs = jax.flatten_util.ravel_pytree(converged_solution)[0]
        return res_fn(dofs)
    
    # Convert to solution list format
    def constraint_fn_sol_to_sol(params):
        con_vec = constraint_fn(params)
        return problem.unflatten_fn_sol_list(con_vec)
    
    # We need to compute VJP w.r.t. parameters
    # Since we don't have get_current_params, we'll work with the parameter structure
    # that was passed to set_params
    
    # Get current parameter values by looking at problem attributes
    # This is a simplified approach - in practice, problems store params differently
    current_params = {}
    if hasattr(problem, 'E'):
        current_params['E'] = problem.E
    if hasattr(problem, 'nu'):
        current_params['nu'] = problem.nu
    
    # Compute VJP: -λ^T (∂c/∂p)
    _, f_vjp = jax.vjp(constraint_fn_sol_to_sol, current_params)
    adjoint_sol_list = problem.unflatten_fn_sol_list(adjoint_vec)
    (vjp_result,) = f_vjp(adjoint_sol_list)
    
    # Return negative (adjoint method convention)
    vjp_result = jax.tree_map(lambda x: -x, vjp_result)
    
    return vjp_result


# Aliases for backward compatibility
jit_solver = _jit_solver


# Legacy pure JIT compilation removed