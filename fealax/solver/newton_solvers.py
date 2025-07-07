"""Newton-Raphson solver with JAX-compatible control flow.

This module provides the core Newton-Raphson solver for nonlinear finite element problems
with JAX-compatible control flow structures for optimal performance and differentiability.
"""

import jax
import jax.numpy as np
import jax.flatten_util
from jax.experimental.sparse import BCOO
import time
from typing import Dict, List, Optional, Callable, Union, Tuple, Any
import gc

from fealax import logger
from .linear_algebra import jax_get_diagonal, zero_rows_jax, jax_matrix_multiply, array_to_jax_vec
from .boundary_conditions import apply_bc_vec, apply_bc, assign_bc, copy_bc, get_flatten_fn, jit_apply_bc_vec
from .jit_solvers import (
    jit_newton_step,
    jit_residual_norm
)

from jax import config
config.update("jax_enable_x64", True)
CHUNK_SIZE = 100000000


def linear_incremental_solver(
    problem: Any, 
    res_vec: np.ndarray, 
    A: BCOO, 
    dofs: np.ndarray, 
    solver_options: Dict[str, Any]
) -> np.ndarray:
    """Solve linear system for Newton-Raphson increment.
    
    Computes the Newton increment by solving the linearized system at each
    Newton iteration. Handles constraint enforcement and optional line search.
    
    Args:
        problem: Finite element problem instance.
        res_vec: Current residual vector.
        A: Jacobian matrix at current solution state.
        dofs: Current solution degrees of freedom.
        solver_options: Solver configuration options.
        
    Returns:
        Updated solution after applying Newton increment.
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

    line_search_flag = solver_options.get("line_search_flag", False)
    
    def with_line_search():
        return line_search(problem, dofs, inc)
    
    def without_line_search():
        return dofs + inc
    
    dofs = jax.lax.cond(
        line_search_flag,
        with_line_search,
        without_line_search
    )

    return dofs


def line_search(problem: Any, dofs: np.ndarray, inc: np.ndarray) -> np.ndarray:
    """Perform line search to optimize Newton step size using JAX while_loop.
    
    Implements a backtracking line search with JAX-compatible control flow
    to find an optimal step size along the Newton direction.
    
    Args:
        problem: Finite element problem instance.
        dofs: Current solution degrees of freedom.
        inc: Newton increment direction.
        
    Returns:
        Updated solution with optimized step size.
    """
    res_fn = problem.compute_residual
    res_fn = get_flatten_fn(res_fn, problem)
    res_fn = apply_bc(res_fn, problem)

    def res_norm_fn(alpha):
        res_vec = res_fn(dofs + alpha * inc)
        return np.linalg.norm(res_vec)

    # Use JAX while_loop for line search
    def line_search_body(state):
        i, alpha, res_norm, should_continue = state
        alpha_new = alpha * 0.5
        res_norm_half = res_norm_fn(alpha_new)
        
        # Update state based on whether we found improvement
        def found_improvement(args):
            _, alpha_new, res_norm, res_norm_half = args
            return alpha_new * 2.0, res_norm, False  # Stop and use doubled alpha
        
        def continue_search(args):
            _, alpha_new, res_norm, res_norm_half = args
            return alpha_new, res_norm_half, True  # Continue with halved alpha
        
        alpha_out, res_norm_out, continue_out = jax.lax.cond(
            res_norm_half > res_norm,
            found_improvement,
            continue_search,
            (i, alpha_new, res_norm, res_norm_half)
        )
        
        return i + 1, alpha_out, res_norm_out, continue_out
    
    def line_search_cond(state):
        i, _, _, should_continue = state
        return (i < 3) & should_continue
    
    # Initial state: (iteration, alpha, res_norm, should_continue)
    init_state = (0, 1.0, res_norm_fn(1.0), True)
    _, final_alpha, _, _ = jax.lax.while_loop(
        line_search_cond, 
        line_search_body, 
        init_state
    )

    return dofs + final_alpha * inc


def get_A(problem: Any) -> Union[BCOO, Tuple[BCOO, BCOO]]:
    """Construct JAX BCOO matrix with boundary condition enforcement.
    
    Converts the assembled sparse matrix to JAX BCOO format and applies
    boundary condition enforcement via row elimination.
    
    Args:
        problem: Finite element problem with assembled sparse matrix.
        
    Returns:
        System matrix. If prolongation matrix is present, returns both 
        original and reduced matrices.
    """
    A = problem.csr_array
    logger.info(
        f"Global sparse matrix takes about {A.data.shape[0]*8*3/2**30} G memory to store."
    )

    # Collect all row indices to zero out
    row_index_arrays = []
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = np.array(
                fe.node_inds_list[i] * fe.vec
                + fe.vec_inds_list[i]
                + problem.offset[ind],
                dtype=np.int32,
            )
            row_index_arrays.append(row_inds)
    
    # Apply boundary conditions to matrix (re-enabled)
    if row_index_arrays:
        rows_to_zero = np.concatenate(row_index_arrays)
        A = zero_rows_jax(A, rows_to_zero)

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
    
    Pre-computes all boundary condition data that cannot be JIT compiled,
    returning JIT-compatible arrays.
    
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
            
            # Use JAX arrays directly without .tolist()
            bc_indices_list.append(global_dof_inds)
            bc_values_list.append(fe.vals_list[i])
    
    # Concatenate arrays instead of using .tolist()
    if bc_indices_list:
        bc_indices = np.concatenate(bc_indices_list).astype(np.int32)
        bc_values = np.concatenate(bc_values_list)
    else:
        bc_indices = np.array([], dtype=np.int32)
        bc_values = np.array([])
    
    return bc_indices, bc_values


def newton_solve(problem: Any, solver_options: Dict[str, Any] = {}) -> List[np.ndarray]:
    """Newton-Raphson solver with JAX-compatible control flow.
    
    This solver separates JIT-compilable operations from assembly/setup operations
    to maximize performance while maintaining compatibility with JAX transformations.
    
    Args:
        problem: Finite element problem instance.
        solver_options: Solver configuration options.
            - tol: Absolute convergence tolerance (default: 1e-6)
            - rel_tol: Relative convergence tolerance (default: 1e-8)
            - max_iter: Maximum Newton iterations (default: 50)
            - method: Linear solver method (default: 'bicgstab')
            - precond: Enable preconditioning (default: True)
            - line_search_flag: Enable line search (default: False)
            
    Returns:
        Solution list with converged degrees of freedom.
        
    Example:
        >>> solution = newton_solve(problem, {'tol': 1e-6, 'precond': True})
    """
    logger.debug("Using Newton solver with JAX-compatible control flow")
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
    res_val_initial = jit_residual_norm(res_vec)
    
    # Newton iteration with JAX-compatible control flow
    max_iter = solver_options.get("max_iter", 50)
    
    # Initial state for while loop
    initial_state = (dofs, res_vec, A, res_val_initial, 0)  # (dofs, res_vec, A, res_val, iteration)
    
    def newton_condition(state):
        """Condition function for Newton while loop."""
        dofs, res_vec, A, res_val, iteration = state
        
        # Check convergence using JAX control flow
        def check_relative_convergence():
            rel_res_val = res_val / res_val_initial
            return (rel_res_val <= rel_tol) | (res_val <= tol)
        
        def check_absolute_convergence():
            return res_val <= tol
            
        converged = jax.lax.cond(
            res_val_initial > 0,
            check_relative_convergence,
            check_absolute_convergence
        )
        
        # Continue if not converged and within max iterations
        return ~converged & (iteration < max_iter)
    
    def newton_body(state):
        """Body function for Newton while loop."""
        dofs, res_vec, A, res_val, iteration = state
        
        # JIT-compiled Newton step
        new_dofs = jit_newton_step(dofs, A, res_vec, solver_options)
        
        # Re-compute residual and matrix (not JIT-compiled due to assembly operations)
        new_res_vec, new_A = compute_residual_and_matrix(new_dofs)
        new_res_val = jit_residual_norm(new_res_vec)
        
        return (new_dofs, new_res_vec, new_A, new_res_val, iteration + 1)
    
    # Check for early convergence before starting iterations
    def run_iterations():
        final_state = jax.lax.while_loop(newton_condition, newton_body, initial_state)
        return final_state[0]  # Return final dofs
        
    def early_convergence():
        return dofs  # Already converged
    
    # Use JAX cond to decide whether to run iterations
    converged_early = res_val_initial < tol
    final_dofs = jax.lax.cond(converged_early, early_convergence, run_iterations)
    
    # Apply post-processing (using Python if since these are setup-time decisions)
    if problem.prolongation_matrix is not None:
        final_dofs = problem.prolongation_matrix @ final_dofs
        
    if problem.macro_term is not None:
        final_dofs = final_dofs + problem.macro_term
    
    sol_list = problem.unflatten_fn_sol_list(final_dofs)
    
    end = time.time()
    logger.debug(f"Newton solver completed in {end - start:.4f} seconds")
    
    return sol_list