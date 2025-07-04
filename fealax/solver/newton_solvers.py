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
    _jit_solver: Legacy JIT-compiled Newton solver
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
    
    >>> solver_options = {'use_jit': True, 'tol': 1e-8}
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
from .linear_solvers import linear_solver
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

    inc = linear_solver(A, b, x0, solver_options)

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
    """[LEGACY] JIT-compatible solver that separates setup from core iteration.
    
    This is the legacy JIT solver implementation. For new code, use newton_solve() with use_jit=True instead.
    
    This solver pre-computes all boundary conditions and problem setup outside
    of JIT, then uses JIT-compiled functions for the Newton iteration loop.
    
    Args:
        problem: Finite element problem instance
        solver_options: Solver configuration
        
    Returns:
        Solution list
    """
    logger.debug("Starting JIT-compatible solver")
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
    res_val_initial = float(jit_residual_norm(res_vec))
    res_val = res_val_initial
    
    # Handle case where initial residual is already very small
    if res_val_initial < tol:
        logger.debug(f"Initial residual {res_val_initial} already below tolerance {tol}")
        # Solution is already converged
        if problem.prolongation_matrix is not None:
            dofs = problem.prolongation_matrix @ dofs
            
        if problem.macro_term is not None:
            dofs = dofs + problem.macro_term
            
        sol_list = problem.unflatten_fn_sol_list(dofs)
        
        end = time.time()
        solve_time = end - start
        logger.info(f"JIT Solve took {solve_time} [s] (converged immediately)")
        return sol_list
    
    logger.debug(f"Before, l_2 res = {res_val}, relative l_2 res = 1.0")
    
    # Newton iteration loop
    max_iter = solver_options.get("max_iter", 50)
    iteration = 0
    
    while iteration < max_iter:
        # Check convergence (not JIT - needs to control Python loop)
        # Safe division: avoid divide by zero when res_val_initial is small
        if res_val_initial > 0:
            rel_res_val = res_val / res_val_initial
            converged = (rel_res_val <= rel_tol) or (res_val <= tol)
        else:
            converged = res_val <= tol
            
        if converged:
            break
            
        # JIT-compiled Newton step
        dofs = jit_newton_step(dofs, A, res_vec, solver_options)
        
        # Re-compute residual and matrix (not JIT)
        res_vec, A = compute_residual_and_matrix(dofs)
        res_val = float(jit_residual_norm(res_vec))
        
        rel_res_str = f"{res_val / res_val_initial}" if res_val_initial > 0 else "N/A"
        logger.debug(f"l_2 res = {res_val}, relative l_2 res = {rel_res_str}")
        iteration += 1
    
    # Finalize solution
    if problem.prolongation_matrix is not None:
        dofs = problem.prolongation_matrix @ dofs
        
    if problem.macro_term is not None:
        dofs = dofs + problem.macro_term
        
    sol_list = problem.unflatten_fn_sol_list(dofs)
    
    end = time.time()
    solve_time = end - start
    logger.info(f"JIT Solve took {solve_time} [s]")
    logger.info(f"max of dofs = {np.max(dofs)}")
    logger.info(f"min of dofs = {np.min(dofs)}")
    
    return sol_list


def _solver(problem: Any, solver_options: Dict[str, Any] = {}) -> List[np.ndarray]:
    """[LEGACY] Solve nonlinear finite element problem using Newton-Raphson method.
    
    This is the conventional/legacy solver implementation. For new code, use newton_solve() instead.
    
    Main nonlinear solver that implements Newton-Raphson iteration with multiple
    linear solver backends. Enforces Dirichlet boundary conditions via row 
    elimination method and supports advanced features like line search, 
    prolongation matrices, and macro terms.
    
    Args:
        problem (Problem): Finite element problem instance containing mesh,
            finite elements, and boundary conditions.
        solver_options (dict, optional): Solver configuration dictionary. Defaults to {}.
            Supported keys:
            - 'jax_solver': JAX iterative solver options
                - 'precond' (bool): Enable Jacobi preconditioning. Defaults to True.
            - 'jax_sparse_solver': JAX sparse direct solver options (empty dict)
            - 'jax_iterative_solver': JAX iterative solver options
                - 'solver_type' (str): Krylov method ('bicgstab', 'cg'). Defaults to 'bicgstab'.
                - 'precond_type' (str): Preconditioner ('jacobi', 'none'). Defaults to 'jacobi'.
            - 'line_search_flag' (bool): Enable line search optimization. Defaults to False.
            - 'initial_guess' (List[np.ndarray]): Initial solution guess. Same shape as output.
            - 'tol' (float): Absolute tolerance for residual L2 norm. Defaults to 1e-6.
            - 'rel_tol' (float): Relative tolerance for residual L2 norm. Defaults to 1e-8.
            
    Returns:
        List[np.ndarray]: Solution list where each array corresponds to a variable.
            For multi-variable problems, returns [u1, u2, ...] where each ui has
            shape (num_nodes, vec_components).
            
    Raises:
        AssertionError: If residual contains NaN values or solver fails to converge.
        
    Note:
        Boundary Condition Enforcement:
        Uses row elimination method where the residual becomes:
        res(u) = D*r(u) + (I - D)*u - u_b
        
        Where:
        - D: Diagonal matrix with zeros at constrained DOFs
        - r(u): Physical residual from weak form
        - u_b: Prescribed boundary values
        
        The Jacobian matrix is modified accordingly:
        A = d(res)/d(u) = D*dr/du + (I - D)
        
        Solver Selection:
        If no solver is specified, JAX solver is used by default.
        Only one solver type should be specified per call.
        
    Example:
        Basic nonlinear solve with JAX solver:
        
        >>> solver_options = {'jax_solver': {'precond': True}}
        >>> solution = _solver(problem, solver_options)
        
        JAX iterative solver with custom tolerances:
        
        >>> options = {
        ...     'jax_iterative_solver': {'solver_type': 'bicgstab', 'precond_type': 'jacobi'},
        ...     'tol': 1e-8,
        ...     'rel_tol': 1e-10
        ... }
        >>> solution = _solver(problem, options)
        
        With initial guess and line search:
        
        >>> options = {
        ...     'jax_sparse_solver': {},
        ...     'initial_guess': initial_solution,
        ...     'line_search_flag': True
        ... }
        >>> solution = _solver(problem, options)
    """
    logger.debug(f"Calling the row elimination solver for imposing Dirichlet B.C.")
    logger.debug("Start timing")
    start = time.time()

    if "initial_guess" in solver_options:
        # We dont't want inititual guess to play a role in the differentiation chain.
        initial_guess = jax.lax.stop_gradient(solver_options["initial_guess"])
        dofs = jax.flatten_util.ravel_pytree(initial_guess)[0]

    else:
        if problem.prolongation_matrix is not None:
            dofs = np.zeros(problem.prolongation_matrix.shape[1])  # reduced dofs
        else:
            dofs = np.zeros(problem.num_total_dofs_all_vars)
            # Apply boundary conditions to initial guess for better convergence
            dofs = assign_bc(dofs, problem)

    rel_tol = solver_options["rel_tol"] if "rel_tol" in solver_options else 1e-8
    tol = solver_options["tol"] if "tol" in solver_options else 1e-6

    def newton_update_helper(dofs):
        if problem.prolongation_matrix is not None:
            logger.debug(
                f"Using prolongation_matrix, shape = {problem.prolongation_matrix.shape}"
            )
            dofs = problem.prolongation_matrix @ dofs

        sol_list = problem.unflatten_fn_sol_list(dofs)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        res_vec = apply_bc_vec(res_vec, dofs, problem)

        problem.compute_csr(CHUNK_SIZE)  # Ensure CSR matrix is computed

        if problem.prolongation_matrix is not None:
            res_vec = problem.prolongation_matrix.T @ res_vec

        A_result = get_A(problem)
        if problem.prolongation_matrix is not None:
            A, A_reduced = A_result
        else:
            A = A_result
            A_reduced = A

        if problem.macro_term is not None:
            macro_term_jax = array_to_jax_vec(problem.macro_term, A.shape[0])
            K_affine_vec = A @ macro_term_jax
            del A
            gc.collect()
            affine_force = problem.prolongation_matrix.T @ K_affine_vec
            res_vec += affine_force

        return res_vec, A_reduced

    res_vec, A = newton_update_helper(dofs)
    res_val = np.linalg.norm(res_vec)
    res_val_initial = res_val
    rel_res_val = res_val / res_val_initial
    logger.debug(f"Before, l_2 res = {res_val}, relative l_2 res = {rel_res_val}")

    while (rel_res_val > rel_tol) and (res_val > tol):
        dofs = linear_incremental_solver(problem, res_vec, A, dofs, solver_options)
        res_vec, A = newton_update_helper(dofs)
        res_val = np.linalg.norm(res_vec)
        rel_res_val = res_val / res_val_initial

        logger.debug(f"l_2 res = {res_val}, relative l_2 res = {rel_res_val}")

    assert np.all(np.isfinite(res_val)), f"res_val contains NaN, stop the program!"
    assert np.all(np.isfinite(dofs)), f"dofs contains NaN, stop the program!"

    if problem.prolongation_matrix is not None:
        dofs = problem.prolongation_matrix @ dofs

    if problem.macro_term is not None:
        dofs = dofs + problem.macro_term

    # If sol_list = [[[u1x, u1y],
    #                 [u2x, u2y],
    #                 [u3x, u3y],
    #                 [u4x, u4y]],
    #                [[p1],
    #                 [p2]]],
    # the flattend DOF vector will be [u1x, u1y, u2x, u2y, u3x, u3y, u4x, u4y, p1, p2]
    sol_list = problem.unflatten_fn_sol_list(dofs)

    end = time.time()
    solve_time = end - start
    logger.info(f"Solve took {solve_time} [s]")
    logger.info(f"max of dofs = {np.max(dofs)}")
    logger.info(f"min of dofs = {np.min(dofs)}")

    return sol_list


# jit_newton_step_full is now imported from jit_solvers module


def newton_solve_jit_core(problem: Any, dofs_init: np.ndarray, bc_indices: np.ndarray, 
                         bc_values: np.ndarray, solver_options: Dict[str, Any]) -> np.ndarray:
    """Core JIT-compatible Newton iteration using hybrid approach.
    
    This function separates the JIT-compilable parts from problem setup.
    Uses JIT-compiled linear solves within a Python loop for the Newton iteration.
    """
    tol = solver_options.get('tol', 1e-6)
    rel_tol = solver_options.get('rel_tol', 1e-8)
    max_iter = solver_options.get('max_iter', 20)
    use_precond = solver_options.get('precond', True)
    linear_tol = solver_options.get('linear_tol', 1e-10)
    linear_atol = solver_options.get('linear_atol', 1e-10) 
    linear_maxiter = solver_options.get('linear_maxiter', 10000)
    
    def compute_residual_and_matrix(dofs):
        """Non-JIT helper to compute residual and matrix - called outside JIT loop."""
        # Handle prolongation matrix if present
        dofs_full = dofs
        if problem.prolongation_matrix is not None:
            dofs_full = problem.prolongation_matrix @ dofs
        
        # Compute residual from weak form
        sol_list = problem.unflatten_fn_sol_list(dofs_full)
        res_list = problem.newton_update(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        
        # Apply boundary conditions to residual
        res_vec = jit_apply_bc_vec(res_vec, dofs_full, bc_indices, bc_values)
        
        # Handle prolongation matrix for residual
        if problem.prolongation_matrix is not None:
            res_vec = problem.prolongation_matrix.T @ res_vec
        
        # Compute system matrix
        problem.compute_csr(CHUNK_SIZE)
        A_result = get_A(problem)
        if problem.prolongation_matrix is not None:
            A, A_reduced = A_result
        else:
            A = A_result
            A_reduced = A
        
        # Handle macro terms if present
        if problem.macro_term is not None:
            macro_term_jax = array_to_jax_vec(problem.macro_term, A.shape[0])
            K_affine_vec = A @ macro_term_jax
            del A
            gc.collect()
            affine_force = problem.prolongation_matrix.T @ K_affine_vec
            res_vec += affine_force
        
        return res_vec, A_reduced
    
    # Initialize
    dofs = dofs_init
    
    # Standard Python loop for residual/matrix computation (can't be JIT compiled)
    for iteration in range(max_iter):
        # Compute residual and matrix (non-JIT)
        res_vec, A = compute_residual_and_matrix(dofs)
        
        # Check convergence
        res_norm = float(jit_residual_norm(res_vec))
        if iteration == 0:
            res_norm_initial = res_norm
            if res_norm < tol:
                break
        else:
            rel_res_norm = res_norm / res_norm_initial if res_norm_initial > 0 else 0
            if res_norm < tol or rel_res_norm < rel_tol:
                break
        
        # Non-JIT Newton step for hybrid mode (avoids boolean tracing issues)
        from .linear_solvers import jax_iterative_solve
        neg_res = -res_vec
        precond_type = 'jacobi' if use_precond else 'none'
        delta_dofs = jax_iterative_solve(A, neg_res, 'bicgstab', precond_type)
        dofs = dofs + delta_dofs
    
    return dofs


def newton_solve_pure_jit(res_fn_jit: Callable, A_assembly_fn_jit: Callable, 
                         dofs_init: np.ndarray, tol: float, rel_tol: float, 
                         max_iter: int, use_precond: bool, linear_tol: float, 
                         linear_atol: float, linear_maxiter: int) -> np.ndarray:
    """Fully JIT-compiled Newton solver for pre-compiled residual and matrix functions.
    
    This is a pure JAX implementation that can be fully JIT-compiled when
    the residual and matrix assembly functions are also JIT-compatible.
    
    Args:
        res_fn_jit: JIT-compatible residual function
        A_assembly_fn_jit: JIT-compatible matrix assembly function  
        dofs_init: Initial DOF vector
        tol: Absolute tolerance
        rel_tol: Relative tolerance
        max_iter: Maximum iterations
        use_precond: Use preconditioning
        linear_tol: Linear solver tolerance
        linear_atol: Linear solver absolute tolerance
        linear_maxiter: Linear solver max iterations
        
    Returns:
        Converged DOF vector
    """
    def newton_cond(state):
        iteration, dofs, res_norm, res_norm_initial, converged = state
        # Continue if not converged and under max iterations
        not_converged = ~converged
        under_max_iter = iteration < max_iter
        return not_converged & under_max_iter
    
    def newton_body(state):
        iteration, dofs, res_norm_prev, res_norm_initial, converged = state
        
        # Compute residual and matrix
        res_vec = res_fn_jit(dofs)
        A = A_assembly_fn_jit(dofs)
        
        # Compute residual norm
        res_norm = jit_residual_norm(res_vec)
        
        # Update initial residual norm on first iteration
        res_norm_initial = np.where(iteration == 0, res_norm, res_norm_initial)
        
        # Check convergence
        rel_res_norm = np.where(res_norm_initial > 0, res_norm / res_norm_initial, 0.0)
        abs_converged = res_norm < tol
        rel_converged = rel_res_norm < rel_tol
        is_converged = abs_converged | rel_converged
        
        # Newton step (only if not converged)
        dofs_new = np.where(
            is_converged,
            dofs,
            jit_newton_step_full(dofs, A, res_vec, use_precond, linear_tol, linear_atol, linear_maxiter)
        )
        
        return (iteration + 1, dofs_new, res_norm, res_norm_initial, is_converged)
    
    # Initial state: (iteration, dofs, res_norm, res_norm_initial, converged)
    initial_state = (0, dofs_init, 0.0, 0.0, False)
    
    # Run Newton iteration using while_loop
    final_state = jax.lax.while_loop(newton_cond, newton_body, initial_state)
    
    # Extract final DOFs
    _, dofs_final, _, _, _ = final_state
    return dofs_final


def newton_solve(problem: Any, solver_options: Dict[str, Any] = {}) -> List[np.ndarray]:
    """Newton solver that is JIT-ready with multiple compilation strategies.
    
    This solver provides three levels of JIT compilation:
    1. Hybrid mode (default): JIT-compiled linear solves within Python Newton loop
    2. Full JIT mode (use_jit=True): Uses existing jit_solver for maximum performance
    3. Pure JIT mode: For fully JIT-compatible residual/matrix functions
    
    Args:
        problem (Problem): Finite element problem instance.
        solver_options (dict, optional): Solver configuration. Defaults to {}.
            Available options:
            - tol (float): Absolute convergence tolerance (default: 1e-6)
            - rel_tol (float): Relative convergence tolerance (default: 1e-8)
            - max_iter (int): Maximum Newton iterations (default: 20)
            - use_jit (bool): Enable full JIT compilation via jit_solver (default: False)
            - method (str): Linear solver method (default: 'bicgstab')
            - precond (bool): Enable preconditioning (default: True)
            - linear_tol (float): Linear solver tolerance (default: 1e-10)
            - linear_atol (float): Linear solver absolute tolerance (default: 1e-10)
            - linear_maxiter (int): Linear solver max iterations (default: 10000)
            - pure_jit_mode (bool): Use pure JIT mode with pre-compiled functions (default: False)
            - res_fn_jit (Callable): JIT-compatible residual function (required for pure_jit_mode)
            - A_fn_jit (Callable): JIT-compatible matrix assembly function (required for pure_jit_mode)
            
    Returns:
        List[np.ndarray]: Solution list in the same format as original solver().
        
    Example:
        Basic usage with hybrid JIT (recommended):
        >>> solution = newton_solve(problem, {
        ...     'tol': 1e-6, 
        ...     'precond': True
        ... })
        
        Full JIT compilation using jit_solver:
        >>> solution = newton_solve(problem, {
        ...     'use_jit': True,
        ...     'tol': 1e-8
        ... })
        
        Pure JIT mode (for advanced users with JIT-compatible functions):
        >>> solution = newton_solve(problem, {
        ...     'pure_jit_mode': True,
        ...     'res_fn_jit': my_jit_residual_fn,
        ...     'A_fn_jit': my_jit_assembly_fn
        ... })
    """
    # Check compilation mode
    use_jit = solver_options.get('use_jit', False)
    pure_jit_mode = solver_options.get('pure_jit_mode', False)
    
    if use_jit:
        logger.debug("Using fully JIT-compiled Newton solver via _jit_solver")
        # Use the existing jit_solver for full JIT compilation
        return _jit_solver(problem, solver_options)
    
    if pure_jit_mode:
        logger.debug("Using pure JIT Newton solver with pre-compiled functions")
        # Requires user to provide JIT-compatible functions
        res_fn_jit = solver_options.get('res_fn_jit')
        A_fn_jit = solver_options.get('A_fn_jit')
        
        if res_fn_jit is None or A_fn_jit is None:
            raise ValueError("pure_jit_mode requires 'res_fn_jit' and 'A_fn_jit' in solver_options")
        
        # Extract parameters for pure JIT solver
        if problem.prolongation_matrix is not None:
            dofs_init = np.zeros(problem.prolongation_matrix.shape[1])
        else:
            dofs_init = np.zeros(problem.num_total_dofs_all_vars)
        
        tol = solver_options.get('tol', 1e-6)
        rel_tol = solver_options.get('rel_tol', 1e-8)
        max_iter = solver_options.get('max_iter', 20)
        use_precond = solver_options.get('precond', True)
        linear_tol = solver_options.get('linear_tol', 1e-10)
        linear_atol = solver_options.get('linear_atol', 1e-10)
        linear_maxiter = solver_options.get('linear_maxiter', 10000)
        
        dofs_final = newton_solve_pure_jit(
            res_fn_jit, A_fn_jit, dofs_init, tol, rel_tol, max_iter,
            use_precond, linear_tol, linear_atol, linear_maxiter
        )
        
        # Finalize solution
        if problem.prolongation_matrix is not None:
            dofs_final = problem.prolongation_matrix @ dofs_final
        
        if problem.macro_term is not None:
            dofs_final = dofs_final + problem.macro_term
        
        sol_list = problem.unflatten_fn_sol_list(dofs_final)
        logger.debug("Pure JIT newton_solve completed successfully")
        return sol_list
    
    # Default: Hybrid JIT mode
    logger.debug("Using hybrid JIT Newton solver (JIT linear solves, Python Newton loop)")
    
    # Initialize DOFs exactly like the original solver
    if problem.prolongation_matrix is not None:
        dofs_init = np.zeros(problem.prolongation_matrix.shape[1])  # reduced dofs
    else:
        dofs_init = np.zeros(problem.num_total_dofs_all_vars)
        # Apply boundary conditions to initial guess for better convergence
        dofs_init = assign_bc(dofs_init, problem)
    
    # Pre-extract boundary condition data for JIT compatibility
    bc_indices, bc_values = extract_solver_data(problem)
    
    # Use the JIT-ready core solver
    dofs_final = newton_solve_jit_core(problem, dofs_init, bc_indices, bc_values, solver_options)
    
    # Finalize solution exactly like the original solver
    if problem.prolongation_matrix is not None:
        dofs_final = problem.prolongation_matrix @ dofs_final
    
    if problem.macro_term is not None:
        dofs_final = dofs_final + problem.macro_term
    
    # Convert back to solution list format
    sol_list = problem.unflatten_fn_sol_list(dofs_final)
    
    logger.debug("Hybrid JIT newton_solve completed successfully")
    return sol_list


# Alias for backward compatibility
jit_solver = _jit_solver


# Apply JIT compilation with static arguments to functions defined in this module
newton_solve_pure_jit = jax.jit(newton_solve_pure_jit, static_argnames=[
    'max_iter', 'use_precond', 'linear_maxiter'
])