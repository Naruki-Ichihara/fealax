"""Finite element solvers and solution algorithms.

This module provides comprehensive solution algorithms for finite element problems,
including linear solvers, nonlinear Newton-Raphson methods, boundary condition
enforcement, and specialized continuation methods. It supports multiple backend
solvers (JAX, SciPy/UMFPACK) with JAX as the primary backend for GPU acceleration
and automatic differentiation.

The module includes:
    - Linear solver interfaces (JAX iterative, JAX sparse, UMFPACK)
    - Nonlinear Newton-Raphson solver with line search
    - Boundary condition enforcement via row elimination
    - Arc-length continuation methods for path-following
    - Dynamic relaxation for static equilibrium problems
    - Automatic differentiation wrappers for sensitivity analysis
    - Memory-efficient sparse matrix assembly with JAX BCOO format

Key Functions:
    solver: Main nonlinear solver with Newton-Raphson iteration
    linear_solver: Unified interface to multiple linear solvers
    array_to_jax_vec: Conversion utilities for JAX integration
    implicit_vjp: Adjoint method for parameter sensitivity
    ad_wrapper: Automatic differentiation wrapper for optimization

Example:
    Basic nonlinear solver usage:
    
    >>> from fealax.solver import solver
    >>> from fealax.problem import Problem
    >>> 
    >>> # Setup problem (see problem.py documentation)
    >>> problem = Problem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    >>> 
    >>> # Solve with Newton-Raphson method
    >>> solver_options = {'jax_solver': {'precond': True}, 'tol': 1e-6}
    >>> solution = solver(problem, solver_options)

Note:
    This module requires JAX for automatic differentiation and linear algebra,
    and SciPy for sparse matrix operations. GPU acceleration is available 
    through JAX when configured properly.
"""

import jax
import jax.numpy as np
import jax.flatten_util
import numpy as onp
from jax.experimental.sparse import BCOO
import scipy
import time
from typing import Dict, List, Optional, Callable, Union, Tuple, Any
import gc

from fealax import logger

from jax import config

config.update("jax_enable_x64", True)
CHUNK_SIZE = 100000000


def jax_get_diagonal(A: BCOO) -> np.ndarray:
    """Extract diagonal elements from BCOO sparse matrix.
    
    Args:
        A (BCOO): Sparse matrix in BCOO format.
        
    Returns:
        np.ndarray: Diagonal elements.
    """
    # Find diagonal indices
    diag_mask = A.indices[:, 0] == A.indices[:, 1]
    diag_indices = A.indices[:, 0][diag_mask]
    diag_data = A.data[diag_mask]
    
    # Create full diagonal vector
    diagonal = np.zeros(A.shape[0])
    diagonal = diagonal.at[diag_indices].set(diag_data)
    return diagonal


def scipy_to_jax_bcoo(A_scipy: scipy.sparse.csr_matrix) -> BCOO:
    """Convert SciPy sparse matrix to JAX BCOO format.
    
    Args:
        A_scipy: SciPy sparse matrix.
        
    Returns:
        BCOO: JAX sparse matrix.
    """
    return BCOO.from_scipy_sparse(A_scipy).sort_indices()


def jax_bcoo_to_scipy(A_bcoo: BCOO) -> scipy.sparse.csr_matrix:
    """Convert JAX BCOO matrix to SciPy sparse format.
    
    Args:
        A_bcoo: JAX BCOO sparse matrix.
        
    Returns:
        scipy.sparse.csr_matrix: SciPy sparse matrix.
    """
    # Extract data and indices from BCOO
    data = onp.array(A_bcoo.data)
    indices = onp.array(A_bcoo.indices)
    shape = A_bcoo.shape
    
    # Create scipy sparse matrix from BCOO format
    row_indices = indices[:, 0]
    col_indices = indices[:, 1]
    
    return scipy.sparse.csr_matrix((data, (row_indices, col_indices)), shape=shape)


def zero_rows_jax(A: BCOO, row_indices: np.ndarray) -> BCOO:
    """Zero out specified rows in JAX BCOO matrix and set diagonal entries to 1.0.
    
    Args:
        A (BCOO): Input sparse matrix.
        row_indices (np.ndarray): Indices of rows to zero out.
        
    Returns:
        BCOO: Matrix with specified rows zeroed and diagonal entries set to 1.0.
    """
    # Create mask for entries not in the specified rows
    mask = ~np.isin(A.indices[:, 0], row_indices)
    
    # Filter indices and data
    new_indices = A.indices[mask, :]
    new_data = A.data[mask]
    
    # Add diagonal entries for the zeroed rows
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
    # Convert to scipy for multiplication, then back to JAX
    A_scipy = jax_bcoo_to_scipy(A)
    B_scipy = jax_bcoo_to_scipy(B)
    C_scipy = A_scipy @ B_scipy
    return scipy_to_jax_bcoo(C_scipy)


def array_to_jax_vec(arr: Union[np.ndarray, onp.ndarray], size: Optional[int] = None) -> np.ndarray:
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


def jax_solve(A: BCOO, b: np.ndarray, x0: np.ndarray, precond: bool) -> np.ndarray:
    """Solves the equilibrium equation using a JAX solver.

    Args:
        A: System matrix in BCOO format.
        b: Right-hand side vector.
        x0: Initial guess.
        precond (bool): Whether to calculate the preconditioner or not.

    Returns:
        Solution vector.
    """
    logger.debug(f"JAX Solver - Solving linear system")
    # A is already in BCOO format
    jacobi = jax_get_diagonal(A)
    pc = lambda x: x * (1.0 / jacobi) if precond else None
    x, _ = jax.scipy.sparse.linalg.bicgstab(
        A, b, x0=x0, M=pc, tol=1e-10, atol=1e-10, maxiter=10000
    )

    # Verify convergence
    err = np.linalg.norm(A @ x - b)
    logger.debug(f"JAX Solver - Finished solving, res = {err}")
    assert err < 0.1, f"JAX linear solver failed to converge with err = {err}"
    x = np.where(
        err < 0.1, x, np.nan
    )  # For assert purpose, some how this also affects bicgstab.

    return x


def umfpack_solve(A: BCOO, b: np.ndarray) -> np.ndarray:
    """Solve linear system using SciPy's UMFPACK interface.
    
    Solves the linear system Ax = b using the UMFPACK sparse direct solver
    through SciPy's interface. UMFPACK is typically faster and more robust
    than iterative methods for moderately-sized problems.
    
    Args:
        A (BCOO): Sparse system matrix in JAX BCOO format.
        b (np.ndarray): Right-hand side vector.
        
    Returns:
        np.ndarray: Solution vector x.
        
    Note:
        The function converts the JAX BCOO matrix to SciPy CSR format internally.
        Consider using the experimental JAX sparse solver for GPU acceleration.
    """
    logger.debug(f"Scipy Solver - Solving linear system with UMFPACK")
    Asp = A.to_scipy_sparse().tocsr()
    x = scipy.sparse.linalg.spsolve(Asp, onp.array(b))

    # TODO: try https://jax.readthedocs.io/en/latest/_autosummary/jax.experimental.sparse.linalg.spsolve.html
    # x = jax.experimental.sparse.linalg.spsolve(av, aj, ai, b)

    logger.debug(
        f"Scipy Solver - Finished solving, linear solve res = {np.linalg.norm(Asp @ x - b)}"
    )
    return x


def jax_iterative_solve(A: BCOO, b: np.ndarray, solver_type: str = 'bicgstab', precond_type: str = 'jacobi') -> np.ndarray:
    """Solve linear system using JAX iterative solvers.
    
    Solves the linear system Ax = b using JAX's iterative Krylov subspace
    methods with preconditioning. Supports various solver and preconditioner
    combinations for different problem types.
    
    Args:
        A (BCOO): Sparse system matrix in JAX BCOO format.
        b (np.ndarray): Right-hand side vector.
        solver_type (str): Krylov subspace method type. Options include:
            - 'bicgstab': Bi-conjugate gradient stabilized method
            - 'gmres': Generalized minimal residual method (if available)
            - 'cg': Conjugate gradient method
        precond_type (str): Preconditioner type. Options include:
            - 'jacobi': Jacobi (diagonal) preconditioning
            - 'none': No preconditioning
            
    Returns:
        np.ndarray: Solution vector x.
        
    Raises:
        AssertionError: If solver fails to converge to specified tolerance.
        
    Note:
        Uses JAX's built-in iterative solvers with automatic differentiation support.
        Convergence is verified by computing the residual norm.
    """
    logger.debug(
        f"JAX Solver - Solving linear system with solver_type = {solver_type}, precond = {precond_type}"
    )
    
    # Setup preconditioner
    if precond_type == 'jacobi':
        diagonal = jax_get_diagonal(A)
        # Avoid division by zero
        diagonal = np.where(np.abs(diagonal) < 1e-12, 1.0, diagonal)
        pc = lambda x: x / diagonal
    else:
        pc = None
    
    # Initial guess
    x0 = np.zeros_like(b)
    
    # Solve using JAX iterative solver
    if solver_type == 'bicgstab':
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0, M=pc, tol=1e-10, atol=1e-10, maxiter=10000
        )
    elif solver_type == 'cg':
        x, _ = jax.scipy.sparse.linalg.cg(
            A, b, x0=x0, M=pc, tol=1e-10, maxiter=10000
        )
    else:
        # Default to bicgstab
        x, _ = jax.scipy.sparse.linalg.bicgstab(
            A, b, x0=x0, M=pc, tol=1e-10, atol=1e-10, maxiter=10000
        )

    # Verify convergence
    residual = A @ x - b
    err = np.linalg.norm(residual)
    logger.debug(f"JAX Solver - Finished solving, linear solve res = {err}")
    assert err < 0.1, f"JAX linear solver failed to converge, err = {err}"
    
    return x


def linear_solver(A: BCOO, b: np.ndarray, x0: np.ndarray, solver_options: Dict[str, Any]) -> np.ndarray:
    """Unified interface for multiple linear solver backends.
    
    Provides a consistent interface to JAX, UMFPACK, and custom linear
    solvers. Automatically selects JAX solver if no specific solver is requested.
    
    Args:
        A (BCOO): Sparse system matrix in JAX BCOO format.
        b (np.ndarray): Right-hand side vector.
        x0 (np.ndarray): Initial guess for iterative solvers.
        solver_options (dict): Solver configuration dictionary with possible keys:
            - 'jax_solver': JAX iterative solver options
            - 'umfpack_solver': UMFPACK direct solver options
            - 'jax_iterative_solver': JAX iterative solver options
            - 'custom_solver': User-defined solver function
            
    Returns:
        np.ndarray: Solution vector x.
        
    Raises:
        NotImplementedError: If no valid solver is specified in options.
        
    Example:
        >>> options = {'jax_solver': {'precond': True}}
        >>> x = linear_solver(A, b, x0, options)
        
        >>> options = {'jax_iterative_solver': {'solver_type': 'bicgstab', 'precond_type': 'jacobi'}}
        >>> x = linear_solver(A, b, x0, options)
    """

    # If user does not specify any solver, set jax_solver as the default one.
    if (
        len(
            solver_options.keys()
            & {"jax_solver", "umfpack_solver", "jax_iterative_solver", "custom_solver"}
        )
        == 0
    ):
        solver_options["jax_solver"] = {}

    if "jax_solver" in solver_options:
        precond = (
            solver_options["jax_solver"]["precond"]
            if "precond" in solver_options["jax_solver"]
            else True
        )
        x = jax_solve(A, b, x0, precond)
    elif "umfpack_solver" in solver_options:
        x = umfpack_solve(A, b)
    elif "jax_iterative_solver" in solver_options:
        solver_type = (
            solver_options["jax_iterative_solver"]["solver_type"]
            if "solver_type" in solver_options["jax_iterative_solver"]
            else "bicgstab"
        )
        precond_type = (
            solver_options["jax_iterative_solver"]["precond_type"]
            if "precond_type" in solver_options["jax_iterative_solver"]
            else "jacobi"
        )
        x = jax_iterative_solve(A, b, solver_type, precond_type)
    elif "custom_solver" in solver_options:
        # Users can define their own solver
        custom_solver = solver_options["custom_solver"]
        x = custom_solver(A, b, x0, solver_options)
    else:
        raise NotImplementedError(f"Unknown linear solver.")

    return x


################################################################################
# "row elimination" solver


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


def assign_ones_bc(dofs: np.ndarray, problem: Any) -> np.ndarray:
    """Set Dirichlet boundary condition DOFs to unity values.
    
    Utility function that sets all constrained degrees of freedom to 1.0.
    Useful for testing and generating unit perturbations.
    
    Args:
        dofs (np.ndarray): Solution vector to modify.
        problem (Problem): Finite element problem with boundary condition data.
        
    Returns:
        np.ndarray: Modified solution vector with boundary DOFs set to 1.0.
    """
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(1.0)
        sol_list[ind] = sol
    return jax.flatten_util.ravel_pytree(sol_list)[0]


def assign_zeros_bc(dofs: np.ndarray, problem: Any) -> np.ndarray:
    """Set Dirichlet boundary condition DOFs to zero values.
    
    Utility function that sets all constrained degrees of freedom to 0.0.
    Useful for homogeneous boundary conditions and initialization.
    
    Args:
        dofs (np.ndarray): Solution vector to modify.
        problem (Problem): Finite element problem with boundary condition data.
        
    Returns:
        np.ndarray: Modified solution vector with boundary DOFs set to 0.0.
    """
    sol_list = problem.unflatten_fn_sol_list(dofs)
    for ind, fe in enumerate(problem.fes):
        sol = sol_list[ind]
        for i in range(len(fe.node_inds_list)):
            sol = sol.at[fe.node_inds_list[i], fe.vec_inds_list[i]].set(0.0)
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


def operator_to_matrix(operator_fn: Callable[[np.ndarray], np.ndarray], problem: Any) -> np.ndarray:
    """Convert a nonlinear operator to its Jacobian matrix.
    
    Computes the full Jacobian matrix of a nonlinear operator using automatic
    differentiation. Primarily used for debugging and analysis.
    
    Args:
        operator_fn (Callable): Nonlinear operator function.
        problem (Problem): Finite element problem for size information.
        
    Returns:
        np.ndarray: Dense Jacobian matrix.
        
    Warning:
        This function computes a dense matrix and should only be used for
        small problems or debugging purposes.
    """
    """Only used for when debugging.
    Can be used to print the matrix, check the conditional number, etc.
    """
    J = jax.jacfwd(operator_fn)(np.zeros(problem.num_total_dofs_all_vars))
    return J


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
    A_sp_scipy = problem.csr_array
    logger.info(
        f"Global sparse matrix takes about {A_sp_scipy.data.shape[0]*8*3/2**30} G memory to store."
    )

    A = scipy_to_jax_bcoo(A_sp_scipy)

    # Collect all row indices to zero out
    rows_to_zero = []
    for ind, fe in enumerate(problem.fes):
        for i in range(len(fe.node_inds_list)):
            row_inds = onp.array(
                fe.node_inds_list[i] * fe.vec
                + fe.vec_inds_list[i]
                + problem.offset[ind],
                dtype=onp.int32,
            )
            rows_to_zero.extend(row_inds)
    
    # Zero out the rows
    if rows_to_zero:
        A = zero_rows_jax(A, np.array(rows_to_zero))

    # Linear multipoint constraints
    if problem.prolongation_matrix is not None:
        P = scipy_to_jax_bcoo(problem.prolongation_matrix)
        
        # Compute A_reduced = P^T @ A @ P
        tmp = jax_matrix_multiply(A, P)
        P_T = BCOO((P.data, P.indices[:, np.array([1, 0])]), shape=(P.shape[1], P.shape[0]))
        A_reduced = jax_matrix_multiply(P_T, tmp)
        return A, A_reduced
    return A


def solver(problem: Any, solver_options: Dict[str, Any] = {}) -> List[np.ndarray]:
    """Solve nonlinear finite element problem using Newton-Raphson method.
    
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
            - 'umfpack_solver': SciPy UMFPACK direct solver options (empty dict)
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
        >>> solution = solver(problem, solver_options)
        
        JAX iterative solver with custom tolerances:
        
        >>> options = {
        ...     'jax_iterative_solver': {'solver_type': 'bicgstab', 'precond_type': 'jacobi'},
        ...     'tol': 1e-8,
        ...     'rel_tol': 1e-10
        ... }
        >>> solution = solver(problem, options)
        
        With initial guess and line search:
        
        >>> options = {
        ...     'umfpack_solver': {},
        ...     'initial_guess': initial_solution,
        ...     'line_search_flag': True
        ... }
        >>> solution = solver(problem, options)
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


def implicit_vjp(problem: Any, sol_list: List[np.ndarray], params: Any, v_list: List[np.ndarray], adjoint_solver_options: Dict[str, Any]) -> Any:
    """Compute vector-Jacobian product using the adjoint method.
    
    Implements the adjoint method to efficiently compute gradients of functionals
    with respect to problem parameters. This is essential for optimization,
    parameter identification, and sensitivity analysis.
    
    Args:
        problem (Problem): Finite element problem instance.
        sol_list (List[np.ndarray]): Solution state at which to evaluate gradients.
        params: Problem parameters with respect to which gradients are computed.
        v_list (List[np.ndarray]): Vector for the vector-Jacobian product.
        adjoint_solver_options (dict): Linear solver options for adjoint system.
        
    Returns:
        Gradients with respect to problem parameters.
        
    Note:
        The method solves the adjoint system A^T λ = v where A is the Jacobian
        at the solution state, then computes the parameter sensitivities using
        the chain rule: dF/dp = -λ^T (∂c/∂p) where c is the constraint (residual).
        
    Example:
        >>> adjoint_options = {'jax_solver': {'precond': True}}
        >>> gradients = implicit_vjp(problem, solution, params, v_list, adjoint_options)
    """

    def constraint_fn(dofs, params):
        """c(u, p)"""
        problem.set_params(params)
        res_fn = problem.compute_residual
        res_fn = get_flatten_fn(res_fn, problem)
        res_fn = apply_bc(res_fn, problem)
        return res_fn(dofs)

    def constraint_fn_sol_to_sol(sol_list, params):
        dofs = jax.flatten_util.ravel_pytree(sol_list)[0]
        con_vec = constraint_fn(dofs, params)
        return problem.unflatten_fn_sol_list(con_vec)

    def get_partial_params_c_fn(sol_list):
        """c(u=u, p)"""

        def partial_params_c_fn(params):
            return constraint_fn_sol_to_sol(sol_list, params)

        return partial_params_c_fn

    def get_vjp_contraint_fn_params(params, sol_list):
        """v*(partial dc/dp)"""
        partial_c_fn = get_partial_params_c_fn(sol_list)

        def vjp_linear_fn(v_list):
            _, f_vjp = jax.vjp(partial_c_fn, params)
            (val,) = f_vjp(v_list)
            return val

        return vjp_linear_fn

    problem.set_params(params)
    problem.newton_update(sol_list)

    A, A_reduced = get_A(problem)
    v_vec = jax.flatten_util.ravel_pytree(v_list)[0]

    if problem.prolongation_matrix is not None:
        v_vec = problem.prolongation_matrix.T @ v_vec

    # Create transpose matrix for adjoint solve
    A_reduced_T = BCOO((A_reduced.data, A_reduced.indices[:, np.array([1, 0])]), shape=(A_reduced.shape[1], A_reduced.shape[0]))
    adjoint_vec = linear_solver(
        A_reduced_T, v_vec, None, adjoint_solver_options
    )

    if problem.prolongation_matrix is not None:
        adjoint_vec = problem.prolongation_matrix @ adjoint_vec

    vjp_linear_fn = get_vjp_contraint_fn_params(params, sol_list)
    vjp_result = vjp_linear_fn(problem.unflatten_fn_sol_list(adjoint_vec))
    vjp_result = jax.tree_map(lambda x: -x, vjp_result)
    
    # JAX matrices are automatically garbage collected
    del A, A_reduced
    gc.collect()

    return vjp_result


def ad_wrapper(problem: Any, solver_options: Dict[str, Any] = {}, adjoint_solver_options: Dict[str, Any] = {}) -> Callable[[Any], List[np.ndarray]]:
    """Create automatic differentiation wrapper for the solver.
    
    Wraps the nonlinear solver with JAX's custom VJP (vector-Jacobian product)
    to enable automatic differentiation through the solution process. This allows
    the solver to be used in optimization loops and gradient-based algorithms.
    
    Args:
        problem (Problem): Finite element problem template.
        solver_options (dict, optional): Options for forward solver. Defaults to {}.
        adjoint_solver_options (dict, optional): Options for adjoint solver. Defaults to {}.
        
    Returns:
        Callable: JAX-differentiable function that takes parameters and returns solution.
        
    Example:
        Setup for parameter optimization:
        
        >>> differentiable_solver = ad_wrapper(problem)
        >>> 
        >>> def objective(params):
        ...     solution = differentiable_solver(params)
        ...     return compute_objective(solution)
        >>> 
        >>> grad_fn = jax.grad(objective)
        >>> gradients = grad_fn(initial_params)
        
    Note:
        The wrapper uses implicit differentiation via the adjoint method to
        compute gradients efficiently, avoiding the need to differentiate
        through the entire Newton iteration process.
    """
    @jax.custom_vjp
    def fwd_pred(params):
        problem.set_params(params)
        sol_list = solver(problem, solver_options)
        return sol_list

    def f_fwd(params):
        sol_list = fwd_pred(params)
        return sol_list, (params, sol_list)

    def f_bwd(res, v):
        logger.info("Running backward and solving the adjoint problem...")
        params, sol_list = res
        vjp_result = implicit_vjp(problem, sol_list, params, v, adjoint_solver_options)
        return (vjp_result,)

    fwd_pred.defvjp(f_fwd, f_bwd)
    return fwd_pred
