"""Solver utility functions for automatic differentiation and parameter sensitivity.

This module provides utility functions for advanced solver capabilities including
automatic differentiation, parameter sensitivity analysis, and optimization support.
It implements the adjoint method for efficient gradient computation through
nonlinear finite element solutions.

The module includes:
    - Adjoint method implementation for parameter sensitivity
    - Automatic differentiation wrappers for solver functions
    - VJP (vector-Jacobian product) computations
    - Integration with JAX's automatic differentiation system

Key Functions:
    implicit_vjp: Compute gradients using the adjoint method
    ad_wrapper: Create differentiable solver functions
    extract_solver_data: Extract JIT-compatible boundary condition data

Example:
    Creating a differentiable solver for optimization:
    
    >>> from fealax.solver.solver_utils import ad_wrapper
    >>> from fealax.problem import Problem
    >>> 
    >>> # Setup problem
    >>> problem = Problem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    >>> 
    >>> # Create differentiable solver
    >>> differentiable_solver = ad_wrapper(problem)
    >>> 
    >>> # Use in optimization
    >>> def objective(params):
    ...     solution = differentiable_solver(params)
    ...     return compute_objective(solution)
    >>> 
    >>> grad_fn = jax.grad(objective)
    >>> gradients = grad_fn(initial_params)

Note:
    This module requires JAX for automatic differentiation and should be used
    in conjunction with the main solver functions for parameter optimization
    and sensitivity analysis.
"""

import jax
import jax.numpy as np
import jax.flatten_util
from jax.experimental.sparse import BCOO
from typing import Dict, List, Optional, Callable, Union, Tuple, Any
import gc

from fealax import logger
from .linear_solvers import linear_solver
from .boundary_conditions import get_flatten_fn, apply_bc
# Imports are done inside functions to avoid circular imports

from jax import config
config.update("jax_enable_x64", True)


# extract_solver_data has been moved to newton_solvers.py to avoid circular imports


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

    # Import locally to avoid circular imports
    from .newton_solvers import get_A
    
    problem.set_params(params)
    problem.newton_update(sol_list)

    A_result = get_A(problem)
    if problem.prolongation_matrix is not None:
        A, A_reduced = A_result
    else:
        A = A_result
        A_reduced = A
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


def _ad_wrapper(problem: Any, solver_options: Dict[str, Any] = {}, adjoint_solver_options: Dict[str, Any] = {}, use_jit: bool = False) -> Callable[[Any], List[np.ndarray]]:
    """Internal automatic differentiation wrapper for the solver.
    
    This is the internal implementation used by the NewtonSolver wrapper class.
    Direct use of this function is discouraged - use the NewtonSolver class instead.
    
    Args:
        problem (Problem): Finite element problem template.
        solver_options (dict, optional): Options for forward solver. Defaults to {}.
        adjoint_solver_options (dict, optional): Options for adjoint solver. Defaults to {}.
        use_jit (bool, optional): Whether to JIT-compile the wrapper. Defaults to False.
        
    Returns:
        Callable: JAX-differentiable function that takes parameters and returns solution.
        
    Note:
        This is an internal function. Use NewtonSolver wrapper class for public API:
        
        >>> from fealax.solver import NewtonSolver
        >>> solver = NewtonSolver(problem, solver_options, differentiable=True)
        >>> solution = solver.solve(params)
    """
    @jax.custom_vjp
    def fwd_pred(params):
        # Import locally to avoid circular imports
        from .newton_solvers import newton_solve, _solver
        
        problem.set_params(params)
        # Choose solver based on JIT option
        if 'use_jit' in solver_options or use_jit:
            # Use JIT-compatible solver
            jit_options = solver_options.copy()
            jit_options['use_jit'] = True
            sol_list = newton_solve(problem, jit_options)
        else:
            # Use regular solver
            sol_list = _solver(problem, solver_options)
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
    
    # Optionally JIT-compile the entire wrapper
    if use_jit:
        logger.debug("JIT-compiling AD wrapper")
        return jax.jit(fwd_pred)
    else:
        return fwd_pred


def ad_wrapper(problem: Any, solver_options: Dict[str, Any] = {}, adjoint_solver_options: Dict[str, Any] = {}, use_jit: bool = False) -> Callable[[Any], List[np.ndarray]]:
    """Create automatic differentiation wrapper for the solver.
    
    This function provides backward compatibility with the original ad_wrapper API.
    For new code, consider using the NewtonSolver wrapper class which provides
    a cleaner, more object-oriented interface.
    
    Args:
        problem (Problem): Finite element problem template.
        solver_options (dict, optional): Options for forward solver. Defaults to {}.
        adjoint_solver_options (dict, optional): Options for adjoint solver. Defaults to {}.
        use_jit (bool, optional): Whether to JIT-compile the wrapper. Defaults to False.
        
    Returns:
        Callable: JAX-differentiable function that takes parameters and returns solution.
        
    Example:
        Traditional usage:
        
        >>> differentiable_solver = ad_wrapper(problem)
        >>> solution = differentiable_solver(params)
        
        Recommended usage with NewtonSolver:
        
        >>> from fealax.solver import NewtonSolver
        >>> solver = NewtonSolver(problem, solver_options, differentiable=True)
        >>> solution = solver.solve(params)
        
    Note:
        This function is provided for backward compatibility. The NewtonSolver
        wrapper class offers a more intuitive API for new projects.
    """
    return _ad_wrapper(problem, solver_options, adjoint_solver_options, use_jit)