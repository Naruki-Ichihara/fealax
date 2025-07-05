"""Solver utility functions for automatic differentiation.

This module provides utility functions for creating JAX-differentiable solver
functions. The implementation uses direct JAX automatic differentiation for
robust and accurate gradient computation through nonlinear finite element
solutions.

Key Functions:
    _differentiable_solver: Create JAX-differentiable solver function (internal)
    ad_wrapper: Create differentiable solver (backward compatibility)

Example:
    Creating a differentiable solver:
    
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

Recommended Usage:
    Use the NewtonSolver wrapper class for new code:
    
    >>> from fealax.solver import NewtonSolver
    >>> solver = NewtonSolver(problem, solver_options)
    >>> solution = solver.solve(params)

Note:
    This module uses direct JAX automatic differentiation, which is more
    robust and accurate than complex custom VJP implementations.
"""

import jax
import jax.numpy as np
from typing import Dict, List, Callable, Any

from fealax import logger
# Imports are done inside functions to avoid circular imports

from jax import config
config.update("jax_enable_x64", True)


# Core solver utilities for automatic differentiation


# implicit_vjp function removed - direct JAX AD is more accurate and simpler


def _differentiable_solver(problem: Any, solver_options: Dict[str, Any] = {}) -> Callable[[Any], List[np.ndarray]]:
    """Create a differentiable solver function using direct JAX automatic differentiation.
    
    This function creates a JAX-differentiable solver that uses direct automatic
    differentiation through the Newton-Raphson solver. This approach is simpler
    and more accurate than complex custom VJP implementations.
    
    Args:
        problem (Problem): Finite element problem template.
        solver_options (dict, optional): Options for Newton solver. Defaults to {}.
        
    Returns:
        Callable: JAX-differentiable function that takes parameters and returns solution.
        
    Note:
        This is an internal function. Use NewtonSolver wrapper class for public API:
        
        >>> from fealax.solver import NewtonSolver
        >>> solver = NewtonSolver(problem, solver_options)
        >>> solution = solver.solve(params)
    """
    
    def solver_fn(params):
        # Import locally to avoid circular imports
        from .newton_solvers import _jit_solver
        
        # Set parameters and solve
        problem.set_params(params)
        sol_list = _jit_solver(problem, solver_options)
        return sol_list
    
    logger.debug("Differentiable solver created - JAX handles AD automatically")
    return solver_fn


def ad_wrapper(problem: Any, solver_options: Dict[str, Any] = {}) -> Callable[[Any], List[np.ndarray]]:
    """Create automatic differentiation wrapper for the solver.
    
    This function provides backward compatibility with the original ad_wrapper API.
    For new code, consider using the NewtonSolver wrapper class which provides
    a cleaner, more object-oriented interface.
    
    Args:
        problem (Problem): Finite element problem template.
        solver_options (dict, optional): Options for Newton solver. Defaults to {}.
        
    Returns:
        Callable: JAX-differentiable function that takes parameters and returns solution.
        
    Example:
        Traditional usage:
        
        >>> differentiable_solver = ad_wrapper(problem)
        >>> solution = differentiable_solver(params)
        
        Recommended usage with NewtonSolver:
        
        >>> from fealax.solver import NewtonSolver
        >>> solver = NewtonSolver(problem, solver_options)
        >>> solution = solver.solve(params)
        
    Note:
        This function is provided for backward compatibility. The NewtonSolver
        wrapper class offers a more intuitive API for new projects.
    """
    return _differentiable_solver(problem, solver_options)