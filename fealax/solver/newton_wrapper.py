"""Newton solver wrapper providing clean API for JAX automation chains.

This module provides a wrapper class that encapsulates the Newton solver functionality
with a clean `.solve()` interface that's always differentiable through JAX transformations.

Example:
    Basic usage:
    >>> solver = NewtonSolver(problem, solver_options)
    >>> solution = solver.solve(params)
    
    With automatic differentiation:
    >>> objective = lambda params: compute_objective(solver.solve(params))
    >>> gradients = jax.grad(objective)(params)
"""

import jax
import jax.numpy as np
from typing import Dict, List, Any
from fealax import logger
from .solver_utils import _ad_wrapper


class NewtonSolver:
    """Newton solver wrapper with clean API for JAX automation chains.
    
    This class provides a wrapper around the Newton solver that is always
    differentiable through JAX transformations. The solver automatically
    supports gradient computation without any special configuration and
    uses JIT compilation for optimal performance.
    
    Args:
        problem: Finite element problem instance
        solver_options: Configuration options for the Newton solver
        adjoint_solver_options: Configuration for adjoint solver (defaults to solver_options)
        
    Example:
        Basic solver usage:
        >>> solver = NewtonSolver(problem, {'tol': 1e-6})
        >>> solution = solver.solve(params)
        
        With automatic differentiation:
        >>> objective = lambda p: compute_objective(solver.solve(p))
        >>> gradients = jax.grad(objective)(params)
    """
    
    def __init__(
        self,
        problem: Any,
        solver_options: Dict[str, Any] = {},
        adjoint_solver_options: Dict[str, Any] = None
    ):
        self.problem = problem
        self.solver_options = solver_options.copy()
        # If no adjoint options provided, use the same as forward solver
        self.adjoint_solver_options = adjoint_solver_options if adjoint_solver_options is not None else solver_options.copy()
        
        # Create the differentiable solver function with JIT always enabled
        logger.debug("Creating differentiable Newton solver with JIT compilation")
        self._solve_fn = _ad_wrapper(
            problem, 
            solver_options, 
            self.adjoint_solver_options
        )
    
    
    def solve(self, params: Any) -> List[np.ndarray]:
        """Solve the finite element problem with given parameters.
        
        Args:
            params: Problem parameters (material properties, boundary conditions, etc.)
            
        Returns:
            Solution list where each array corresponds to a variable.
            
        Example:
            >>> solution = solver.solve(material_params)
            >>> displacement = solution[0]  # First variable (e.g., displacement)
            
            With gradients:
            >>> grad_fn = jax.grad(lambda p: jnp.sum(solver.solve(p)[0]))
            >>> gradients = grad_fn(material_params)
        """
        return self._solve_fn(params)
    
    def update_solver_options(self, new_options: Dict[str, Any]):
        """Update solver options and recreate solver function.
        
        Args:
            new_options: New solver configuration options
            
        Note:
            This recreates the internal solver function with new options.
        """
        self.solver_options.update(new_options)
        
        # Recreate solver function with new options
        self._solve_fn = _ad_wrapper(
            self.problem, 
            self.solver_options, 
            self.adjoint_solver_options
        )
    
    def update_adjoint_options(self, new_options: Dict[str, Any]):
        """Update adjoint solver options.
        
        Args:
            new_options: New adjoint solver configuration options
        """
        self.adjoint_solver_options.update(new_options)
        
        # Recreate solver function with new adjoint options
        self._solve_fn = _ad_wrapper(
            self.problem, 
            self.solver_options, 
            self.adjoint_solver_options
        )
    
    @property
    def is_jit_compiled(self) -> bool:
        """Check if solver uses JIT compilation."""
        return True  # Always JIT compiled
    
    def solve_batch(self, params_batch: List[Dict[str, Any]]) -> List[List[np.ndarray]]:
        """Solve for a batch of parameter sets using sequential solving.
        
        Args:
            params_batch: List of parameter dictionaries
            
        Returns:
            List of solution lists, one for each parameter set
            
        Example:
            >>> param_batch = [{'E': 200e9, 'nu': 0.3}, {'E': 300e9, 'nu': 0.25}]
            >>> solutions = solver.solve_batch(param_batch)
        """
        if not params_batch:
            return []
        
        logger.info(f"Using sequential solver for batch of {len(params_batch)} parameter sets")
        # Use sequential solving with existing solver
        solutions = []
        for params in params_batch:
            solution = self.solve(params)
            solutions.append(solution)
        return solutions
    
    def solve_parameter_sweep(
        self, 
        param_name: str, 
        param_values: List[float], 
        base_params: Dict[str, Any] = {}
    ) -> Dict[str, Any]:
        """Convenient parameter sweep for a single parameter.
        
        Args:
            param_name: Name of parameter to sweep
            param_values: List of values for the parameter
            base_params: Base parameter dictionary (other fixed parameters)
            
        Returns:
            Dictionary with parameter values, solutions, and analysis
        """
        # Create parameter batch
        param_batch = []
        for value in param_values:
            params = base_params.copy()
            params[param_name] = value
            param_batch.append(params)
        
        # Solve batch
        solutions = self.solve_batch(param_batch)
        
        # Analyze results  
        max_displacements = []
        for solution in solutions:
            displacement = solution[0]  # Assume first variable is displacement
            max_disp = float(np.max(np.abs(displacement)))
            max_displacements.append(max_disp)
        
        return {
            'parameter_name': param_name,
            'parameter_values': param_values,
            'solutions': solutions,
            'max_displacements': max_displacements
        }


# Convenience function for creating solver instances
def create_newton_solver(
    problem: Any,
    solver_options: Dict[str, Any] = {},
    adjoint_solver_options: Dict[str, Any] = None,
    differentiable: bool = None  # Deprecated parameter for backward compatibility
) -> NewtonSolver:
    """Create a Newton solver instance with specified configuration.
    
    The solver is always differentiable and supports automatic differentiation
    through JAX transformations. JIT compilation is always enabled for optimal
    performance.
    
    Args:
        problem: Finite element problem instance
        solver_options: Configuration options for the Newton solver
        adjoint_solver_options: Configuration for adjoint solver (defaults to solver_options)
        differentiable: Deprecated - kept for backward compatibility
        
    Returns:
        Configured NewtonSolver instance
        
    Example:
        Create solver:
        >>> solver = create_newton_solver(problem, {'tol': 1e-6})
        >>> solution = solver.solve(params)
    """
    if differentiable is not None:
        logger.warning("The 'differentiable' parameter is deprecated. NewtonSolver is always differentiable.")
    
    return NewtonSolver(
        problem=problem,
        solver_options=solver_options,
        adjoint_solver_options=adjoint_solver_options
    )