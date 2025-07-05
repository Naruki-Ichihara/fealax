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
from .solver_utils import _differentiable_solver


class NewtonSolver:
    """Newton solver wrapper with clean API for JAX automation chains.
    
    This class provides a wrapper around the Newton solver that is always
    differentiable through JAX transformations. The solver automatically
    supports gradient computation through direct JAX automatic differentiation
    and uses JIT compilation for optimal performance.
    
    Args:
        problem: Finite element problem instance
        solver_options: Configuration options for the Newton solver
        
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
        solver_options: Dict[str, Any] = {}
    ):
        self.problem = problem
        self.solver_options = solver_options.copy()
        
        # Create the differentiable solver function
        logger.debug("Creating differentiable Newton solver with JIT compilation")
        self._solve_fn = _differentiable_solver(problem, solver_options)
        
        # Initialize parameter names and vmap solver as None (created on first solve)
        self._param_names = None
        self._vmap_solve_fn = None
    
    def _extract_param_names(self, params: Any) -> List[str]:
        """Extract parameter names from a parameter dictionary or list."""
        if isinstance(params, dict):
            return list(params.keys())
        elif isinstance(params, list) and len(params) > 0 and isinstance(params[0], dict):
            return list(params[0].keys())
        else:
            raise ValueError("Cannot extract parameter names from parameters. Expected dict or list of dicts.")
    
    def _create_vmap_solver(self, param_names: List[str]):
        """Create vmap-compatible solver functions for any parameter set."""
        
        def solve_with_dynamic_params(*param_values):
            """Solver that takes individual parameters dynamically (works with vmap)."""
            params = dict(zip(param_names, param_values))
            # Use the regular solver path - it should now be vmap-compatible
            return self._solve_fn(params)
        
        # Create vmapped version with appropriate in_axes (all parameters are batched)
        in_axes = tuple(0 for _ in param_names)
        self._vmap_solve_fn = jax.vmap(solve_with_dynamic_params, in_axes=in_axes)
        self._param_names = param_names
        
        logger.debug(f"Created vmap solver for parameters: {param_names}")
    
    def solve(self, params: Any) -> List[np.ndarray]:
        """Solve the finite element problem with given parameters.
        
        Automatically applies vmap for multiple parameter sets, excluding assembly.
        
        Args:
            params: Problem parameters. Can be:
                - Single parameter dict: {'param1': value1, 'param2': value2}
                - List of parameter dicts: [{'param1': val1, 'param2': val2}, ...]
                - Dict with batched arrays: {'param1': [val1, val2], 'param2': [val3, val4]}
            
        Returns:
            Solution list where each array corresponds to a variable.
            For batched inputs, returns batched solutions with leading batch dimension.
            
        Example:
            Single solve:
            >>> solution = solver.solve({'E': 200e9, 'nu': 0.3})
            >>> displacement = solution[0]  # Shape: (n_dofs,)
            
            Batch solve (automatic vmap):
            >>> batch_params = [{'E': 200e9, 'nu': 0.3}, {'E': 300e9, 'nu': 0.25}]
            >>> solutions = solver.solve(batch_params)
            >>> displacements = solutions[0]  # Shape: (batch_size, n_dofs)
        """
        # Initialize vmap solver on first call
        if self._param_names is None:
            self._param_names = self._extract_param_names(params)
            self._create_vmap_solver(self._param_names)
            logger.debug(f"Initialized Newton solver for parameters: {self._param_names}")
        
        # Detect if we have multiple parameter sets
        is_batch = self._is_batch_params(params)
        
        if is_batch:
            logger.info(f"Detected batch parameters, applying batch solving")
            return self._solve_batch_vmap(params)
        else:
            return self._solve_fn(params)
    
    def _is_batch_params(self, params: Any) -> bool:
        """Detect if parameters represent multiple parameter sets."""
        # Case 1: List of parameter dictionaries
        if isinstance(params, list) and len(params) > 0:
            if isinstance(params[0], dict):
                return True
        
        # Case 2: Dictionary with batched arrays (all values are arrays with same leading dimension > 1)
        if isinstance(params, dict):
            array_lengths = []
            for value in params.values():
                if hasattr(value, '__len__') and not isinstance(value, str):
                    try:
                        value_array = np.asarray(value)
                        if value_array.ndim >= 1 and value_array.shape[0] > 1:
                            array_lengths.append(value_array.shape[0])
                    except:
                        pass
            
            # If all arrays have same length > 1, it's a batch
            if array_lengths and len(set(array_lengths)) == 1 and array_lengths[0] > 1:
                return True
        
        return False
    
    def _solve_batch_vmap(self, params: Any) -> List[np.ndarray]:
        """Solve batch of parameters using vmap for parallel execution."""
        # Convert to standardized format: dict with batched arrays
        if isinstance(params, list):
            # Convert list of dicts to arrays
            batch_size = len(params)
            param_arrays = {}
            for param_name in self._param_names:
                param_arrays[param_name] = np.array([p[param_name] for p in params])
        else:
            # Already in dict format with arrays
            param_arrays = {}
            for param_name in self._param_names:
                param_arrays[param_name] = np.array(params[param_name])
            batch_size = len(param_arrays[self._param_names[0]])
        
        logger.debug(f"Running vmap solver for batch size: {batch_size}")
        
        # Try vmap first with the new vmap-compatible solver
        try:
            logger.info("Attempting vmap solving for batch parameters")
            # Pass parameters in the order expected by vmap function
            param_values = [param_arrays[name] for name in self._param_names]
            solutions = self._vmap_solve_fn(*param_values)
            logger.info("✓ Vmap solving successful!")
            return solutions
        except Exception as e:
            logger.warning(f"Vmap failed: {e}, falling back to sequential solving")
            # Fall back to sequential solving
            param_dicts = []
            for i in range(batch_size):
                param_dict = {name: param_arrays[name][i] for name in self._param_names}
                param_dicts.append(param_dict)
            return self._solve_batch_sequential(param_dicts)
    
    def _solve_batch_sequential(self, param_list: List[Dict[str, Any]]) -> List[List[np.ndarray]]:
        """Fallback sequential solving for batch parameters."""
        solutions = []
        for params in param_list:
            solution = self._solve_fn(params)
            solutions.append(solution)
        
        # Stack solutions to match vmap output format
        if solutions:
            stacked_solutions = []
            for var_idx in range(len(solutions[0])):
                var_solutions = [sol[var_idx] for sol in solutions]
                stacked_solutions.append(np.stack(var_solutions, axis=0))
            return stacked_solutions
        
        return []
    
    def update_solver_options(self, new_options: Dict[str, Any]):
        """Update solver options and recreate solver function.
        
        Args:
            new_options: New solver configuration options
            
        Note:
            This recreates the internal solver function with new options.
        """
        self.solver_options.update(new_options)
        
        # Recreate solver function with new options
        self._solve_fn = _differentiable_solver(self.problem, self.solver_options)
    
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
    differentiable: bool = None  # Deprecated parameter for backward compatibility
) -> NewtonSolver:
    """Create a Newton solver instance with specified configuration.
    
    The solver is always differentiable and supports automatic differentiation
    through JAX transformations. JIT compilation is always enabled for optimal
    performance.
    
    Args:
        problem: Finite element problem instance
        solver_options: Configuration options for the Newton solver
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
    
    return NewtonSolver(problem=problem, solver_options=solver_options)