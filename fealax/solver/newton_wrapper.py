"""Newton solver wrapper providing clean API for JAX automation chains.

This module provides a wrapper class that encapsulates the Newton solver functionality
with a clean `.solve()` interface that's compatible with JAX transformations.
The wrapper supports both regular and automatic differentiation modes.

Example:
    Basic usage:
    >>> solver = NewtonSolver(problem, solver_options)
    >>> solution = solver.solve(params)
    
    In JAX automation chain:
    >>> solver = NewtonSolver(problem, solver_options, differentiable=True)
    >>> objective = lambda params: compute_objective(solver.solve(params))
    >>> gradients = jax.grad(objective)(params)
"""

import jax
import jax.numpy as np
from typing import Dict, List, Any, Callable
from fealax import logger
from .newton_solvers import newton_solve
from .solver_utils import _ad_wrapper


class NewtonSolver:
    """Newton solver wrapper with clean API for JAX automation chains.
    
    This class provides a wrapper around the Newton solver that supports both
    regular solving and automatic differentiation through JAX transformations.
    
    Args:
        problem: Finite element problem instance
        solver_options: Configuration options for the Newton solver
        differentiable: If True, creates a differentiable solver using ad_wrapper
        adjoint_solver_options: Configuration for adjoint solver (if differentiable=True)
        use_jit: Enable JIT compilation for better performance
        
    Example:
        Basic solver usage:
        >>> solver = NewtonSolver(problem, {'tol': 1e-6})
        >>> solution = solver.solve(params)
        
        Differentiable solver for optimization:
        >>> solver = NewtonSolver(problem, {'tol': 1e-6}, differentiable=True)
        >>> objective = lambda p: compute_objective(solver.solve(p))
        >>> gradients = jax.grad(objective)(params)
    """
    
    def __init__(
        self,
        problem: Any,
        solver_options: Dict[str, Any] = {},
        differentiable: bool = False,
        adjoint_solver_options: Dict[str, Any] = {},
        use_jit: bool = False
    ):
        self.problem = problem
        self.solver_options = solver_options.copy()
        self.differentiable = differentiable
        self.adjoint_solver_options = adjoint_solver_options
        self.use_jit = use_jit
        
        # Create the appropriate solver function
        if differentiable:
            logger.debug("Creating differentiable Newton solver")
            self._solve_fn = _ad_wrapper(
                problem, 
                solver_options, 
                adjoint_solver_options, 
                use_jit
            )
        else:
            logger.debug("Creating standard Newton solver")
            self._solve_fn = self._create_standard_solver()
    
    def _create_standard_solver(self) -> Callable[[Any], List[np.ndarray]]:
        """Create standard (non-differentiable) solver function."""
        def solver_fn(params):
            # Set parameters on the problem
            self.problem.set_params(params)
            
            # Choose solver based on options
            if self.solver_options.get('use_jit', False) or self.use_jit:
                # Use JIT-compatible solver
                jit_options = self.solver_options.copy()
                jit_options['use_jit'] = True
                return newton_solve(self.problem, jit_options)
            else:
                # Use regular solver  
                return newton_solve(self.problem, self.solver_options)
        
        # Optionally JIT-compile the solver
        if self.use_jit and not self.differentiable:
            logger.debug("JIT-compiling standard solver")
            return jax.jit(solver_fn)
        else:
            return solver_fn
    
    def solve(self, params: Any) -> List[np.ndarray]:
        """Solve the finite element problem with given parameters.
        
        Args:
            params: Problem parameters (material properties, boundary conditions, etc.)
            
        Returns:
            Solution list where each array corresponds to a variable.
            
        Example:
            >>> solution = solver.solve(material_params)
            >>> displacement = solution[0]  # First variable (e.g., displacement)
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
        if self.differentiable:
            self._solve_fn = _ad_wrapper(
                self.problem, 
                self.solver_options, 
                self.adjoint_solver_options, 
                self.use_jit
            )
        else:
            self._solve_fn = self._create_standard_solver()
    
    def update_adjoint_options(self, new_options: Dict[str, Any]):
        """Update adjoint solver options (only for differentiable solvers).
        
        Args:
            new_options: New adjoint solver configuration options
            
        Raises:
            ValueError: If solver is not differentiable
        """
        if not self.differentiable:
            raise ValueError("Cannot update adjoint options for non-differentiable solver")
        
        self.adjoint_solver_options.update(new_options)
        
        # Recreate differentiable solver function
        self._solve_fn = _ad_wrapper_legacy(
            self.problem, 
            self.solver_options, 
            self.adjoint_solver_options, 
            self.use_jit
        )
    
    def make_differentiable(self, adjoint_solver_options: Dict[str, Any] = {}):
        """Convert to differentiable solver.
        
        Args:
            adjoint_solver_options: Configuration for adjoint solver
            
        Note:
            This recreates the solver as a differentiable version.
        """
        if self.differentiable:
            logger.warning("Solver is already differentiable")
            return
        
        self.differentiable = True
        self.adjoint_solver_options = adjoint_solver_options
        
        # Recreate as differentiable solver
        self._solve_fn = _ad_wrapper_legacy(
            self.problem, 
            self.solver_options, 
            self.adjoint_solver_options, 
            self.use_jit
        )
        
        logger.debug("Converted to differentiable solver")
    
    def make_non_differentiable(self):
        """Convert to non-differentiable solver.
        
        Note:
            This recreates the solver as a standard (non-differentiable) version.
        """
        if not self.differentiable:
            logger.warning("Solver is already non-differentiable")
            return
        
        self.differentiable = False
        
        # Recreate as standard solver
        self._solve_fn = self._create_standard_solver()
        
        logger.debug("Converted to non-differentiable solver")
    
    @property
    def is_differentiable(self) -> bool:
        """Check if solver is differentiable."""
        return self.differentiable
    
    @property
    def is_jit_compiled(self) -> bool:
        """Check if solver uses JIT compilation."""
        return self.use_jit or self.solver_options.get('use_jit', False)


# Convenience function for creating solver instances
def create_newton_solver(
    problem: Any,
    solver_options: Dict[str, Any] = {},
    differentiable: bool = False,
    adjoint_solver_options: Dict[str, Any] = {},
    use_jit: bool = False
) -> NewtonSolver:
    """Create a Newton solver instance with specified configuration.
    
    Args:
        problem: Finite element problem instance
        solver_options: Configuration options for the Newton solver
        differentiable: If True, creates a differentiable solver
        adjoint_solver_options: Configuration for adjoint solver
        use_jit: Enable JIT compilation
        
    Returns:
        Configured NewtonSolver instance
        
    Example:
        Standard solver:
        >>> solver = create_newton_solver(problem, {'tol': 1e-6})
        
        Differentiable solver with JIT:
        >>> solver = create_newton_solver(
        ...     problem, 
        ...     {'tol': 1e-6}, 
        ...     differentiable=True, 
        ...     use_jit=True
        ... )
    """
    return NewtonSolver(
        problem=problem,
        solver_options=solver_options,
        differentiable=differentiable,
        adjoint_solver_options=adjoint_solver_options,
        use_jit=use_jit
    )