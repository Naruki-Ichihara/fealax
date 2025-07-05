"""Unified solver module providing all solver functionality.

This module provides a clean interface to all solver components:
    - Linear solvers for assembled finite element systems
    - Newton-Raphson solvers for nonlinear problems  
    - Boundary condition handling utilities
    - Linear algebra utilities
    - Automatic differentiation wrappers

The module is organized into submodules but provides a flat import interface
for convenience and backward compatibility.
"""

# Import all linear solver functions
from .linear_solvers import (
    jax_sparse_direct_solve,
    jax_iterative_solve,
    solve,
    solve_jit  # alias for solve
)

# Import all Newton solver functions
from .newton_solvers import (
    newton_solve,
    _jit_solver,
    linear_incremental_solver,
    line_search,
    get_A,
    jit_solver,  # alias for backward compatibility
    extract_solver_data
)

# Import boundary condition utilities
from .boundary_conditions import (
    apply_bc_vec,
    apply_bc,
    assign_bc,
    copy_bc,
    get_flatten_fn,
    jit_apply_bc_vec
)

# Import linear algebra utilities
from .linear_algebra import (
    jax_get_diagonal,
    zero_rows_jax,
    jax_matrix_multiply,
    array_to_jax_vec
)

# Import JIT solver functions
from .jit_solvers import (
    jit_newton_step_bicgstab_precond,
    jit_newton_step_bicgstab_no_precond,
    jit_newton_step_sparse,
    jit_newton_step_full,
    jit_newton_step,
    jit_residual_norm
)

# Import solver utilities
from .solver_utils import (
    implicit_vjp,
    ad_wrapper,
    _ad_wrapper
)

# Import Newton solver wrapper
from .newton_wrapper import (
    NewtonSolver,
    create_newton_solver
)

# Vmap and batch solvers removed - use standard NewtonSolver for all cases

# Export list for clarity
__all__ = [
    # Linear solvers
    'jax_sparse_direct_solve', 
    'jax_iterative_solve',
    'solve',
    'solve_jit',
    
    # Newton solvers
    'newton_solve',
    '_jit_solver', 
    'linear_incremental_solver',
    'line_search',
    'get_A',
    'jit_solver',
    
    # JIT solvers
    'jit_newton_step_bicgstab_precond',
    'jit_newton_step_bicgstab_no_precond',
    'jit_newton_step_sparse',
    'jit_newton_step_full',
    'jit_newton_step',
    'jit_residual_norm',
    
    # Solver utilities
    'extract_solver_data',
    
    # Boundary conditions
    'apply_bc_vec',
    'apply_bc',
    'assign_bc',
    'copy_bc',
    'get_flatten_fn',
    'jit_apply_bc_vec',
    
    # Linear algebra
    'jax_get_diagonal',
    'zero_rows_jax',
    'jax_matrix_multiply',
    'array_to_jax_vec',
    
    # Automatic differentiation
    'implicit_vjp',
    'ad_wrapper',
    '_ad_wrapper',
    
    # Newton solver wrapper
    'NewtonSolver',
    'create_newton_solver',
    
]