"""Finite element solvers and solution algorithms.

This module provides comprehensive solution algorithms for finite element problems,
including linear solvers, nonlinear Newton-Raphson methods, boundary condition
enforcement, and specialized continuation methods. It supports multiple backend
solvers with JAX as the primary backend for GPU acceleration
and automatic differentiation.

The module includes:
    - Linear solver interfaces (JAX iterative, JAX sparse direct)
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
    
    >>> from fealax.solver import _solver
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

# Import linear solver functions from refactored module
from fealax.solver.linear_solvers import (
    jax_solve,
    jax_sparse_direct_solve,
    jax_iterative_solve,
    linear_solver,
    solve,
    solve_jit
)

# Import Newton solver functions from refactored module
from fealax.solver.newton_solvers import (
    newton_solve,
    linear_incremental_solver,
    line_search,
    get_A,
    extract_solver_data
)

from jax import config

config.update("jax_enable_x64", True)
CHUNK_SIZE = 100000000


# Import linear algebra utilities from refactored module
from fealax.solver.linear_algebra import (
    jax_get_diagonal,
    zero_rows_jax,
    jax_matrix_multiply,
    array_to_jax_vec
)

# Import boundary condition utilities from refactored module  
from fealax.solver.boundary_conditions import (
    apply_bc_vec,
    apply_bc,
    assign_bc,
    copy_bc,
    get_flatten_fn
)

# Import solver utilities from refactored module
from fealax.solver.solver_utils import (
    implicit_vjp,
    ad_wrapper,
    extract_solver_data
)










# All solver functions have been moved to appropriate submodules:
# - Linear solvers: fealax.solver.linear_solvers
# - Newton solvers: fealax.solver.newton_solvers
# - JIT solvers: fealax.solver.jit_solvers
# - Boundary conditions: fealax.solver.boundary_conditions
# - Linear algebra: fealax.solver.linear_algebra
# - Solver utilities: fealax.solver.solver_utils
#
# This module is kept for backward compatibility and imports everything
# from the solver submodules for a unified interface.
