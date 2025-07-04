"""Problem module for finite element analysis.

This module contains the refactored problem components extracted from the
monolithic problem_core.py file. It provides a clean, modular structure for
finite element problem setup and solution.

Key Components:
    boundary_conditions: Boundary condition management and application
    kernels: Kernel generation for finite element weak forms
    
Public API:
    DirichletBC: Dirichlet boundary condition specification
    BoundaryConditionManager: Manages boundary condition application
    KernelGenerator: Generates computational kernels for weak forms
    Problem: Main finite element problem class
"""

from .boundary_conditions import (
    DirichletBC,
    BoundaryConditionManager,
    process_dirichlet_bcs,
    validate_dirichlet_bcs,
)
from .kernels import KernelGenerator

# Import Problem class from the core module
from fealax.problem_core import Problem

__all__ = [
    'DirichletBC',
    'BoundaryConditionManager', 
    'process_dirichlet_bcs',
    'validate_dirichlet_bcs',
    'KernelGenerator',
    'Problem',
]