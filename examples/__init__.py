"""
Fealax Examples Package

This package contains example problems demonstrating the capabilities of the fealax
finite element analysis library. The examples progress from basic linear problems
to advanced nonlinear and performance-oriented demonstrations.

Available Examples:
- simple_elasticity: Basic linear elasticity on a unit cube
- hyperelasticity: Neo-Hookean hyperelastic material with large deformation
- solver_comparison: Comprehensive solver API testing and comparison
- speed_comparison: Performance benchmarking across problem sizes  
- large_mesh_comparison: Memory management for high-resolution problems

To run an example:
    cd examples
    python simple_elasticity.py

For detailed information, see the README.md in this directory.
"""

__version__ = "1.0.0"
__author__ = "Fealax Development Team"

# Import key example functions for programmatic access
try:
    from .simple_elasticity import ElasticityProblem as LinearElasticityProblem
    from .hyperelasticity import HyperelasticProblem
except ImportError:
    # Handle case where examples are run as scripts
    pass

__all__ = [
    "LinearElasticityProblem",
    "HyperelasticProblem",
]