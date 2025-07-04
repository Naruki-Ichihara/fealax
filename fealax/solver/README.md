# Solver Module Refactoring

This directory contains the refactored solver components:

## Module Structure

- `linear_algebra.py` - Linear algebra utilities and matrix operations
- `boundary_conditions.py` - Boundary condition handling and enforcement  
- `linear_solvers.py` - Linear solver implementations (JAX, iterative, direct)
- `newton_solvers.py` - Newton-Raphson solver implementations
- `jit_solvers.py` - JIT-compiled solver variants
- `solver_utils.py` - Utility functions and helpers
- `__init__.py` - Main solver interface and exports

## Design Principles

1. **Single Responsibility**: Each module has a clear, focused purpose
2. **Minimal Dependencies**: Reduce circular imports and coupling
3. **Clean APIs**: Well-defined interfaces between modules
4. **Backward Compatibility**: Maintain existing API for users