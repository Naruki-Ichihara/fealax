# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

fealax is a GPU-accelerated finite element analysis (FEA) library built with JAX for computational mechanics and numerical material testing. The project leverages JAX for automatic differentiation and GPU acceleration, making it suitable for high-performance scientific computing applications.

## Development Commands

### Installation
```bash
# Development installation
pip install -e .

# Production installation
pip install .

# Note: JAX is not listed in dependencies but is required - install manually:
pip install jax[cuda]  # For GPU support

# Works with latest numpy - no version constraints
```

### Testing
```bash
# Run all tests (pytest configuration in pyproject.toml)
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_basis.py

# Run single test function
pytest tests/test_basis.py::test_basis_function_evaluation
```

### Running Examples
```bash
# Run the simple elasticity example
cd examples
python simple_elasticity.py

# Run the hyperelasticity example
python hyper_elasticity.py
```

### Docker Development
```bash
# Build Docker image
docker build -t fealax:latest .

# Run with Docker Compose (includes GPU support)
docker-compose up -d

# Access container
docker exec -it fealax_fealax_1 bash
```

## Architecture and Key Components

The codebase follows a modular architecture centered around finite element analysis:

### Core Modules

1. **basis.py**: Implements finite element basis functions using the Basix library
   - Supports multiple element types: HEX8/20, TET4/10, QUAD4/8, TRI3/6
   - Provides shape functions and their gradients for numerical integration

2. **mesh.py**: Handles mesh generation and management
   - `Mesh` class stores mesh topology and geometry
   - Provides structured mesh generation (`box_mesh`)
   - Integrates with meshio for mesh I/O operations

3. **fe.py**: Core finite element implementation
   - `FiniteElement` class defines finite element spaces
   - Handles degrees of freedom management
   - Supports vector-valued problems

4. **problem.py**: Problem definition framework
   - `Problem` class assembles finite element problems
   - `DirichletBC` class for boundary conditions
   - Implements weak form computation and sparse matrix assembly

5. **solver.py**: Solution algorithms
   - Linear solvers: JAX iterative, JAX sparse, UMFPACK
   - Nonlinear Newton-Raphson solver with line search
   - Arc-length continuation for path following
   - Dynamic relaxation for quasi-static problems

### Key Design Patterns

- Heavy use of JAX for automatic differentiation and GPU acceleration
- Functional programming style with JAX transformations (jit, vmap, grad)
- Sparse matrix operations optimized for GPU execution
- Modular solver architecture with pluggable linear solvers

## Important Notes

1. **Fixed Naming Issues**: Previously the package had inconsistent naming between "cellax" and "fealax". All references have been updated to use "fealax" consistently.

2. **Missing JAX Dependency**: JAX is not listed in setup.cfg dependencies despite being essential. Install it manually when setting up the development environment.

3. **Test Suite**: The test suite has been cleaned up for fealax. Available tests:
   - `test_environment.py`: Tests basic package importing and structure
   - `test_basis.py`: Tests finite element basis functions

4. **GPU Requirements**: The project is optimized for NVIDIA GPUs. The Docker setup uses NVIDIA's JAX container with CUDA support.

5. **License Inconsistency**: setup.cfg states MIT license, but LICENSE file contains GPL v3. Verify correct license before contributions.

## Examples

The `examples/` directory contains example problems demonstrating fealax usage:

- `simple_elasticity.py`: 3D linear elasticity on a unit cube with uniaxial compression
- `hyper_elasticity.py`: Hyperelastic material model using Neo-Hookean constitutive law
- Shows mesh creation, boundary conditions, material properties, and solver usage

## Common Development Tasks

### Creating a New Physics Problem
1. Subclass the `Problem` class
2. Implement `get_tensor_map()` for gradient-based terms (like elasticity, diffusion)
3. Optionally implement `get_mass_map()` for reaction/source terms
4. Optionally implement `get_surface_maps()` for boundary flux terms
5. Follow the pattern shown in `examples/simple_elasticity.py`

### Adding a New Element Type
1. Add the element definition to `basis.py`
2. Update the element type mappings in the basis functions
3. Ensure proper integration rules are defined

### Implementing a New Solver
1. Add the solver function to `solver.py`
2. Follow the existing pattern of accepting a problem instance and solver options
3. Return solutions in the expected format (JAX arrays)

### Working with JAX
- Use `jax.numpy` instead of `numpy` for GPU-compatible operations
- Apply `@jax.jit` decorator for performance-critical functions
- Be mindful of JAX's functional programming constraints (no in-place modifications)