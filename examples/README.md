# Fealax Examples

This directory contains example problems demonstrating how to use the fealax finite element analysis library.

## 📋 Available Examples

### 1. Linear Elasticity (`simple_elasticity.py`)

**Basic linear elasticity problem on a unit cube**

- **Physics**: Linear elastic deformation under compression
- **Domain**: 1×1×1 unit cube with HEX8 elements
- **Material**: Isotropic linear elastic (steel-like: E=200 GPa, ν=0.3)
- **Loading**: 5% compression applied to top face
- **Features**: Demonstrates basic fealax workflow, boundary conditions, and solver usage

```bash
python simple_elasticity.py
```

### 2. Hyperelasticity (`hyperelasticity.py`)

**Neo-Hookean hyperelastic material with large deformation**

- **Physics**: Nonlinear hyperelastic deformation with rotation and compression
- **Domain**: 1×1×1 unit cube with HEX8 elements  
- **Material**: Neo-Hookean hyperelastic (soft rubber: E=1 kPa, ν=0.3)
- **Loading**: Combined rotation (30°) and compression (10%) of top face
- **Features**: Large deformation mechanics, automatic differentiation, JIT compilation

```bash
python hyperelasticity.py
```

### 3. Solver Comparison (`solver_comparison.py`)

**Comprehensive comparison of different solver APIs and methods**

- **Purpose**: Compare newton_solve API vs conventional solver API
- **Features**: Performance benchmarking, API compatibility testing, solution verification
- **Solvers tested**: JIT vs non-JIT Newton solvers, different linear solver backends

```bash
python solver_comparison.py
```

### 4. Performance Comparison (`speed_comparison.py`)

**Performance benchmarking across different problem sizes**

- **Purpose**: Evaluate solver performance at different mesh resolutions
- **Features**: Memory usage analysis, convergence timing, scalability testing
- **Methods**: Hybrid JIT, Full JIT, and conventional solver comparison

```bash
python speed_comparison.py
```

### 5. Large Mesh Handling (`large_mesh_comparison.py`)

**Memory management for high-resolution finite element problems**

- **Purpose**: Demonstrate GPU memory management for large problems
- **Features**: Automatic mesh size selection, memory estimation, chunked assembly
- **Applications**: High-resolution simulations, GPU memory optimization

```bash
python large_mesh_comparison.py
```

## 🚀 Quick Start

### Prerequisites

Install fealax and its dependencies:

```bash
# Install fealax in development mode
pip install -e .

# Install JAX with GPU support (optional but recommended)
pip install jax[cuda]

# Verify installation
python -c "import fealax; print('Fealax installed successfully!')"
```

### Running Examples

All examples can be run directly from the examples directory:

```bash
cd examples

# Start with the linear elasticity example
python simple_elasticity.py

# Try nonlinear hyperelasticity
python hyperelasticity.py

# Compare solver performance
python speed_comparison.py
```

## 📊 Expected Output

Each example provides:

1. **Problem setup information**: Mesh size, material properties, boundary conditions
2. **Solver progress**: Newton iteration convergence, linear solver performance
3. **Solution analysis**: Displacement statistics, physical interpretation
4. **VTU output files**: For visualization in ParaView or similar tools

## 🔧 Customization

### Mesh Resolution

Adjust mesh density by modifying the mesh creation parameters:

```python
# In any example, change:
mesh = box_mesh(nx=10, ny=10, nz=10, ...)  # Coarse mesh
mesh = box_mesh(nx=50, ny=50, nz=50, ...)  # Fine mesh
```

### Material Properties

Modify material parameters in the problem definition:

```python
# Linear elasticity
E = 200e9  # Young's modulus (Pa)
nu = 0.3   # Poisson's ratio

# Hyperelasticity  
E = 1e3    # Softer material for better convergence
nu = 0.45  # Nearly incompressible
```

### Solver Options

Tune solver behavior for your specific problem:

```python
solver_options = {
    'tol': 1e-6,           # Convergence tolerance
    'max_iter': 20,        # Maximum Newton iterations
    # JIT compilation is always enabled for optimal performance
    'precond': True,       # Use preconditioning
    'line_search': True    # Enable line search
}
```

## 🎯 Learning Path

**Recommended order for learning fealax:**

1. **Start with `simple_elasticity.py`** - Learn basic concepts
2. **Try `hyperelasticity.py`** - Explore nonlinear mechanics
3. **Run `speed_comparison.py`** - Understand performance options
4. **Experiment with `solver_comparison.py`** - Compare different APIs
5. **Scale up with `large_mesh_comparison.py`** - Handle large problems

## 📚 Key Concepts Demonstrated

- **Problem Definition**: Subclassing `Problem` and implementing physics
- **Boundary Conditions**: Using `DirichletBC` for essential boundary conditions
- **Material Models**: Linear elasticity and hyperelastic constitutive laws
- **Solver APIs**: Modern `newton_solve()` vs conventional `solver()` 
- **Performance**: JIT compilation, GPU acceleration, memory management
- **Post-processing**: Solution analysis and VTK output for visualization

## 🐛 Troubleshooting

### Common Issues

1. **Out of memory errors**: Reduce mesh size or use `large_mesh_comparison.py` approach
2. **Convergence problems**: Lower applied loads, enable line search, use softer materials
3. **Slow performance**: JIT compilation is now always enabled for optimal performance
4. **Import errors**: Verify fealax installation with `pip install -e .`

### Getting Help

- Check the main repository documentation
- Review solver options in the API reference
- Look at the modular solver and problem documentation

Each example includes detailed comments and docstrings to help you understand the implementation details.