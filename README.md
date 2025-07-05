# fealax

GPU-accelerated finite element analysis (FEA) library built with JAX for computational mechanics and numerical material testing.

## Quick Start

### Problem Setup

```python
import jax.numpy as jnp
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import NewtonSolver

# Create mesh
mesh = box_mesh(10, 10, 10, 1.0, 1.0, 1.0, ele_type="HEX8")

# Define boundary conditions
bcs = [
    DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 2, lambda x: 0.0),       # Fix bottom
    DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 2, lambda x: -0.05), # Compress top
]

# Define elasticity problem with parameterized materials
class ElasticityProblem(Problem):
    def __init__(self, mesh, **kwargs):
        self.E = None   # Young's modulus (set via parameters)
        self.nu = None  # Poisson's ratio (set via parameters)
        super().__init__(mesh=mesh, **kwargs)
        
    def set_params(self, params):
        """Set material parameters."""
        self.E = params['E']
        self.nu = params['nu']
        
    def get_tensor_map(self):
        """Linear elasticity constitutive law."""
        def stress_strain(u_grads):
            strain = 0.5 * (u_grads + jnp.transpose(u_grads))
            lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            mu = self.E / (2 * (1 + self.nu))
            stress = 2 * mu * strain + lam * jnp.trace(strain) * jnp.eye(3)
            return stress
        return stress_strain

# Create problem
problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
```

### Newton Solver Usage

#### Basic Solving

```python
# Create Newton solver
solver = NewtonSolver(problem, {'tol': 1e-6, 'max_iter': 10})

# Material parameters
material_params = {
    'E': 200e9,  # Young's modulus (Pa)
    'nu': 0.3    # Poisson's ratio
}

# Solve the problem
solution = solver.solve(material_params)
displacement_field = solution[0]  # Extract displacement field

print(f"Solution shape: {displacement_field.shape}")
print(f"Max displacement: {jnp.max(jnp.abs(displacement_field)):.6f} m")
```

#### Performance Options

```python
# JIT-compiled solver for better performance
solver = NewtonSolver(problem, {
    'tol': 1e-6,
    'use_jit': True,     # Enable JIT compilation
    'precond': True,     # Use preconditioning
    'method': 'bicgstab' # Linear solver method
})

solution = solver.solve(material_params)
```

#### Differentiable Solver for Optimization

```python
import jax

# Create differentiable solver
diff_solver = NewtonSolver(problem, {
    'tol': 1e-6,
    'max_iter': 10
}, differentiable=True)  # Enable automatic differentiation

# Define objective function (e.g., compliance minimization)
def compliance_objective(params):
    solution = diff_solver.solve(params)
    displacement = solution[0]
    return jnp.sum(displacement**2)  # Simple compliance measure

# Compute gradients with respect to material parameters
grad_fn = jax.grad(compliance_objective)
gradients = grad_fn(material_params)

print(f"∂C/∂E = {gradients['E']:.2e}")
print(f"∂C/∂ν = {gradients['nu']:.2e}")
```

#### Parameter Studies and Optimization

```python
# Parameter sweep
E_values = jnp.linspace(100e9, 300e9, 10)
compliances = []

for E in E_values:
    params = {'E': float(E), 'nu': 0.3}
    solution = solver.solve(params)
    compliance = jnp.sum(solution[0]**2)
    compliances.append(compliance)

# Optimization loop
def optimize_material():
    """Simple gradient-based optimization."""
    params = {'E': 200e9, 'nu': 0.3}
    learning_rate = 1e-10
    
    for i in range(10):
        gradients = grad_fn(params)
        # Update parameters (gradient descent)
        params['E'] -= learning_rate * gradients['E']
        params['nu'] -= learning_rate * gradients['nu']
        
        # Compute objective
        obj_value = compliance_objective(params)
        print(f"Iteration {i}: Objective = {obj_value:.6e}")
    
    return params

optimized_params = optimize_material()
```

#### Batch Parameter Solving

```python
# Efficient batch processing of multiple parameter sets
from fealax.solver import BatchNewtonSolver, create_batch_solver

# Create batch solver for efficient parameter processing
batch_solver = create_batch_solver(problem, {
    'tol': 1e-6,
    'max_iter': 10,
    'use_jit': True
}, differentiable=True)

# Parameter batch: different material properties
param_batch = [
    {'E': 200e9, 'nu': 0.3},   # Steel
    {'E': 70e9,  'nu': 0.33},  # Aluminum
    {'E': 210e9, 'nu': 0.28},  # Carbon steel
    {'E': 110e9, 'nu': 0.35},  # Brass
]

# Solve all parameter sets efficiently
solutions = batch_solver.solve_batch(param_batch, show_progress=True)

# Process results
for i, (params, solution) in enumerate(zip(param_batch, solutions)):
    displacement = solution[0]
    max_disp = jnp.max(jnp.abs(displacement))
    print(f"Material {i+1}: E={params['E']/1e9:.0f} GPa → max_disp={max_disp:.4f} m")

# Parameter sweep utility
sweep_results = batch_solver.solve_parameter_sweep(
    param_name='E',
    param_values=jnp.linspace(100e9, 300e9, 10),
    base_params={'nu': 0.3}
)

# Multi-parameter grid
grid_results = batch_solver.solve_multi_parameter_grid(
    param_grids={
        'E': [150e9, 200e9, 250e9],
        'nu': [0.25, 0.3, 0.35]
    }
)
```

#### Performance Benchmarking

```python
from fealax.solver import benchmark_batch_solving

# Compare individual vs batch solving performance
results = benchmark_batch_solving(
    problem=problem,
    params_batch=param_batch,
    solver_options={'tol': 1e-6, 'use_jit': True}
)

print(f"Individual time: {results['individual_time']:.3f} s")
print(f"Batch time: {results['batch_time']:.3f} s") 
print(f"Performance ratio: {results['speedup_ratio']:.2f}x")
```

#### Advanced Configuration

```python
# Advanced solver options
advanced_solver = NewtonSolver(
    problem=problem,
    solver_options={
        'tol': 1e-8,              # Convergence tolerance
        'rel_tol': 1e-10,         # Relative tolerance
        'max_iter': 20,           # Maximum Newton iterations
        'method': 'bicgstab',     # Linear solver: 'bicgstab', 'cg'
        'precond': True,          # Enable Jacobi preconditioning
        'line_search_flag': True, # Enable line search
        'use_jit': True           # JIT compilation
    },
    differentiable=True,          # Enable automatic differentiation
    adjoint_solver_options={      # Options for adjoint system
        'tol': 1e-10,
        'method': 'bicgstab'
    },
    use_jit=True                  # JIT-compile the entire wrapper
)

# Dynamic configuration updates
advanced_solver.update_solver_options({'tol': 1e-10})
advanced_solver.update_adjoint_options({'method': 'cg'})

# Convert between differentiable and non-differentiable modes
advanced_solver.make_non_differentiable()  # Disable AD
advanced_solver.make_differentiable()      # Re-enable AD

# Check solver properties
print(f"Is differentiable: {advanced_solver.is_differentiable}")
print(f"Is JIT compiled: {advanced_solver.is_jit_compiled}")
```

### Legacy Solver Interface

```python
# Backward compatibility with original API
from fealax.solver import newton_solve, ad_wrapper

# Original newton_solve function
solution = newton_solve(problem, {'tol': 1e-6, 'use_jit': True})

# Original ad_wrapper for automatic differentiation
differentiable_solver = ad_wrapper(problem, {'tol': 1e-6})
solution = differentiable_solver(material_params)
```

## Key Features

### 🚀 Modern Solver API
- **`NewtonSolver` wrapper**: Clean `solver.solve(params)` interface
- **JAX automation chains**: Seamless integration with `jax.grad`, `jax.jit`
- **Parameter-driven**: Material properties and parameters passed to solve
- **Dynamic configuration**: Update solver options on the fly

### 🎯 Automatic Differentiation
- **Adjoint method**: Efficient gradient computation through FE solutions
- **Parameter sensitivity**: Gradients w.r.t. material properties, geometry
- **Optimization ready**: Drop-in compatibility with JAX optimizers
- **Hybrid approach**: Manual adjoint solve + JAX VJP for optimal performance

### ⚡ High Performance
- **GPU acceleration**: JAX-based implementation with NVIDIA GPU support
- **JIT compilation**: High-performance compiled solver kernels
- **Memory efficient**: Optimized sparse matrix operations
- **Scalable**: Handles large finite element problems (100k+ elements)

### 🔧 Comprehensive FE Capabilities
- **Multiple element types**: HEX8/20, TET4/10, QUAD4/8, TRI3/6
- **Advanced solvers**: Newton-Raphson with line search, BiCGSTAB, CG
- **Boundary conditions**: Flexible Dirichlet BC specification
- **Material models**: Linear elasticity, hyperelasticity support

### 🔄 Flexible Architecture
- **Backward compatible**: Existing `newton_solve`, `ad_wrapper` still work
- **Modular design**: Clean separation of concerns
- **Easy migration**: Smooth transition from legacy to new API

## Use Cases

### Research & Development
```python
# Quick prototyping with automatic differentiation
solver = NewtonSolver(problem, options, differentiable=True)
sensitivity = jax.grad(objective)(material_params)
```

### Parameter Optimization
```python
# Material property optimization
def objective(params):
    return compliance(solver.solve(params))

optimizer = optax.adam(1e-3)
optimized_params = optimization_loop(jax.grad(objective), initial_params)
```

### High-Performance Computing
```python
# Large-scale problems with JIT compilation
solver = NewtonSolver(problem, {'use_jit': True}, use_jit=True)
solution = solver.solve(params)  # GPU-accelerated, JIT-compiled
```

### Design Space Exploration
```python
# Parameter sweeps and sensitivity studies
for E in E_range:
    solution = solver.solve({'E': E, 'nu': 0.3})
    results.append(post_process(solution))
```

### Batch Parameter Studies
```python
# Efficient batch processing with optimized compilation
batch_solver = create_batch_solver(problem, options)
param_batch = [{'E': E, 'nu': 0.3} for E in E_range]
solutions = batch_solver.solve_batch(param_batch)  # Optimized batch solving
```

## Architecture

**Solver Module** (`fealax.solver`):
- `NewtonSolver` - Modern wrapper with clean API
- `BatchNewtonSolver` - Efficient batch parameter processing
- `SimpleVmapSolver` - Experimental vmap-based parallel solving
- `newton_solve()` - Legacy Newton solver interface  
- `ad_wrapper()` - Legacy automatic differentiation wrapper
- `linear_solvers` - JAX-based linear solvers (BiCGSTAB, CG, sparse direct)
- `newton_solvers` - Newton-Raphson implementations with line search
- `jit_solvers` - JIT-compiled versions for performance
- `boundary_conditions` - Boundary condition enforcement
- `solver_utils` - Automatic differentiation utilities

**Problem Module** (`fealax.problem`):
- `Problem` - Main finite element problem class
- `DirichletBC` - Boundary condition specification  
- `kernels` - Weak form kernel generation
- `assembly` - Global assembly and sparse matrix construction
- `boundary_conditions` - BC management and processing

## Examples

See the `examples/` directory for complete working examples:

### Basic Examples
- **`simple_elasticity.py`** - 3D linear elasticity with compression loading
  - NewtonSolver wrapper usage
  - Parameter-driven material properties
  - Automatic differentiation for sensitivity analysis
  - Performance optimization with JIT compilation

- **`simple_vmap_example.py`** - Parallel parameter solving with vmap
  - SimpleVmapSolver for batch processing
  - Material property parameter sweeps
  - Performance comparison: sequential vs parallel
  - Automatic differentiation with parameter batches

### Migration Guide

**From Legacy API:**
```python
# Old approach
from fealax.solver import newton_solve, ad_wrapper

solution = newton_solve(problem, solver_options)

# For optimization
diff_solver = ad_wrapper(problem, solver_options)
solution = diff_solver(params)
```

**To New API:**
```python
# New approach - cleaner and more flexible
from fealax.solver import NewtonSolver

solver = NewtonSolver(problem, solver_options)
solution = solver.solve(params)

# For optimization - same interface
diff_solver = NewtonSolver(problem, solver_options, differentiable=True)
solution = diff_solver.solve(params)
```

### Performance Tips

1. **Enable JIT compilation** for repeated solves:
   ```python
   solver = NewtonSolver(problem, {'use_jit': True}, use_jit=True)
   ```

2. **Use preconditioning** for faster convergence:
   ```python
   solver = NewtonSolver(problem, {'precond': True, 'method': 'bicgstab'})
   ```

3. **Batch parameter studies** efficiently:
   ```python
   # JIT-compile once, solve many times
   solver = NewtonSolver(problem, options, use_jit=True)
   results = [solver.solve(params) for params in param_list]
   ```

4. **Optimize solver tolerances** for your problem:
   ```python
   # Tighter tolerances for accuracy
   solver = NewtonSolver(problem, {'tol': 1e-8, 'rel_tol': 1e-10})
   
   # Looser tolerances for speed
   solver = NewtonSolver(problem, {'tol': 1e-4, 'max_iter': 5})
   ```

## Installation

```bash
# Development installation
pip install -e .

# JAX with GPU support (install separately)
pip install jax[cuda]

# For examples and visualization
pip install matplotlib vtk meshio
```

## GPU Requirements

- NVIDIA GPU with CUDA support
- JAX CUDA installation
- Sufficient GPU memory for problem size

The library automatically uses GPU acceleration when available. For large problems (>100k DOFs), GPU acceleration provides significant speedup over CPU-only computation.