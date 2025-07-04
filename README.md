# fealax

GPU-accelerated finite element analysis (FEA) library built with JAX for computational mechanics and numerical material testing.

## Quick Start

### Problem Setup

```python
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import newton_solve

# Create mesh
mesh = box_mesh(10, 10, 10, 1.0, 1.0, 1.0, ele_type="HEX8")

# Define boundary conditions
bcs = [
    DirichletBC(lambda x: x[2] < 1e-6, 2, lambda x: 0.0),  # Fix bottom in z
    DirichletBC(lambda x: x[2] > 1.0 - 1e-6, 2, lambda x: -0.05),  # Compress top
]

# Setup problem
problem = Problem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)

# Define material properties in custom_init
class ElasticityProblem(Problem):
    def custom_init(self, E, nu):
        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        
    def get_tensor_map(self):
        def stress_strain(u_grad):
            # Linear elasticity constitutive law
            eps = 0.5 * (u_grad + jnp.transpose(u_grad, axes=(0, 2, 1)))
            lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))
            mu = self.E / (2 * (1 + self.nu))
            return lam * jnp.trace(eps, axis1=1, axis2=2)[:, None, None] * jnp.eye(3) + 2 * mu * eps
        return stress_strain

# Create problem with material properties
problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs, 
                           additional_info=(200e9, 0.3))  # Steel properties
```

### Solver Usage

```python
# Simple Newton solver
sol = newton_solve(problem, solver_options={'max_iter': 10, 'atol': 1e-8})

# JIT-compiled solver for performance
sol = newton_solve(problem, solver_options={'jit': True})

# Custom solver options
solver_options = {
    'max_iter': 20,
    'atol': 1e-10,
    'rtol': 1e-8,
    'linear_solver': 'bicgstab',
    'line_search': True,
    'jit': True
}
sol = newton_solve(problem, solver_options=solver_options)

# Low-level solver access
from fealax.solver import _solver, _jit_solver
sol = _solver(problem, solver_options)  # Legacy solver
sol = _jit_solver(problem, solver_options)  # JIT-compiled legacy solver
```

### Modular Architecture

The library is organized into clean, modular components:

**Solver Module** (`fealax.solver`):
- `newton_solve()` - Main Newton solver interface
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

## Features

- **GPU Acceleration**: JAX-based implementation with NVIDIA GPU support
- **Automatic Differentiation**: Automatic Jacobian computation for Newton methods
- **Multiple Element Types**: HEX8/20, TET4/10, QUAD4/8, TRI3/6
- **Advanced Solvers**: Newton-Raphson with line search, arc-length continuation
- **Memory Management**: Efficient handling of large finite element problems
- **JIT Compilation**: High-performance JIT-compiled solver kernels