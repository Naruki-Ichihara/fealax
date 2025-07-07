# Batch Gradients with fealax NewtonSolver

This guide demonstrates how to efficiently compute gradients with batch parameters using fealax's NewtonSolver for hyperelastic finite element problems.

## Quick Start

```python
import jax
from fealax.solver import NewtonSolver

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Create solver (see batch_gradients_simple.py for full setup)
solver = create_hyperelastic_solver()

# 1. Batch solving (efficient forward pass)
batch_params = [
    {'E': 5e4, 'nu': 0.25},
    {'E': 1e5, 'nu': 0.3},
    {'E': 2e5, 'nu': 0.35}
]
solutions = solver.solve(batch_params)

# 2. Individual gradients (reliable backward pass)
def objective(params):
    solution = solver.solve(params)
    displacement = solution[0]
    return jax.numpy.sum(displacement**2)

grad_fn = jax.grad(objective)
gradients = grad_fn({'E': 1e5, 'nu': 0.3})
```

## Key Features

### ✅ What Works Well

- **Batch solving**: `solver.solve(batch_params)` - Fast parallel forward passes
- **Individual gradients**: `jax.grad(objective)` - Reliable automatic differentiation
- **JIT compilation**: Both approaches can be JIT-compiled for performance
- **No tracer leaks**: Proper JAX tracer handling in parameter setting

### 🔧 Recommended Workflow

1. **Forward pass**: Use batch solving for multiple parameter sets
2. **Gradient computation**: Use individual `jax.grad()` calls 
3. **Performance**: JIT-compile gradient functions for repeated use
4. **Optimization**: Combine both approaches for parameter identification

## Examples

### Batch Solving
```python
# Solve multiple material parameter sets efficiently
batch_params = [
    {'E': 5e4, 'nu': 0.25},
    {'E': 1e5, 'nu': 0.3}, 
    {'E': 2e5, 'nu': 0.35}
]
solutions = solver.solve(batch_params)
print(f"Solved {len(batch_params)} cases, shape: {solutions[0].shape}")
```

### Gradient Computation
```python
# Compute gradients for optimization
def objective(params):
    solution = solver.solve(params)
    return jax.numpy.sum(solution[0]**2)

gradients = jax.grad(objective)({'E': 1e5, 'nu': 0.3})
print(f"∂J/∂E: {gradients['E']:.2e}")
```

### Optimization Loop
```python
# JIT-compiled optimization
grad_fn = jax.jit(jax.grad(objective))

params = {'E': 1e5, 'nu': 0.3}
for step in range(10):
    obj_val = objective(params)
    gradients = grad_fn(params)
    
    # Update parameters
    step_size = 1e-12
    params['E'] -= step_size * gradients['E']
    params['nu'] -= step_size * gradients['nu']
```

## Technical Details

### JAX Tracer Handling

The `set_params` method properly handles JAX tracers during gradient computation:

```python
def set_params(self, params):
    def is_tracer(x):
        return hasattr(x, 'aval') or str(type(x)).find('Tracer') != -1
    
    if 'E' in params:
        self.E = params['E'] if is_tracer(params['E']) else jnp.asarray(params['E'])
    # ... update material parameters safely
```

### Performance Characteristics

- **Batch solving**: ~5s for 3 parameter sets (4×4×4 mesh)
- **Individual gradients**: ~4s per gradient with JIT compilation
- **Memory usage**: ~0.35MB per problem (efficient sparse matrices)
- **Scalability**: Excellent for parameter studies and optimization

## Problem Setup

The examples use a Neo-Hookean hyperelastic material:

```python
class NeoHookeanProblem(Problem):
    def get_tensor_map(self):
        def tensor_map(u_grads, *internal_vars):
            F = u_grads + jnp.eye(3)  # Deformation gradient
            J = jnp.linalg.det(F)     # Jacobian
            C = jnp.transpose(F) @ F   # Right Cauchy-Green tensor
            
            # Neo-Hookean stress
            I1 = jnp.trace(C)
            S = self.mu * (jnp.eye(3) - jnp.linalg.inv(C)) + self.lam * jnp.log(J) * jnp.linalg.inv(C)
            return F @ S  # First Piola-Kirchhoff stress
        return tensor_map
```

## Running the Examples

```bash
# Run the simple example
python batch_gradients_simple.py

# Expected output:
# Batch Solving Example
# ========================================
# Solved 3 parameter sets
# Solution shape: (3, 125, 3)
# 
# Gradient Computation Example  
# ========================================
# Objective: 2.57e-03
# ∂J/∂E:  -7.35e-21
# ∂J/∂nu: 4.57e-04
```

## Files

- `batch_gradients_simple.py` - Clean, minimal example with hyperelasticity
- `working_batch_gradients.py` - Complete solution with detailed demonstrations
- `practical_batch_gradients.py` - Multiple approaches and performance comparison

## Best Practices

1. **Always enable 64-bit precision**: `jax.config.update("jax_enable_x64", True)`
2. **JIT-compile gradient functions** for repeated optimization
3. **Use batch solving** for forward passes when possible
4. **Handle JAX tracers properly** in `set_params` methods
5. **Consider finite differences** for very large batch sizes where individual gradients become slow

This approach provides both the performance benefits of batch solving and the reliability of individual gradient computation, making it ideal for parameter identification, optimization, and sensitivity analysis in finite element problems.