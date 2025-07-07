#!/usr/bin/env python3
"""
Simple example: Computing gradients with batch parameters in hyperelasticity problems.

This demonstrates the recommended approach for gradient computation with fealax NewtonSolver:
- Use solver.solve(batch_params) for efficient batch forward passes
- Use jax.grad() on individual parameter sets for reliable gradients
"""

import jax
import jax.numpy as jnp
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import NewtonSolver


class NeoHookeanProblem(Problem):
    """Neo-Hookean hyperelastic material problem."""
    
    def __init__(self, mesh, E=1e5, nu=0.3, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.E = E
        self.nu = nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def get_tensor_map(self):
        def tensor_map(u_grads, *internal_vars):
            # Deformation gradient
            F = u_grads + jnp.eye(3)
            J = jnp.linalg.det(F)
            C = jnp.transpose(F) @ F
            
            # Neo-Hookean stress
            I1 = jnp.trace(C)
            S = self.mu * (jnp.eye(3) - jnp.linalg.inv(C)) + self.lam * jnp.log(J) * jnp.linalg.inv(C)
            P = F @ S
            return P
        return tensor_map
    
    def set_params(self, params):
        # Handle JAX tracers safely
        def is_tracer(x):
            return hasattr(x, 'aval') or str(type(x)).find('Tracer') != -1
        
        if 'E' in params:
            self.E = params['E'] if is_tracer(params['E']) else jnp.asarray(params['E'], dtype=jnp.float64)
        if 'nu' in params:
            self.nu = params['nu'] if is_tracer(params['nu']) else jnp.asarray(params['nu'], dtype=jnp.float64)
        
        # Update material parameters
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def create_hyperelastic_solver():
    """Create hyperelastic problem solver."""
    # Create mesh
    mesh = box_mesh(4, 4, 4, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    # Boundary conditions: fix bottom, compress top
    bcs = [
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 0, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 1, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 2, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 2, lambda x: -0.1),
    ]
    
    # Create problem
    problem = NeoHookeanProblem(
        mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs, E=1e5, nu=0.3
    )
    
    return NewtonSolver(problem, {'tol': 1e-8, 'max_iter': 10})


def batch_solve_example():
    """Example: Batch solving for multiple material parameters."""
    print("Batch Solving Example")
    print("=" * 40)
    
    solver = create_hyperelastic_solver()
    
    # Multiple material parameters
    batch_params = [
        {'E': 5e4, 'nu': 0.25},
        {'E': 1e5, 'nu': 0.3},
        {'E': 2e5, 'nu': 0.35}
    ]
    
    # Batch solve (efficient)
    solutions = solver.solve(batch_params)
    
    print(f"Solved {len(batch_params)} parameter sets")
    print(f"Solution shape: {solutions[0].shape}")
    
    return solutions


def gradient_computation_example():
    """Example: Computing gradients for optimization."""
    print("\nGradient Computation Example")
    print("=" * 40)
    
    solver = create_hyperelastic_solver()
    
    def objective(params):
        """Objective: minimize displacement energy."""
        solution = solver.solve(params)
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # Parameters for gradient computation
    params = {'E': 1e5, 'nu': 0.3}
    
    # Compute objective and gradients
    obj_value = objective(params)
    grad_fn = jax.grad(objective)
    gradients = grad_fn(params)
    
    print(f"Objective: {obj_value:.6e}")
    print(f"∂J/∂E:  {gradients['E']:.6e}")
    print(f"∂J/∂nu: {gradients['nu']:.6e}")
    
    return gradients


def optimization_example():
    """Example: Simple optimization workflow."""
    print("\nOptimization Example")
    print("=" * 40)
    
    solver = create_hyperelastic_solver()
    
    def objective(params):
        solution = solver.solve(params)
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # JIT-compiled gradient function
    grad_fn = jax.jit(jax.grad(objective))
    
    # Initial parameters
    params = {'E': 1e5, 'nu': 0.3}
    
    # Gradient descent steps
    for step in range(3):
        obj_val = objective(params)
        gradients = grad_fn(params)
        
        print(f"Step {step+1}: obj={obj_val:.2e}, ∂E={gradients['E']:.2e}")
        
        # Update parameters
        step_size = 1e-12
        params = {
            'E': params['E'] - step_size * gradients['E'],
            'nu': max(0.1, min(0.45, params['nu'] - step_size * gradients['nu']))
        }


def main():
    """Run all examples."""
    jax.config.update("jax_enable_x64", True)
    
    # Run examples
    batch_solve_example()
    gradient_computation_example()
    optimization_example()
    
    print("\n" + "=" * 40)
    print("✅ All examples completed successfully!")


if __name__ == "__main__":
    main()