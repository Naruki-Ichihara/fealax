#!/usr/bin/env python3
"""Practical example of vmap gradient computation with fealax."""

import sys
import os
sys.path.insert(0, '/workspace')

import jax.numpy as jnp
import jax
import numpy as np
import time
from fealax.solver.newton_wrapper import NewtonSolver
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC

def create_test_problem():
    """Create a simple elasticity problem."""
    mesh = box_mesh(4, 4, 4, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    class ElasticityProblem(Problem):
        def __init__(self, mesh, **kwargs):
            super().__init__(mesh=mesh, **kwargs)
            self.E = 200e9
            self.nu = 0.3
            self.mu = self.E / (2.0 * (1.0 + self.nu))
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
            
        def set_params(self, params):
            self.E = params['E']
            self.nu = params['nu']
            self.mu = self.E / (2.0 * (1.0 + self.nu))
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        
        def get_tensor_map(self):
            def stress(u_grad):
                epsilon = 0.5 * (u_grad + u_grad.T)
                sigma = self.lam * jnp.trace(epsilon) * jnp.eye(3) + 2.0 * self.mu * epsilon
                return sigma
            return stress
    
    bcs = [
        DirichletBC(subdomain=lambda x: jnp.abs(x[2]) < 1e-6, vec=2, eval=lambda x: 0.0),
        DirichletBC(subdomain=lambda x: jnp.abs(x[2] - 1.0) < 1e-6, vec=2, eval=lambda x: 0.02)
    ]
    
    problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    return problem

def practical_vmap_gradients_demo():
    """Demonstrate practical vmap gradient computation."""
    print("Practical Vmap Gradient Computation Demo")
    print("=" * 50)
    
    # Create problem and solver
    problem = create_test_problem()
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    solver = NewtonSolver(problem, solver_options)
    
    # Define objective function with individual parameters
    def compliance_objective(E, nu):
        params = {'E': E, 'nu': nu}
        sol = solver.solve(params)
        # Compliance: sum of absolute displacements (proportional to strain energy)
        return jnp.sum(jnp.abs(sol[0]))
    
    # Create gradient function
    print("1. Creating gradient function...")
    grad_fn = jax.grad(compliance_objective, argnums=(0, 1))
    
    # Create vmapped version for batch computation
    print("2. Creating vmapped gradient function...")
    vmap_grad_fn = jax.vmap(grad_fn, in_axes=(0, 0))
    
    # Test parameters for optimization/parameter study
    n_params = 5
    E_range = jnp.linspace(150e9, 300e9, n_params)  # Young's modulus range
    nu_range = jnp.linspace(0.25, 0.35, n_params)   # Poisson's ratio range
    
    print(f"3. Computing gradients for {n_params} parameter combinations...")
    
    # Time sequential vs batch gradient computation
    print("\n--- Sequential Gradient Computation ---")
    start_time = time.time()
    
    sequential_gradients = []
    for i in range(n_params):
        dE, dnu = grad_fn(E_range[i], nu_range[i])
        sequential_gradients.append((dE, dnu))
    
    sequential_time = time.time() - start_time
    print(f"Sequential time: {sequential_time:.4f} seconds")
    print(f"Time per gradient: {sequential_time/n_params:.4f} seconds")
    
    # Batch gradient computation
    print("\n--- Batch (Vmap) Gradient Computation ---")
    start_time = time.time()
    
    batch_dE, batch_dnu = vmap_grad_fn(E_range, nu_range)
    
    batch_time = time.time() - start_time
    print(f"Batch time: {batch_time:.4f} seconds")
    print(f"Time per gradient: {batch_time/n_params:.4f} seconds")
    print(f"Speedup: {sequential_time/batch_time:.2f}x")
    
    # Display results
    print(f"\n--- Gradient Results ---")
    print(f"{'E (GPa)':<10} {'nu':<6} {'dJ/dE':<15} {'dJ/dnu':<15}")
    print("-" * 50)
    
    for i in range(n_params):
        E_gpa = E_range[i] / 1e9
        nu_val = nu_range[i]
        
        # Sequential results
        seq_dE, seq_dnu = sequential_gradients[i]
        
        # Batch results
        batch_dE_val = float(batch_dE[i])
        batch_dnu_val = float(batch_dnu[i])
        
        print(f"{E_gpa:<10.1f} {nu_val:<6.3f} {batch_dE_val:<15.6e} {batch_dnu_val:<15.6e}")
        
        # Verify consistency
        diff_E = abs(seq_dE - batch_dE_val)
        diff_nu = abs(seq_dnu - batch_dnu_val)
        assert diff_E < 1e-10, f"E gradient mismatch: {diff_E}"
        assert diff_nu < 1e-10, f"nu gradient mismatch: {diff_nu}"
    
    print(f"\n✅ All gradients match between sequential and batch!")
    
    return batch_time < sequential_time

def optimization_example():
    """Example: Using vmap gradients for optimization."""
    print(f"\n{'='*50}")
    print("Optimization Example with Vmap Gradients")
    print(f"{'='*50}")
    
    # Create problem and solver
    problem = create_test_problem()
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    solver = NewtonSolver(problem, solver_options)
    
    # Define objective: minimize compliance subject to material cost constraint
    def objective_with_constraint(E, nu):
        params = {'E': E, 'nu': nu}
        sol = solver.solve(params)
        
        # Compliance (structural flexibility - want to minimize)
        compliance = jnp.sum(jnp.abs(sol[0]))
        
        # Material cost penalty (higher E costs more)
        cost_penalty = (E / 200e9 - 1.0)**2 * 0.1
        
        return compliance + cost_penalty
    
    # Create gradient function
    grad_fn = jax.grad(objective_with_constraint, argnums=(0, 1))
    vmap_grad_fn = jax.vmap(grad_fn, in_axes=(0, 0))
    
    # Gradient-based optimization loop (simplified)
    print("Running simplified gradient-based optimization...")
    
    # Initial design points
    E_current = jnp.array([200e9, 220e9, 250e9])
    nu_current = jnp.array([0.3, 0.28, 0.32])
    
    learning_rate = 1e8  # Adjusted for E scale
    
    for iteration in range(3):
        # Compute batch gradients
        dE_batch, dnu_batch = vmap_grad_fn(E_current, nu_current)
        
        # Update parameters (gradient descent)
        E_current = E_current - learning_rate * dE_batch
        nu_current = nu_current - 0.001 * dnu_batch  # Different scale for nu
        
        # Clamp to reasonable bounds
        E_current = jnp.clip(E_current, 100e9, 400e9)
        nu_current = jnp.clip(nu_current, 0.2, 0.4)
        
        print(f"Iteration {iteration + 1}:")
        for i in range(len(E_current)):
            print(f"  Design {i}: E={E_current[i]/1e9:.1f} GPa, nu={nu_current[i]:.3f}")
            print(f"              dE={dE_batch[i]:.2e}, dnu={dnu_batch[i]:.2e}")
    
    print("✅ Optimization example completed!")

def main():
    """Run practical vmap gradient demonstrations."""
    print("Fealax Vmap Gradient Computation")
    print("=" * 60)
    
    # Demo 1: Basic vmap gradients
    is_faster = practical_vmap_gradients_demo()
    
    # Demo 2: Optimization example
    optimization_example()
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    print("✅ Vmap gradient computation works!")
    print("✅ Perfect accuracy - sequential and batch results match")
    if is_faster:
        print("✅ Batch gradients are faster than sequential")
    else:
        print("⚠️  Batch gradients have similar speed (overhead may dominate for small batches)")
    
    print("\n🚀 Key Benefits:")
    print("  • Enables efficient parameter studies")
    print("  • Perfect for optimization algorithms")
    print("  • Maintains full accuracy of gradients")
    print("  • Compatible with existing JAX ecosystem")
    
    print("\n📝 Usage Pattern:")
    print("  def objective(E, nu):")
    print("      params = {'E': E, 'nu': nu}")
    print("      sol = solver.solve(params)")
    print("      return some_function(sol)")
    print("  ")
    print("  grad_fn = jax.grad(objective, argnums=(0, 1))")
    print("  vmap_grad_fn = jax.vmap(grad_fn, in_axes=(0, 0))")
    print("  batch_gradients = vmap_grad_fn(E_batch, nu_batch)")

if __name__ == "__main__":
    main()