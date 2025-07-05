#!/usr/bin/env python3
"""Test the gradient solver concept with a simplified approach."""

import sys
import os
sys.path.insert(0, '/workspace')

import jax.numpy as jnp
import jax
from fealax.solver.newton_wrapper import NewtonSolver
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC

def create_simple_problem():
    """Create a very simple elasticity problem."""
    mesh = box_mesh(2, 2, 2, 1.0, 1.0, 1.0, ele_type="HEX8")
    
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
        DirichletBC(subdomain=lambda x: jnp.abs(x[2] - 1.0) < 1e-6, vec=2, eval=lambda x: 0.01)
    ]
    
    problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    return problem

def test_current_implementation():
    """Test the current implementation without the new gradient solver."""
    print("Testing Current Implementation")
    print("=" * 40)
    
    # Create problem and solver
    problem = create_simple_problem()
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    solver = NewtonSolver(problem, solver_options)
    
    # Test single gradient
    def objective(params):
        sol = solver.solve(params)
        return jnp.sum(sol[0]**2)
    
    params = {'E': 200e9, 'nu': 0.3}
    
    try:
        print("Testing single gradient...")
        grad_fn = jax.grad(objective)
        gradients = grad_fn(params)
        print(f"✓ Single gradient works: dE={gradients['E']:.2e}, dnu={gradients['nu']:.2e}")
        return True
    except Exception as e:
        print(f"✗ Single gradient failed: {e}")
        return False

def test_simple_vmap_workaround():
    """Test a simple workaround for vmap gradients."""
    print("\nTesting Simple Vmap Workaround")
    print("=" * 40)
    
    # Create problem and solver
    problem = create_simple_problem()
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    solver = NewtonSolver(problem, solver_options)
    
    # Define objective with individual parameters (this worked before)
    def objective_individual(E, nu):
        params = {'E': E, 'nu': nu}
        sol = solver.solve(params)
        return jnp.sum(sol[0]**2)
    
    try:
        print("Testing individual parameter gradient...")
        grad_fn = jax.grad(objective_individual, argnums=(0, 1))
        dE, dnu = grad_fn(200e9, 0.3)
        print(f"✓ Individual gradient works: dE={dE:.2e}, dnu={dnu:.2e}")
        
        print("Testing vmap of individual gradients...")
        vmap_grad_fn = jax.vmap(grad_fn, in_axes=(0, 0))
        
        E_batch = jnp.array([200e9, 250e9])
        nu_batch = jnp.array([0.3, 0.28])
        
        dE_batch, dnu_batch = vmap_grad_fn(E_batch, nu_batch)
        print(f"✓ Vmap gradients work!")
        print(f"  Batch 0: dE={dE_batch[0]:.2e}, dnu={dnu_batch[0]:.2e}")
        print(f"  Batch 1: dE={dE_batch[1]:.2e}, dnu={dnu_batch[1]:.2e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Vmap workaround failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test both current implementation and vmap workaround."""
    print("Testing Gradient Solver Concepts")
    print("=" * 50)
    
    # Test current implementation
    current_works = test_current_implementation()
    
    # Test vmap workaround
    vmap_works = test_simple_vmap_workaround()
    
    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    
    if current_works:
        print("✅ Current implementation: Single gradients work")
    else:
        print("❌ Current implementation: Single gradients fail")
    
    if vmap_works:
        print("✅ Vmap workaround: Batch gradients work")
        print("✅ This provides a practical solution for optimization!")
    else:
        print("❌ Vmap workaround: Batch gradients fail")
    
    if vmap_works:
        print(f"\n🎯 RECOMMENDATION:")
        print(f"The individual parameter approach already provides vmap gradients!")
        print(f"This may be sufficient for most optimization use cases.")
        print(f"\nUsage pattern:")
        print(f"  def objective(E, nu):")
        print(f"      params = {{'E': E, 'nu': nu}}")
        print(f"      return compute_objective(solver.solve(params))")
        print(f"  ")
        print(f"  grad_fn = jax.grad(objective, argnums=(0, 1))")
        print(f"  vmap_grad_fn = jax.vmap(grad_fn)")
        
    return current_works and vmap_works

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎉 Both approaches work - practical vmap gradients available!")
    else:
        print(f"\n⚠️  Some limitations found.")