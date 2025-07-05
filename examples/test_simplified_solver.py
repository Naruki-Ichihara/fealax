#!/usr/bin/env python3
"""Test the simplified solver implementation after removing implicit_vjp."""

import sys
import os
sys.path.insert(0, '/workspace')

import jax.numpy as jnp
import jax
from fealax.solver.newton_wrapper import NewtonSolver
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC

def create_test_problem():
    """Create a simple elasticity problem."""
    mesh = box_mesh(3, 3, 3, 1.0, 1.0, 1.0, ele_type="HEX8")
    
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

def test_simplified_solver():
    """Test the simplified solver implementation."""
    print("Testing Simplified Solver (No implicit_vjp)")
    print("=" * 50)
    
    # Test parameters
    params = {'E': 200e9, 'nu': 0.3}
    
    try:
        # Create problem and solver
        problem = create_test_problem()
        solver_options = {'tol': 1e-8, 'max_iter': 30}
        
        print("Creating NewtonSolver...")
        solver = NewtonSolver(problem, solver_options)
        
        print("✓ Solver created successfully")
        
        # Test forward solve
        print("\nTesting forward solve...")
        solution = solver.solve(params)
        print(f"✓ Forward solve successful, solution shape: {solution[0].shape}")
        print(f"  Max displacement: {jnp.max(jnp.abs(solution[0])):.6e}")
        
        # Test gradient computation
        print("\nTesting gradient computation...")
        def objective(p):
            sol = solver.solve(p)
            return jnp.sum(jnp.abs(sol[0]))
        
        grad_fn = jax.grad(objective)
        gradients = grad_fn(params)
        
        print(f"✓ Gradient computation successful")
        print(f"  dJ/dE = {gradients['E']:.6e}")
        print(f"  dJ/dnu = {gradients['nu']:.6e}")
        
        # Test batch solving
        print("\nTesting batch solving...")
        params_batch = [
            {'E': 200e9, 'nu': 0.3},
            {'E': 250e9, 'nu': 0.28},
            {'E': 300e9, 'nu': 0.25}
        ]
        
        batch_solutions = solver.solve(params_batch)
        print(f"✓ Batch solve successful, batch shape: {batch_solutions[0].shape}")
        
        # Test solver options update
        print("\nTesting solver options update...")
        solver.update_solver_options({'tol': 1e-6})
        updated_solution = solver.solve(params)
        print(f"✓ Solver options update successful")
        
        print(f"\n{'='*50}")
        print("✅ ALL TESTS PASSED!")
        print("✅ Simplified solver implementation works correctly")
        print("✅ No more complex implicit_vjp - just clean JAX AD")
        
        return True
        
    except Exception as e:
        print(f"\n{'='*50}")
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simplified_solver()
    if success:
        print("\n🎉 Refactoring successful! The solver is now cleaner and more robust.")
    else:
        print("\n💥 Issues found in refactored code.")