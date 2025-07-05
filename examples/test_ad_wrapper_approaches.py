#!/usr/bin/env python3
"""Test script to compare the two AD wrapper approaches."""

import sys
import os
sys.path.insert(0, '/workspace')

import jax.numpy as jnp
import jax
from fealax.solver.newton_wrapper import NewtonSolver
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC

def create_simple_problem():
    """Create a simple elasticity problem for testing."""
    # Create a small mesh for fast testing
    mesh = box_mesh(5, 5, 5, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    # Create elasticity problem
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
    
    # Add boundary conditions
    bcs = [
        DirichletBC(subdomain=lambda x: jnp.abs(x[2]) < 1e-6, vec=2, eval=lambda x: 0.0),  # Fixed bottom
        DirichletBC(subdomain=lambda x: jnp.abs(x[2] - 1.0) < 1e-6, vec=2, eval=lambda x: 0.01)  # Load top
    ]
    
    problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    return problem

def test_approach(approach_name, use_implicit_vjp):
    """Test one of the AD wrapper approaches."""
    print(f"\n{'='*50}")
    print(f"Testing {approach_name}")
    print(f"{'='*50}")
    
    try:
        # Create problem and solver
        problem = create_simple_problem()
        
        # Configure adjoint solver options
        adjoint_options = {'use_implicit_vjp': use_implicit_vjp}
        solver_options = {'tol': 1e-6, 'max_iter': 20}
        
        solver = NewtonSolver(problem, solver_options, adjoint_options)
        
        # Test parameters
        params = {'E': 200e9, 'nu': 0.3}
        
        # Solve forward problem
        print("Running forward solve...")
        solution = solver.solve(params)
        print(f"✓ Forward solve successful, solution shape: {solution[0].shape}")
        
        # Test gradient computation
        print("Testing gradient computation...")
        def objective(params_dict):
            sol = solver.solve(params_dict)
            # Simple objective: sum of squared displacements
            return jnp.sum(sol[0]**2)
        
        grad_fn = jax.grad(objective)
        gradients = grad_fn(params)
        
        print(f"✓ Gradient computation successful")
        print(f"  dJ/dE = {gradients['E']:.2e}")
        print(f"  dJ/dnu = {gradients['nu']:.2e}")
        
        return True, gradients
        
    except Exception as e:
        print(f"✗ {approach_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def main():
    """Test both AD wrapper approaches."""
    print("Testing different AD wrapper approaches for fealax")
    print("=" * 60)
    
    # Test the new direct approach (default)
    success1, grad1 = test_approach("Direct JAX AD (new default)", use_implicit_vjp=False)
    
    # Test the original implicit_vjp approach
    success2, grad2 = test_approach("Custom VJP with implicit_vjp (original)", use_implicit_vjp=True)
    
    # Compare results
    print(f"\n{'='*50}")
    print("COMPARISON RESULTS")
    print(f"{'='*50}")
    
    if success1 and success2:
        print("✓ Both approaches work!")
        
        # Compare gradients
        diff_E = abs(grad1['E'] - grad2['E'])
        diff_nu = abs(grad1['nu'] - grad2['nu'])
        
        print(f"\nGradient comparison:")
        print(f"  dJ/dE difference: {diff_E:.2e}")
        print(f"  dJ/dnu difference: {diff_nu:.2e}")
        
        if diff_E < 1e-6 and diff_nu < 1e-6:
            print("✓ Gradients match within tolerance!")
        else:
            print("⚠ Gradient differences are significant")
            
    elif success1 and not success2:
        print("✓ Direct JAX AD works, implicit_vjp fails")
        print("  Recommendation: Use direct approach (default)")
        
    elif not success1 and success2:
        print("✓ implicit_vjp works, direct JAX AD fails")
        print("  Recommendation: Use implicit_vjp approach")
        
    else:
        print("✗ Both approaches failed")
        
    print(f"\nTo use the implicit_vjp approach, set:")
    print(f"  adjoint_options = {{'use_implicit_vjp': True}}")
    print(f"  solver = NewtonSolver(problem, solver_options, adjoint_options)")

if __name__ == "__main__":
    main()