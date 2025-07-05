#!/usr/bin/env python3
"""Simple test to compare derivative accuracy between implicit_vjp and direct JAX modes."""

import sys
import os
sys.path.insert(0, '/workspace')

import jax.numpy as jnp
import jax
import numpy as np
from fealax.solver.newton_wrapper import NewtonSolver
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC

def create_test_problem():
    """Create a simple elasticity problem for derivative testing."""
    # Create a very small mesh for fast testing
    mesh = box_mesh(3, 3, 3, 1.0, 1.0, 1.0, ele_type="HEX8")
    
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

def finite_difference_gradient(solver, params, h=1e-5):
    """Compute gradient using finite differences."""
    def objective(p):
        sol = solver.solve(p)
        return jnp.sum(sol[0]**2)
    
    gradients = {}
    
    for param_name in params.keys():
        # Forward difference
        params_plus = params.copy()
        params_plus[param_name] = params[param_name] + h
        f_plus = objective(params_plus)
        
        # Backward difference  
        params_minus = params.copy()
        params_minus[param_name] = params[param_name] - h
        f_minus = objective(params_minus)
        
        # Central difference
        grad = (f_plus - f_minus) / (2 * h)
        gradients[param_name] = float(grad)
    
    return gradients

def test_single_approach(approach_name, use_implicit_vjp, params):
    """Test a single AD approach."""
    print(f"\n{'-'*50}")
    print(f"Testing {approach_name}")
    print(f"{'-'*50}")
    
    try:
        # Create fresh problem and solver for this test
        problem = create_test_problem()
        solver_options = {'tol': 1e-8, 'max_iter': 30}
        adjoint_options = {'use_implicit_vjp': use_implicit_vjp}
        
        solver = NewtonSolver(problem, solver_options, adjoint_options)
        
        # Test forward solve first
        print("Testing forward solve...")
        solution = solver.solve(params)
        print(f"✓ Forward solve successful, solution shape: {solution[0].shape}")
        
        # Define simple objective
        def objective(p):
            sol = solver.solve(p)
            return jnp.sum(sol[0]**2)
        
        # Test gradient computation
        print("Computing analytical gradients...")
        try:
            grad_fn = jax.grad(objective)
            gradients = grad_fn(params)
            print(f"✓ Analytical gradients computed")
            print(f"  dJ/dE = {gradients['E']:.6e}")
            print(f"  dJ/dnu = {gradients['nu']:.6e}")
            
            # Compute finite differences for comparison
            print("Computing finite difference gradients...")
            fd_gradients = finite_difference_gradient(solver, params)
            print(f"✓ Finite difference gradients computed")
            print(f"  dJ/dE (FD) = {fd_gradients['E']:.6e}")
            print(f"  dJ/dnu (FD) = {fd_gradients['nu']:.6e}")
            
            # Compare accuracy
            error_E = abs(gradients['E'] - fd_gradients['E']) / (abs(fd_gradients['E']) + 1e-12)
            error_nu = abs(gradients['nu'] - fd_gradients['nu']) / (abs(fd_gradients['nu']) + 1e-12)
            
            print(f"\nAccuracy vs finite differences:")
            print(f"  Relative error dJ/dE: {error_E:.2e}")
            print(f"  Relative error dJ/dnu: {error_nu:.2e}")
            print(f"  Total relative error: {error_E + error_nu:.2e}")
            
            return True, gradients, fd_gradients, error_E + error_nu
            
        except Exception as grad_error:
            print(f"✗ Gradient computation failed: {grad_error}")
            return False, None, None, float('inf')
            
    except Exception as e:
        print(f"✗ Approach failed: {e}")
        return False, None, None, float('inf')

def test_derivative_accuracy():
    """Compare derivative accuracy between different AD approaches."""
    print("Testing Derivative Accuracy: implicit_vjp vs Direct JAX")
    print("=" * 60)
    
    # Test parameters
    params = {'E': 200e9, 'nu': 0.3}
    
    # Test both approaches
    success_direct, grad_direct, fd_direct, error_direct = test_single_approach(
        "Direct JAX AD", use_implicit_vjp=False, params=params)
    
    success_implicit, grad_implicit, fd_implicit, error_implicit = test_single_approach(
        "Implicit VJP", use_implicit_vjp=True, params=params)
    
    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON RESULTS")
    print(f"{'='*60}")
    
    if success_direct and success_implicit:
        print("✓ Both approaches work!")
        
        print(f"\nGradient comparison:")
        print(f"{'Method':<15} {'dJ/dE':<15} {'dJ/dnu':<15} {'Total Error':<15}")
        print("-" * 60)
        print(f"{'Direct JAX':<15} {grad_direct['E']:<15.6e} {grad_direct['nu']:<15.6e} {error_direct:<15.2e}")
        print(f"{'Implicit VJP':<15} {grad_implicit['E']:<15.6e} {grad_implicit['nu']:<15.6e} {error_implicit:<15.2e}")
        
        # Check differences between methods
        diff_E = abs(grad_direct['E'] - grad_implicit['E'])
        diff_nu = abs(grad_direct['nu'] - grad_implicit['nu'])
        
        print(f"\nDifference between methods:")
        print(f"  dJ/dE difference: {diff_E:.2e}")
        print(f"  dJ/dnu difference: {diff_nu:.2e}")
        
        # Determine which is more accurate
        if error_direct < error_implicit * 0.9:
            print(f"\n✓ Direct JAX is more accurate (error: {error_direct:.2e} vs {error_implicit:.2e})")
            print(f"  Recommendation: Use default settings (direct JAX)")
        elif error_implicit < error_direct * 0.9:
            print(f"\n✓ Implicit VJP is more accurate (error: {error_implicit:.2e} vs {error_direct:.2e})")
            print(f"  Recommendation: Set adjoint_options = {{'use_implicit_vjp': True}}")
        else:
            print(f"\n= Both methods have similar accuracy")
            print(f"  Direct JAX error: {error_direct:.2e}")
            print(f"  Implicit VJP error: {error_implicit:.2e}")
            print(f"  Recommendation: Use default (direct JAX) for simplicity")
            
    elif success_direct and not success_implicit:
        print("✓ Direct JAX works, Implicit VJP fails")
        print(f"  Direct JAX accuracy: {error_direct:.2e}")
        print("  Recommendation: Use direct approach (default)")
        
    elif not success_direct and success_implicit:
        print("✓ Implicit VJP works, Direct JAX fails")
        print(f"  Implicit VJP accuracy: {error_implicit:.2e}")
        print("  Recommendation: Use implicit_vjp approach")
        
    else:
        print("✗ Both approaches failed")
        
    return success_direct, success_implicit, error_direct, error_implicit

if __name__ == "__main__":
    test_derivative_accuracy()