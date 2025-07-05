#!/usr/bin/env python3
"""Test derivative accuracy with non-zero gradients."""

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
    """Create a simple elasticity problem that will have non-zero gradients."""
    # Create a small mesh
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
    
    # Add boundary conditions - apply force dependent on material properties
    bcs = [
        DirichletBC(subdomain=lambda x: jnp.abs(x[2]) < 1e-6, vec=2, eval=lambda x: 0.0),  # Fixed bottom
        DirichletBC(subdomain=lambda x: jnp.abs(x[2] - 1.0) < 1e-6, vec=2, eval=lambda x: 0.02)  # Load top
    ]
    
    problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    return problem

def finite_difference_gradient(solver, params, objective_fn, h=1e-5):
    """Compute gradient using finite differences."""
    gradients = {}
    
    for param_name in params.keys():
        # Forward difference
        params_plus = params.copy()
        params_plus[param_name] = params[param_name] + h
        f_plus = objective_fn(solver, params_plus)
        
        # Backward difference  
        params_minus = params.copy()
        params_minus[param_name] = params[param_name] - h
        f_minus = objective_fn(solver, params_minus)
        
        # Central difference
        grad = (f_plus - f_minus) / (2 * h)
        gradients[param_name] = float(grad)
    
    return gradients

def compliance_objective(solver, params):
    """Compliance objective: u^T * K * u (related to strain energy)."""
    sol = solver.solve(params)
    # Simplified compliance - sum of displacement components
    return jnp.sum(jnp.abs(sol[0]))

def weighted_displacement_objective(solver, params):
    """Weighted displacement objective that depends on material properties."""
    sol = solver.solve(params)
    # Weight displacements by inverse stiffness
    weight = 1.0 / (params['E'] * (1 - params['nu']**2))
    return weight * jnp.sum(sol[0]**2)

def test_objective_derivatives(objective_name, objective_fn):
    """Test derivatives for a specific objective function."""
    print(f"\n{'='*70}")
    print(f"Testing objective: {objective_name}")
    print(f"{'='*70}")
    
    # Test parameters
    params = {'E': 200e9, 'nu': 0.3}
    
    results = {}
    
    # Test both approaches
    for approach_name, use_implicit_vjp in [("Direct JAX", False), ("Implicit VJP", True)]:
        print(f"\n{'-'*50}")
        print(f"Testing {approach_name}")
        print(f"{'-'*50}")
        
        try:
            # Create fresh problem and solver
            problem = create_test_problem()
            solver_options = {'tol': 1e-8, 'max_iter': 30}
            adjoint_options = {'use_implicit_vjp': use_implicit_vjp}
            
            solver = NewtonSolver(problem, solver_options, adjoint_options)
            
            # Define JAX-compatible objective
            def jax_objective(p):
                return objective_fn(solver, p)
            
            # Test forward evaluation
            print("Testing forward evaluation...")
            obj_value = objective_fn(solver, params)
            print(f"✓ Objective value: {obj_value:.6e}")
            
            # Compute analytical gradients
            print("Computing analytical gradients...")
            grad_fn = jax.grad(jax_objective)
            gradients = grad_fn(params)
            print(f"✓ Analytical gradients computed")
            print(f"  dJ/dE = {gradients['E']:.6e}")
            print(f"  dJ/dnu = {gradients['nu']:.6e}")
            
            # Compute finite differences
            print("Computing finite difference gradients...")
            fd_gradients = finite_difference_gradient(solver, params, objective_fn, h=1e-6)
            print(f"✓ Finite difference gradients computed")
            print(f"  dJ/dE (FD) = {fd_gradients['E']:.6e}")
            print(f"  dJ/dnu (FD) = {fd_gradients['nu']:.6e}")
            
            # Calculate relative errors
            error_E = abs(gradients['E'] - fd_gradients['E']) / (abs(fd_gradients['E']) + 1e-12)
            error_nu = abs(gradients['nu'] - fd_gradients['nu']) / (abs(fd_gradients['nu']) + 1e-12)
            total_error = error_E + error_nu
            
            print(f"\nAccuracy vs finite differences:")
            print(f"  Relative error dJ/dE: {error_E:.2e}")
            print(f"  Relative error dJ/dnu: {error_nu:.2e}")
            print(f"  Total relative error: {total_error:.2e}")
            
            results[approach_name] = {
                'success': True,
                'gradients': gradients,
                'fd_gradients': fd_gradients,
                'error': total_error,
                'obj_value': obj_value
            }
            
        except Exception as e:
            print(f"✗ {approach_name} failed: {e}")
            results[approach_name] = {
                'success': False,
                'error': float('inf')
            }
    
    # Compare results
    print(f"\n{'-'*50}")
    print("COMPARISON")
    print(f"{'-'*50}")
    
    if results["Direct JAX"]["success"] and results["Implicit VJP"]["success"]:
        print("✓ Both approaches work!")
        
        grad_direct = results["Direct JAX"]["gradients"]
        grad_implicit = results["Implicit VJP"]["gradients"]
        error_direct = results["Direct JAX"]["error"]
        error_implicit = results["Implicit VJP"]["error"]
        
        print(f"\nGradient comparison:")
        print(f"{'Method':<15} {'dJ/dE':<15} {'dJ/dnu':<15} {'Error':<15}")
        print("-" * 60)
        print(f"{'Direct JAX':<15} {grad_direct['E']:<15.6e} {grad_direct['nu']:<15.6e} {error_direct:<15.2e}")
        print(f"{'Implicit VJP':<15} {grad_implicit['E']:<15.6e} {grad_implicit['nu']:<15.6e} {error_implicit:<15.2e}")
        
        # Check differences between methods
        diff_E = abs(grad_direct['E'] - grad_implicit['E'])
        diff_nu = abs(grad_direct['nu'] - grad_implicit['nu'])
        
        print(f"\nDifference between methods:")
        print(f"  dJ/dE difference: {diff_E:.2e}")
        print(f"  dJ/dnu difference: {diff_nu:.2e}")
        
        # Determine winner
        if error_direct < error_implicit * 0.9:
            print(f"✓ Direct JAX is more accurate")
            return "Direct JAX"
        elif error_implicit < error_direct * 0.9:
            print(f"✓ Implicit VJP is more accurate") 
            return "Implicit VJP"
        else:
            print(f"= Similar accuracy")
            return "Tie"
    else:
        print("One or both approaches failed")
        return "Failed"

def main():
    """Test derivative accuracy for multiple objectives."""
    print("Comprehensive Derivative Accuracy Test")
    print("=" * 70)
    
    objectives = [
        ("Compliance", compliance_objective),
        ("Weighted Displacement", weighted_displacement_objective)
    ]
    
    winners = []
    
    for obj_name, obj_fn in objectives:
        winner = test_objective_derivatives(obj_name, obj_fn)
        winners.append(winner)
    
    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    direct_wins = winners.count("Direct JAX")
    implicit_wins = winners.count("Implicit VJP")
    ties = winners.count("Tie")
    failed = winners.count("Failed")
    
    print(f"Results across {len(objectives)} objectives:")
    print(f"  Direct JAX more accurate: {direct_wins}")
    print(f"  Implicit VJP more accurate: {implicit_wins}")
    print(f"  Similar accuracy: {ties}")
    print(f"  Failed tests: {failed}")
    
    if direct_wins > implicit_wins:
        print(f"\n✓ Overall recommendation: Use Direct JAX (default)")
    elif implicit_wins > direct_wins:
        print(f"\n✓ Overall recommendation: Use Implicit VJP")
        print(f"  Set: adjoint_options = {{'use_implicit_vjp': True}}")
    else:
        print(f"\n= Both methods perform similarly")
        print(f"  Recommendation: Use default (Direct JAX) for simplicity")

if __name__ == "__main__":
    main()