#!/usr/bin/env python3
"""Test script to compare derivative accuracy between implicit_vjp and direct JAX modes."""

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
    # Create a small mesh for fast testing
    mesh = box_mesh(4, 4, 4, 1.0, 1.0, 1.0, ele_type="HEX8")
    
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
        DirichletBC(subdomain=lambda x: jnp.abs(x[2] - 1.0) < 1e-6, vec=2, eval=lambda x: 0.02)  # Load top
    ]
    
    problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    return problem

def create_objective_function(solver):
    """Create various objective functions to test derivatives."""
    
    def displacement_norm(params):
        """Objective: L2 norm of displacements."""
        sol = solver.solve(params)
        return jnp.sum(sol[0]**2)
    
    def max_displacement(params):
        """Objective: Maximum displacement magnitude."""
        sol = solver.solve(params)
        disp_mag = jnp.sqrt(jnp.sum(sol[0]**2, axis=1))
        return jnp.max(disp_mag)
    
    def tip_displacement(params):
        """Objective: Displacement at a specific point (tip of the structure)."""
        sol = solver.solve(params)
        # Take displacement at the last node (tip)
        return jnp.sum(sol[0][-1, :]**2)
    
    def strain_energy(params):
        """Objective: Strain energy (more complex objective)."""
        sol = solver.solve(params)
        # Simplified strain energy calculation
        return 0.5 * params['E'] * jnp.sum(sol[0]**2) / (1 + params['nu'])
    
    return {
        'displacement_norm': displacement_norm,
        'max_displacement': max_displacement,
        'tip_displacement': tip_displacement,
        'strain_energy': strain_energy
    }

def finite_difference_gradient(objective, params, h=1e-6):
    """Compute gradient using finite differences."""
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

def test_derivative_accuracy():
    """Compare derivative accuracy between different AD approaches."""
    print("Testing Derivative Accuracy: implicit_vjp vs Direct JAX vs Finite Differences")
    print("=" * 80)
    
    # Test parameters
    params = {'E': 200e9, 'nu': 0.3}
    
    # Create problem
    problem = create_test_problem()
    
    # Create solvers with different AD approaches
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    
    print("Setting up solvers...")
    solver_direct = NewtonSolver(problem, solver_options, {'use_implicit_vjp': False})
    solver_implicit = NewtonSolver(problem, solver_options, {'use_implicit_vjp': True})
    
    # Create objective functions
    objectives_direct = create_objective_function(solver_direct)
    objectives_implicit = create_objective_function(solver_implicit)
    
    results = {}
    
    # Test each objective function
    for obj_name in objectives_direct.keys():
        print(f"\n{'='*60}")
        print(f"Testing objective: {obj_name}")
        print(f"{'='*60}")
        
        obj_direct = objectives_direct[obj_name]
        obj_implicit = objectives_implicit[obj_name]
        
        # Compute gradients using different methods
        print("Computing gradients...")
        
        # 1. Direct JAX AD
        print("  - Direct JAX AD...")
        grad_direct_fn = jax.grad(obj_direct)
        grad_direct = grad_direct_fn(params)
        
        # 2. Implicit VJP
        print("  - Implicit VJP...")
        grad_implicit_fn = jax.grad(obj_implicit)
        grad_implicit = grad_implicit_fn(params)
        
        # 3. Finite differences (reference)
        print("  - Finite differences (reference)...")
        grad_fd = finite_difference_gradient(obj_direct, params, h=1e-6)
        
        # Store results
        results[obj_name] = {
            'direct': grad_direct,
            'implicit': grad_implicit,
            'finite_diff': grad_fd
        }
        
        # Compare results
        print(f"\nResults for {obj_name}:")
        print(f"{'Method':<15} {'dJ/dE':<15} {'dJ/dnu':<15}")
        print("-" * 45)
        print(f"{'Direct JAX':<15} {grad_direct['E']:<15.6e} {grad_direct['nu']:<15.6e}")
        print(f"{'Implicit VJP':<15} {grad_implicit['E']:<15.6e} {grad_implicit['nu']:<15.6e}")
        print(f"{'Finite Diff':<15} {grad_fd['E']:<15.6e} {grad_fd['nu']:<15.6e}")
        
        # Compute errors relative to finite differences
        error_direct_E = abs(grad_direct['E'] - grad_fd['E']) / (abs(grad_fd['E']) + 1e-12)
        error_direct_nu = abs(grad_direct['nu'] - grad_fd['nu']) / (abs(grad_fd['nu']) + 1e-12)
        error_implicit_E = abs(grad_implicit['E'] - grad_fd['E']) / (abs(grad_fd['E']) + 1e-12)
        error_implicit_nu = abs(grad_implicit['nu'] - grad_fd['nu']) / (abs(grad_fd['nu']) + 1e-12)
        
        print(f"\nRelative errors vs finite differences:")
        print(f"{'Method':<15} {'Error dJ/dE':<15} {'Error dJ/dnu':<15}")
        print("-" * 45)
        print(f"{'Direct JAX':<15} {error_direct_E:<15.2e} {error_direct_nu:<15.2e}")
        print(f"{'Implicit VJP':<15} {error_implicit_E:<15.2e} {error_implicit_nu:<15.2e}")
        
        # Determine which is more accurate
        total_error_direct = error_direct_E + error_direct_nu
        total_error_implicit = error_implicit_E + error_implicit_nu
        
        if total_error_direct < total_error_implicit:
            print(f"✓ Direct JAX is more accurate (total rel. error: {total_error_direct:.2e})")
        elif total_error_implicit < total_error_direct:
            print(f"✓ Implicit VJP is more accurate (total rel. error: {total_error_implicit:.2e})")
        else:
            print(f"= Both methods have similar accuracy")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL ACCURACY SUMMARY")
    print(f"{'='*80}")
    
    # Count wins for each method
    direct_wins = 0
    implicit_wins = 0
    ties = 0
    
    for obj_name, result in results.items():
        grad_direct = result['direct']
        grad_implicit = result['implicit']
        grad_fd = result['finite_diff']
        
        # Compute total relative errors
        error_direct = (abs(grad_direct['E'] - grad_fd['E']) / (abs(grad_fd['E']) + 1e-12) + 
                       abs(grad_direct['nu'] - grad_fd['nu']) / (abs(grad_fd['nu']) + 1e-12))
        error_implicit = (abs(grad_implicit['E'] - grad_fd['E']) / (abs(grad_fd['E']) + 1e-12) + 
                         abs(grad_implicit['nu'] - grad_fd['nu']) / (abs(grad_fd['nu']) + 1e-12))
        
        if error_direct < error_implicit * 0.95:  # 5% tolerance
            direct_wins += 1
        elif error_implicit < error_direct * 0.95:
            implicit_wins += 1
        else:
            ties += 1
    
    print(f"Accuracy comparison across {len(results)} objective functions:")
    print(f"  Direct JAX more accurate: {direct_wins}")
    print(f"  Implicit VJP more accurate: {implicit_wins}")
    print(f"  Similar accuracy: {ties}")
    
    if direct_wins > implicit_wins:
        print(f"\n✓ Overall winner: Direct JAX AD")
        print(f"  Recommendation: Use default settings (direct JAX)")
    elif implicit_wins > direct_wins:
        print(f"\n✓ Overall winner: Implicit VJP")
        print(f"  Recommendation: Set adjoint_options = {{'use_implicit_vjp': True}}")
    else:
        print(f"\n= Both methods have similar overall accuracy")
        print(f"  Recommendation: Use default (direct JAX) for simplicity")
    
    return results

if __name__ == "__main__":
    results = test_derivative_accuracy()