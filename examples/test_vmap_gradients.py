#!/usr/bin/env python3
"""Test if gradient computation is vmappable with the simplified solver."""

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

def test_vmap_gradients():
    """Test if gradient computation can be vmapped."""
    print("Testing Vmap Gradient Computation")
    print("=" * 50)
    
    # Create problem and solver
    problem = create_test_problem()
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    solver = NewtonSolver(problem, solver_options)
    
    # Define objective function
    def objective(params):
        sol = solver.solve(params)
        return jnp.sum(jnp.abs(sol[0]))  # L1 norm of displacements
    
    # Create gradient function
    grad_fn = jax.grad(objective)
    
    # Test single gradient computation
    print("1. Testing single gradient computation...")
    single_params = {'E': 200e9, 'nu': 0.3}
    
    try:
        single_grad = grad_fn(single_params)
        print(f"✓ Single gradient successful")
        print(f"  dJ/dE = {single_grad['E']:.6e}")
        print(f"  dJ/dnu = {single_grad['nu']:.6e}")
    except Exception as e:
        print(f"✗ Single gradient failed: {e}")
        return False
    
    # Test batch gradient computation using vmap
    print("\n2. Testing batch gradient computation with vmap...")
    
    # Create batch of parameters
    batch_params = {
        'E': jnp.array([200e9, 250e9, 300e9]),
        'nu': jnp.array([0.3, 0.28, 0.25])
    }
    
    try:
        # Create vmapped gradient function
        # vmap over the parameter dictionary
        vmap_grad_fn = jax.vmap(grad_fn)
        
        print("  Created vmap gradient function...")
        
        # Compute batch gradients
        batch_gradients = vmap_grad_fn(batch_params)
        
        print(f"✓ Batch gradient computation successful!")
        print(f"  Batch shape - dJ/dE: {batch_gradients['E'].shape}")
        print(f"  Batch shape - dJ/dnu: {batch_gradients['nu'].shape}")
        
        print(f"\n  Batch gradients:")
        for i in range(len(batch_params['E'])):
            E_val = batch_params['E'][i]
            nu_val = batch_params['nu'][i]
            dE = batch_gradients['E'][i]
            dnu = batch_gradients['nu'][i]
            print(f"    Params[{i}]: E={E_val:.0e}, nu={nu_val:.3f}")
            print(f"      → dJ/dE={dE:.6e}, dJ/dnu={dnu:.6e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vmap_gradients_alternative():
    """Test alternative vmap approach - vmap over individual parameters."""
    print("\n3. Testing alternative vmap approach...")
    
    # Create problem and solver
    problem = create_test_problem()
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    solver = NewtonSolver(problem, solver_options)
    
    # Define objective function that takes individual parameters
    def objective_individual(E, nu):
        params = {'E': E, 'nu': nu}
        sol = solver.solve(params)
        return jnp.sum(jnp.abs(sol[0]))
    
    try:
        # Create gradient function for individual parameters
        grad_fn = jax.grad(objective_individual, argnums=(0, 1))
        
        # Create vmapped version
        vmap_grad_fn = jax.vmap(grad_fn, in_axes=(0, 0))
        
        # Test parameters
        E_values = jnp.array([200e9, 250e9, 300e9])
        nu_values = jnp.array([0.3, 0.28, 0.25])
        
        # Compute batch gradients
        dE_batch, dnu_batch = vmap_grad_fn(E_values, nu_values)
        
        print(f"✓ Alternative vmap approach successful!")
        print(f"  dE batch shape: {dE_batch.shape}")
        print(f"  dnu batch shape: {dnu_batch.shape}")
        
        print(f"\n  Alternative batch gradients:")
        for i in range(len(E_values)):
            print(f"    Params[{i}]: E={E_values[i]:.0e}, nu={nu_values[i]:.3f}")
            print(f"      → dJ/dE={dE_batch[i]:.6e}, dJ/dnu={dnu_batch[i]:.6e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Alternative vmap approach failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_higher_order_vmap():
    """Test higher-order transformations: vmap of grad of solve."""
    print("\n4. Testing higher-order transformations...")
    
    # Create problem and solver
    problem = create_test_problem()
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    solver = NewtonSolver(problem, solver_options)
    
    def objective(params):
        sol = solver.solve(params)
        return jnp.sum(sol[0]**2)  # L2 norm squared
    
    try:
        # Test: vmap(grad(solve))
        grad_fn = jax.grad(objective)
        vmap_grad_fn = jax.vmap(grad_fn)
        
        # Test: grad(vmap(solve)) - probably won't work but let's see
        def batch_objective(batch_params):
            batch_sol = solver.solve(batch_params)  # This uses vmap internally
            return jnp.sum(batch_sol[0]**2, axis=(1, 2))  # Sum over each solution
        
        # Test parameters
        batch_params = [
            {'E': 200e9, 'nu': 0.3},
            {'E': 250e9, 'nu': 0.28},
            {'E': 300e9, 'nu': 0.25}
        ]
        
        print("  Testing vmap(grad(solve))...")
        gradients = vmap_grad_fn(batch_params)
        print(f"✓ vmap(grad(solve)) works!")
        print(f"  Gradient shapes: dE={gradients['E'].shape}, dnu={gradients['nu'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Higher-order transformations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all vmap gradient tests."""
    print("Testing Vmap Compatibility of Gradient Computation")
    print("=" * 60)
    
    results = []
    
    # Test 1: Basic vmap gradients
    results.append(test_vmap_gradients())
    
    # Test 2: Alternative vmap approach  
    results.append(test_vmap_gradients_alternative())
    
    # Test 3: Higher-order transformations
    results.append(test_higher_order_vmap())
    
    # Summary
    print(f"\n{'='*60}")
    print("VMAP GRADIENT TEST SUMMARY")
    print(f"{'='*60}")
    
    success_count = sum(results)
    total_tests = len(results)
    
    print(f"Successful tests: {success_count}/{total_tests}")
    
    if success_count == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Gradient computation is fully vmappable!")
        print("✅ This enables efficient batch gradient computation")
        print("✅ Perfect for optimization and parameter studies")
    elif success_count > 0:
        print("⚠️  Some tests passed - partial vmap support")
        print("✅ Basic functionality works")
    else:
        print("❌ No vmap support for gradients")
    
    return success_count == total_tests

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🚀 Ready for high-performance batch gradient computations!")
    else:
        print("\n🔧 Some limitations found in vmap gradient support.")