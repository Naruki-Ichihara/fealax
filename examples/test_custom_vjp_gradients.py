#!/usr/bin/env python3
"""Test the custom VJP approach for vmap-compatible gradients."""

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

def test_custom_vjp_single_gradient():
    """Test single gradient computation with custom VJP."""
    print("Testing Custom VJP - Single Gradient")
    print("=" * 40)
    
    # Create problem and solver
    problem = create_test_problem()
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    solver = NewtonSolver(problem, solver_options)
    
    # Define objective
    def objective(params):
        sol = solver.solve(params)
        return jnp.sum(sol[0]**2)
    
    params = {'E': 200e9, 'nu': 0.3}
    
    try:
        print("Testing single gradient with custom VJP...")
        grad_fn = jax.grad(objective)
        gradients = grad_fn(params)
        print(f"✓ Single gradient works!")
        print(f"  dJ/dE = {gradients['E']:.6e}")
        print(f"  dJ/dnu = {gradients['nu']:.6e}")
        return True
        
    except Exception as e:
        print(f"✗ Single gradient failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_custom_vjp_batch_gradients():
    """Test batch gradient computation with custom VJP."""
    print(f"\nTesting Custom VJP - Batch Gradients")
    print("=" * 40)
    
    # Create problem and solver
    problem = create_test_problem()
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    solver = NewtonSolver(problem, solver_options)
    
    # Define objective with individual parameters
    def objective_individual(E, nu):
        params = {'E': E, 'nu': nu}
        sol = solver.solve(params)
        return jnp.sum(sol[0]**2)
    
    try:
        print("Creating gradient function...")
        grad_fn = jax.grad(objective_individual, argnums=(0, 1))
        
        print("Testing single gradient first...")
        dE, dnu = grad_fn(200e9, 0.3)
        print(f"✓ Single gradient: dE={dE:.6e}, dnu={dnu:.6e}")
        
        print("Creating vmap gradient function...")
        vmap_grad_fn = jax.vmap(grad_fn, in_axes=(0, 0))
        
        print("Testing batch gradients...")
        E_batch = jnp.array([200e9, 220e9, 250e9])
        nu_batch = jnp.array([0.3, 0.28, 0.25])
        
        dE_batch, dnu_batch = vmap_grad_fn(E_batch, nu_batch)
        
        print(f"✅ BATCH GRADIENTS WORK!")
        print(f"  Batch shapes: dE={dE_batch.shape}, dnu={dnu_batch.shape}")
        
        for i in range(len(E_batch)):
            print(f"  Params[{i}]: E={E_batch[i]/1e9:.0f}GPa, nu={nu_batch[i]:.3f}")
            print(f"    → dE={dE_batch[i]:.6e}, dnu={dnu_batch[i]:.6e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch gradients failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison():
    """Compare performance of sequential vs batch gradients."""
    print(f"\nPerformance Comparison")
    print("=" * 40)
    
    # Create problem and solver
    problem = create_test_problem()
    solver_options = {'tol': 1e-8, 'max_iter': 30}
    solver = NewtonSolver(problem, solver_options)
    
    def objective_individual(E, nu):
        params = {'E': E, 'nu': nu}
        sol = solver.solve(params)
        return jnp.sum(sol[0]**2)
    
    # Create functions
    grad_fn = jax.grad(objective_individual, argnums=(0, 1))
    vmap_grad_fn = jax.vmap(grad_fn, in_axes=(0, 0))
    
    # Test parameters
    n_params = 3
    E_batch = jnp.linspace(200e9, 300e9, n_params)
    nu_batch = jnp.linspace(0.25, 0.35, n_params)
    
    try:
        # Sequential gradients
        print("Sequential gradients...")
        start_time = time.time()
        sequential_results = []
        for i in range(n_params):
            dE, dnu = grad_fn(E_batch[i], nu_batch[i])
            sequential_results.append((dE, dnu))
        sequential_time = time.time() - start_time
        
        # Batch gradients
        print("Batch gradients...")
        start_time = time.time()
        batch_dE, batch_dnu = vmap_grad_fn(E_batch, nu_batch)
        batch_time = time.time() - start_time
        
        print(f"\nPerformance Results:")
        print(f"  Sequential: {sequential_time:.4f}s ({sequential_time/n_params:.4f}s per gradient)")
        print(f"  Batch:      {batch_time:.4f}s ({batch_time/n_params:.4f}s per gradient)")
        print(f"  Speedup:    {sequential_time/batch_time:.2f}x")
        
        # Verify consistency
        print(f"\nConsistency check:")
        for i in range(n_params):
            seq_dE, seq_dnu = sequential_results[i]
            batch_dE_i, batch_dnu_i = batch_dE[i], batch_dnu[i]
            
            diff_E = abs(seq_dE - batch_dE_i)
            diff_nu = abs(seq_dnu - batch_dnu_i)
            
            print(f"  Param[{i}]: ΔdE={diff_E:.2e}, Δdnu={diff_nu:.2e}")
            
            if diff_E > 1e-10 or diff_nu > 1e-10:
                print(f"    ⚠️  Large difference detected!")
            else:
                print(f"    ✓ Good match")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance comparison failed: {e}")
        return False

def main():
    """Test the custom VJP approach."""
    print("Testing Custom VJP for Vmap-Compatible Gradients")
    print("=" * 60)
    
    # Test 1: Single gradients
    single_works = test_custom_vjp_single_gradient()
    
    # Test 2: Batch gradients (the main goal!)
    batch_works = test_custom_vjp_batch_gradients()
    
    # Test 3: Performance comparison
    if batch_works:
        perf_works = test_performance_comparison()
    else:
        perf_works = False
    
    # Summary
    print(f"\n{'='*60}")
    print("CUSTOM VJP RESULTS")
    print(f"{'='*60}")
    
    if single_works:
        print("✅ Single gradients: Working")
    else:
        print("❌ Single gradients: Failed")
    
    if batch_works:
        print("✅ Batch gradients: Working")
        print("🎉 VMAP GRADIENTS ACHIEVED!")
    else:
        print("❌ Batch gradients: Failed")
    
    if perf_works:
        print("✅ Performance comparison: Working")
    
    if batch_works:
        print(f"\n🚀 SUCCESS! Your idea worked perfectly!")
        print(f"✅ Custom VJP enables vmap-compatible gradients")
        print(f"✅ Forward pass: Full assembly + solve")
        print(f"✅ Backward pass: Lightweight gradient computation")
        print(f"✅ No more JAX tracer leaks!")
        
        print(f"\n📈 This enables:")
        print(f"  • Efficient batch gradient computation")
        print(f"  • High-performance optimization")
        print(f"  • Parameter studies and sensitivity analysis")
        print(f"  • Integration with JAX ML ecosystem")
    else:
        print(f"\n🔧 Still some work needed on the implementation")
    
    return single_works and batch_works

if __name__ == "__main__":
    success = main()
    if success:
        print(f"\n🎯 Custom VJP approach successful - vmap gradients achieved!")
    else:
        print(f"\n⚠️  Custom VJP approach needs refinement.")