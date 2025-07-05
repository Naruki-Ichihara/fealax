#!/usr/bin/env python3
"""Test gradient computation of objective function containing solver.solve(params_batch)."""

import sys
import os
sys.path.insert(0, '/workspace')

import jax
import jax.numpy as jnp
import numpy as np
from fealax.solver.newton_wrapper import NewtonSolver
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC


def create_test_problem():
    """Create a simple elasticity problem for testing."""
    # Create small mesh for fast testing
    mesh = box_mesh(3, 3, 3, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    # Create elasticity problem
    class TestElasticityProblem(Problem):
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
    
    problem = TestElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    return problem


def test_single_param_grad():
    """Test gradient computation for single parameter set."""
    print("Testing gradient computation for single parameter set...")
    
    problem = create_test_problem()
    solver = NewtonSolver(problem, {'tol': 1e-6, 'max_iter': 10})
    
    def objective(params):
        """Simple objective function: sum of squared displacements."""
        solution = solver.solve(params)
        displacement = solution[0]  # First variable (displacement)
        return jnp.sum(displacement**2)
    
    # Test parameters
    params = {'E': 200e9, 'nu': 0.3}
    
    # Compute gradient
    try:
        grad_fn = jax.grad(objective)
        gradients = grad_fn(params)
        
        print(f"✓ Single parameter gradient computation successful!")
        print(f"  Gradients: {gradients}")
        print(f"  Gradient w.r.t. E: {gradients['E']:.2e}")
        print(f"  Gradient w.r.t. nu: {gradients['nu']:.2e}")
        
        # Check if gradients are finite
        all_finite = all(jnp.isfinite(v) for v in gradients.values())
        print(f"  All gradients finite: {all_finite}")
        
        return True, gradients
    except Exception as e:
        print(f"✗ Single parameter gradient computation failed: {e}")
        return False, None


def test_batch_param_grad():
    """Test gradient computation for batch parameter sets."""
    print("\nTesting gradient computation for batch parameter sets...")
    
    problem = create_test_problem()
    solver = NewtonSolver(problem, {'tol': 1e-6, 'max_iter': 10})
    
    def objective(params_batch):
        """Objective function for batch parameters: sum of all squared displacements."""
        solutions = solver.solve(params_batch)
        # solutions[0] should have shape (batch_size, n_dofs, 3)
        displacement = solutions[0]  # First variable (displacement)
        return jnp.sum(displacement**2)
    
    # Test batch parameters
    params_batch = [
        {'E': 200e9, 'nu': 0.3},
        {'E': 250e9, 'nu': 0.25}
    ]
    
    # Compute gradient
    try:
        grad_fn = jax.grad(objective)
        gradients = grad_fn(params_batch)
        
        print(f"✓ Batch parameter gradient computation successful!")
        print(f"  Gradients shape: {[{k: v.shape for k, v in g.items()} for g in gradients]}")
        
        # Check if gradients are finite
        all_finite = True
        for i, grad_dict in enumerate(gradients):
            for k, v in grad_dict.items():
                if not jnp.isfinite(v):
                    all_finite = False
                    print(f"  Non-finite gradient in batch {i}, parameter {k}: {v}")
        
        print(f"  All gradients finite: {all_finite}")
        
        return True, gradients
    except Exception as e:
        print(f"✗ Batch parameter gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_batched_dict_grad():
    """Test gradient computation for batched dictionary parameters."""
    print("\nTesting gradient computation for batched dictionary parameters...")
    
    problem = create_test_problem()
    solver = NewtonSolver(problem, {'tol': 1e-6, 'max_iter': 10})
    
    def objective(params_dict):
        """Objective function for batched dict parameters."""
        solutions = solver.solve(params_dict)
        displacement = solutions[0]  # First variable (displacement)
        return jnp.sum(displacement**2)
    
    # Test batched dictionary parameters
    params_dict = {
        'E': jnp.array([200e9, 250e9]),
        'nu': jnp.array([0.3, 0.25])
    }
    
    # Compute gradient
    try:
        grad_fn = jax.grad(objective)
        gradients = grad_fn(params_dict)
        
        print(f"✓ Batched dictionary gradient computation successful!")
        print(f"  Gradients: {gradients}")
        print(f"  Gradient w.r.t. E: {gradients['E']}")
        print(f"  Gradient w.r.t. nu: {gradients['nu']}")
        
        # Check if gradients are finite
        all_finite = all(jnp.all(jnp.isfinite(v)) for v in gradients.values())
        print(f"  All gradients finite: {all_finite}")
        
        return True, gradients
    except Exception as e:
        print(f"✗ Batched dictionary gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_compliance_objective():
    """Test a more realistic compliance objective function."""
    print("\nTesting compliance objective function...")
    
    problem = create_test_problem()
    solver = NewtonSolver(problem, {'tol': 1e-6, 'max_iter': 10})
    
    def compliance_objective(params):
        """Compliance objective: force * displacement."""
        solution = solver.solve(params)
        displacement = solution[0]  # Shape: (n_dofs, 3)
        
        # Simple compliance: sum of top surface displacement * load
        # In a real case, you'd compute the actual compliance
        top_displacement = jnp.mean(displacement[:, 2])  # Average z-displacement
        load = 0.01  # Applied load
        compliance = top_displacement * load
        
        return compliance
    
    # Test parameters
    params = {'E': 200e9, 'nu': 0.3}
    
    # Compute gradient
    try:
        grad_fn = jax.grad(compliance_objective)
        gradients = grad_fn(params)
        
        print(f"✓ Compliance objective gradient computation successful!")
        print(f"  Gradients: {gradients}")
        print(f"  Gradient w.r.t. E: {gradients['E']:.2e}")
        print(f"  Gradient w.r.t. nu: {gradients['nu']:.2e}")
        
        # Test physical meaning: compliance should decrease with increasing E
        # So gradient w.r.t. E should be negative
        dC_dE = gradients['E']
        print(f"  Physical check - dC/dE < 0: {dC_dE < 0}")
        
        return True, gradients
    except Exception as e:
        print(f"✗ Compliance objective gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    """Run all gradient tests."""
    print("=" * 60)
    print("Testing JAX gradient computation with solver.solve()")
    print("=" * 60)
    
    results = []
    
    # Test 1: Single parameter gradient
    success, grad = test_single_param_grad()
    results.append(("Single parameter", success))
    
    # Test 2: Batch parameter gradient
    success, grad = test_batch_param_grad()
    results.append(("Batch parameters", success))
    
    # Test 3: Batched dictionary gradient
    success, grad = test_batched_dict_grad()
    results.append(("Batched dictionary", success))
    
    # Test 4: Compliance objective
    success, grad = test_compliance_objective()
    results.append(("Compliance objective", success))
    
    # Summary
    print("\n" + "=" * 60)
    print("GRADIENT TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:<25} {status}")
    
    total_passed = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")
    
    if total_passed == len(results):
        print("\n✓ All gradient tests passed! JAX can differentiate through solver.solve()")
    else:
        print(f"\n⚠ {len(results) - total_passed} tests failed")
    
    return results


if __name__ == "__main__":
    main()