#!/usr/bin/env python3
"""
Test script for the automatic vmap batch solver functionality.

This script tests the enhanced NewtonSolver that automatically detects
multiple parameter sets and applies vmap for parallel solving.
"""

import jax.numpy as jnp
import jax
import numpy as np

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import NewtonSolver


class ElasticityProblem(Problem):
    def __init__(self, mesh, **kwargs):
        self.E = None
        self.nu = None
        self.mu = None
        self.lam = None
        super().__init__(mesh=mesh, **kwargs)
    
    def set_params(self, params):
        self.E = params['E']
        self.nu = params['nu']
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
    
    def get_tensor_map(self):
        def tensor_map(u_grads, *internal_vars):
            strain = 0.5 * (u_grads + jnp.transpose(u_grads))
            trace_strain = jnp.trace(strain)
            stress = (2.0 * self.mu * strain + 
                     self.lam * trace_strain * jnp.eye(3))
            return stress
        return tensor_map


def create_test_setup():
    """Create a simple test setup."""
    # Small mesh for fast testing
    mesh = box_mesh(5, 5, 5, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    # Simple boundary conditions
    bcs = [
        DirichletBC(
            subdomain=lambda x: jnp.abs(x[2]) < 1e-6,
            vec=2,
            eval=lambda x: 0.0
        ),
        DirichletBC(
            subdomain=lambda x: jnp.abs(x[0]) < 1e-6,
            vec=0,
            eval=lambda x: 0.0
        ),
        DirichletBC(
            subdomain=lambda x: jnp.abs(x[1]) < 1e-6,
            vec=1,
            eval=lambda x: 0.0
        ),
        DirichletBC(
            subdomain=lambda x: jnp.abs(x[2] - 1.0) < 1e-6,
            vec=2,
            eval=lambda x: -0.01  # Small compression
        ),
    ]
    
    problem = ElasticityProblem(
        mesh=mesh,
        vec=3,
        dim=3,
        dirichlet_bcs=bcs
    )
    
    solver_options = {
        'tol': 1e-4,
        'rel_tol': 1e-5,
        'max_iter': 5,
        'method': 'bicgstab'
    }
    
    solver = NewtonSolver(problem, solver_options)
    
    return solver


def test_single_params():
    """Test single parameter solving (existing functionality)."""
    print("Testing single parameter solve...")
    
    solver = create_test_setup()
    
    single_params = {'E': 200e9, 'nu': 0.3}
    solution = solver.solve(single_params)
    
    print(f"  Single solve - solution shape: {solution[0].shape}")
    print(f"  Max displacement: {jnp.max(jnp.abs(solution[0])):.6f}")
    
    return solution


def test_batch_params_list():
    """Test batch parameter solving with list of dicts."""
    print("\nTesting batch parameter solve (list of dicts)...")
    
    solver = create_test_setup()
    
    batch_params = [
        {'E': 200e9, 'nu': 0.3},
        {'E': 300e9, 'nu': 0.25},
        {'E': 150e9, 'nu': 0.35}
    ]
    
    solutions = solver.solve(batch_params)
    
    print(f"  Batch solve - solutions shape: {solutions[0].shape}")
    print(f"  Batch size: {solutions[0].shape[0]}")
    print(f"  Max displacements per case:")
    for i in range(len(batch_params)):
        max_disp = jnp.max(jnp.abs(solutions[0][i]))
        E = batch_params[i]['E'] / 1e9
        print(f"    Case {i+1} (E={E:.0f} GPa): {max_disp:.6f}")
    
    return solutions


def test_batch_params_dict():
    """Test batch parameter solving with dict of arrays."""
    print("\nTesting batch parameter solve (dict of arrays)...")
    
    solver = create_test_setup()
    
    batch_params = {
        'E': jnp.array([200e9, 300e9, 150e9]),
        'nu': jnp.array([0.3, 0.25, 0.35])
    }
    
    solutions = solver.solve(batch_params)
    
    print(f"  Batch solve - solutions shape: {solutions[0].shape}")
    print(f"  Batch size: {solutions[0].shape[0]}")
    print(f"  Max displacements per case:")
    for i in range(len(batch_params['E'])):
        max_disp = jnp.max(jnp.abs(solutions[0][i]))
        E = batch_params['E'][i] / 1e9
        print(f"    Case {i+1} (E={E:.0f} GPa): {max_disp:.6f}")
    
    return solutions


def test_consistency():
    """Test that batch and single solving give consistent results."""
    print("\nTesting consistency between single and batch solving...")
    
    solver = create_test_setup()
    
    # Test parameters
    test_params = {'E': 250e9, 'nu': 0.28}
    
    # Single solve
    single_solution = solver.solve(test_params)
    
    # Batch solve with same parameters
    batch_solution = solver.solve([test_params])
    
    # Compare results
    diff = jnp.max(jnp.abs(single_solution[0] - batch_solution[0][0]))
    print(f"  Maximum difference: {diff:.2e}")
    
    if diff < 1e-10:
        print("  ✓ Results are consistent!")
    else:
        print("  ✗ Results differ significantly")
    
    return diff < 1e-10


def main():
    """Run all tests."""
    jax.config.update("jax_enable_x64", True)
    
    print("="*60)
    print("TESTING AUTOMATIC VMAP BATCH SOLVER")
    print("="*60)
    
    try:
        # Test single parameter solving
        single_solution = test_single_params()
        
        # Test batch parameter solving (list format)
        batch_solutions_list = test_batch_params_list()
        
        # Test batch parameter solving (dict format)
        batch_solutions_dict = test_batch_params_dict()
        
        # Test consistency
        consistent = test_consistency()
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print("✓ Single parameter solving: PASSED")
        print("✓ Batch solving (list of dicts): PASSED")
        print("✓ Batch solving (dict of arrays): PASSED")
        print(f"{'✓' if consistent else '✗'} Consistency test: {'PASSED' if consistent else 'FAILED'}")
        print("\nAll tests completed!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()