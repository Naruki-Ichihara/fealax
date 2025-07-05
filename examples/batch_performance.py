#!/usr/bin/env python3
"""Performance comparison between vmap mode and sequential mode for elasticity problems."""

import sys
import os
sys.path.insert(0, '/workspace')

import time
import numpy as np
import jax.numpy as jnp
from fealax.solver.newton_wrapper import NewtonSolver
from fealax.mesh import Mesh
from fealax.fe import FiniteElement
from fealax.problem import Problem, DirichletBC


def create_elasticity_problem():
    """Create a simple elasticity problem for performance testing."""
    # Create a smaller mesh for faster iterations using box_mesh
    from fealax.mesh import box_mesh
    mesh = box_mesh(30, 30, 30, 1.0, 1.0, 1.0, ele_type="HEX8")  # Small 3x3x3 mesh for speed
    
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
        DirichletBC(subdomain=lambda x: jnp.abs(x[2] - 1.0) < 1e-6, vec=2, eval=lambda x: 0.05)  # Load top
    ]
    
    problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    
    return problem


def time_solver_mode(solver, params_batch, mode_name):
    """Time a solver in a specific mode."""
    print(f"\n{'='*50}")
    print(f"Testing {mode_name} mode with batch size {len(params_batch)}")
    print(f"{'='*50}")
    
    # Warm up with a single solve
    print("Warming up...")
    solver.solve(params_batch[0])
    
    # Time the batch solve
    print(f"Running {mode_name} batch solve...")
    start_time = time.time()
    solutions = solver.solve(params_batch)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    time_per_solve = elapsed_time / len(params_batch)
    
    print(f"✓ {mode_name} completed:")
    print(f"  Total time: {elapsed_time:.4f} seconds")
    print(f"  Time per solve: {time_per_solve:.4f} seconds")
    print(f"  Solutions shape: {[sol.shape for sol in solutions]}")
    
    return elapsed_time, time_per_solve, solutions


def force_sequential_mode(solver, params_batch):
    """Force sequential mode by using solve_batch method."""
    print(f"\n{'='*50}")
    print(f"Testing SEQUENTIAL mode (forced) with batch size {len(params_batch)}")
    print(f"{'='*50}")
    
    # Warm up
    print("Warming up...")
    solver.solve(params_batch[0])
    
    # Time the sequential batch solve
    print("Running sequential batch solve...")
    start_time = time.time()
    solutions = solver.solve_batch(params_batch)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    time_per_solve = elapsed_time / len(params_batch)
    
    print(f"✓ Sequential completed:")
    print(f"  Total time: {elapsed_time:.4f} seconds")
    print(f"  Time per solve: {time_per_solve:.4f} seconds")
    print(f"  Solutions count: {len(solutions)}")
    
    return elapsed_time, time_per_solve, solutions


def test_batch_size(batch_size):
    """Test performance for a specific batch size."""
    print(f"\n{'#'*60}")
    print(f"BATCH SIZE: {batch_size}")
    print(f"{'#'*60}")
    
    # Create problem
    problem = create_elasticity_problem()
    solver = NewtonSolver(problem, {'tol': 1e-6, 'max_iter': 20})
    
    # Create parameter batch with varying E and nu values
    params_batch = []
    for i in range(batch_size):
        E = 200e9 + i * 10e9  # Vary E from 200 to 200+10*batch_size GPa
        nu = 0.3 + i * 0.01   # Vary nu from 0.3 to 0.3+0.01*batch_size
        params_batch.append({'E': E, 'nu': nu})
    
    # Test automatic mode (should use vmap for batch_size > 1)
    auto_time, auto_per_solve, auto_solutions = time_solver_mode(solver, params_batch, "AUTO")
    
    # Test forced sequential mode
    seq_time, seq_per_solve, seq_solutions = force_sequential_mode(solver, params_batch)
    
    # Calculate speedup
    speedup = seq_time / auto_time if auto_time > 0 else 0
    
    print(f"\n{'*'*50}")
    print(f"PERFORMANCE SUMMARY (Batch Size: {batch_size})")
    print(f"{'*'*50}")
    print(f"Auto mode:       {auto_time:.4f}s ({auto_per_solve:.4f}s per solve)")
    print(f"Sequential mode: {seq_time:.4f}s ({seq_per_solve:.4f}s per solve)")
    print(f"Speedup:         {speedup:.2f}x")
    
    if speedup > 1.1:
        print("✓ Auto mode (vmap) is faster!")
    elif speedup < 0.9:
        print("⚠ Sequential mode is faster!")
    else:
        print("≈ Performance is similar")
    
    # Verify solutions are numerically similar
    if len(auto_solutions) > 0 and len(seq_solutions) > 0:
        # Compare first solution
        auto_sol = auto_solutions[0]
        seq_sol = seq_solutions[0][0]  # Sequential returns list of lists
        
        max_diff = float(np.max(np.abs(auto_sol - seq_sol)))
        print(f"Max solution difference: {max_diff:.2e}")
        
        if max_diff < 1e-10:
            print("✓ Solutions are numerically identical")
        elif max_diff < 1e-6:
            print("✓ Solutions are numerically similar")
        else:
            print("⚠ Solutions differ significantly!")
    
    return {
        'batch_size': batch_size,
        'auto_time': auto_time,
        'seq_time': seq_time,
        'speedup': speedup,
        'auto_per_solve': auto_per_solve,
        'seq_per_solve': seq_per_solve
    }


def main():
    """Main performance comparison."""
    print("Performance Comparison: vmap vs Sequential Mode")
    print("=" * 60)
    
    # Test different batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 20]
    results = []
    
    for batch_size in batch_sizes:
        try:
            result = test_batch_size(batch_size)
            results.append(result)
        except Exception as e:
            print(f"Error with batch size {batch_size}: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"{'Batch Size':<10} {'Auto (s)':<10} {'Sequential (s)':<13} {'Speedup':<8}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['batch_size']:<10} {result['auto_time']:<10.4f} {result['seq_time']:<13.4f} {result['speedup']:<8.2f}x")
    
    # Analysis
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")
    
    if len(results) > 1:
        avg_speedup = np.mean([r['speedup'] for r in results if r['batch_size'] > 1])
        max_speedup = np.max([r['speedup'] for r in results if r['batch_size'] > 1])
        
        print(f"Average speedup for batch sizes > 1: {avg_speedup:.2f}x")
        print(f"Maximum speedup: {max_speedup:.2f}x")
        
        if avg_speedup > 1.2:
            print("✓ vmap mode shows significant performance benefits for batch operations")
        elif avg_speedup > 1.05:
            print("✓ vmap mode shows modest performance benefits")
        else:
            print("≈ Performance is similar between modes")
    
    print("\nNote: Performance depends on:")
    print("- Problem size (mesh complexity)")
    print("- Hardware (CPU vs GPU)")
    print("- JAX compilation overhead")
    print("- Memory bandwidth")


if __name__ == "__main__":
    main()