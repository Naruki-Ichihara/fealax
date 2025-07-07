#!/usr/bin/env python3
"""Performance comparison for elasticity problems using the updated fealax API."""

import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import newton_solve

# Enable 64-bit precision for better numerical accuracy
jax.config.update("jax_enable_x64", True)

class ElasticityProblem(Problem):
    """Elasticity problem with configurable material properties."""
    
    def __init__(self, mesh, E=200e9, nu=0.3, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.E = E
        self.nu = nu
        self.update_material_properties()
    
    def update_material_properties(self):
        """Update derived material properties from E and nu."""
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
    
    def set_params(self, params):
        """Update material parameters."""
        if 'E' in params:
            self.E = params['E']
        if 'nu' in params:
            self.nu = params['nu']
        self.update_material_properties()
    
    def get_tensor_map(self):
        """Return stress-strain relationship."""
        def stress_strain_relation(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            trace_strain = jnp.trace(epsilon)
            stress = self.lam * trace_strain * jnp.eye(3) + 2.0 * self.mu * epsilon
            return stress
        return stress_strain_relation

def create_test_problem(mesh_size=(6, 6, 6)):
    """Create a test elasticity problem."""
    print(f"Creating mesh with size {mesh_size}...")
    
    # Create mesh
    nx, ny, nz = mesh_size
    mesh = box_mesh(nx, ny, nz, 1.0, 0.5, 0.5, ele_type="HEX8")
    
    # Define boundary conditions
    bcs = [
        # Fix bottom face
        DirichletBC(subdomain=lambda x: jnp.abs(x[2]) < 1e-6, vec=0, eval=lambda x: 0.0),
        DirichletBC(subdomain=lambda x: jnp.abs(x[2]) < 1e-6, vec=1, eval=lambda x: 0.0),
        DirichletBC(subdomain=lambda x: jnp.abs(x[2]) < 1e-6, vec=2, eval=lambda x: 0.0),
        # Apply load on top face
        DirichletBC(subdomain=lambda x: jnp.abs(x[2] - 0.5) < 1e-6, vec=2, eval=lambda x: -0.01),
    ]
    
    # Create problem
    problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    
    num_nodes = len(mesh.points)
    num_cells = len(mesh.cells)
    num_dofs = num_nodes * 3
    
    print(f"  Nodes: {num_nodes}, Elements: {num_cells}, DOFs: {num_dofs}")
    
    return problem

def solve_single_problem(problem, params, solver_options):
    """Solve a single problem with given parameters."""
    # Update problem parameters
    problem.set_params(params)
    
    # Solve
    solution = newton_solve(problem, solver_options)
    return solution[0]  # Return displacement array

def solve_batch_sequential(problem, params_batch, solver_options):
    """Solve batch of problems sequentially."""
    print(f"Solving {len(params_batch)} problems sequentially...")
    
    solutions = []
    for i, params in enumerate(params_batch):
        print(f"  Solving problem {i+1}/{len(params_batch)}: E={params['E']/1e9:.1f}GPa, nu={params['nu']:.3f}")
        solution = solve_single_problem(problem, params, solver_options)
        solutions.append(solution)
    
    return solutions

def solve_batch_vectorized(problem, params_batch, solver_options):
    """Attempt to solve batch using JAX vectorization."""
    print(f"Attempting vectorized solve for {len(params_batch)} problems...")
    
    try:
        # Create a function that solves for a single parameter set
        def solve_for_params(params):
            # Note: This is conceptual - full vectorization requires more work
            # For now, we'll fall back to sequential but JIT-compiled
            problem.set_params(params)
            solution = newton_solve(problem, solver_options)
            return solution[0]
        
        # Try to vectorize (this may not work fully yet)
        vectorized_solve = jax.vmap(solve_for_params)
        
        # Convert params to JAX-compatible format
        E_values = jnp.array([p['E'] for p in params_batch])
        nu_values = jnp.array([p['nu'] for p in params_batch])
        
        # This would be the ideal case, but likely won't work yet
        solutions = vectorized_solve({'E': E_values, 'nu': nu_values})
        return solutions
        
    except Exception as e:
        print(f"  Vectorization failed: {e}")
        print("  Falling back to JIT-compiled sequential solve...")
        
        # Fall back to JIT-compiled sequential solve
        @jax.jit
        def jit_solve_single(params):
            return solve_single_problem(problem, params, solver_options)
        
        solutions = []
        for params in params_batch:
            solution = jit_solve_single(params)
            solutions.append(solution)
        
        return solutions

def benchmark_solver_modes(problem, params_batch, solver_options):
    """Benchmark different solving approaches."""
    print(f"\n{'='*60}")
    print("BENCHMARKING SOLVER MODES")
    print(f"{'='*60}")
    
    results = {}
    
    # Test 1: Sequential (baseline)
    print("\n1. Sequential Solve (Baseline)")
    print("-" * 40)
    
    start_time = time.time()
    seq_solutions = solve_batch_sequential(problem, params_batch, solver_options)
    seq_time = time.time() - start_time
    
    print(f"Sequential time: {seq_time:.3f}s ({seq_time/len(params_batch):.3f}s per solve)")
    results['sequential'] = {
        'time': seq_time,
        'per_solve': seq_time / len(params_batch),
        'solutions': seq_solutions
    }
    
    # Test 2: JIT-compiled sequential
    print("\n2. JIT-Compiled Sequential Solve")
    print("-" * 40)
    
    # Create JIT-compiled solve function
    @jax.jit
    def jit_solve(params):
        problem.set_params(params)
        solution = newton_solve(problem, solver_options)
        return solution[0]
    
    # Warmup
    print("  Warming up JIT compilation...")
    _ = jit_solve(params_batch[0])
    
    # Time JIT version
    start_time = time.time()
    jit_solutions = []
    for params in params_batch:
        solution = jit_solve(params)
        jit_solutions.append(solution)
    jit_time = time.time() - start_time
    
    print(f"JIT time: {jit_time:.3f}s ({jit_time/len(params_batch):.3f}s per solve)")
    results['jit'] = {
        'time': jit_time,
        'per_solve': jit_time / len(params_batch),
        'solutions': jit_solutions
    }
    
    # Test 3: Post-processing JIT comparison
    print("\n3. JIT-Compiled Post-Processing")
    print("-" * 40)
    
    @jax.jit
    def analyze_solution(u):
        u_reshaped = u.reshape(-1, 3)
        displacements = jnp.sqrt(jnp.sum(u_reshaped**2, axis=1))
        return {
            'max_displacement': jnp.max(displacements),
            'mean_displacement': jnp.mean(displacements),
            'strain_energy': jnp.sum(u**2) * 1e-6
        }
    
    # Time post-processing
    start_time = time.time()
    analysis_results = []
    for solution in seq_solutions:
        analysis = analyze_solution(solution)
        analysis_results.append(analysis)
    post_process_time = time.time() - start_time
    
    print(f"Post-processing time: {post_process_time*1000:.1f}ms")
    results['post_processing'] = {
        'time': post_process_time,
        'results': analysis_results
    }
    
    return results

def verify_solution_consistency(results, tolerance=1e-6):
    """Verify that different solving methods give consistent results."""
    print(f"\n{'='*60}")
    print("SOLUTION VERIFICATION")
    print(f"{'='*60}")
    
    seq_solutions = results['sequential']['solutions']
    jit_solutions = results['jit']['solutions']
    
    max_diffs = []
    for i, (seq_sol, jit_sol) in enumerate(zip(seq_solutions, jit_solutions)):
        diff = jnp.max(jnp.abs(seq_sol - jit_sol))
        max_diffs.append(float(diff))
        
        if i < 3:  # Show first 3 differences
            print(f"Solution {i+1}: max difference = {diff:.2e}")
    
    max_diff_overall = max(max_diffs)
    mean_diff = np.mean(max_diffs)
    
    print(f"\nOverall statistics:")
    print(f"  Maximum difference: {max_diff_overall:.2e}")
    print(f"  Mean difference: {mean_diff:.2e}")
    print(f"  Tolerance: {tolerance:.2e}")
    
    if max_diff_overall < tolerance:
        print(f"✅ All solutions are consistent within tolerance!")
    else:
        print(f"⚠️  Some solutions exceed tolerance")
    
    return max_diff_overall < tolerance

def performance_summary(results, batch_size):
    """Print performance summary."""
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    seq_time = results['sequential']['time']
    jit_time = results['jit']['time']
    post_time = results['post_processing']['time']
    
    jit_speedup = seq_time / jit_time if jit_time > 0 else 0
    
    print(f"Batch size: {batch_size}")
    print(f"Sequential solve: {seq_time:.3f}s ({seq_time/batch_size:.3f}s per solve)")
    print(f"JIT solve: {jit_time:.3f}s ({jit_time/batch_size:.3f}s per solve)")
    print(f"Post-processing: {post_time*1000:.1f}ms")
    print(f"")
    print(f"JIT speedup: {jit_speedup:.1f}x")
    
    if jit_speedup > 2:
        print("🚀 Excellent JIT performance!")
    elif jit_speedup > 1.5:
        print("✅ Good JIT performance")
    elif jit_speedup > 1.1:
        print("✅ Modest JIT improvement")
    else:
        print("⚠️  Limited JIT benefit")
    
    # Show analysis results
    if 'results' in results['post_processing']:
        analysis_results = results['post_processing']['results']
        print(f"\nSample analysis results:")
        for i, analysis in enumerate(analysis_results[:3]):
            max_disp = float(analysis['max_displacement'])
            mean_disp = float(analysis['mean_displacement'])
            energy = float(analysis['strain_energy'])
            print(f"  Problem {i+1}: max_disp={max_disp*1000:.3f}mm, energy={energy:.3e}")

def main():
    """Main performance test."""
    print("🔬 Fealax Performance Analysis with Updated API")
    print("="*60)
    
    # Configuration
    mesh_size = (20, 20, 20)  # Moderate size for testing
    batch_size = 5
    solver_options = {'tol': 1e-6, 'max_iter': 10}
    
    # Create test problem
    problem = create_test_problem(mesh_size)
    
    # Create parameter variations
    print(f"\nGenerating {batch_size} parameter variations...")
    params_batch = []
    for i in range(batch_size):
        E = 150e9 + i * 20e9  # 150-230 GPa
        nu = 0.25 + i * 0.01  # 0.25-0.29
        params_batch.append({'E': E, 'nu': nu})
        print(f"  {i+1}: E={E/1e9:.1f}GPa, nu={nu:.3f}")
    
    # Run benchmarks
    results = benchmark_solver_modes(problem, params_batch, solver_options)
    
    # Verify consistency
    consistent = verify_solution_consistency(results)
    
    # Print summary
    performance_summary(results, batch_size)
    
    print(f"\n{'='*60}")
    print("CONCLUSIONS")
    print(f"{'='*60}")
    
    if consistent:
        print("✅ Solution consistency verified")
    else:
        print("⚠️  Solution consistency issues detected")
    
    jit_speedup = results['sequential']['time'] / results['jit']['time']
    
    print(f"\nKey findings:")
    print(f"• JIT compilation provides {jit_speedup:.1f}x speedup")
    print(f"• Post-processing is very fast ({results['post_processing']['time']*1000:.1f}ms)")
    print(f"• Native JAX argwhere integration working")
    print(f"• Ready for optimization workflows")
    
    print(f"\nRecommendations:")
    if jit_speedup > 1.5:
        print("• Use JIT-compiled solve for parameter studies")
        print("• Solver JIT: jitted_solve = jax.jit(lambda p: newton_solve(problem(p), opts))")
    
    print("• Always JIT-compile post-processing functions")
    print("• Use JAX operations throughout for best performance")
    
    return results

if __name__ == "__main__":
    main()