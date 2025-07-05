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
from fealax.utils import save_as_vtk
import meshio


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


def verify_convergence(solutions, params_batch, mode_name):
    """Verify that all solutions in the batch converged properly."""
    print(f"\n{'='*30}")
    print(f"Verifying {mode_name} Solutions")
    print(f"{'='*30}")
    
    all_converged = True
    convergence_details = []
    
    # Handle different solution formats
    if mode_name == "AUTO" and len(solutions) == 1 and solutions[0].ndim == 3:
        # vmap mode returns a single array with shape (batch_size, num_dofs, vec_dim)
        batch_solutions = solutions[0]
        for i, params in enumerate(params_batch):
            sol_array = batch_solutions[i]
            
            has_nan = np.any(np.isnan(sol_array))
            has_inf = np.any(np.isinf(sol_array))
            max_val = float(np.max(np.abs(sol_array)))
            mean_val = float(np.mean(np.abs(sol_array)))
            
            converged = not has_nan and not has_inf and max_val < 1e3
            all_converged &= converged
            
            convergence_details.append({
                'index': i,
                'E': params['E'],
                'nu': params['nu'],
                'converged': converged,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'max_abs_value': max_val,
                'mean_abs_value': mean_val
            })
            
            if not converged:
                print(f"  ⚠ Solution {i} FAILED: E={params['E']:.2e}, nu={params['nu']:.3f}")
                print(f"    - NaN: {has_nan}, Inf: {has_inf}, Max: {max_val:.2e}")
    else:
        # Sequential mode or single solve
        for i, (sol, params) in enumerate(zip(solutions, params_batch)):
            # Check if solution has reasonable values (not NaN or Inf)
            if isinstance(sol, list):
                sol_array = sol[0]  # For sequential mode which returns list of lists
            else:
                sol_array = sol
                
            has_nan = np.any(np.isnan(sol_array))
            has_inf = np.any(np.isinf(sol_array))
            max_val = float(np.max(np.abs(sol_array)))
            mean_val = float(np.mean(np.abs(sol_array)))
            
            converged = not has_nan and not has_inf and max_val < 1e3
            all_converged &= converged
            
            convergence_details.append({
                'index': i,
                'E': params['E'],
                'nu': params['nu'],
                'converged': converged,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'max_abs_value': max_val,
                'mean_abs_value': mean_val
            })
            
            if not converged:
                print(f"  ⚠ Solution {i} FAILED: E={params['E']:.2e}, nu={params['nu']:.3f}")
                print(f"    - NaN: {has_nan}, Inf: {has_inf}, Max: {max_val:.2e}")
        
    converged_count = sum(1 for d in convergence_details if d['converged'])
    print(f"\nConvergence Summary:")
    print(f"  ✓ Converged: {converged_count}/{len(params_batch)}")
    print(f"  ✗ Failed: {len(params_batch) - converged_count}/{len(params_batch)}")
    
    if all_converged:
        print(f"  ✓ All {mode_name} solutions converged successfully!")
    
    return all_converged, convergence_details


def compare_solutions_detailed(auto_solutions, seq_solutions, params_batch):
    """Detailed comparison between auto (vmap) and sequential solutions."""
    print(f"\n{'='*50}")
    print("DETAILED SOLUTION COMPARISON")
    print(f"{'='*50}")
    
    differences = []
    
    # Handle different solution formats for vmap mode
    if len(auto_solutions) == 1 and auto_solutions[0].ndim == 3:
        # vmap mode returns a single array with shape (batch_size, num_dofs, vec_dim)
        batch_auto_solutions = auto_solutions[0]
    else:
        batch_auto_solutions = auto_solutions
    
    for i in range(len(params_batch)):
        # Get sequential solution
        if isinstance(seq_solutions[i], list):
            seq_sol = seq_solutions[i][0]
        else:
            seq_sol = seq_solutions[i]
        
        # Get auto solution
        if len(auto_solutions) == 1 and auto_solutions[0].ndim == 3:
            auto_sol = batch_auto_solutions[i]
        else:
            auto_sol = auto_solutions[i]
        
        # Compute various difference metrics
        abs_diff = np.abs(auto_sol - seq_sol)
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))
        rel_diff = abs_diff / (np.abs(seq_sol) + 1e-10)
        max_rel_diff = float(np.max(rel_diff))
        
        differences.append({
            'index': i,
            'max_abs_diff': max_diff,
            'mean_abs_diff': mean_diff,
            'max_rel_diff': max_rel_diff,
            'E': params_batch[i]['E'],
            'nu': params_batch[i]['nu']
        })
        
    # Print summary statistics
    max_diffs = [d['max_abs_diff'] for d in differences]
    mean_diffs = [d['mean_abs_diff'] for d in differences]
    rel_diffs = [d['max_rel_diff'] for d in differences]
    
    print(f"\nDifference Statistics:")
    print(f"  Maximum absolute difference:")
    print(f"    - Max: {np.max(max_diffs):.2e}")
    print(f"    - Mean: {np.mean(max_diffs):.2e}")
    print(f"    - Min: {np.min(max_diffs):.2e}")
    
    print(f"\n  Mean absolute difference:")
    print(f"    - Max: {np.max(mean_diffs):.2e}")
    print(f"    - Mean: {np.mean(mean_diffs):.2e}")
    print(f"    - Min: {np.min(mean_diffs):.2e}")
    
    print(f"\n  Maximum relative difference:")
    print(f"    - Max: {np.max(rel_diffs):.2e}")
    print(f"    - Mean: {np.mean(rel_diffs):.2e}")
    print(f"    - Min: {np.min(rel_diffs):.2e}")
    
    # Check consistency
    tolerance = 1e-6
    consistent = all(d['max_abs_diff'] < tolerance for d in differences)
    
    if consistent:
        print(f"\n✓ All solutions are consistent within tolerance {tolerance}")
    else:
        inconsistent_count = sum(1 for d in differences if d['max_abs_diff'] >= tolerance)
        print(f"\n⚠ {inconsistent_count} solutions exceed tolerance {tolerance}")
        
        # Show worst cases
        worst_cases = sorted(differences, key=lambda x: x['max_abs_diff'], reverse=True)[:3]
        print("\nWorst cases:")
        for case in worst_cases:
            print(f"  - Index {case['index']}: max_diff={case['max_abs_diff']:.2e}, E={case['E']:.2e}, nu={case['nu']:.3f}")
    
    return differences


def save_vtu_output(mesh_data, solutions, params_batch, mode_name, batch_size):
    """Save VTU files for visualization of batch solutions."""
    print(f"\n{'='*30}")
    print(f"Saving VTU Output for {mode_name}")
    print(f"{'='*30}")
    
    output_dir = f"batch_output_{mode_name.lower()}_size{batch_size}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle different solution formats
    if mode_name == "AUTO" and len(solutions) == 1 and solutions[0].ndim == 3:
        # vmap mode returns a single array with shape (batch_size, num_dofs, vec_dim)
        batch_solutions = solutions[0]
        solution_list = [batch_solutions[i] for i in range(batch_size)]
    else:
        # Sequential mode
        solution_list = []
        for sol in solutions:
            if isinstance(sol, list):
                solution_list.append(sol[0])
            else:
                solution_list.append(sol)
    
    # Save each solution
    for i, (sol, params) in enumerate(zip(solution_list, params_batch)):
        # Create mesh data for meshio using saved mesh data
        points = mesh_data['points']
        cells = [("hexahedron", mesh_data['cells'])]
        
        # Prepare point data
        point_data = {
            "displacement": sol,
            "displacement_magnitude": np.linalg.norm(sol, axis=1)
        }
        
        # Prepare cell data showing material properties
        E_array = np.full(len(mesh_data['cells']), params['E'])
        nu_array = np.full(len(mesh_data['cells']), params['nu'])
        
        cell_data = {
            "E": [E_array],
            "nu": [nu_array]
        }
        
        # Create mesh object
        mesh_out = meshio.Mesh(
            points=points,
            cells=cells,
            point_data=point_data,
            cell_data=cell_data
        )
        
        # Save to VTU file
        filename = os.path.join(output_dir, f"solution_{i:03d}_E{params['E']:.0e}_nu{params['nu']:.3f}.vtu")
        mesh_out.write(filename)
        
        if i < 3:  # Show first 3 filenames
            print(f"  Saved: {filename}")
    
    if batch_size > 3:
        print(f"  ... and {batch_size - 3} more files")
    
    print(f"\n✓ Saved {batch_size} VTU files to '{output_dir}/'")
    
    return output_dir


def save_vtu_output_using_fealax(problem, solutions, params_batch, mode_name, batch_size):
    """Save VTU files using fealax.utils.save_as_vtk function."""
    print(f"\n{'='*30}")
    print(f"Saving VTU Output for {mode_name} (using fealax.utils.save_as_vtk)")
    print(f"{'='*30}")
    
    output_dir = f"batch_output_{mode_name.lower()}_size{batch_size}_fealax"
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle mesh object which might be a list
    if isinstance(problem.mesh, list):
        mesh_obj = problem.mesh[0]
    else:
        mesh_obj = problem.mesh
    
    # Create a fresh finite element object with the original mesh
    fe = FiniteElement(mesh_obj, vec=3, dim=3)
    
    # Handle different solution formats
    if mode_name == "AUTO" and len(solutions) == 1 and solutions[0].ndim == 3:
        # vmap mode returns a single array with shape (batch_size, num_dofs, vec_dim)
        batch_solutions = solutions[0]
        solution_list = [batch_solutions[i] for i in range(batch_size)]
    else:
        # Sequential mode
        solution_list = []
        for sol in solutions:
            if isinstance(sol, list):
                solution_list.append(sol[0])
            else:
                solution_list.append(sol)
    
    # Save each solution
    for i, (sol, params) in enumerate(zip(solution_list, params_batch)):
        # Create point data
        point_infos = [("displacement", sol)]
        
        # Create cell data showing material properties
        E_array = np.full(fe.num_cells, params['E'])
        nu_array = np.full(fe.num_cells, params['nu'])
        cell_infos = [("E", E_array), ("nu", nu_array)]
        
        # Save using fealax built-in function
        filename = os.path.join(output_dir, f"solution_{i:03d}_E{params['E']:.0e}_nu{params['nu']:.3f}.vtu")
        save_as_vtk(fe, filename, cell_infos=cell_infos, point_infos=point_infos)
        
        print(f"  Saved: {filename}")
    
    print(f"\n✓ Saved {len(solution_list)} VTU files to '{output_dir}/' using fealax.utils.save_as_vtk")
    
    return output_dir


def test_batch_size(batch_size):
    """Test performance for a specific batch size."""
    print(f"\n{'#'*60}")
    print(f"BATCH SIZE: {batch_size}")
    print(f"{'#'*60}")
    
    # Create problem
    problem = create_elasticity_problem()
    
    # Handle mesh object which might be a list
    if isinstance(problem.mesh, list):
        mesh_obj = problem.mesh[0]  # Use first mesh in list
    else:
        mesh_obj = problem.mesh
    
    # Save mesh data before solver potentially modifies it
    mesh_points = np.array(mesh_obj.points)
    mesh_cells = np.array(mesh_obj.cells)
    mesh_data = {'points': mesh_points, 'cells': mesh_cells}
    
    solver = NewtonSolver(problem, {'tol': 1e-6, 'max_iter': 20})
    
    # Create parameter batch with varying E and nu values
    params_batch = []
    for i in range(batch_size):
        E = 200e9 + i * 1e7  # Vary E from 200 to 200+batch_size*0.1 GPa
        nu = 0.3 + i * 0.001   # Vary nu from 0.3 to 0.3+0.01*batch_size
        params_batch.append({'E': E, 'nu': nu})
    
    # Test automatic mode (should use vmap for batch_size > 1)
    auto_time, auto_per_solve, auto_solutions = time_solver_mode(solver, params_batch, "AUTO")
    
    # Test forced sequential mode
    seq_time, seq_per_solve, seq_solutions = force_sequential_mode(solver, params_batch)
    
    # Verify convergence for both modes
    auto_converged, auto_details = verify_convergence(auto_solutions, params_batch, "AUTO")
    seq_converged, seq_details = verify_convergence(seq_solutions, params_batch, "SEQUENTIAL")
    
    # Detailed comparison
    differences = compare_solutions_detailed(auto_solutions, seq_solutions, params_batch)
    
    # Save VTU outputs for both modes
    auto_output_dir = save_vtu_output(mesh_data, auto_solutions, params_batch, "AUTO", batch_size)
    seq_output_dir = save_vtu_output(mesh_data, seq_solutions, params_batch, "SEQUENTIAL", batch_size)
    
    # Also save using fealax built-in utility for comparison
    try:
        save_vtu_output_using_fealax(problem, auto_solutions, params_batch, "AUTO", batch_size)
        save_vtu_output_using_fealax(problem, seq_solutions, params_batch, "SEQUENTIAL", batch_size)
    except Exception as e:
        print(f"Note: Could not save using fealax.utils.save_as_vtk: {e}")
    
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
    
    print(f"\nOutput directories:")
    print(f"  - Auto mode: {auto_output_dir}/")
    print(f"  - Sequential mode: {seq_output_dir}/")
    
    return {
        'batch_size': batch_size,
        'auto_time': auto_time,
        'seq_time': seq_time,
        'speedup': speedup,
        'auto_per_solve': auto_per_solve,
        'seq_per_solve': seq_per_solve,
        'auto_converged': auto_converged,
        'seq_converged': seq_converged,
        'auto_convergence_details': auto_details,
        'seq_convergence_details': seq_details,
        'solution_differences': differences,
        'auto_output_dir': auto_output_dir,
        'seq_output_dir': seq_output_dir
    }


def main():
    """Main performance comparison."""
    print("Performance Comparison: vmap vs Sequential Mode")
    print("=" * 60)
    
    # Test different batch sizes
    batch_sizes = [15]
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