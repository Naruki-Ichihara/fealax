#!/usr/bin/env python3
"""Fixed performance comparison for elasticity problems using the updated fealax API."""

import time
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import NewtonSolver

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

def benchmark_solver_modes(problem, params_batch, solver_options):
    """Benchmark different solving approaches with correct parameter handling."""
    print(f"\n{'='*60}")
    print("BENCHMARKING SOLVER MODES")
    print(f"{'='*60}")
    
    results = {}
    
    # Test 1: Using NewtonSolver wrapper (recommended approach)
    print("\n1. NewtonSolver with batch parameters")
    print("-" * 40)
    
    solver = NewtonSolver(problem, solver_options)
    
    start_time = time.time()
    batch_solutions = solver.solve(params_batch)
    batch_time = time.time() - start_time
    
    print(f"Batch solve time: {batch_time:.3f}s ({batch_time/len(params_batch):.3f}s per solve)")
    results['batch'] = {
        'time': batch_time,
        'per_solve': batch_time / len(params_batch),
        'solutions': batch_solutions
    }
    
    # Test 2: Individual solves with NewtonSolver
    print("\n2. Individual solves with NewtonSolver")
    print("-" * 40)
    
    start_time = time.time()
    individual_solutions = []
    for i, params in enumerate(params_batch):
        print(f"  Solving problem {i+1}/{len(params_batch)}: E={params['E']/1e9:.1f}GPa, nu={params['nu']:.3f}")
        solution = solver.solve(params)
        individual_solutions.append(solution)
    individual_time = time.time() - start_time
    
    print(f"Individual solve time: {individual_time:.3f}s ({individual_time/len(params_batch):.3f}s per solve)")
    results['individual'] = {
        'time': individual_time,
        'per_solve': individual_time / len(params_batch),
        'solutions': individual_solutions
    }
    
    # Test 3: Direct vmap approach
    print("\n3. Direct vmap approach")
    print("-" * 40)
    
    def solve_single(E, nu):
        return solver.solve({'E': E, 'nu': nu})
    
    E_values = jnp.array([p['E'] for p in params_batch])
    nu_values = jnp.array([p['nu'] for p in params_batch])
    
    vmap_solve = jax.vmap(solve_single)
    
    start_time = time.time()
    vmap_solutions = vmap_solve(E_values, nu_values)
    vmap_time = time.time() - start_time
    
    print(f"Vmap solve time: {vmap_time:.3f}s ({vmap_time/len(params_batch):.3f}s per solve)")
    results['vmap'] = {
        'time': vmap_time,
        'per_solve': vmap_time / len(params_batch),
        'solutions': vmap_solutions
    }
    
    # Test 4: Post-processing analysis
    print("\n4. Post-processing Analysis")
    print("-" * 40)
    
    @jax.jit
    def analyze_solution(u):
        u_reshaped = u.reshape(-1, 3)
        displacements = jnp.sqrt(jnp.sum(u_reshaped**2, axis=1))
        return {
            'max_displacement': jnp.max(displacements),
            'mean_displacement': jnp.mean(displacements),
            'strain_energy': jnp.sum(u**2)
        }
    
    # Analyze batch solutions
    start_time = time.time()
    analysis_results = []
    
    # Use batch solutions for analysis
    if isinstance(batch_solutions, list):
        # List of solutions
        for i, solution in enumerate(batch_solutions):
            if len(solution) > 0:  # Check if solution exists
                u = solution[0] if isinstance(solution, list) else solution
                analysis = analyze_solution(u)
                analysis_results.append(analysis)
                if i < 3:  # Show first 3 results
                    max_disp = float(analysis['max_displacement'])
                    energy = float(analysis['strain_energy'])
                    print(f"  Problem {i+1}: max_disp={max_disp*1000:.3f}mm, energy={energy:.3e}")
    else:
        # Array of solutions
        for i in range(len(params_batch)):
            u = batch_solutions[0][i]  # Assuming shape (1, batch, n_dofs)
            analysis = analyze_solution(u)
            analysis_results.append(analysis)
            if i < 3:  # Show first 3 results
                max_disp = float(analysis['max_displacement'])
                energy = float(analysis['strain_energy'])
                print(f"  Problem {i+1}: max_disp={max_disp*1000:.3f}mm, energy={energy:.3e}")
    
    post_process_time = time.time() - start_time
    print(f"Post-processing time: {post_process_time*1000:.1f}ms")
    
    results['post_processing'] = {
        'time': post_process_time,
        'results': analysis_results
    }
    
    return results

def verify_solutions_differ(results, params_batch):
    """Verify that solutions actually differ for different parameters."""
    print(f"\n{'='*60}")
    print("SOLUTION VERIFICATION")
    print(f"{'='*60}")
    
    # Get analysis results
    analysis_results = results['post_processing']['results']
    
    print(f"\nChecking if solutions differ for different parameters:")
    print(f"{'Params':<30} {'Max Disp (mm)':<15} {'Energy':<15}")
    print("-" * 60)
    
    max_disps = []
    energies = []
    
    for i, (params, analysis) in enumerate(zip(params_batch, analysis_results)):
        max_disp = float(analysis['max_displacement']) * 1000  # Convert to mm
        energy = float(analysis['strain_energy'])
        max_disps.append(max_disp)
        energies.append(energy)
        
        print(f"E={params['E']/1e9:.1f}GPa, nu={params['nu']:.3f}"
              f"{max_disp:>15.3f}{energy:>15.3e}")
    
    # Check variation
    disp_variation = (max(max_disps) - min(max_disps)) / np.mean(max_disps) * 100
    energy_variation = (max(energies) - min(energies)) / np.mean(energies) * 100
    
    print(f"\nVariation in results:")
    print(f"  Displacement variation: {disp_variation:.1f}%")
    print(f"  Energy variation: {energy_variation:.1f}%")
    
    if disp_variation < 0.1 and energy_variation < 0.1:
        print(f"⚠️  WARNING: Solutions are nearly identical!")
        print(f"   This suggests parameters are not being properly updated.")
        return False
    else:
        print(f"✅ Solutions vary with parameters as expected.")
        return True

def performance_summary(results, batch_size):
    """Print performance summary."""
    print(f"\n{'='*60}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    batch_time = results['batch']['time']
    individual_time = results['individual']['time']
    vmap_time = results['vmap']['time']
    
    print(f"Batch size: {batch_size}")
    print(f"Batch solve: {batch_time:.3f}s ({batch_time/batch_size:.3f}s per solve)")
    print(f"Individual solve: {individual_time:.3f}s ({individual_time/batch_size:.3f}s per solve)")
    print(f"Vmap solve: {vmap_time:.3f}s ({vmap_time/batch_size:.3f}s per solve)")
    
    # Find fastest method
    times = [
        ('Batch', batch_time),
        ('Individual', individual_time),
        ('Vmap', vmap_time)
    ]
    fastest = min(times, key=lambda x: x[1])
    
    print(f"\n🏆 Fastest method: {fastest[0]} ({fastest[1]:.3f}s)")
    
    # Calculate speedups
    print(f"\nSpeedup ratios:")
    print(f"  Batch vs Individual: {individual_time/batch_time:.2f}x")
    print(f"  Vmap vs Individual: {individual_time/vmap_time:.2f}x")

def main():
    """Main performance test with fixed parameter handling."""
    print("🔬 Fealax Performance Analysis - Fixed Parameter Handling")
    print("="*60)
    
    # Configuration
    mesh_size = (10, 10, 10)  # Moderate size for testing
    batch_size = 5
    solver_options = {'tol': 1e-6, 'max_iter': 10}
    
    # Create test problem
    problem = create_test_problem(mesh_size)
    
    # Create parameter variations with larger differences
    print(f"\nGenerating {batch_size} parameter variations...")
    params_batch = []
    for i in range(batch_size):
        E = 100e9 + i * 50e9  # 100-300 GPa (larger variation)
        nu = 0.20 + i * 0.05  # 0.20-0.40 (larger variation)
        params_batch.append({'E': E, 'nu': nu})
        print(f"  {i+1}: E={E/1e9:.1f}GPa, nu={nu:.3f}")
    
    # Run benchmarks
    results = benchmark_solver_modes(problem, params_batch, solver_options)
    
    # Verify solutions differ
    solutions_vary = verify_solutions_differ(results, params_batch)
    
    # Print summary
    performance_summary(results, batch_size)
    
    print(f"\n{'='*60}")
    print("CONCLUSIONS")
    print(f"{'='*60}")
    
    if solutions_vary:
        print("✅ Parameter updates working correctly")
        print("✅ Different parameters produce different solutions")
    else:
        print("⚠️  Parameter updates may not be working correctly")
        print("   Consider using NewtonSolver with proper parameter handling")
    
    print("\nRecommendations:")
    print("• Use NewtonSolver for clean parameter handling")
    print("• Use batch solving for multiple parameter sets")
    print("• Use vmap for maximum flexibility")
    
    return results

if __name__ == "__main__":
    main()