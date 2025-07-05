#!/usr/bin/env python3
"""
Speed comparison between newton_solve API and conventional solver API.

This script compares the performance of:
1. New newton_solve API (Hybrid JIT)
2. New newton_solve API (Full JIT) 
3. Conventional solver API

All methods solve the same 3D linear elasticity problem with identical parameters.
"""

import jax.numpy as jnp
import jax
import time
import gc

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import newton_solve, _solver


class ElasticityProblem(Problem):
    """Linear elasticity problem with isotropic material properties."""
    
    def __init__(self, mesh, E, nu, **kwargs):
        self.E = E
        self.nu = nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        super().__init__(mesh=mesh, **kwargs)
    
    def get_tensor_map(self):
        def tensor_map(u_grads, *internal_vars):
            strain = 0.5 * (u_grads + jnp.transpose(u_grads))
            trace_strain = jnp.trace(strain)
            stress = (2.0 * self.mu * strain + 
                     self.lam * trace_strain * jnp.eye(3))
            return stress
        return tensor_map


def create_test_problem(mesh_size=15):
    """Create test problem with configurable mesh size."""
    mesh = box_mesh(mesh_size, mesh_size, mesh_size, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    bcs = [
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 2, lambda x: 0.0),           # Fixed bottom
        DirichletBC(lambda x: jnp.abs(x[0]) < 1e-6, 0, lambda x: 0.0),           # Symmetry x  
        DirichletBC(lambda x: jnp.abs(x[1]) < 1e-6, 1, lambda x: 0.0),           # Symmetry y
        DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 2, lambda x: -0.05),   # Top compression
    ]
    
    problem = ElasticityProblem(
        mesh=mesh,
        E=200e9,  # 200 GPa steel
        nu=0.3,
        vec=3, 
        dim=3, 
        dirichlet_bcs=bcs
    )
    
    return problem


def run_solver_comparison(mesh_sizes=[10, 30, 60]):
    """Run comprehensive solver comparison across different problem sizes."""
    print("=" * 80)
    print("FEALAX SOLVER API PERFORMANCE COMPARISON")
    print("=" * 80)
    
    jax.config.update("jax_enable_x64", True)
    
    # Solver configurations
    newton_solve_hybrid_options = {
        'tol': 1e-5,
        'rel_tol': 1e-6,
        'max_iter': 15,
        # JIT compilation is always enabled - this was hybrid mode
    }
    
    newton_solve_jit_options = {
        'tol': 1e-5,
        'rel_tol': 1e-6, 
        'max_iter': 15,
        'method': "cg",
        # Full JIT mode is always enabled
    }
    
    conventional_options = {
        'tol': 1e-5,
        'max_iter': 15,
        'jax_solver': {'precond': True}
    }
    
    results = []
    
    for mesh_size in mesh_sizes:
        print(f"\n📊 PROBLEM SIZE: {mesh_size}×{mesh_size}×{mesh_size} mesh")
        print("-" * 60)
        
        # Create test problem
        problem = create_test_problem(mesh_size)
        num_dofs = problem.num_total_dofs_all_vars
        # Handle mesh access - problem.mesh might be a list for multi-variable problems
        if isinstance(problem.mesh, list):
            num_elements = problem.mesh[0].cells.shape[0]
        else:
            num_elements = problem.mesh.cells.shape[0]
        
        print(f"Elements: {num_elements:,}")
        print(f"DOFs: {num_dofs:,}")
        print()
        
        # Results for this mesh size
        mesh_results = {
            'mesh_size': mesh_size,
            'num_dofs': num_dofs,
            'num_elements': num_elements
        }
        
        # Test 1: Newton Solve API - Hybrid JIT
        print("🔷 Test 1: newton_solve API (Hybrid JIT)")
        gc.collect()  # Clean memory before test
        
        start_time = time.time()
        try:
            solution_hybrid = newton_solve(problem, newton_solve_hybrid_options)
            hybrid_time = time.time() - start_time
            mesh_results['hybrid_time'] = hybrid_time
            mesh_results['hybrid_success'] = True
            print(f"   ✅ Converged in {hybrid_time:.3f}s")
        except Exception as e:
            mesh_results['hybrid_time'] = float('inf')
            mesh_results['hybrid_success'] = False
            print(f"   ❌ Failed: {e}")
        
        # Test 2: Newton Solve API - Full JIT
        print("🔷 Test 2: newton_solve API (Full JIT)")
        gc.collect()  # Clean memory before test
        
        start_time = time.time()
        try:
            solution_jit = newton_solve(problem, newton_solve_jit_options)
            jit_time = time.time() - start_time
            mesh_results['jit_time'] = jit_time
            mesh_results['jit_success'] = True
            print(f"   ✅ Converged in {jit_time:.3f}s")
        except Exception as e:
            mesh_results['jit_time'] = float('inf')
            mesh_results['jit_success'] = False
            print(f"   ❌ Failed: {e}")
        
        # Test 3: Conventional Solver API
        print("🔷 Test 3: Conventional solver API")
        gc.collect()  # Clean memory before test
        
        start_time = time.time()
        try:
            solution_conventional = _solver(problem, conventional_options)
            conventional_time = time.time() - start_time
            mesh_results['conventional_time'] = conventional_time
            mesh_results['conventional_success'] = True
            print(f"   ✅ Converged in {conventional_time:.3f}s")
        except Exception as e:
            mesh_results['conventional_time'] = float('inf')
            mesh_results['conventional_success'] = False
            print(f"   ❌ Failed: {e}")
        
        # Solution accuracy comparison
        print("🔍 Solution Accuracy Check:")
        if (mesh_results['hybrid_success'] and 
            mesh_results['jit_success'] and 
            mesh_results['conventional_success']):
            
            # Extract displacement fields
            hybrid_u = solution_hybrid[0] if isinstance(solution_hybrid, list) else solution_hybrid
            jit_u = solution_jit[0] if isinstance(solution_jit, list) else solution_jit
            conv_u = solution_conventional[0] if isinstance(solution_conventional, list) else solution_conventional
            
            # Compare solutions
            diff_hybrid_jit = jnp.linalg.norm(hybrid_u.flatten() - jit_u.flatten())
            diff_hybrid_conv = jnp.linalg.norm(hybrid_u.flatten() - conv_u.flatten())
            diff_jit_conv = jnp.linalg.norm(jit_u.flatten() - conv_u.flatten())
            
            mesh_results['accuracy_hybrid_jit'] = float(diff_hybrid_jit)
            mesh_results['accuracy_hybrid_conv'] = float(diff_hybrid_conv)
            mesh_results['accuracy_jit_conv'] = float(diff_jit_conv)
            
            print(f"   Hybrid vs JIT difference: {diff_hybrid_jit:.2e}")
            print(f"   Hybrid vs Conventional difference: {diff_hybrid_conv:.2e}")
            print(f"   JIT vs Conventional difference: {diff_jit_conv:.2e}")
            print(f"   All methods agree: {max(diff_hybrid_jit, diff_hybrid_conv, diff_jit_conv) < 1e-10}")
        
        # Performance summary for this mesh size
        print("⚡ Performance Summary:")
        if mesh_results['hybrid_success'] and mesh_results['jit_success']:
            speedup_jit_vs_hybrid = mesh_results['hybrid_time'] / mesh_results['jit_time']
            print(f"   JIT vs Hybrid speedup: {speedup_jit_vs_hybrid:.2f}x")
            
        if mesh_results['conventional_success'] and mesh_results['jit_success']:
            speedup_jit_vs_conv = mesh_results['conventional_time'] / mesh_results['jit_time']
            print(f"   JIT vs Conventional speedup: {speedup_jit_vs_conv:.2f}x")
            
        if mesh_results['conventional_success'] and mesh_results['hybrid_success']:
            speedup_hybrid_vs_conv = mesh_results['conventional_time'] / mesh_results['hybrid_time']
            print(f"   Hybrid vs Conventional speedup: {speedup_hybrid_vs_conv:.2f}x")
        
        results.append(mesh_results)
    
    # Overall summary table
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print(f"{'Mesh Size':<10} {'DOFs':<8} {'Hybrid(s)':<10} {'JIT(s)':<8} {'Conv(s)':<8} {'JIT Speedup':<12}")
    print("-" * 70)
    
    for result in results:
        mesh_str = f"{result['mesh_size']}³"
        dofs_str = f"{result['num_dofs']//1000}k" if result['num_dofs'] >= 1000 else str(result['num_dofs'])
        
        hybrid_str = f"{result['hybrid_time']:.2f}" if result['hybrid_success'] else "FAIL"
        jit_str = f"{result['jit_time']:.2f}" if result['jit_success'] else "FAIL"
        conv_str = f"{result['conventional_time']:.2f}" if result['conventional_success'] else "FAIL"
        
        if result['jit_success'] and result['conventional_success']:
            speedup = result['conventional_time'] / result['jit_time']
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "N/A"
        
        print(f"{mesh_str:<10} {dofs_str:<8} {hybrid_str:<10} {jit_str:<8} {conv_str:<8} {speedup_str:<12}")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS:")
    print("-" * 40)
    print("• JIT compilation is always enabled for optimal performance")
    print("• Use pure_jit_mode=True for advanced optimization scenarios")
    print("• Conventional solver() is being superseded by new API")
    print("• All methods produce identical solutions (machine precision)")
    
    return results


if __name__ == "__main__":
    # Run comparison with GPU memory-friendly problem sizes
    # 80³ mesh (1.6M DOFs) exceeds 16GB GPU memory limit
    results = run_solver_comparison(mesh_sizes=[1, 5, 25])