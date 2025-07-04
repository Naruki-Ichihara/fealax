#!/usr/bin/env python3
"""
High-resolution mesh performance comparison with memory management.

This script demonstrates how to handle large finite element problems
that approach GPU memory limits using adaptive memory management.
"""

import jax.numpy as jnp
import jax
import time
import gc
from typing import List, Dict, Any

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import newton_solve
from fealax.large_solver import solve_large_problem, check_problem_feasibility
from fealax.memory_utils import get_gpu_memory_info, clear_jax_memory


class ElasticityProblem(Problem):
    """Linear elasticity problem for performance testing."""
    
    def __init__(self, mesh, E=200e9, nu=0.3, **kwargs):
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


def create_test_problem(mesh_size: int) -> ElasticityProblem:
    """Create elasticity test problem."""
    mesh = box_mesh(mesh_size, mesh_size, mesh_size, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    # Simple boundary conditions
    bcs = [
        DirichletBC(
            subdomain=lambda x: jnp.abs(x[2]) < 1e-6,
            vec=2,
            eval=lambda x: 0.0
        ),
        DirichletBC(
            subdomain=lambda x: jnp.abs(x[2] - 1.0) < 1e-6,
            vec=2,
            eval=lambda x: -0.01
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
    ]
    
    return ElasticityProblem(
        mesh=mesh,
        vec=3,
        dim=3,
        dirichlet_bcs=bcs
    )


def test_solver_method(problem: ElasticityProblem, method_name: str, 
                      solver_options: Dict[str, Any]) -> Dict[str, Any]:
    """Test a specific solver method."""
    print(f"   Testing {method_name}...")
    
    # Record initial memory
    memory_before = get_gpu_memory_info()
    
    start_time = time.time()
    try:
        if method_name == "Large Problem Solver":
            solution = solve_large_problem(problem, solver_options)
        else:
            solution = newton_solve(problem, solver_options)
            
        solve_time = time.time() - start_time
        
        # Verify solution
        displacement_field = solution[0]
        max_displacement = jnp.max(jnp.abs(displacement_field))
        
        memory_after = get_gpu_memory_info()
        memory_used = (memory_after['used'] - memory_before['used']) / 1e9
        
        return {
            'success': True,
            'time': solve_time,
            'max_displacement': float(max_displacement),
            'memory_used_gb': memory_used,
            'peak_memory_gb': memory_after['used'] / 1e9
        }
        
    except Exception as e:
        solve_time = time.time() - start_time
        memory_after = get_gpu_memory_info()
        
        return {
            'success': False,
            'time': solve_time,
            'error': str(e),
            'peak_memory_gb': memory_after['used'] / 1e9
        }
    finally:
        # Cleanup
        clear_jax_memory()


def run_large_mesh_comparison():
    """Run comprehensive comparison for large mesh problems."""
    print("=" * 80)
    print("LARGE MESH PERFORMANCE COMPARISON")
    print("=" * 80)
    
    jax.config.update("jax_enable_x64", True)
    
    # Test progressively larger mesh sizes
    mesh_sizes = [10, 50]  # Will test feasibility for each
    
    # Solver configurations
    solver_configs = {
        "Standard Hybrid": {
            'use_jit': False,
            'tol': 1e-5,
            'max_iter': 15,
            'method': 'cg'
        },
        "Standard JIT": {
            'use_jit': True,
            'tol': 1e-5,
            'max_iter': 15,
            'method': 'cg'
        },
        "Memory Optimized": {
            'use_jit': False,
            'tol': 1e-4,
            'max_iter': 10,
            'method': 'cg',
            'line_search_flag': True
        }
    }
    
    results = []
    
    for mesh_size in mesh_sizes:
        print(f"\n📊 TESTING MESH SIZE: {mesh_size}³")
        
        # Create problem
        problem = create_test_problem(mesh_size)
        mesh = problem.mesh[0]  # Get the actual mesh from the problem
        num_elements = mesh.cells.shape[0]
        num_dofs = problem.num_total_dofs_all_vars
        
        print(f"   Elements: {num_elements:,}")
        print(f"   DOFs: {num_dofs:,}")
        
        # Check feasibility
        feasibility = check_problem_feasibility(mesh)
        print(f"   Feasible: {feasibility['feasible']}")
        print(f"   Estimated memory: {feasibility['estimated_memory_gb']:.2f} GB")
        print(f"   Available memory: {feasibility['available_memory_gb']:.2f} GB")
        
        if not feasibility['feasible']:
            print(f"   ⚠️  {feasibility['recommendation']}")
            # Still try with large problem solver
            
        mesh_results = {
            'mesh_size': mesh_size,
            'num_elements': num_elements,
            'num_dofs': num_dofs,
            'feasible': feasibility['feasible'],
            'estimated_memory_gb': feasibility['estimated_memory_gb']
        }
        
        # Test each solver method
        for method_name, options in solver_configs.items():
            if not feasibility['feasible'] and method_name != "Memory Optimized":
                # Skip non-optimized methods for infeasible problems
                continue
                
            result = test_solver_method(problem, method_name, options)
            mesh_results[method_name] = result
            
            if result['success']:
                print(f"      ✅ {method_name}: {result['time']:.2f}s, "
                      f"Memory: {result['memory_used_gb']:.2f} GB")
            else:
                print(f"      ❌ {method_name}: Failed - {result['error']}")
        
        # Always test large problem solver for comparison
        large_solver_result = test_solver_method(problem, "Large Problem Solver", {})
        mesh_results["Large Problem Solver"] = large_solver_result
        
        if large_solver_result['success']:
            print(f"      ✅ Large Problem Solver: {large_solver_result['time']:.2f}s, "
                  f"Memory: {large_solver_result['memory_used_gb']:.2f} GB")
        else:
            print(f"      ❌ Large Problem Solver: Failed - {large_solver_result['error']}")
        
        results.append(mesh_results)
        
        # If this size failed completely, don't try larger sizes
        all_failed = all(not result.get('success', False) 
                        for key, result in mesh_results.items() 
                        if isinstance(result, dict) and 'success' in result)
        if all_failed:
            print(f"   🛑 All solvers failed for {mesh_size}³ mesh, stopping here")
            break
    
    # Print summary table
    print_comparison_summary(results)
    
    return results


def print_comparison_summary(results: List[Dict]):
    """Print comprehensive summary table."""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE PERFORMANCE SUMMARY")
    print("=" * 100)
    
    # Header
    print(f"{'Size':<6} {'DOFs':<8} {'Feasible':<9} {'Standard':<10} {'JIT':<10} "
          f"{'Optimized':<10} {'Large Solver':<12} {'Best':<15}")
    print("-" * 100)
    
    for result in results:
        size_str = f"{result['mesh_size']}³"
        dofs_str = f"{result['num_dofs']//1000}k" if result['num_dofs'] >= 1000 else str(result['num_dofs'])
        feasible_str = "✅" if result['feasible'] else "❌"
        
        # Get times for each method
        methods = ["Standard Hybrid", "Standard JIT", "Memory Optimized", "Large Problem Solver"]
        times = []
        best_time = float('inf')
        best_method = "None"
        
        for method in methods:
            if method in result and result[method].get('success', False):
                time_val = result[method]['time']
                times.append(f"{time_val:.2f}s")
                if time_val < best_time:
                    best_time = time_val
                    best_method = method
            else:
                times.append("FAIL")
        
        best_str = f"{best_method[:12]}" if best_method != "None" else "None"
        
        print(f"{size_str:<6} {dofs_str:<8} {feasible_str:<9} {times[0]:<10} {times[1]:<10} "
              f"{times[2]:<10} {times[3]:<12} {best_str:<15}")
    
    # Recommendations
    print("\n🎯 RECOMMENDATIONS FOR LARGE MESHES:")
    print("-" * 50)
    print("• Use Large Problem Solver for meshes approaching memory limits")
    print("• Disable JIT compilation for very large problems to save memory")
    print("• Enable line search for better convergence with relaxed tolerances")
    print("• Monitor memory usage and use adaptive chunking")
    print("• Consider mesh refinement strategies for very large problems")


if __name__ == "__main__":
    run_large_mesh_comparison()