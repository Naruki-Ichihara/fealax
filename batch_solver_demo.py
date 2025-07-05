#!/usr/bin/env python3
"""
Demonstration of the automatic batch solving functionality in fealax.

This script shows how the NewtonSolver now automatically detects when
multiple parameter sets are provided and handles them appropriately.
"""

import jax.numpy as jnp
import jax
import time

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


def main():
    print("="*70)
    print("FEALAX AUTOMATIC BATCH SOLVING DEMONSTRATION")
    print("="*70)
    
    jax.config.update("jax_enable_x64", True)
    
    # Create a simple test setup
    print("Setting up finite element problem...")
    mesh = box_mesh(8, 8, 8, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    bcs = [
        DirichletBC(subdomain=lambda x: jnp.abs(x[2]) < 1e-6, vec=2, eval=lambda x: 0.0),
        DirichletBC(subdomain=lambda x: jnp.abs(x[0]) < 1e-6, vec=0, eval=lambda x: 0.0),
        DirichletBC(subdomain=lambda x: jnp.abs(x[1]) < 1e-6, vec=1, eval=lambda x: 0.0),
        DirichletBC(subdomain=lambda x: jnp.abs(x[2] - 1.0) < 1e-6, vec=2, eval=lambda x: -0.02),
    ]
    
    problem = ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    solver = NewtonSolver(problem, {'tol': 1e-4, 'max_iter': 5})
    
    print(f"✓ Problem setup complete: {mesh.cells.shape[0]} elements, {mesh.points.shape[0]} nodes")
    
    # Demonstration 1: Single parameter solving (unchanged behavior)
    print("\n" + "-"*50)
    print("1. SINGLE PARAMETER SOLVING")
    print("-"*50)
    
    single_params = {'E': 200e9, 'nu': 0.3}
    
    start_time = time.time()
    solution = solver.solve(single_params)
    single_time = time.time() - start_time
    
    max_disp = jnp.max(jnp.abs(solution[0]))
    print(f"✓ Single solve completed in {single_time:.2f}s")
    print(f"  Material: E = {single_params['E']/1e9:.0f} GPa, ν = {single_params['nu']}")
    print(f"  Max displacement: {max_disp:.6f} m")
    
    
    # Demonstration 2: Batch solving with list of dictionaries
    print("\n" + "-"*50)
    print("2. BATCH SOLVING - LIST OF DICTIONARIES")
    print("-"*50)
    
    batch_params_list = [
        {'E': 150e9, 'nu': 0.25},
        {'E': 200e9, 'nu': 0.30},
        {'E': 250e9, 'nu': 0.35},
        {'E': 300e9, 'nu': 0.28}
    ]
    
    print(f"Solving for {len(batch_params_list)} different material configurations...")
    
    start_time = time.time()
    batch_solutions = solver.solve(batch_params_list)
    batch_time = time.time() - start_time
    
    print(f"✓ Batch solve completed in {batch_time:.2f}s")
    print(f"  Average time per solve: {batch_time/len(batch_params_list):.2f}s")
    print(f"  Solution shape: {batch_solutions[0].shape}")
    print("\n  Results:")
    for i, params in enumerate(batch_params_list):
        max_disp = jnp.max(jnp.abs(batch_solutions[0][i]))
        E_gpa = params['E'] / 1e9
        print(f"    Case {i+1}: E={E_gpa:3.0f} GPa, ν={params['nu']:.2f} → max disp = {max_disp:.6f} m")
    
    
    # Demonstration 3: Batch solving with dictionary of arrays
    print("\n" + "-"*50)
    print("3. BATCH SOLVING - DICTIONARY OF ARRAYS")
    print("-"*50)
    
    batch_params_dict = {
        'E': jnp.array([180e9, 220e9, 260e9]),
        'nu': jnp.array([0.27, 0.32, 0.29])
    }
    
    print(f"Solving for {len(batch_params_dict['E'])} material configurations...")
    
    start_time = time.time()
    batch_solutions_dict = solver.solve(batch_params_dict)
    batch_dict_time = time.time() - start_time
    
    print(f"✓ Batch solve completed in {batch_dict_time:.2f}s")
    print(f"  Solution shape: {batch_solutions_dict[0].shape}")
    print("\n  Results:")
    for i in range(len(batch_params_dict['E'])):
        max_disp = jnp.max(jnp.abs(batch_solutions_dict[0][i]))
        E_gpa = batch_params_dict['E'][i] / 1e9
        nu = batch_params_dict['nu'][i]
        print(f"    Case {i+1}: E={E_gpa:3.0f} GPa, ν={nu:.2f} → max disp = {max_disp:.6f} m")
    
    
    # Performance summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Single solve:           {single_time:.2f}s")
    print(f"Batch solve (list):     {batch_time:.2f}s ({len(batch_params_list)} cases)")
    print(f"Batch solve (dict):     {batch_dict_time:.2f}s ({len(batch_params_dict['E'])} cases)")
    print(f"Sequential efficiency:  ~{single_time/batch_time*len(batch_params_list):.1f}x speedup")
    
    print("\n✓ All demonstrations completed successfully!")
    print("\nKey features demonstrated:")
    print("  • Automatic detection of batch parameters")
    print("  • Support for list of dictionaries format")
    print("  • Support for dictionary of arrays format")
    print("  • Automatic batch processing with proper result stacking")
    print("  • Backward compatibility with single parameter solving")


if __name__ == "__main__":
    main()