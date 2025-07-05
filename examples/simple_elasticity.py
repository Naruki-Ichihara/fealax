#!/usr/bin/env python3
"""
Simple linear elasticity problem using the new fealax Newton solver wrapper API.

This example demonstrates a clean, modern approach to solving finite element
problems using the new Newton solver wrapper. It solves a 3D linear
elasticity problem with:

- 1x1x1 unit cube domain
- Hexahedral elements (HEX8)  
- Fixed displacement boundary conditions
- Applied compression on top face
- Linear elastic material (steel-like properties)

Key features demonstrated:
- Clean NewtonSolver wrapper with solver.solve(params) API
- Parameter-driven material properties for sensitivity analysis
- Boundary condition specification using lambda functions
- JAX-compatible solver for optimization workflows with automatic JIT compilation
- Physical interpretation of results

The example focuses on demonstrating the new wrapper API's simplicity and
its compatibility with JAX automation chains for optimization and sensitivity analysis.
All solvers automatically use JIT compilation for optimal performance.
"""

import jax.numpy as jnp
import jax

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import NewtonSolver
from fealax.utils import save_as_vtk


def create_mesh(nx=15, ny=15, nz=15):
    return box_mesh(nx, ny, nz, 1.0, 1.0, 1.0, ele_type="HEX8")


def define_boundary_conditions():
    bcs = []
    
    # Fix bottom face in z-direction (prevent vertical motion)
    bcs.append(DirichletBC(
        subdomain=lambda x: jnp.abs(x[2]) < 1e-6,  # z = 0 plane
        vec=2,  # z-component
        eval=lambda x: 0.0
    ))
    
    # Symmetry condition: fix x=0 plane in x-direction
    bcs.append(DirichletBC(
        subdomain=lambda x: jnp.abs(x[0]) < 1e-6,  # x = 0 plane
        vec=0,  # x-component
        eval=lambda x: 0.0
    ))
    
    # Symmetry condition: fix y=0 plane in y-direction  
    bcs.append(DirichletBC(
        subdomain=lambda x: jnp.abs(x[1]) < 1e-6,  # y = 0 plane
        vec=1,  # y-component
        eval=lambda x: 0.0
    ))
    
    # Applied compression on top face
    bcs.append(DirichletBC(
        subdomain=lambda x: jnp.abs(x[2] - 1.0) < 1e-6,  # z = 1 plane
        vec=2,  # z-component
        eval=lambda x: -0.05  # 5% compression
    ))
    
    return bcs


class ElasticityProblem(Problem):
    def __init__(self, mesh, **kwargs):
        # Material properties will be set via parameters
        self.E = None  # Young's modulus (Pa)
        self.nu = None  # Poisson's ratio
        self.mu = None  # Shear modulus
        self.lam = None  # First Lame parameter
        
        super().__init__(mesh=mesh, **kwargs)
    
    def set_params(self, params):
        self.E = params['E']
        self.nu = params['nu']
        
        # Compute Lame parameters for constitutive law
        self.mu = self.E / (2.0 * (1.0 + self.nu))  # Shear modulus
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))  # First Lame parameter
    
    def get_tensor_map(self):
        def tensor_map(u_grads, *internal_vars):
            # Strain tensor (symmetric gradient): ε = 0.5 * (∇u + ∇u^T)
            strain = 0.5 * (u_grads + jnp.transpose(u_grads))
            
            # Stress tensor using Hooke's law: σ = 2μ*ε + λ*tr(ε)*I
            trace_strain = jnp.trace(strain)  # scalar
            stress = (2.0 * self.mu * strain + 
                     self.lam * trace_strain * jnp.eye(3))
            
            return stress
        
        return tensor_map


def main():
    # Enable JAX 64-bit precision for better numerical accuracy
    jax.config.update("jax_enable_x64", True)

    # Create mesh
    print("Creating 1x1x1 box mesh...")
    mesh = create_mesh(nx=10, ny=10, nz=10)
    
    # Define boundary conditions
    print("Setting up boundary conditions...")
    bcs = define_boundary_conditions()
    
    # Create elasticity problem (without material properties initially)
    problem = ElasticityProblem(
        mesh=mesh,
        vec=3,  # 3D displacement field
        dim=3,  # 3D problem
        dirichlet_bcs=bcs
    )
    
    # Solver options
    solver_options = {
        'tol': 1e-5,           # Convergence tolerance  
        'rel_tol': 1e-6,       # Relative convergence tolerance
        'max_iter': 10,        # Maximum Newton iterations
        'method': 'bicgstab'  # Linear solver method
    }
    
    # Create Newton solver using the new wrapper API
    solver = NewtonSolver(problem, solver_options)
    
    # Material properties (steel-like)
    material_params = {
        'E': 200e9,  # Young's modulus (Pa) - 200 GPa
        'nu': 0.3    # Poisson's ratio
    }
    
    solution = solver.solve(material_params)
        
    # Post-process results
    displacement_field = solution[0]  # Primary variable (displacement)
    print(f"Solution shape: {displacement_field.shape}")
        
    # Reshape to get displacement components
    num_nodes = mesh.points.shape[0]
    displacements = displacement_field.reshape((num_nodes, 3))
        
    # Display physical results
    max_displacement = jnp.max(jnp.abs(displacements))
    print(f"Maximum displacement magnitude: {max_displacement:.6f} m")
    
    # Compute von Mises stress at nodes (simplified)
    print(f"Applied compression: 5% (0.05 m)")
    print(f"Material: E = {material_params['E']/1e9:.0f} GPa, ν = {material_params['nu']}")
    
    # Save results for visualization
    print("Saving results to VTK format...")
        
    # Create VTK helper object
    class VTKHelper:
        def __init__(self, mesh, ele_type):
            self.points = mesh.points
            self.cells = mesh.cells
            self.ele_type = ele_type
            self.num_cells = mesh.cells.shape[0]
            
    vtk_helper = VTKHelper(mesh, "HEX8")
            
    # Prepare data for visualization
    point_data = [
        ("displacement", displacements)
    ]
            
    # Save to VTK file
    import os
    vtk_filename = os.path.join(os.getcwd(), "simple_elasticity_results.vtu")
    save_as_vtk(vtk_helper, vtk_filename, point_infos=point_data)
    
    print("Example completed successfully!")
    
    return solver, material_params, solution


def demonstrate_optimization():
    """Demonstrate optimization capabilities with the new API."""
    
    print("\n" + "="*50)
    print("OPTIMIZATION DEMO")
    print("="*50)
    
    # Enable JAX 64-bit precision
    jax.config.update("jax_enable_x64", True)
    
    # Create simpler mesh for optimization demo
    print("Creating smaller mesh for optimization demo...")
    mesh = create_mesh(nx=10, ny=10, nz=10)
    bcs = define_boundary_conditions()
    
    # Create problem
    problem = ElasticityProblem(
        mesh=mesh,
        vec=3,
        dim=3,
        dirichlet_bcs=bcs
    )
    
    # Create differentiable solver
    print("Creating differentiable solver...")
    solver_options = {
        'tol': 1e-4,
        'rel_tol': 1e-5,
        'max_iter': 5,
        # JIT compilation is always enabled
    }
    
    # Create differentiable solver for optimization
    # Note: All NewtonSolver instances are automatically differentiable
    diff_solver = NewtonSolver(problem, solver_options)
    
    # Define objective function (compliance minimization)
    def compliance_objective(params):
        """Compute structural compliance (inverse of stiffness)."""
        solution = diff_solver.solve(params)
        displacement_field = solution[0]
        
        # Simple compliance measure: sum of squared displacements
        return jnp.sum(displacement_field**2)
    
    # Test gradient computation
    print("Computing gradients with respect to material parameters...")
    test_params = {'E': 200e9, 'nu': 0.3}
    
    try:
        # Compute gradient
        grad_fn = jax.grad(compliance_objective)
        gradients = grad_fn(test_params)
        
        print(f"Gradients computed successfully:")
        print(f"  ∂C/∂E = {gradients['E']:.2e}")
        print(f"  ∂C/∂ν = {gradients['nu']:.2e}")
        
        # Verify gradient signs make physical sense
        print("\nPhysical interpretation:")
        print("  - Increase in E should decrease compliance (negative gradient)")
        print("  - Gradient signs indicate sensitivity to material parameters")
        
    except Exception as e:
        print(f"Gradient computation failed: {e}")
        print("This may be due to missing problem methods or boundary condition setup.")
    
    print("\nOptimization demo completed!")
    
    return diff_solver


if __name__ == "__main__":
    # Run main elasticity example
    solver, material_params, solution = main()
    demonstrate_optimization()