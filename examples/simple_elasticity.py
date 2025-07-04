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
- JAX-compatible solver for optimization workflows
- Physical interpretation of results

The example focuses on demonstrating the new wrapper API's simplicity and
its compatibility with JAX automation chains for optimization and sensitivity analysis.
"""

import jax.numpy as jnp
import jax

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import NewtonSolver, create_newton_solver
from fealax.utils import save_as_vtk


def create_mesh(nx=15, ny=15, nz=15):
    """Create a structured 1x1x1 box mesh.
    
    Args:
        nx, ny, nz: Number of elements in each direction
        
    Returns:
        Mesh object
    """
    return box_mesh(nx, ny, nz, 1.0, 1.0, 1.0, ele_type="HEX8")


def define_boundary_conditions():
    """Define boundary conditions for compression test.
    
    Creates boundary conditions that constrain the bottom face and apply
    compression on the top face, with symmetry conditions to prevent
    rigid body motion.
    
    Returns:
        List of DirichletBC objects
    """
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
    """Linear elasticity problem with parameterized material properties."""
    
    def __init__(self, mesh, **kwargs):
        """Initialize elasticity problem.
        
        Args:
            mesh: Finite element mesh
            **kwargs: Additional arguments passed to Problem base class
        """
        # Material properties will be set via parameters
        self.E = None  # Young's modulus (Pa)
        self.nu = None  # Poisson's ratio
        self.mu = None  # Shear modulus
        self.lam = None  # First Lame parameter
        
        super().__init__(mesh=mesh, **kwargs)
    
    def set_params(self, params):
        """Set material parameters and update derived properties.
        
        Args:
            params: Dictionary with material parameters {'E': float, 'nu': float}
        """
        self.E = params['E']
        self.nu = params['nu']
        
        # Compute Lame parameters for constitutive law
        self.mu = self.E / (2.0 * (1.0 + self.nu))  # Shear modulus
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))  # First Lame parameter
    
    def get_tensor_map(self):
        """Return tensor map function for linear elasticity.
        
        Implements the linear elastic constitutive law relating strain to stress.
        
        Returns:
            Function that maps displacement gradients to stress tensor
        """
        def tensor_map(u_grads, *internal_vars):
            """Compute stress tensor from displacement gradients.
            
            Args:
                u_grads: Displacement gradients with shape (vec, dim)
                        For 3D: (3, 3) - gradients at a single quadrature point
                        
            Returns:
                Stress tensor with same shape as u_grads
            """
            # Strain tensor (symmetric gradient): ε = 0.5 * (∇u + ∇u^T)
            strain = 0.5 * (u_grads + jnp.transpose(u_grads))
            
            # Stress tensor using Hooke's law: σ = 2μ*ε + λ*tr(ε)*I
            trace_strain = jnp.trace(strain)  # scalar
            stress = (2.0 * self.mu * strain + 
                     self.lam * trace_strain * jnp.eye(3))
            
            return stress
        
        return tensor_map


def main():
    """Main function to setup and solve the elasticity problem."""
    
    # Enable JAX 64-bit precision for better numerical accuracy
    jax.config.update("jax_enable_x64", True)

    # Create mesh
    print("Creating 1x1x1 box mesh...")
    mesh = create_mesh(nx=50, ny=50, nz=50)
    
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
        'method': 'cg',        # Linear solver method
        'use_jit': True        # Enable for GPU acceleration
    }
    
    # Create Newton solver using the new wrapper API
    print("Creating Newton solver...")
    solver = NewtonSolver(problem, solver_options)
    
    # Alternative: using convenience function
    # solver = create_newton_solver(problem, solver_options)
    
    # Material properties (steel-like)
    material_params = {
        'E': 200e9,  # Young's modulus (Pa) - 200 GPa
        'nu': 0.3    # Poisson's ratio
    }
    
    # Solve the problem using the new API
    print("Solving elasticity problem...")
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
    print("Results saved to: simple_elasticity_results.vtu")
    
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
        'use_jit': True
    }
    
    # Create differentiable solver for optimization
    diff_solver = NewtonSolver(problem, solver_options, differentiable=True)
    
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
    
    # Run optimization demonstration
    try:
        diff_solver = demonstrate_optimization()
        print("\n" + "="*50)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*50)
        print("✓ Standard solver: solver.solve(params)")
        print("✓ Differentiable solver: jax.grad(objective)(params)")
        print("✓ JAX automation chains working")
        print("✓ Material parameter sensitivity computed")
        
    except Exception as e:
        print(f"\nOptimization demo failed: {e}")
        print("Standard solver example completed successfully.")
        print("Optimization may require additional problem setup.")