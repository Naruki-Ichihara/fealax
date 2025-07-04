#!/usr/bin/env python3
"""
Simple linear elasticity problem using the new fealax API.

This example demonstrates a clean, modern approach to solving finite element
problems using the new separated assembly/solve API. It solves a 3D linear
elasticity problem with:

- 1x1x1 unit cube domain
- Hexahedral elements (HEX8)  
- Fixed displacement boundary conditions
- Applied compression on top face
- Linear elastic material (steel-like properties)

Key features demonstrated:
- Clean newton_solve() wrapper for complete Newton iteration
- Simple problem setup with material properties
- Boundary condition specification using lambda functions
- Physical interpretation of results

The example focuses on demonstrating the new API's simplicity and clarity
for typical finite element analysis workflows.
"""

import jax.numpy as jnp
import jax

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import newton_solve
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
    """Linear elasticity problem with isotropic material properties."""
    
    def __init__(self, mesh, E, nu, **kwargs):
        """Initialize elasticity problem.
        
        Args:
            mesh: Finite element mesh
            E: Young's modulus (Pa)
            nu: Poisson's ratio
            **kwargs: Additional arguments passed to Problem base class
        """
        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        
        # Compute Lame parameters for constitutive law
        self.mu = E / (2.0 * (1.0 + nu))  # Shear modulus
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # First Lame parameter
        
        super().__init__(mesh=mesh, **kwargs)
    
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
    
    # Material properties (steel-like)
    E = 200e9  # Young's modulus (Pa) - 200 GPa
    nu = 0.3   # Poisson's ratio
    
    # Create elasticity problem
    problem = ElasticityProblem(
        mesh=mesh,
        E=E,
        nu=nu,
        vec=3,  # 3D displacement field
        dim=3,  # 3D problem
        dirichlet_bcs=bcs
    )
    
    # Solver options
    solver_options = {
        'tol': 1e-5,           # Convergence tolerance  
        'rel_tol': 1e-6,       # Relative convergence tolerance
        'max_iter': 10,        # Maximum Newton iterations
        'method': 'cg',  # Linear solver method
        'use_jit': True       # Enable for GPU acceleration
    }
    
    # Solve the problem using the new API
    solution = newton_solve(problem, solver_options)
        
    # Post-process results
    displacement_field = solution[0]  # Primary variable (displacement)
    print(f"Solution shape: {displacement_field.shape}")
        
    # Reshape to get displacement components
    num_nodes = mesh.points.shape[0]
    displacements = displacement_field.reshape((num_nodes, 3))
        
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


if __name__ == "__main__":
    main()