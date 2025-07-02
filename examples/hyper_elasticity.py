#!/usr/bin/env python3
"""
Hyperelasticity problem on a 1x1x1 box mesh.

This example demonstrates a hyperelastic material model with:
- Neo-Hookean material model
- 3D deformation with rotation and stretch
- Nonlinear finite element formulation
- First Piola-Kirchhoff stress tensor
"""

import jax
import jax.numpy as jnp
import os

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import solver
from fealax.utils import save_as_vtk


class HyperElasticity(Problem):
    """Hyperelastic material model using Neo-Hookean constitutive law.
    
    This class implements a finite strain hyperelasticity problem using
    the Neo-Hookean material model. The strain energy density function
    is defined and automatic differentiation is used to compute the
    first Piola-Kirchhoff stress tensor.
    """
    
    def __init__(self, mesh, material_params, **kwargs):
        """Initialize hyperelasticity problem.
        
        Args:
            mesh: Finite element mesh
            material_params: Dict with 'E' (Young's modulus) and 'nu' (Poisson's ratio)
            **kwargs: Additional arguments passed to Problem base class
        """
        self.E = material_params['E']  # Young's modulus
        self.nu = material_params['nu']  # Poisson's ratio
        
        # Compute material parameters
        self.mu = self.E / (2.0 * (1.0 + self.nu))  # Shear modulus
        self.kappa = self.E / (3.0 * (1.0 - 2.0 * self.nu))  # Bulk modulus
        
        super().__init__(mesh=mesh, **kwargs)
    
    def get_tensor_map(self):
        """Return tensor map function for hyperelasticity.
        
        The tensor map computes the first Piola-Kirchhoff stress tensor P
        from the deformation gradient F using the Neo-Hookean strain energy.
        
        Returns:
            Function that maps displacement gradients to first PK stress tensor
        """
        def strain_energy_density(F):
            """Neo-Hookean strain energy density function.
            
            Args:
                F: Deformation gradient tensor
                
            Returns:
                Strain energy density (scalar)
            """
            J = jnp.linalg.det(F)  # Jacobian determinant
            Jinv = J**(-2.0 / 3.0)  # J^(-2/3) for isochoric part
            I1 = jnp.trace(F.T @ F)  # First invariant of right Cauchy-Green tensor
            
            # Neo-Hookean strain energy
            energy = (self.mu / 2.0) * (Jinv * I1 - 3.0) + (self.kappa / 2.0) * (J - 1.0)**2
            return energy
        
        # Compute first Piola-Kirchhoff stress using automatic differentiation
        P_fn = jax.grad(strain_energy_density)
        
        def tensor_map(u_grads, *internal_vars):
            """Compute first Piola-Kirchhoff stress from displacement gradients.
            
            Args:
                u_grads: Displacement gradients with shape (vec, dim)
                        For 3D: (3, 3) - gradients at a single quadrature point
                        
            Returns:
                First Piola-Kirchhoff stress tensor with same shape as u_grads
            """
            # Deformation gradient: F = I + ∇u
            I = jnp.eye(self.dim)
            F = u_grads + I
            
            # Compute first Piola-Kirchhoff stress tensor
            P = P_fn(F)
            return P
        
        return tensor_map


def create_mesh(nx=20, ny=20, nz=20):
    """Create a structured 1x1x1 box mesh.
    
    Args:
        nx, ny, nz: Number of elements in each direction
        
    Returns:
        Mesh object
    """
    return box_mesh(nx, ny, nz, 1.0, 1.0, 1.0, ele_type="HEX8")


def define_boundary_conditions():
    """Define boundary conditions for hyperelastic deformation.
    
    This example applies a complex deformation involving rotation and stretch.
    The left face is fixed, while the right face undergoes prescribed displacements
    that combine rotation and stretching.
    
    Returns:
        List of DirichletBC objects
    """
    bcs = []
    Lx = 1.0  # Box dimension
    
    # Left face (x = 0): completely fixed
    def left_face(x):
        return jnp.abs(x[0]) < 1e-6
    
    # Fix all components on left face
    for component in range(3):
        bcs.append(DirichletBC(
            subdomain=left_face,
            vec=component,
            eval=lambda x: 0.0
        ))
    
    # Right face (x = Lx): prescribed displacement with rotation and stretch
    def right_face(x):
        return jnp.abs(x[0] - Lx) < 1e-6
    
    # Rotation angle (60 degrees)
    theta = jnp.pi / 3.0
    
    # y-component: rotation + small stretch
    def right_disp_y(x):
        return (0.5 + (x[1] - 0.5) * jnp.cos(theta) - 
                (x[2] - 0.5) * jnp.sin(theta) - x[1]) / 2.0
    
    # z-component: rotation + small stretch
    def right_disp_z(x):
        return (0.5 + (x[1] - 0.5) * jnp.sin(theta) + 
                (x[2] - 0.5) * jnp.cos(theta) - x[2]) / 2.0
    
    bcs.append(DirichletBC(
        subdomain=right_face,
        vec=0,  # x-component
        eval=lambda x: 0.0
    ))
    
    bcs.append(DirichletBC(
        subdomain=right_face,
        vec=1,  # y-component
        eval=right_disp_y
    ))
    
    bcs.append(DirichletBC(
        subdomain=right_face,
        vec=2,  # z-component
        eval=right_disp_z
    ))
    
    return bcs


def main():
    """Main function to setup and solve the hyperelasticity problem."""
    print("Setting up hyperelasticity problem...")
    
    # Create mesh
    print("Creating 1x1x1 box mesh...")
    mesh = create_mesh(nx=40, ny=40, nz=40)
    print(f"Mesh created: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")
    
    # Define boundary conditions
    print("Setting up boundary conditions...")
    bcs = define_boundary_conditions()
    print(f"Defined {len(bcs)} boundary conditions")
    
    # Material properties (softer material for large deformations)
    material_params = {
        'E': 1e6,      # Young's modulus (Pa) - 1 MPa for soft material
        'nu': 0.3      # Poisson's ratio
    }
    print(f"Material: E = {material_params['E']/1e6:.0f} MPa, ν = {material_params['nu']}")
    
    # Create hyperelasticity problem
    print("Setting up finite element problem...")
    problem = HyperElasticity(
        mesh=mesh,
        material_params=material_params,
        vec=3,  # 3D displacement field
        dim=3,  # 3D problem
        dirichlet_bcs=bcs
    )
    
    # Solver options for nonlinear problem
    solver_options = {
        'max_iter': 20,
        'tol': 1e-6,
        'newton_raphson': {}
    }
    
    print("Solving hyperelasticity problem...")
    print(f"Solver tolerance: {solver_options['tol']}")
    
    # Solve the problem
    try:
        solution = solver(problem, solver_options)
        print("✓ Solution converged!")
        
        # Post-process results
        final_solution = solution[-1] if isinstance(solution, list) else solution
        print(f"Solution vector shape: {final_solution.shape}")
        
        # Reshape solution to get displacement components
        num_nodes = mesh.points.shape[0]
        displacements = final_solution.reshape((num_nodes, 3))
        
        # Print some results
        print(f"Maximum displacement magnitude: {jnp.max(jnp.linalg.norm(displacements, axis=1)):.6f}")
        print(f"Maximum displacement components:")
        print(f"  x: {jnp.max(jnp.abs(displacements[:, 0])):.6f}")
        print(f"  y: {jnp.max(jnp.abs(displacements[:, 1])):.6f}")
        print(f"  z: {jnp.max(jnp.abs(displacements[:, 2])):.6f}")
        
        # Save results to VTK format
        print("Saving results to VTK...")
        try:
            class VTKHelper:
                def __init__(self, mesh, ele_type):
                    self.points = mesh.points
                    self.cells = mesh.cells
                    self.ele_type = ele_type
                    self.num_cells = mesh.cells.shape[0]
            
            vtk_helper = VTKHelper(mesh, "HEX8")
            
            # Prepare displacement data
            point_infos = [("displacement", displacements)]
            
            # Calculate displacement magnitude
            displacement_magnitude = jnp.linalg.norm(displacements, axis=1)
            point_infos.append(("displacement_magnitude", displacement_magnitude))
            
            # Save to VTK file
            vtk_filename = os.path.join(os.getcwd(), "hyper_elasticity_results.vtu")
            save_as_vtk(vtk_helper, vtk_filename, point_infos=point_infos)
            print(f"✓ Results saved to {vtk_filename}")
            
        except Exception as e:
            print(f"✗ Failed to save VTK: {e}")
        
        return solution, displacements
        
    except Exception as e:
        print(f"✗ Solver failed: {e}")
        return None, None


if __name__ == "__main__":
    # Enable JAX 64-bit precision for better numerical accuracy
    jax.config.update("jax_enable_x64", True)
    
    print("=" * 60)
    print("Hyperelasticity Example")
    print("=" * 60)
    
    solution, displacements = main()
    
    if solution is not None:
        print("\n" + "=" * 60)
        print("Problem solved successfully!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Problem failed to solve.")
        print("=" * 60)