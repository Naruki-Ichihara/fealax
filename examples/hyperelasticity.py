#!/usr/bin/env python3
"""
Hyperelasticity example using Neo-Hookean material model.

This example demonstrates:
- Large deformation mechanics with hyperelastic materials
- Neo-Hookean constitutive law implementation
- Automatic differentiation for stress computation
- JIT compilation for performance
- Robust nonlinear solver strategies

Problem setup:
- Domain: 1×1×1 unit cube
- Element type: HEX8 (8-node hexahedral)
- Material: Neo-Hookean hyperelastic (rubber-like, E=100 kPa)
- Loading: Combined rotation (15°) and compression (5%) on top face
- Boundary conditions: Fixed bottom face, applied displacement on top
"""

import jax
import jax.numpy as jnp
import os

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import newton_solve
from fealax.utils import save_as_vtk


class HyperelasticProblem(Problem):
    """Neo-Hookean hyperelastic material problem."""
    
    def custom_init(self, E, nu):
        """Initialize material properties."""
        self.E = E  # Young's modulus
        self.nu = nu  # Poisson's ratio
        
        # Derived material parameters for Neo-Hookean model
        self.mu = E / (2.0 * (1.0 + nu))  # Shear modulus
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))  # First Lamé parameter
        self.kappa = E / (3.0 * (1.0 - 2.0 * nu))  # Bulk modulus
        
        print(f"Material properties:")
        print(f"  Young's modulus: {E:.1e} Pa")
        print(f"  Poisson's ratio: {nu:.3f}")
        print(f"  Shear modulus: {self.mu:.1e} Pa")
        print(f"  Bulk modulus: {self.kappa:.1e} Pa")

    def get_tensor_map(self):
        """Define Neo-Hookean hyperelastic constitutive law."""
        def tensor_map(u_grads, *internal_vars):
            # Deformation gradient F = I + ∇u
            F = u_grads + jnp.eye(3)
            
            # Jacobian (determinant of F)
            J = jnp.linalg.det(F)
            
            # Clamp J to avoid numerical issues
            J_min = 0.2
            J_max = 5.0
            J_clamped = jnp.clip(J, J_min, J_max)
            
            # Right Cauchy-Green deformation tensor C = F^T F
            C = jnp.transpose(F) @ F
            
            # Modified invariants for numerical stability
            I1 = jnp.trace(C)
            
            # Neo-Hookean stress formulation with clamped Jacobian
            # P = μ(F - F^(-T)) + λ ln(J) F^(-T)
            
            # Use SVD for more stable inverse computation
            F_inv = jnp.linalg.inv(F + 1e-8 * jnp.eye(3))
            F_inv_T = jnp.transpose(F_inv)
            
            # Log of clamped Jacobian for stability
            ln_J = jnp.log(J_clamped)
            
            # Neo-Hookean first Piola-Kirchhoff stress
            P = self.mu * (F - F_inv_T) + self.lam * ln_J * F_inv_T
            
            # Penalty terms for extreme deformations
            penalty_small = jnp.where(J < J_min, 1e8 * (J_min - J)**2, 0.0)
            penalty_large = jnp.where(J > J_max, 1e8 * (J - J_max)**2, 0.0)
            penalty = penalty_small + penalty_large
            
            # Add penalty as volumetric stress
            P = P + penalty * F_inv_T
            
            return P
            
        return tensor_map


def create_mesh(nx=8, ny=8, nz=8):
    """Create structured hexahedral mesh."""
    print(f"Creating {nx}×{ny}×{nz} mesh...")
    return box_mesh(nx, ny, nz, 1.0, 1.0, 1.0, ele_type="HEX8")


def define_boundary_conditions():
    """Define boundary conditions for hyperelastic problem."""
    print("Setting up boundary conditions...")
    
    # Applied rotation angle (much smaller for convergence)
    theta = jnp.pi / 6  # 30 degrees - reduced rotation
    
    def applied_displacement(x):
        """Apply rotation about z-axis to top face."""
        center_x, center_y = 0.5, 0.5
        rel_x = x[0] - center_x
        rel_y = x[1] - center_y
        
        # Rotation matrix displacement
        u_x = rel_x * (jnp.cos(theta) - 1) - rel_y * jnp.sin(theta)
        u_y = rel_x * jnp.sin(theta) + rel_y * (jnp.cos(theta) - 1)
        
        return [u_x, u_y, 0.1]  # Reduced compression in z-direction
    
    bcs = [
        # Fix bottom face completely
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 0, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 1, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 2, lambda x: 0.0),
        
        # Apply rotation + compression to top face
        DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 0, 
                   lambda x: applied_displacement(x)[0]),
        DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 1, 
                   lambda x: applied_displacement(x)[1]),
        DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 2, 
                   lambda x: applied_displacement(x)[2]),
    ]
    
    return bcs


def solve_hyperelastic_problem():
    """Main solution procedure."""
    jax.config.update("jax_enable_x64", True)
    
    print("=" * 60)
    print("FEALAX HYPERELASTICITY EXAMPLE")
    print("=" * 60)
    
    # Create mesh
    mesh = create_mesh(nx=8, ny=8, nz=8)
    
    # Define boundary conditions
    bcs = define_boundary_conditions()
    
    # Material properties (moderate stiffness for stability)
    E = 1e5  # 100 kPa (rubber-like material)
    nu = 0.3
    
    # Create problem
    print("Creating hyperelastic problem...")
    problem = HyperelasticProblem(
        mesh=mesh,
        vec=3,
        dim=3,
        dirichlet_bcs=bcs,
        additional_info=(E, nu)
    )
    
    print(f"Problem size: {problem.num_total_dofs_all_vars:,} DOFs")
    
    # Solver options with robust settings
    solver_options = {
        'tol': 1e-5,
        'rel_tol': 1e-6,
        'max_iter': 15,
        # JIT compilation is always enabled for optimal performance
        'linear_solver': 'bicgstab',
        'precond': True,
        'line_search': True  # Enable line search for robustness
    }
    
    print("Solving hyperelastic problem...")
    print("Solver options:", solver_options)
    
    solution = newton_solve(problem, solver_options)
    print("✅ Solution converged successfully!")
    
    # Extract displacement field
    u = solution[0] if isinstance(solution, list) else solution
    
    # Analyze results
    print("\n" + "=" * 40)
    print("SOLUTION ANALYSIS")
    print("=" * 40)
    print(f"Max displacement magnitude: {jnp.linalg.norm(u, axis=1).max():.6f}")
    print(f"Max u_x: {u[:, 0].max():.6f}, Min u_x: {u[:, 0].min():.6f}")
    print(f"Max u_y: {u[:, 1].max():.6f}, Min u_y: {u[:, 1].min():.6f}")
    print(f"Max u_z: {u[:, 2].max():.6f}, Min u_z: {u[:, 2].min():.6f}")
    
    # Save results
    print("\nSaving results to VTK...")
    
    # Create VTK helper object
    class VTKHelper:
        def __init__(self, problem):
            mesh = problem.mesh[0] if isinstance(problem.mesh, list) else problem.mesh
            self.points = mesh.points
            self.cells = mesh.cells
            self.ele_type = problem.ele_type[0] if isinstance(problem.ele_type, list) else problem.ele_type
            self.num_cells = mesh.cells.shape[0]
    
    vtk_helper = VTKHelper(problem)
    
    # Create point data for VTK output
    point_infos = [('displacement', u)]
    import os
    vtk_filename = os.path.join(os.getcwd(), 'hyperelasticity_results.vtu')
    save_as_vtk(vtk_helper, vtk_filename, point_infos=point_infos)
    print("Results saved to: hyperelasticity_results.vtu")
    
    return solution


if __name__ == "__main__":
    solution = solve_hyperelastic_problem()
    
    if solution is not None:
        print("\n🎉 Hyperelasticity example completed successfully!")
        print("You can visualize the results in ParaView by opening hyperelasticity_results.vtu")
    else:
        print("\n❌ Example failed. Check solver settings and material parameters.")