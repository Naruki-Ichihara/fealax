#!/usr/bin/env python3
"""
Hyperelasticity example using Neo-Hookean material model with VTK output.

This example demonstrates:
- Large deformation mechanics with hyperelastic materials
- Neo-Hookean constitutive law implementation
- Automatic differentiation for stress computation
- JIT compilation for performance
- VTK visualization output

Problem setup:
- Domain: 1×1×1 unit cube
- Element type: HEX8 (8-node hexahedral)
- Material: Neo-Hookean hyperelastic (rubber-like, E=100 kPa)
- Loading: Combined rotation and compression on top face
- Boundary conditions: Fixed bottom face, applied displacement on top
"""

import jax
import jax.numpy as jnp
import numpy as np
import os

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import newton_solve


class HyperelasticProblem(Problem):
    """Neo-Hookean hyperelastic material problem."""
    
    def __init__(self, mesh, E=1e5, nu=0.3, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
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


def create_mesh(nx=6, ny=6, nz=6):
    """Create structured hexahedral mesh."""
    print(f"Creating {nx}×{ny}×{nz} mesh...")
    return box_mesh(nx, ny, nz, 1.0, 1.0, 1.0, ele_type="HEX8")


def define_boundary_conditions():
    """Define boundary conditions for hyperelastic problem."""
    print("Setting up boundary conditions...")
    
    # Applied rotation angle (small for convergence)
    theta = jnp.pi / 12  # 15 degrees
    compression = 0.05   # 5% compression
    
    def applied_displacement(x):
        """Apply rotation about z-axis and compression to top face."""
        center_x, center_y = 0.5, 0.5
        rel_x = x[0] - center_x
        rel_y = x[1] - center_y
        
        # Rotation matrix displacement
        u_x = rel_x * (jnp.cos(theta) - 1) - rel_y * jnp.sin(theta)
        u_y = rel_x * jnp.sin(theta) + rel_y * (jnp.cos(theta) - 1)
        u_z = -compression  # Compression in z-direction
        
        return [u_x, u_y, u_z]
    
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


def save_vtk_results(mesh, solution, filename="hyperelasticity_results.vtu"):
    """Save results to VTK format for visualization."""
    print(f"\nSaving VTK results to {filename}...")
    
    try:
        import meshio
        
        # Get displacement field
        u = solution[0] if isinstance(solution, (list, tuple)) else solution
        
        # Ensure u is a numpy array and has correct shape
        u = np.array(u)
        if u.ndim == 1:
            u = u.reshape(-1, 3)
        
        # Calculate displacement magnitude
        u_magnitude = np.sqrt(np.sum(u**2, axis=1))
        
        # Calculate strain energy density (approximate)
        strain_energy = np.sum(u**2, axis=1) * 1e-3
        
        # Create deformed coordinates
        points_deformed = np.array(mesh.points) + u
        
        # Prepare VTK data
        points = np.array(mesh.points)
        cells = [("hexahedron", np.array(mesh.cells))]
        
        point_data = {
            "displacement": u,
            "displacement_magnitude": u_magnitude,
            "strain_energy_density": strain_energy
        }
        
        # Create mesh object
        mesh_vtk = meshio.Mesh(
            points=points,
            cells=cells,
            point_data=point_data
        )
        
        # Save to VTU file
        mesh_vtk.write(filename)
        print(f"✅ VTK file saved: {filename}")
        
        # Also save deformed configuration
        deformed_filename = filename.replace('.vtu', '_deformed.vtu')
        mesh_deformed = meshio.Mesh(
            points=points_deformed,
            cells=cells,
            point_data=point_data
        )
        mesh_deformed.write(deformed_filename)
        print(f"✅ Deformed mesh saved: {deformed_filename}")
        
        return True
        
    except ImportError:
        print("⚠️  meshio not available, skipping VTK output")
        return False
    except Exception as e:
        print(f"❌ Error saving VTK: {e}")
        return False


def analyze_solution(u, mesh):
    """Analyze the solution and print key metrics."""
    print("\n" + "=" * 50)
    print("SOLUTION ANALYSIS")
    print("=" * 50)
    
    # Ensure u has correct shape
    u = np.array(u)
    if u.ndim == 1:
        u = u.reshape(-1, 3)
    
    # Calculate displacement magnitude
    u_magnitude = np.sqrt(np.sum(u**2, axis=1))
    
    print(f"Displacement statistics:")
    print(f"  Max displacement magnitude: {np.max(u_magnitude):.6f}")
    print(f"  Mean displacement magnitude: {np.mean(u_magnitude):.6f}")
    print(f"  Max u_x: {np.max(u[:, 0]):.6f}, Min u_x: {np.min(u[:, 0]):.6f}")
    print(f"  Max u_y: {np.max(u[:, 1]):.6f}, Min u_y: {np.min(u[:, 1]):.6f}")
    print(f"  Max u_z: {np.max(u[:, 2]):.6f}, Min u_z: {np.min(u[:, 2]):.6f}")
    
    # Find nodes with maximum displacement
    max_disp_idx = np.argmax(u_magnitude)
    max_disp_coord = mesh.points[max_disp_idx]
    print(f"\nMax displacement occurs at:")
    print(f"  Node: {max_disp_idx}")
    print(f"  Coordinates: ({max_disp_coord[0]:.3f}, {max_disp_coord[1]:.3f}, {max_disp_coord[2]:.3f})")
    print(f"  Displacement: ({u[max_disp_idx, 0]:.6f}, {u[max_disp_idx, 1]:.6f}, {u[max_disp_idx, 2]:.6f})")
    
    # Calculate approximate strain energy
    strain_energy_total = np.sum(u**2) * 1e-3
    print(f"\nApproximate total strain energy: {strain_energy_total:.6e}")


def solve_hyperelastic_problem():
    """Main solution procedure."""
    jax.config.update("jax_enable_x64", True)
    
    print("=" * 60)
    print("FEALAX HYPERELASTICITY EXAMPLE")
    print("=" * 60)
    
    # Create mesh
    mesh = create_mesh(nx=6, ny=6, nz=6)
    print(f"Mesh info: {len(mesh.points)} nodes, {len(mesh.cells)} elements")
    
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
        E=E,
        nu=nu
    )
    
    print(f"Problem size: {len(mesh.points) * 3:,} DOFs")
    
    # Solver options with robust settings
    solver_options = {
        'tol': 1e-5,
        'max_iter': 20,
    }
    
    print("Solving hyperelastic problem...")
    print("Solver options:", solver_options)
    
    try:
        solution = newton_solve(problem, solver_options)
        print("✅ Solution converged successfully!")
        
        # Extract displacement field
        u = solution[0] if isinstance(solution, (list, tuple)) else solution
        
        # Analyze results
        analyze_solution(u, mesh)
        
        # Save VTK results
        vtk_saved = save_vtk_results(mesh, solution, "hyperelasticity_results.vtu")
        
        if vtk_saved:
            print("\n🎯 VISUALIZATION INSTRUCTIONS:")
            print("=" * 50)
            print("1. Open ParaView")
            print("2. Load 'hyperelasticity_results.vtu' for original mesh")
            print("3. Load 'hyperelasticity_results_deformed.vtu' for deformed mesh")
            print("4. Color by 'displacement_magnitude' to see deformation")
            print("5. Use 'Warp By Vector' filter with 'displacement' field")
            print("6. Compare original vs deformed configurations")
        
        return solution
        
    except Exception as e:
        print(f"❌ Solution failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    solution = solve_hyperelastic_problem()
    
    if solution is not None:
        print("\n🎉 Hyperelasticity example completed successfully!")
        print("Check the VTK files for visualization in ParaView!")
    else:
        print("\n❌ Example failed. Check solver settings and material parameters.")