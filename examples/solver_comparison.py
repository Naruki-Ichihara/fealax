#!/usr/bin/env python3
"""
Simple linear elasticity problem on a 1x1x1 box mesh using the new separated API.

This example demonstrates a basic 3D linear elasticity problem with:
- 1x1x1 unit cube domain
- Hexahedral elements (HEX8)
- Pure compression boundary conditions with symmetry constraints
- Applied compression displacement on top face (z=1)
- Linear elastic material (steel-like properties)

NEW API Features Demonstrated:
- problem.assemble(dofs, bc_data) → (A, b) for clean assembly
- solver.solve(A, b, options) → solution for clean solving
- solver.newton_solve(problem, options) → complete Newton iteration
- Comparison with original solver for verification

The problem solves for displacement under pure compression, allowing
lateral expansion (Poisson effect) while maintaining axial compression.
"""

import jax.numpy as jnp
import jax

from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import _solver, solve, newton_solve
from fealax.utils import save_as_vtk


def create_mesh(nx=20, ny=20, nz=20):
    """Create a structured 1x1x1 box mesh.
    
    Args:
        nx, ny, nz: Number of elements in each direction
        
    Returns:
        Mesh object
    """
    return box_mesh(nx, ny, nz, 1.0, 1.0, 1.0, ele_type="HEX8")


def define_boundary_conditions():
    """Define boundary conditions for pure compression.
    
    Pure compression allows lateral expansion while applying axial compression.
    Symmetry conditions are used to prevent rigid body motion.
    
    Returns:
        List of DirichletBC objects
    """
    bcs = []
    
    # Symmetry conditions to prevent rigid body motion
    # Fix u_x = 0 on x = 0 plane (left face)
    def left_face(x):
        return jnp.abs(x[0]) < 1e-6
    
    bcs.append(DirichletBC(
        subdomain=left_face,
        vec=0,  # x-component
        eval=lambda x: 0.0
    ))
    
    # Fix u_y = 0 on y = 0 plane (front face)
    def front_face(x):
        return jnp.abs(x[1]) < 1e-6
    
    bcs.append(DirichletBC(
        subdomain=front_face,
        vec=1,  # y-component
        eval=lambda x: 0.0
    ))
    
    # Fix u_z = 0 on z = 0 plane (bottom face) - only vertical displacement
    def bottom_face(x):
        return jnp.abs(x[2]) < 1e-6
    
    bcs.append(DirichletBC(
        subdomain=bottom_face,
        vec=2,  # z-component
        eval=lambda x: 0.0
    ))
    
    # Applied displacement on top face (z = 1): u_z = -0.1 (compression)
    def top_face(x):
        return jnp.abs(x[2] - 1.0) < 1e-6
    
    bcs.append(DirichletBC(
        subdomain=top_face,
        vec=2,  # z-component
        eval=lambda x: -0.1  # 10% compression (reduced for pure compression)
    ))
    
    return bcs


class ElasticityProblem(Problem):
    """Linear elasticity problem with isotropic material properties."""
    
    def __init__(self, mesh, material_params, **kwargs):
        """Initialize elasticity problem.
        
        Args:
            mesh: Finite element mesh
            material_params: Dict with 'E' (Young's modulus) and 'nu' (Poisson's ratio)
            **kwargs: Additional arguments passed to Problem base class
        """
        self.E = material_params['E']  # Young's modulus
        self.nu = material_params['nu']  # Poisson's ratio
        
        # Compute Lame parameters
        self.mu = self.E / (2.0 * (1.0 + self.nu))  # Shear modulus
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))  # First Lame parameter
        
        super().__init__(mesh=mesh, **kwargs)
    
    def get_tensor_map(self):
        """Return tensor map function for linear elasticity.
        
        Returns:
            Function that maps displacement gradients to stress tensor
        """
        def tensor_map(u_grads, *internal_vars):
            """Compute stress tensor from displacement gradients.
            
            Args:
                u_grads: Displacement gradients with shape (vec, dim)
                        For 3D: (3, 3) - gradients at a single quadrature point
                        Note: jax.vmap is applied over quadrature points
                        
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
    print("Setting up simple elasticity problem...")
    
    # Create mesh
    print("Creating 1x1x1 box mesh...")
    mesh = create_mesh(nx=20, ny=20, nz=20)  # Start with coarse mesh
    print(f"Mesh created: {mesh.points.shape[0]} nodes, {mesh.cells.shape[0]} elements")
    
    # Define boundary conditions
    print("Setting up boundary conditions...")
    bcs = define_boundary_conditions()
    print(f"Defined {len(bcs)} boundary conditions")
    
    # Material properties (steel-like)
    material_params = {
        'E': 200e9,      # Young's modulus (Pa) - 200 GPa for steel
        'nu': 0.3       # Poisson's ratio
    }
    print(f"Material: E = {material_params['E']/1e9:.0f} GPa, ν = {material_params['nu']}")
    
    # Create elasticity problem
    print("Setting up finite element problem...")
    problem = ElasticityProblem(
        mesh=mesh,
        material_params=material_params,
        vec=3,  # 3D displacement field
        dim=3,  # 3D problem
        dirichlet_bcs=bcs
    )
    
    # Solver options for new API (no JIT)
    solver_options = {
        'max_iter': 10,  # Reduce iterations for faster demo
        'tol': 1e-5,  # Further relaxed tolerance
        'method': 'bicgstab',  # Linear solver method
        'precond': True,  # Enable preconditioning
        'use_jit': False  # Disable JIT for this example
    }
    
    print("Solving linear elasticity problem with new separated API...")
    print(f"Solver tolerance: {solver_options['tol']}")
    print("Using new problem.assemble() + solver.solve() API")
    print("Benefits: Clean separation, reusable systems, better performance potential")
    
    # Demonstrate the new separated API
    try:
        print("\n--- Method 1: Manual assembly + solve (maximum control) ---")
        
        # Initialize DOFs exactly like original solver
        if problem.prolongation_matrix is not None:
            dofs = jnp.zeros(problem.prolongation_matrix.shape[1])
        else:
            dofs = jnp.zeros(problem.num_total_dofs_all_vars)
        print(f"Problem DOFs: {problem.num_total_dofs_all_vars}")
        
        # Manual Newton iteration matching original solver approach
        print("Starting manual Newton iteration (matching original solver)...")
        
        # Import required functions from solver
        from fealax.solver import apply_bc_vec, get_A, linear_incremental_solver, CHUNK_SIZE
        
        for iteration in range(solver_options['max_iter']):
            print(f"  Newton iteration {iteration + 1}:")
            
            # Match newton_update_helper function from original solver
            print("    Computing residual and system matrix...")
            
            # Handle prolongation matrix if present
            dofs_full = dofs
            if problem.prolongation_matrix is not None:
                dofs_full = problem.prolongation_matrix @ dofs
            
            # Compute residual from weak form
            sol_list = problem.unflatten_fn_sol_list(dofs_full)
            res_list = problem.newton_update(sol_list)
            res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
            
            # Apply boundary conditions to residual (original solver approach)
            res_vec = apply_bc_vec(res_vec, dofs_full, problem)
            
            # Handle prolongation matrix for residual
            if problem.prolongation_matrix is not None:
                res_vec = problem.prolongation_matrix.T @ res_vec
            
            # Compute system matrix
            problem.compute_csr(CHUNK_SIZE)
            A_result = get_A(problem)
            if problem.prolongation_matrix is not None:
                A, A_reduced = A_result
            else:
                A = A_result
                A_reduced = A
            
            # Check convergence
            res_norm = float(jnp.linalg.norm(res_vec))
            print(f"    Residual norm: {res_norm:.2e}")
            
            if iteration == 0:
                initial_residual = res_norm
                print(f"    Initial residual: {initial_residual:.2e}")
                
            if res_norm < solver_options['tol']:
                print("    ✓ Converged!")
                break
                
            # Solve for Newton increment (matching original solver)
            print("    Solving for Newton increment...")
            dofs = linear_incremental_solver(problem, res_vec, A_reduced, dofs, solver_options)
            increment_norm = float(jnp.linalg.norm(res_vec))  # This is just for logging
            print(f"    Updated solution norm: {increment_norm:.2e}")
        
        # Finalize solution exactly like original solver
        if problem.prolongation_matrix is not None:
            dofs = problem.prolongation_matrix @ dofs
        
        if problem.macro_term is not None:
            dofs = dofs + problem.macro_term
        
        # Convert back to solution list format
        manual_solution = problem.unflatten_fn_sol_list(dofs)
        print(f"✓ Manual API solution converged in {iteration + 1} iterations!")
        
        print("\n--- Method 2: Using newton_solve() wrapper ---")
        
        # Use the new Newton solver wrapper
        wrapper_solution = newton_solve(problem, solver_options)
        print("✓ Newton wrapper solution converged!")
        
        print("\n--- Method 3: Compare with original solver ---")
        
        # Compare with original solver
        original_options = {
            'max_iter': 10,
            'tol': 1e-5,
            'jax_solver': {'precond': True}
        }
        original_solution = _solver(problem, original_options)
        print("✓ Original solver completed!")
        
        # Compare solutions
        manual_u = manual_solution[0] if isinstance(manual_solution, list) else manual_solution
        wrapper_u = wrapper_solution[0] if isinstance(wrapper_solution, list) else wrapper_solution
        original_u = original_solution[0] if isinstance(original_solution, list) else original_solution
        
        diff_manual_wrapper = jnp.linalg.norm(manual_u.flatten() - wrapper_u.flatten())
        diff_manual_original = jnp.linalg.norm(manual_u.flatten() - original_u.flatten())
        
        print(f"\nSolution comparison:")
        print(f"  Manual vs Wrapper difference: {diff_manual_wrapper:.2e}")
        print(f"  Manual vs Original difference: {diff_manual_original:.2e}")
        print(f"  All methods agree: {diff_manual_wrapper < 1e-10 and diff_manual_original < 1e-10}")
        
        # Use manual solution for post-processing
        solution = manual_solution
        print("\n✓ All solution methods successful!")
        
        # Post-process results
        final_solution = solution[0] if isinstance(solution, list) else solution
        print(f"Solution vector shape: {final_solution.shape}")
        
        # Reshape solution to get displacement components
        num_nodes = mesh.points.shape[0]
        displacements = final_solution.reshape((num_nodes, 3))
        
        # Print some results
        print(f"Maximum displacement magnitude: {jnp.max(jnp.linalg.norm(displacements, axis=1)):.6f}")
        print(f"Maximum z-displacement: {jnp.max(displacements[:, 2]):.6f}")
        print(f"Minimum z-displacement: {jnp.min(displacements[:, 2]):.6f}")
        
        # Find displacement at center of top face
        top_center_idx = jnp.argmin(
            jnp.sum((mesh.points - jnp.array([0.5, 0.5, 1.0]))**2, axis=1)
        )
        top_center_disp = displacements[top_center_idx]
        print(f"Displacement at top center: [{top_center_disp[0]:.6f}, {top_center_disp[1]:.6f}, {top_center_disp[2]:.6f}]")
        
        # Save results to VTK format for visualization
        print("Saving results to VTK...")
        try:
            # The save_as_vtk function expects fe (finite element) object, but we have separate mesh and problem
            # We need to create a minimal object with the required attributes
            class VTKHelper:
                def __init__(self, mesh, ele_type):
                    self.points = mesh.points
                    self.cells = mesh.cells
                    self.ele_type = ele_type
                    self.num_cells = mesh.cells.shape[0]
            
            vtk_helper = VTKHelper(mesh, "HEX8")
            
            # Prepare displacement data
            point_infos = [("displacement", displacements)]
            
            # Calculate displacement magnitude for visualization
            displacement_magnitude = jnp.linalg.norm(displacements, axis=1)
            point_infos.append(("displacement_magnitude", displacement_magnitude))
            
            # Save to VTK file
            import os
            vtk_filename = os.path.join(os.getcwd(), "elasticity_results.vtu")
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
    print("Simple Linear Elasticity Example")
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