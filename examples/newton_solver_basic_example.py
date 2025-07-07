#!/usr/bin/env python3
"""
Basic NewtonSolver interface example.

This example shows the minimal usage pattern for the NewtonSolver interface
and demonstrates the key improvements over the old newton_solve function.
"""

import jax
import jax.numpy as jnp
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import NewtonSolver


class SimpleProblem(Problem):
    """Simple linear elastic problem for demonstration."""
    
    def __init__(self, mesh, E=1e5, nu=0.3, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.E = E
        self.nu = nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def get_tensor_map(self):
        """Linear elastic constitutive law."""
        def tensor_map(u_grads, *internal_vars):
            strain = 0.5 * (u_grads + jnp.transpose(u_grads, (1, 0)))
            stress = 2.0 * self.mu * strain + self.lam * jnp.trace(strain) * jnp.eye(3)
            return stress
        return tensor_map
    
    def set_params(self, params):
        """Set material parameters."""
        if 'E' in params:
            # Handle both regular values and JAX tracers
            self.E = jnp.asarray(params['E'], dtype=jnp.float64)
            self.mu = self.E / (2.0 * (1.0 + self.nu))
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        if 'nu' in params:
            # Handle both regular values and JAX tracers
            self.nu = jnp.asarray(params['nu'], dtype=jnp.float64)
            self.mu = self.E / (2.0 * (1.0 + self.nu))
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def main():
    """Demonstrate basic NewtonSolver usage."""
    print("NEWTON SOLVER BASIC EXAMPLE")
    print("=" * 40)
    
    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)
    
    # Create a simple mesh
    mesh = box_mesh(10, 10, 10, 1.0, 1.0, 1.0, ele_type="HEX8")
    print(f"Mesh: {len(mesh.points)} nodes, {len(mesh.cells)} elements")
    
    # Define boundary conditions
    bcs = [
        # Fix bottom face
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 0, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 1, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 2, lambda x: 0.0),
        # Apply displacement to top face
        DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 2, lambda x: -0.01),
    ]
    
    # Create problem
    problem = SimpleProblem(
        mesh=mesh,
        vec=3,
        dim=3,
        dirichlet_bcs=bcs,
        E=1e5,
        nu=0.3
    )
    
    print(f"Problem: {len(mesh.points) * 3} DOFs")
    
    
    # Create solver
    solver_options = {'tol': 1e-6, 'max_iter': 15}
    solver = NewtonSolver(problem, solver_options)
    
    # Single parameter solve
    print("\n   a) Single parameter solve:")
    params = {'E': 1e5, 'nu': 0.3}
    solution = solver.solve(params)
    max_disp = float(jnp.max(jnp.abs(solution[0])))
    print(f"      Max displacement: {max_disp:.6f}")
    
    # Batch parameter solve
    print("\n   b) Batch parameter solve:")
    batch_params = [
        {'E': 5e4, 'nu': 0.3},
        {'E': 1e5, 'nu': 0.3},
        {'E': 2e5, 'nu': 0.3}
    ]
    batch_solutions = solver.solve(batch_params)
    print(f"      Solved {len(batch_params)} parameter sets")
    print(f"      Solution shape: {batch_solutions[0].shape}")


if __name__ == "__main__":
    main()