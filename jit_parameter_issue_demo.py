#!/usr/bin/env python3
"""Demonstration of the JIT parameter issue and its fix."""

import jax
import jax.numpy as jnp
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import newton_solve, NewtonSolver

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

class ElasticityProblem(Problem):
    """Simple elasticity problem."""
    
    def __init__(self, mesh, E=200e9, nu=0.3, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.E = E
        self.nu = nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    
    def set_params(self, params):
        """Update material parameters."""
        if 'E' in params:
            self.E = params['E']
            self.mu = self.E / (2.0 * (1.0 + self.nu))
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        if 'nu' in params:
            self.nu = params['nu']
            self.mu = self.E / (2.0 * (1.0 + self.nu))
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
    
    def get_tensor_map(self):
        """Return stress-strain relationship."""
        def stress_strain_relation(u_grad):
            epsilon = 0.5 * (u_grad + u_grad.T)
            trace_strain = jnp.trace(epsilon)
            stress = self.lam * trace_strain * jnp.eye(3) + 2.0 * self.mu * epsilon
            return stress
        return stress_strain_relation

def create_simple_problem():
    """Create a simple test problem."""
    mesh = box_mesh(5, 5, 5, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    bcs = [
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 0, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 1, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 2, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 2, lambda x: -0.01),
    ]
    
    return ElasticityProblem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)

def demonstrate_issue():
    """Demonstrate the JIT parameter update issue."""
    print("DEMONSTRATING JIT PARAMETER UPDATE ISSUE")
    print("=" * 50)
    
    problem = create_simple_problem()
    solver_options = {'tol': 1e-6, 'max_iter': 10}
    
    # Test parameters
    params_list = [
        {'E': 100e9, 'nu': 0.2},  # Softer material
        {'E': 200e9, 'nu': 0.3},  # Default material
        {'E': 300e9, 'nu': 0.4},  # Stiffer material
    ]
    
    print("\n1. NON-JIT APPROACH (Works correctly):")
    print("-" * 40)
    
    def solve_non_jit(params):
        problem.set_params(params)
        solution = newton_solve(problem, solver_options)
        return solution[0]
    
    for i, params in enumerate(params_list):
        solution = solve_non_jit(params)
        max_disp = float(jnp.max(jnp.abs(solution))) * 1000
        print(f"  E={params['E']/1e9:.0f}GPa: max_disp = {max_disp:.3f}mm")
    
    print("\n2. JIT APPROACH (Incorrectly returns same result):")
    print("-" * 40)
    
    @jax.jit
    def solve_jit_wrong(params):
        problem.set_params(params)  # This doesn't work inside JIT!
        solution = newton_solve(problem, solver_options)
        return solution[0]
    
    for i, params in enumerate(params_list):
        solution = solve_jit_wrong(params)
        max_disp = float(jnp.max(jnp.abs(solution))) * 1000
        print(f"  E={params['E']/1e9:.0f}GPa: max_disp = {max_disp:.3f}mm ❌")
    
    print("\n3. CORRECT APPROACH USING NewtonSolver:")
    print("-" * 40)
    
    solver = NewtonSolver(problem, solver_options)
    
    for i, params in enumerate(params_list):
        solution = solver.solve(params)
        max_disp = float(jnp.max(jnp.abs(solution[0]))) * 1000
        print(f"  E={params['E']/1e9:.0f}GPa: max_disp = {max_disp:.3f}mm ✅")
    
    print("\n4. BATCH SOLVING (Most efficient):")
    print("-" * 40)
    
    batch_solutions = solver.solve(params_list)
    for i, params in enumerate(params_list):
        max_disp = float(jnp.max(jnp.abs(batch_solutions[0][i]))) * 1000
        print(f"  E={params['E']/1e9:.0f}GPa: max_disp = {max_disp:.3f}mm ✅")
    
    print("\nSUMMARY:")
    print("=" * 50)
    print("❌ JIT-compiling functions that modify problem state doesn't work")
    print("✅ Use NewtonSolver for proper parameter handling")
    print("✅ Batch solving is most efficient for multiple parameters")

if __name__ == "__main__":
    demonstrate_issue()