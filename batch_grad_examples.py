#!/usr/bin/env python3
"""
Examples of computing gradients through batch parameter solving with NewtonSolver.

This demonstrates different approaches for gradient computation with batch parameters,
including individual gradients, batch gradients, and optimization workflows.
"""

import jax
import jax.numpy as jnp
import numpy as np
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import NewtonSolver


class SimpleElasticProblem(Problem):
    """Simple elastic problem for gradient demonstrations."""
    
    def __init__(self, mesh, E=1e5, nu=0.3, **kwargs):
        super().__init__(mesh=mesh, **kwargs)
        self.E = E
        self.nu = nu
        self.mu = E / (2.0 * (1.0 + nu))
        self.lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))

    def get_tensor_map(self):
        def tensor_map(u_grads, *internal_vars):
            strain = 0.5 * (u_grads + jnp.transpose(u_grads, (1, 0)))
            stress = 2.0 * self.mu * strain + self.lam * jnp.trace(strain) * jnp.eye(3)
            return stress
        return tensor_map
    
    def set_params(self, params):
        if 'E' in params:
            self.E = jnp.asarray(params['E'], dtype=jnp.float64)
            self.mu = self.E / (2.0 * (1.0 + self.nu))
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        if 'nu' in params:
            self.nu = jnp.asarray(params['nu'], dtype=jnp.float64)
            self.mu = self.E / (2.0 * (1.0 + self.nu))
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def setup_problem():
    """Create a simple problem for gradient demonstrations."""
    # Small mesh for fast computation
    mesh = box_mesh(3, 3, 3, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    # Simple boundary conditions
    bcs = [
        # Fix bottom face
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 0, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 1, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 2, lambda x: 0.0),
        # Apply small displacement to top face
        DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 2, lambda x: -0.01),
    ]
    
    problem = SimpleElasticProblem(
        mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs, E=1e5, nu=0.3
    )
    
    solver = NewtonSolver(problem, {'tol': 1e-8, 'max_iter': 10})
    
    return solver, mesh


def example_1_single_parameter_gradient():
    """Example 1: Gradient with respect to single parameter set."""
    print("=" * 60)
    print("EXAMPLE 1: SINGLE PARAMETER GRADIENT")
    print("=" * 60)
    
    solver, mesh = setup_problem()
    
    def objective(params_dict):
        """Objective function: sum of squared displacements."""
        solution = solver.solve(params_dict)
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # Define parameters
    params = {'E': 1e5, 'nu': 0.3}
    
    print(f"Problem size: {len(mesh.points) * 3} DOFs")
    print(f"Parameters: {params}")
    
    try:
        # Compute objective value
        obj_value = objective(params)
        print(f"Objective value: {obj_value:.6e}")
        
        # Compute gradients
        grad_fn = jax.grad(objective)
        gradients = grad_fn(params)
        
        print("✅ Gradient computation successful!")
        print(f"∂J/∂E  = {gradients['E']:.6e}")
        print(f"∂J/∂nu = {gradients['nu']:.6e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        return False


def example_2_batch_parameter_gradients():
    """Example 2: Gradients for batch parameters using vmap."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: BATCH PARAMETER GRADIENTS (VMAP)")
    print("=" * 60)
    
    solver, mesh = setup_problem()
    
    def objective_single(params_dict):
        """Objective for a single parameter set."""
        solution = solver.solve(params_dict)
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # Batch objective function using vmap
    def batch_objective(E_array, nu_array):
        """Objective for batch parameters using structured arrays."""
        # Create parameter dicts for each batch element
        def single_solve(E, nu):
            params = {'E': E, 'nu': nu}
            return objective_single(params)
        
        # Use vmap to vectorize over batch
        return jax.vmap(single_solve)(E_array, nu_array)
    
    # Define batch parameters as arrays
    E_values = jnp.array([5e4, 1e5, 2e5])
    nu_values = jnp.array([0.25, 0.3, 0.35])
    
    print(f"Batch size: {len(E_values)}")
    print(f"E values: {E_values}")
    print(f"nu values: {nu_values}")
    
    try:
        # Compute batch objectives
        objectives = batch_objective(E_values, nu_values)
        print(f"Batch objectives: {objectives}")
        
        # Compute gradients for each parameter in the batch
        grad_fn = jax.grad(batch_objective, argnums=(0, 1))  # Gradients w.r.t. both arrays
        grad_E, grad_nu = grad_fn(E_values, nu_values)
        
        print("✅ Batch gradient computation successful!")
        print(f"∂J/∂E  = {grad_E}")
        print(f"∂J/∂nu = {grad_nu}")
        
        return True
        
    except Exception as e:
        print(f"❌ Batch gradient computation failed: {e}")
        return False


def example_3_optimization_workflow():
    """Example 3: Optimization workflow with parameter constraints."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: OPTIMIZATION WORKFLOW")
    print("=" * 60)
    
    solver, mesh = setup_problem()
    
    def objective(params_array):
        """Objective function for optimization (takes array input)."""
        E, nu = params_array
        params_dict = {'E': E, 'nu': nu}
        solution = solver.solve(params_dict)
        displacement = solution[0]
        
        # Objective: minimize displacement + regularization
        disp_term = jnp.sum(displacement**2)
        reg_term = 1e-10 * (E - 1e5)**2  # Regularization around reference value
        
        return disp_term + reg_term
    
    # Initial parameters
    x0 = jnp.array([1e5, 0.3])
    
    print(f"Initial parameters: E={x0[0]:.0e}, nu={x0[1]:.3f}")
    
    try:
        # Compute initial objective and gradient
        obj_value = objective(x0)
        grad_fn = jax.grad(objective)
        gradient = grad_fn(x0)
        
        print(f"Initial objective: {obj_value:.6e}")
        print(f"Initial gradient: [{gradient[0]:.6e}, {gradient[1]:.6e}]")
        
        # Simple gradient descent step
        step_size = 1e-12
        x1 = x0 - step_size * gradient
        obj_value_1 = objective(x1)
        
        print(f"After gradient step:")
        print(f"  Parameters: E={x1[0]:.0e}, nu={x1[1]:.3f}")
        print(f"  Objective: {obj_value_1:.6e}")
        print(f"  Improvement: {obj_value - obj_value_1:.6e}")
        
        print("✅ Optimization workflow successful!")
        return True
        
    except Exception as e:
        print(f"❌ Optimization workflow failed: {e}")
        return False


def example_4_individual_batch_gradients():
    """Example 4: Compute gradients for individual elements in a batch."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: INDIVIDUAL GRADIENTS FROM BATCH")
    print("=" * 60)
    
    solver, mesh = setup_problem()
    
    def objective_single(params_dict):
        """Objective for a single parameter set."""
        solution = solver.solve(params_dict)
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # Batch parameters as list of dicts
    batch_params = [
        {'E': 5e4, 'nu': 0.3},
        {'E': 1e5, 'nu': 0.3}, 
        {'E': 2e5, 'nu': 0.3}
    ]
    
    print(f"Computing individual gradients for {len(batch_params)} parameter sets...")
    
    try:
        # Compute gradients for each parameter set individually
        grad_fn = jax.grad(objective_single)
        
        individual_gradients = []
        for i, params in enumerate(batch_params):
            gradients = grad_fn(params)
            individual_gradients.append(gradients)
            print(f"  Set {i+1}: E={params['E']:.0e}, ∂J/∂E={gradients['E']:.2e}, ∂J/∂nu={gradients['nu']:.2e}")
        
        print("✅ Individual gradient computation successful!")
        return True
        
    except Exception as e:
        print(f"❌ Individual gradient computation failed: {e}")
        return False


def main():
    """Run all gradient computation examples."""
    print("BATCH PARAMETER GRADIENT COMPUTATION EXAMPLES")
    print("=" * 60)
    
    jax.config.update("jax_enable_x64", True)
    
    results = []
    
    # Run all examples
    results.append(example_1_single_parameter_gradient())
    results.append(example_2_batch_parameter_gradients()) 
    results.append(example_3_optimization_workflow())
    results.append(example_4_individual_batch_gradients())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF GRADIENT COMPUTATION APPROACHES")
    print("=" * 60)
    
    approaches = [
        "1. Single parameter gradient (dict input)",
        "2. Batch parameter gradients (vmap)",
        "3. Optimization workflow (array input)",
        "4. Individual gradients from batch"
    ]
    
    for i, (approach, success) in enumerate(zip(approaches, results)):
        status = "✅ Working" if success else "❌ Failed"
        print(f"{approach}: {status}")
    
    print("\n📝 KEY INSIGHTS:")
    print("• Single parameter gradients work well with dict inputs")
    print("• Batch gradients require structured array inputs for vmap")
    print("• Optimization workflows prefer array parameterization")
    print("• Individual gradients from batch provide maximum flexibility")
    print("• Success depends on problem size and transformation complexity")


if __name__ == "__main__":
    main()