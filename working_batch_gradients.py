#!/usr/bin/env python3
"""
Working solution for computing gradients with batch parameters.

This demonstrates the practical approach that works with the current NewtonSolver:
- Use solver.solve() for batch solving (forward pass)
- Use individual jax.grad() calls for gradients (works reliably)
- JIT compile for performance
"""

import jax
import jax.numpy as jnp
import time
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
        # Check if we're dealing with JAX tracers and handle appropriately
        def is_tracer(x):
            return hasattr(x, 'aval') or str(type(x)).find('Tracer') != -1
        
        # For JAX tracers, we need to be careful about state modifications
        if 'E' in params:
            E_val = params['E']
            if is_tracer(E_val):
                # During gradient computation, avoid state modification
                # Store as JAX array but handle operations carefully
                self.E = E_val
            else:
                self.E = jnp.asarray(E_val, dtype=jnp.float64)
                
        if 'nu' in params:
            nu_val = params['nu']
            if is_tracer(nu_val):
                self.nu = nu_val
            else:
                self.nu = jnp.asarray(nu_val, dtype=jnp.float64)
        
        # Compute derived properties safely
        try:
            self.mu = self.E / (2.0 * (1.0 + self.nu))
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))
        except:
            # If tracers cause issues, use a more robust approach
            # This should not happen with the tracer detection above, but just in case
            E_safe = self.E if hasattr(self, 'E') else 1e5
            nu_safe = self.nu if hasattr(self, 'nu') else 0.3
            self.mu = E_safe / (2.0 * (1.0 + nu_safe))
            self.lam = E_safe * nu_safe / ((1.0 + nu_safe) * (1.0 - 2.0 * nu_safe))


def create_solver():
    """Create a simple solver for demonstration."""
    mesh = box_mesh(3, 3, 3, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    bcs = [
        # Fix bottom face
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 0, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 1, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 2, lambda x: 0.0),
        # Apply displacement to top face
        DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 2, lambda x: -0.01),
    ]
    
    problem = SimpleElasticProblem(
        mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs, E=1e5, nu=0.3
    )
    
    return NewtonSolver(problem, {'tol': 1e-8, 'max_iter': 10})


def demonstrate_working_approach():
    """Demonstrate the working approach for batch gradients."""
    print("WORKING APPROACH: BATCH SOLVING + INDIVIDUAL GRADIENTS")
    print("=" * 60)
    
    jax.config.update("jax_enable_x64", True)
    
    # Create solver
    solver = create_solver()
    print("✅ Solver created")
    
    # Define batch parameters
    batch_params = [
        {'E': 5e4, 'nu': 0.3},
        {'E': 1e5, 'nu': 0.3},
        {'E': 2e5, 'nu': 0.3}
    ]
    
    print(f"📦 Batch parameters: {len(batch_params)} sets")
    for i, params in enumerate(batch_params):
        print(f"   Set {i+1}: E={params['E']:.0e}, nu={params['nu']:.2f}")
    
    # STEP 1: Batch solving (this works great!)
    print(f"\n🚀 STEP 1: Batch solving...")
    
    start = time.time()
    batch_solutions = solver.solve(batch_params)
    batch_time = time.time() - start
    
    print(f"✅ Batch solving completed in {batch_time:.3f}s")
    print(f"   Solution shape: {batch_solutions[0].shape}")
    
    # Compute objectives from batch solutions
    batch_objectives = []
    for i in range(len(batch_params)):
        displacement = batch_solutions[0][i]
        obj = jnp.sum(displacement**2)
        batch_objectives.append(float(obj))
    
    print(f"   Objectives: {[f'{obj:.6e}' for obj in batch_objectives]}")
    
    # STEP 2: Individual gradients (this works!)
    print(f"\n🎯 STEP 2: Individual gradient computation...")
    
    # Create a fresh solver for gradient computation
    grad_solver = create_solver()
    
    def objective(params_dict):
        """Objective function for gradient computation."""
        solution = grad_solver.solve(params_dict)
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # JIT compile for performance
    grad_fn = jax.jit(jax.grad(objective))
    
    # Compute gradients for each parameter set
    print("   Computing gradients...")
    
    start = time.time()
    batch_gradients = []
    
    for i, params in enumerate(batch_params):
        gradients = grad_fn(params)
        batch_gradients.append(gradients)
        
        print(f"   Set {i+1}: ∂J/∂E={gradients['E']:.6e}, ∂J/∂nu={gradients['nu']:.6e}")
    
    grad_time = time.time() - start
    print(f"✅ Gradient computation completed in {grad_time:.3f}s")
    
    # STEP 3: Summary and analysis
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"   Batch solve time:     {batch_time:.3f}s")
    print(f"   Gradient compute time: {grad_time:.3f}s")
    print(f"   Total time:           {batch_time + grad_time:.3f}s")
    print(f"   Average per gradient: {grad_time / len(batch_params):.3f}s")
    
    # Combine results
    results = []
    for i, (params, obj, grad) in enumerate(zip(batch_params, batch_objectives, batch_gradients)):
        results.append({
            'parameters': params,
            'objective': obj,
            'gradients': grad
        })
    
    return results


def optimization_example():
    """Show how to use this for optimization."""
    print(f"\n🎯 OPTIMIZATION EXAMPLE:")
    print("=" * 40)
    
    solver = create_solver()
    
    def objective(params_dict):
        solution = solver.solve(params_dict)
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # JIT compile for performance
    grad_fn = jax.jit(jax.grad(objective))
    
    # Initial parameters
    params = {'E': 1e5, 'nu': 0.3}
    
    print(f"Initial: E={params['E']:.0e}, nu={params['nu']:.3f}")
    
    # Gradient descent steps
    for step in range(3):
        obj_val = objective(params)
        gradients = grad_fn(params)
        
        print(f"Step {step+1}: obj={obj_val:.6e}, ∂E={gradients['E']:.2e}, ∂nu={gradients['nu']:.2e}")
        
        # Update (small step)
        step_size = 1e-12
        params = {
            'E': params['E'] - step_size * gradients['E'],
            'nu': max(0.1, min(0.45, params['nu'] - step_size * gradients['nu']))
        }
    
    print("✅ Optimization example completed")


def main():
    """Main demonstration."""
    print("PRACTICAL SOLUTION FOR BATCH GRADIENTS WITH NEWTONSOLVER")
    print("=" * 70)
    
    # Run the working approach
    results = demonstrate_working_approach()
    
    # Show optimization example
    optimization_example()
    
    # Summary
    print(f"\n" + "=" * 70)
    print("🎉 SOLUTION SUMMARY:")
    print("=" * 70)
    
    print("""
✅ WHAT WORKS:
• solver.solve(batch_params) → Fast batch solving
• jax.grad(individual_objective) → Reliable gradients  
• JIT compilation → Excellent performance
• Sequential gradient computation → Always works

📝 RECOMMENDED WORKFLOW:
1. Use solver.solve(batch_params) for forward pass
2. Create separate solver for gradient computation  
3. Use jax.grad() on individual parameter sets
4. JIT compile gradient function for performance
5. Combine for complete optimization workflows

🔧 PERFORMANCE TIPS:
• JIT compile gradient functions for speed
• Reuse solver instances when possible
• Use small problems for gradient computation
• Consider finite differences for large batches

This approach provides both the performance benefits of batch solving
and the reliability of individual gradient computation!
""")


if __name__ == "__main__":
    main()