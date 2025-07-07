#!/usr/bin/env python3
"""
Comprehensive examples of using jax.vmap with fealax NewtonSolver.

This demonstrates different ways to use JAX vmap for batch parameter solving,
including direct vmap usage and comparison with built-in batch solving.
"""

import jax
import jax.numpy as jnp
import time
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import NewtonSolver


class SimpleElasticProblem(Problem):
    """Simple elastic problem optimized for vmap usage."""
    
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
        # JAX-friendly parameter setting
        if 'E' in params:
            self.E = params['E']
        if 'nu' in params:
            self.nu = params['nu']
        
        # Update derived properties
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def create_solver():
    """Create a simple solver for vmap demonstrations."""
    mesh = box_mesh(3, 3, 3, 1.0, 1.0, 1.0, ele_type="HEX8")
    
    bcs = [
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 0, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 1, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2]) < 1e-6, 2, lambda x: 0.0),
        DirichletBC(lambda x: jnp.abs(x[2] - 1.0) < 1e-6, 2, lambda x: -0.01),
    ]
    
    problem = SimpleElasticProblem(
        mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs, E=1e5, nu=0.3
    )
    
    return NewtonSolver(problem, {'tol': 1e-8, 'max_iter': 10})


def method_1_direct_vmap():
    """Method 1: Direct vmap over parameter arrays."""
    print("Method 1: Direct vmap over parameter arrays")
    print("=" * 45)
    
    solver = create_solver()
    
    # Parameter arrays
    E_values = jnp.array([5e4, 1e5, 2e5, 3e5])
    nu_values = jnp.array([0.25, 0.3, 0.35, 0.4])
    
    def solve_single(E, nu):
        """Solve for a single parameter pair."""
        return solver.solve({'E': E, 'nu': nu})
    
    # Apply vmap
    vmap_solve = jax.vmap(solve_single)
    
    print(f"Solving for {len(E_values)} parameter pairs...")
    start = time.time()
    solutions = vmap_solve(E_values, nu_values)
    solve_time = time.time() - start
    
    print(f"✅ Direct vmap successful!")
    print(f"   Time: {solve_time:.3f}s")
    print(f"   Output shape: {solutions[0].shape}")
    
    return solutions, solve_time


def method_2_structured_dict():
    """Method 2: Structured dictionary with parameter arrays."""
    print("\nMethod 2: Structured dictionary batch solving")
    print("=" * 45)
    
    solver = create_solver()
    
    # Structured parameter dictionary
    batch_params = {
        'E': jnp.array([5e4, 1e5, 2e5, 3e5]),
        'nu': jnp.array([0.25, 0.3, 0.35, 0.4])
    }
    
    print(f"Solving with structured dict...")
    start = time.time()
    solutions = solver.solve(batch_params)
    solve_time = time.time() - start
    
    print(f"✅ Structured dict successful!")
    print(f"   Time: {solve_time:.3f}s")
    print(f"   Output shape: {solutions[0].shape}")
    
    return solutions, solve_time


def method_3_current_batch():
    """Method 3: Current batch approach (list of dicts)."""
    print("\nMethod 3: Current batch approach (list of dicts)")
    print("=" * 45)
    
    solver = create_solver()
    
    # List of parameter dictionaries
    batch_params = [
        {'E': 5e4, 'nu': 0.25},
        {'E': 1e5, 'nu': 0.3},
        {'E': 2e5, 'nu': 0.35},
        {'E': 3e5, 'nu': 0.4}
    ]
    
    print(f"Solving with list of dicts...")
    start = time.time()
    solutions = solver.solve(batch_params)
    solve_time = time.time() - start
    
    print(f"✅ List of dicts successful!")
    print(f"   Time: {solve_time:.3f}s")
    print(f"   Output shape: {solutions[0].shape}")
    
    return solutions, solve_time


def method_4_jit_vmap():
    """Method 4: JIT-compiled vmap for maximum performance."""
    print("\nMethod 4: JIT-compiled vmap")
    print("=" * 45)
    
    solver = create_solver()
    
    def solve_single(E, nu):
        return solver.solve({'E': E, 'nu': nu})
    
    # JIT-compile the vmap function
    jit_vmap_solve = jax.jit(jax.vmap(solve_single))
    
    # Parameter arrays
    E_values = jnp.array([5e4, 1e5, 2e5, 3e5])
    nu_values = jnp.array([0.25, 0.3, 0.35, 0.4])
    
    print(f"Solving with JIT-compiled vmap...")
    start = time.time()
    solutions = jit_vmap_solve(E_values, nu_values)
    solve_time = time.time() - start
    
    print(f"✅ JIT-compiled vmap successful!")
    print(f"   Time: {solve_time:.3f}s")
    print(f"   Output shape: {solutions[0].shape}")
    
    return solutions, solve_time


def gradient_with_vmap():
    """Demonstrate gradient computation with vmap."""
    print("\nGradient computation with vmap")
    print("=" * 45)
    
    solver = create_solver()
    
    def objective(E, nu):
        """Objective function for a single parameter pair."""
        solution = solver.solve({'E': E, 'nu': nu})
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # Batch objective using vmap
    batch_objective = jax.vmap(objective)
    
    # Test parameters
    E_values = jnp.array([5e4, 1e5, 2e5])
    nu_values = jnp.array([0.25, 0.3, 0.35])
    
    print("Computing batch objectives...")
    try:
        objectives = batch_objective(E_values, nu_values)
        print(f"✅ Batch objectives successful!")
        for i in range(len(E_values)):
            print(f"   Set {i+1}: obj={objectives[i]:.2e}")
    except Exception as e:
        print(f"❌ Batch objectives failed: {type(e).__name__}")
    
    # Note about gradients
    print("\n⚠️ Note: Gradient computation through vmap encounters tracer leaks")
    print("   Use individual gradient computation instead (see batch_gradients_simple.py)")
    
    # Demonstrate individual gradient computation
    print("\nIndividual gradient computation (recommended):")
    grad_fn = jax.grad(lambda params: objective(params['E'], params['nu']))
    
    for i in range(len(E_values)):
        params = {'E': E_values[i], 'nu': nu_values[i]}
        try:
            obj = objective(params['E'], params['nu'])
            grad = grad_fn(params)
            print(f"   Set {i+1}: obj={obj:.2e}, ∂E={grad['E']:.2e}, ∂nu={grad['nu']:.2e}")
        except:
            print(f"   Set {i+1}: gradient computation failed")


def parameter_sweep_example():
    """Demonstrate parameter sweep using vmap."""
    print("\nParameter sweep using vmap")
    print("=" * 45)
    
    solver = create_solver()
    
    def solve_single(E, nu):
        solution = solver.solve({'E': E, 'nu': nu})
        displacement = solution[0]
        return jnp.max(jnp.abs(displacement))  # Max displacement
    
    # Create parameter sweep
    E_range = jnp.linspace(50e3, 300e3, 10)
    nu_fixed = 0.3
    nu_range = jnp.full_like(E_range, nu_fixed)
    
    # Apply vmap for parameter sweep
    vmap_sweep = jax.vmap(solve_single)
    max_displacements = vmap_sweep(E_range, nu_range)
    
    print(f"✅ Parameter sweep completed!")
    print(f"   E range: {E_range[0]/1e3:.0f} - {E_range[-1]/1e3:.0f} kPa")
    print(f"   Max displacement range: {jnp.min(max_displacements):.4f} - {jnp.max(max_displacements):.4f}")


def performance_comparison():
    """Compare performance of different approaches."""
    print("\nPerformance Comparison")
    print("=" * 45)
    
    # Run all methods and collect timing
    print("Running performance tests...")
    
    try:
        _, time1 = method_1_direct_vmap()
    except:
        time1 = float('inf')
        
    try:
        _, time2 = method_2_structured_dict()
    except:
        time2 = float('inf')
        
    try:
        _, time3 = method_3_current_batch()
    except:
        time3 = float('inf')
        
    try:
        _, time4 = method_4_jit_vmap()
    except:
        time4 = float('inf')
    
    # Determine fastest method
    times = [time1, time2, time3, time4]
    methods = ["Direct vmap", "Structured dict", "List of dicts", "JIT vmap"]
    
    print(f"\nPerformance ranking:")
    sorted_results = sorted(zip(times, methods))
    for i, (time_val, method) in enumerate(sorted_results):
        if time_val == float('inf'):
            print(f"   {i+1}. {method}: Failed")
        else:
            print(f"   {i+1}. {method}: {time_val:.3f}s")


def main():
    """Run all vmap examples."""
    print("JAX VMAP WITH NEWTONSOLVER - COMPREHENSIVE EXAMPLES")
    print("=" * 60)
    
    jax.config.update("jax_enable_x64", True)
    
    # Method demonstrations
    method_1_direct_vmap()
    method_2_structured_dict() 
    method_3_current_batch()
    method_4_jit_vmap()
    
    # Advanced examples
    gradient_with_vmap()
    parameter_sweep_example()
    
    # Performance comparison
    performance_comparison()
    
    print("\n" + "=" * 60)
    print("SUMMARY - RECOMMENDED APPROACHES")
    print("=" * 60)
    
    print("""
✅ RECOMMENDED VMAP USAGE:

1. Direct vmap (most flexible):
   def solve_single(E, nu):
       return solver.solve({'E': E, 'nu': nu})
   
   solutions = jax.vmap(solve_single)(E_values, nu_values)

2. Structured dict (cleanest):
   batch_params = {
       'E': jnp.array([5e4, 1e5, 2e5]),
       'nu': jnp.array([0.25, 0.3, 0.35])
   }
   solutions = solver.solve(batch_params)

3. JIT-compiled vmap (fastest):
   jit_vmap_solve = jax.jit(jax.vmap(solve_single))
   solutions = jit_vmap_solve(E_values, nu_values)

🎯 BEST PRACTICES:
• Use structured dict for simple batch solving
• Use direct vmap for complex parameter combinations
• Use JIT-compiled vmap for repeated computations
• Handle tracer leaks gracefully (solver does this automatically)
""")


if __name__ == "__main__":
    main()