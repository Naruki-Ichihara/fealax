#!/usr/bin/env python3
"""
Practical approaches for computing gradients with batch parameters.

Since direct JAX grad through batch solver.solve() encounters tracer leaks,
this demonstrates practical workarounds that achieve the same goals.
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


def setup_solver():
    """Create solver for demonstrations."""
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
    
    return NewtonSolver(problem, {'tol': 1e-8, 'max_iter': 10}), mesh


def approach_1_individual_gradients():
    """Approach 1: Compute gradients for each parameter set individually."""
    print("=" * 60)
    print("APPROACH 1: INDIVIDUAL GRADIENTS (RECOMMENDED)")
    print("=" * 60)
    
    solver, mesh = setup_solver()
    
    def objective(params_dict):
        solution = solver.solve(params_dict)
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # Batch parameters
    batch_params = [
        {'E': 5e4, 'nu': 0.3},
        {'E': 1e5, 'nu': 0.3},
        {'E': 2e5, 'nu': 0.3}
    ]
    
    print(f"Computing gradients for {len(batch_params)} parameter sets individually...")
    
    # Compute gradients individually (this works!)
    grad_fn = jax.grad(objective)
    
    results = []
    for i, params in enumerate(batch_params):
        try:
            obj_val = objective(params)
            gradients = grad_fn(params)
            
            results.append({
                'params': params,
                'objective': obj_val,
                'gradients': gradients
            })
            
            print(f"  Set {i+1}: E={params['E']:.0e}")
            print(f"    Objective: {obj_val:.6e}")
            print(f"    ∂J/∂E:  {gradients['E']:.6e}")
            print(f"    ∂J/∂nu: {gradients['nu']:.6e}")
            
        except Exception as e:
            print(f"  Set {i+1}: ❌ Failed - {e}")
            return False
    
    print("✅ Individual gradients computed successfully!")
    return results


def approach_2_finite_differences():
    """Approach 2: Finite difference gradients for batch parameters."""
    print("\n" + "=" * 60)
    print("APPROACH 2: FINITE DIFFERENCE GRADIENTS")
    print("=" * 60)
    
    solver, mesh = setup_solver()
    
    def batch_objective(batch_params):
        """Objective for batch parameters."""
        batch_solutions = solver.solve(batch_params)
        objectives = []
        for i in range(len(batch_params)):
            displacement = batch_solutions[0][i]  # Get i-th solution
            obj = jnp.sum(displacement**2)
            objectives.append(obj)
        return jnp.array(objectives)
    
    # Batch parameters
    batch_params = [
        {'E': 5e4, 'nu': 0.3},
        {'E': 1e5, 'nu': 0.3},
        {'E': 2e5, 'nu': 0.3}
    ]
    
    print(f"Computing finite difference gradients for batch of {len(batch_params)}...")
    
    try:
        # Central difference gradients
        eps = 1e3  # Finite difference step for E
        eps_nu = 1e-6  # Finite difference step for nu
        
        # Base objectives
        obj_base = batch_objective(batch_params)
        
        # Gradients w.r.t. E
        batch_params_E_plus = [
            {'E': params['E'] + eps, 'nu': params['nu']} 
            for params in batch_params
        ]
        batch_params_E_minus = [
            {'E': params['E'] - eps, 'nu': params['nu']} 
            for params in batch_params
        ]
        
        obj_E_plus = batch_objective(batch_params_E_plus)
        obj_E_minus = batch_objective(batch_params_E_minus)
        grad_E = (obj_E_plus - obj_E_minus) / (2 * eps)
        
        # Gradients w.r.t. nu
        batch_params_nu_plus = [
            {'E': params['E'], 'nu': params['nu'] + eps_nu} 
            for params in batch_params
        ]
        batch_params_nu_minus = [
            {'E': params['E'], 'nu': params['nu'] - eps_nu} 
            for params in batch_params
        ]
        
        obj_nu_plus = batch_objective(batch_params_nu_plus)
        obj_nu_minus = batch_objective(batch_params_nu_minus)
        grad_nu = (obj_nu_plus - obj_nu_minus) / (2 * eps_nu)
        
        print("✅ Finite difference gradients computed successfully!")
        
        for i, params in enumerate(batch_params):
            print(f"  Set {i+1}: E={params['E']:.0e}")
            print(f"    Objective: {obj_base[i]:.6e}")
            print(f"    ∂J/∂E:  {grad_E[i]:.6e}")
            print(f"    ∂J/∂nu: {grad_nu[i]:.6e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Finite difference gradients failed: {e}")
        return False


def approach_3_jit_individual():
    """Approach 3: JIT-compile individual gradient computations."""
    print("\n" + "=" * 60)
    print("APPROACH 3: JIT-COMPILED INDIVIDUAL GRADIENTS")
    print("=" * 60)
    
    solver, mesh = setup_solver()
    
    def objective(params_dict):
        solution = solver.solve(params_dict)
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # JIT compile the gradient function
    grad_fn = jax.jit(jax.grad(objective))
    
    batch_params = [
        {'E': 5e4, 'nu': 0.3},
        {'E': 1e5, 'nu': 0.3},
        {'E': 2e5, 'nu': 0.3}
    ]
    
    print(f"Computing JIT-compiled gradients for {len(batch_params)} parameter sets...")
    
    try:
        import time
        
        # First call includes JIT compilation
        start = time.time()
        grad_0 = grad_fn(batch_params[0])
        first_time = time.time() - start
        
        # Subsequent calls are fast
        start = time.time()
        all_gradients = []
        for params in batch_params:
            gradients = grad_fn(params)
            all_gradients.append(gradients)
        batch_time = time.time() - start
        
        print(f"✅ JIT gradients computed successfully!")
        print(f"   First call (with compilation): {first_time:.3f}s")
        print(f"   Batch computation time: {batch_time:.3f}s")
        print(f"   Average per gradient: {batch_time/len(batch_params):.3f}s")
        
        for i, (params, gradients) in enumerate(zip(batch_params, all_gradients)):
            print(f"  Set {i+1}: E={params['E']:.0e}, ∂J/∂E={gradients['E']:.6e}, ∂J/∂nu={gradients['nu']:.6e}")
        
        return True
        
    except Exception as e:
        print(f"❌ JIT gradient computation failed: {e}")
        return False


def approach_4_optimization_example():
    """Approach 4: Practical optimization workflow."""
    print("\n" + "=" * 60)
    print("APPROACH 4: OPTIMIZATION WORKFLOW EXAMPLE")
    print("=" * 60)
    
    solver, mesh = setup_solver()
    
    def objective(params_dict):
        solution = solver.solve(params_dict)
        displacement = solution[0]
        return jnp.sum(displacement**2)
    
    # Create JIT-compiled gradient function
    grad_fn = jax.jit(jax.grad(objective))
    
    # Optimization parameters
    params = {'E': 1e5, 'nu': 0.3}
    step_size = 1e-10
    n_steps = 3
    
    print(f"Running {n_steps} gradient descent steps...")
    print(f"Initial parameters: {params}")
    
    try:
        history = [params.copy()]
        
        for step in range(n_steps):
            # Compute objective and gradient
            obj_val = objective(params)
            gradients = grad_fn(params)
            
            print(f"\nStep {step + 1}:")
            print(f"  Objective: {obj_val:.6e}")
            print(f"  Gradients: E={gradients['E']:.6e}, nu={gradients['nu']:.6e}")
            
            # Update parameters
            params = {
                'E': params['E'] - step_size * gradients['E'],
                'nu': params['nu'] - step_size * gradients['nu']
            }
            
            # Clamp nu to valid range
            params['nu'] = max(0.1, min(0.45, params['nu']))
            
            print(f"  Updated: E={params['E']:.0e}, nu={params['nu']:.3f}")
            history.append(params.copy())
        
        print("\n✅ Optimization workflow completed successfully!")
        print("This demonstrates how to use gradients in practical optimization")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization workflow failed: {e}")
        return False


def main():
    """Run all practical gradient approaches."""
    print("PRACTICAL BATCH GRADIENT COMPUTATION APPROACHES")
    print("=" * 60)
    
    jax.config.update("jax_enable_x64", True)
    
    results = []
    
    # Run all approaches
    results.append(approach_1_individual_gradients() is not False)
    results.append(approach_2_finite_differences())
    results.append(approach_3_jit_individual())
    results.append(approach_4_optimization_example())
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("=" * 60)
    
    approaches = [
        "1. Individual gradients (exact, reliable)",
        "2. Finite difference gradients (approximate)",
        "3. JIT-compiled individual gradients (fast)",
        "4. Optimization workflow (practical)"
    ]
    
    for approach, success in zip(approaches, results):
        status = "✅ Working" if success else "❌ Failed"
        print(f"{approach}: {status}")
    
    print("\n🎯 RECOMMENDED WORKFLOW:")
    print("For batch parameter gradients with NewtonSolver:")
    print("")
    print("1. Use individual gradients (Approach 1) for:")
    print("   • Exact gradients when batch size is small")
    print("   • Maximum reliability and accuracy")
    print("")
    print("2. Use finite differences (Approach 2) for:")
    print("   • Large batch sizes where individual gradients are slow")
    print("   • When approximate gradients are acceptable")
    print("")
    print("3. Use JIT-compiled individual (Approach 3) for:")
    print("   • Repeated gradient computations")
    print("   • Performance-critical applications")
    print("")
    print("4. Use optimization workflows (Approach 4) for:")
    print("   • Parameter identification problems")
    print("   • Material property estimation")


if __name__ == "__main__":
    main()