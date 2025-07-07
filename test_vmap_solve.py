#!/usr/bin/env python3
"""
Test if we can use jax.vmap directly with solver.solve()
"""

import jax
import jax.numpy as jnp
from fealax.mesh import box_mesh
from fealax.problem import Problem, DirichletBC
from fealax.solver import NewtonSolver


class SimpleElasticProblem(Problem):
    """Simple elastic problem for vmap testing."""
    
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
        def is_tracer(x):
            return hasattr(x, 'aval') or str(type(x)).find('Tracer') != -1
        
        if 'E' in params:
            self.E = params['E'] if is_tracer(params['E']) else jnp.asarray(params['E'], dtype=jnp.float64)
        if 'nu' in params:
            self.nu = params['nu'] if is_tracer(params['nu']) else jnp.asarray(params['nu'], dtype=jnp.float64)
        
        self.mu = self.E / (2.0 * (1.0 + self.nu))
        self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))


def create_test_solver():
    """Create a simple solver for testing."""
    mesh = box_mesh(2, 2, 2, 1.0, 1.0, 1.0, ele_type="HEX8")
    
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


def test_vmap_with_dict_params():
    """Test Case 1: vmap with dictionary parameters (structured arrays)."""
    print("Test Case 1: vmap with structured dictionary arrays")
    print("=" * 50)
    
    jax.config.update("jax_enable_x64", True)
    solver = create_test_solver()
    
    # Create batch parameters as structured arrays
    E_values = jnp.array([5e4, 1e5, 2e5])
    nu_values = jnp.array([0.25, 0.3, 0.35])
    
    try:
        # Define function that takes individual parameter values
        def solve_single(E, nu):
            params = {'E': E, 'nu': nu}
            return solver.solve(params)
        
        # Try vmap over the parameter arrays
        print("Attempting jax.vmap(solve_single)(E_values, nu_values)...")
        vmap_solve = jax.vmap(solve_single)
        solutions = vmap_solve(E_values, nu_values)
        
        print(f"✅ vmap successful!")
        print(f"   Input shapes: E{E_values.shape}, nu{nu_values.shape}")
        print(f"   Output shape: {solutions[0].shape}")
        return True
        
    except Exception as e:
        print(f"❌ vmap failed: {e}")
        return False


def test_vmap_with_dict_batch():
    """Test Case 2: vmap over solve function that takes dict parameters."""
    print("\nTest Case 2: vmap over function that takes full dict")
    print("=" * 50)
    
    solver = create_test_solver()
    
    # Batch parameters as individual dictionaries
    param_dicts = [
        {'E': 5e4, 'nu': 0.25},
        {'E': 1e5, 'nu': 0.3},
        {'E': 2e5, 'nu': 0.35}
    ]
    
    try:
        # Define function that takes a parameter dict
        def solve_dict(params_dict):
            return solver.solve(params_dict)
        
        # Try vmap over the parameter dicts
        print("Attempting jax.vmap(solve_dict)(param_dicts)...")
        vmap_solve = jax.vmap(solve_dict)
        solutions = vmap_solve(param_dicts)
        
        print(f"✅ vmap successful!")
        print(f"   Input: {len(param_dicts)} parameter dicts")
        print(f"   Output shape: {solutions[0].shape}")
        return True
        
    except Exception as e:
        print(f"❌ vmap failed: {e}")
        return False


def test_current_batch_approach():
    """Test Case 3: Current batch parameter approach."""
    print("\nTest Case 3: Current batch parameter approach")
    print("=" * 50)
    
    solver = create_test_solver()
    
    # Current approach: list of parameter dicts
    batch_params = [
        {'E': 5e4, 'nu': 0.25},
        {'E': 1e5, 'nu': 0.3},
        {'E': 2e5, 'nu': 0.35}
    ]
    
    try:
        print("Attempting solver.solve(batch_params)...")
        solutions = solver.solve(batch_params)
        
        print(f"✅ Batch solve successful!")
        print(f"   Input: {len(batch_params)} parameter sets")
        print(f"   Output shape: {solutions[0].shape}")
        return True
        
    except Exception as e:
        print(f"❌ Batch solve failed: {e}")
        return False


def test_vmap_structured_params():
    """Test Case 4: vmap with structured parameter dict."""
    print("\nTest Case 4: vmap with structured parameter dictionary")
    print("=" * 50)
    
    solver = create_test_solver()
    
    # Structured parameter dict with arrays
    batch_params_structured = {
        'E': jnp.array([5e4, 1e5, 2e5]),
        'nu': jnp.array([0.25, 0.3, 0.35])
    }
    
    try:
        print("Attempting solver.solve(batch_params_structured)...")
        solutions = solver.solve(batch_params_structured)
        
        print(f"✅ Structured batch solve successful!")
        print(f"   Input arrays: E{batch_params_structured['E'].shape}, nu{batch_params_structured['nu'].shape}")
        print(f"   Output shape: {solutions[0].shape}")
        return True
        
    except Exception as e:
        print(f"❌ Structured batch solve failed: {e}")
        return False


def compare_performance():
    """Compare performance of different approaches."""
    print("\nPerformance Comparison")
    print("=" * 50)
    
    import time
    solver = create_test_solver()
    
    # Test data
    E_values = jnp.array([5e4, 1e5, 2e5])
    nu_values = jnp.array([0.25, 0.3, 0.35])
    
    batch_params_list = [
        {'E': 5e4, 'nu': 0.25},
        {'E': 1e5, 'nu': 0.3},
        {'E': 2e5, 'nu': 0.35}
    ]
    
    batch_params_structured = {
        'E': E_values,
        'nu': nu_values
    }
    
    # Test 1: Current batch approach
    try:
        start = time.time()
        solutions1 = solver.solve(batch_params_list)
        time1 = time.time() - start
        print(f"Batch list approach:      {time1:.3f}s")
    except:
        print(f"Batch list approach:      Failed")
    
    # Test 2: Structured dict approach
    try:
        start = time.time()
        solutions2 = solver.solve(batch_params_structured)
        time2 = time.time() - start
        print(f"Structured dict approach: {time2:.3f}s")
    except:
        print(f"Structured dict approach: Failed")
    
    # Test 3: Manual vmap
    try:
        def solve_single(E, nu):
            return solver.solve({'E': E, 'nu': nu})
        
        vmap_solve = jax.vmap(solve_single)
        
        start = time.time()
        solutions3 = vmap_solve(E_values, nu_values)
        time3 = time.time() - start
        print(f"Manual vmap approach:     {time3:.3f}s")
    except Exception as e:
        print(f"Manual vmap approach:     Failed - {e}")


def main():
    """Run all tests."""
    print("TESTING JAX VMAP WITH NEWTONSOLVER")
    print("=" * 60)
    
    jax.config.update("jax_enable_x64", True)
    
    results = []
    results.append(test_vmap_with_dict_params())
    results.append(test_vmap_with_dict_batch())
    results.append(test_current_batch_approach())
    results.append(test_vmap_structured_params())
    
    compare_performance()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    test_names = [
        "vmap with structured arrays",
        "vmap over dict parameters", 
        "Current batch approach",
        "Structured dict batch"
    ]
    
    for name, success in zip(test_names, results):
        status = "✅ Working" if success else "❌ Failed"
        print(f"{name}: {status}")
    
    print("\n📝 CONCLUSION:")
    if results[0]:  # vmap with structured arrays
        print("✅ JAX vmap can be used directly with solver.solve()!")
        print("   Recommended: jax.vmap(lambda E, nu: solver.solve({'E': E, 'nu': nu}))")
    else:
        print("❌ Direct vmap not working, use current batch approach")


if __name__ == "__main__":
    main()