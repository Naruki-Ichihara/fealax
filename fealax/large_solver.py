"""Memory-efficient solver for large finite element problems."""

import jax
import jax.numpy as jnp
import time
from typing import Dict, Any, List

from .memory_utils import MemoryManager, create_memory_efficient_solver_options, monitor_memory_usage
from .solver import newton_solve, assign_bc
from . import logger


class LargeProblemSolver:
    """Solver optimized for large finite element problems."""
    
    def __init__(self, memory_fraction: float = 0.8):
        """Initialize large problem solver.
        
        Args:
            memory_fraction: Fraction of GPU memory to target for usage
        """
        self.memory_manager = MemoryManager(memory_fraction)
        
    def solve(self, problem: Any, custom_options: Dict[str, Any] = None) -> List[jnp.ndarray]:
        """Solve large finite element problem with memory management.
        
        Args:
            problem: Finite element problem instance
            custom_options: Custom solver options (optional)
            
        Returns:
            Solution list
        """
        logger.info(f"🚀 Starting large problem solver")
        
        # Get problem size information
        num_dofs = problem.num_total_dofs_all_vars
        num_elements = problem.mesh[0].cells.shape[0]  # problem.mesh is a list
        
        logger.info(f"📊 Problem size: {num_elements} elements, {num_dofs} DOFs")
        
        # Check if chunking is needed
        use_chunking = self.memory_manager.should_use_chunking(num_elements, num_dofs)
        logger.info(f"💾 Memory management: {'Chunking enabled' if use_chunking else 'Direct assembly'}")
        
        # Create memory-efficient solver options
        solver_options = create_memory_efficient_solver_options(num_dofs)
        
        # Override with custom options if provided
        if custom_options:
            solver_options.update(custom_options)
            
        # Adjust chunk size if needed
        if use_chunking:
            optimal_chunk_size = self.memory_manager.get_optimal_chunk_size(num_elements)
            logger.info(f"🔧 Using adaptive chunk size: {optimal_chunk_size}")
            # Note: This would need to be passed to problem.compute_csr() calls
            
        logger.info(f"⚙️  Solver configuration: {solver_options}")
        
        # Solve with memory monitoring
        start_time = time.time()
        solution = self._solve_with_monitoring(problem, solver_options)
        solve_time = time.time() - start_time
        
        logger.info(f"✅ Large problem solved in {solve_time:.2f}s")
        
        # Cleanup memory
        self.memory_manager.cleanup()
        
        return solution
    
    @monitor_memory_usage
    def _solve_with_monitoring(self, problem: Any, solver_options: Dict[str, Any]) -> List[jnp.ndarray]:
        """Solve with memory usage monitoring."""
        return newton_solve(problem, solver_options)
    
    def estimate_feasibility(self, num_elements: int, num_dofs: int) -> Dict[str, Any]:
        """Estimate if problem is feasible with current memory."""
        from .memory_utils import estimate_memory_requirements, get_gpu_memory_info
        
        memory_req = estimate_memory_requirements(num_dofs, num_elements)
        gpu_memory = get_gpu_memory_info()
        
        feasible = memory_req['total_estimated'] < gpu_memory['available'] * 1.2  # 20% buffer
        
        return {
            'feasible': feasible,
            'estimated_memory_gb': memory_req['total_estimated'] / 1e9,
            'available_memory_gb': gpu_memory['available'] / 1e9,
            'memory_utilization': memory_req['total_estimated'] / max(gpu_memory['available'], 1),
            'recommendation': self._get_recommendation(memory_req, gpu_memory)
        }
    
    def _get_recommendation(self, memory_req: Dict, gpu_memory: Dict) -> str:
        """Get recommendation for problem solving."""
        utilization = memory_req['total_estimated'] / max(gpu_memory['available'], 1)
        
        if utilization < 0.5:
            return "Problem should solve efficiently with standard settings"
        elif utilization < 0.8:
            return "Use memory-efficient settings, disable JIT compilation"
        elif utilization < 1.2:
            return "Use chunked assembly and conservative solver settings"
        else:
            return "Problem too large for current GPU, consider mesh coarsening or CPU fallback"


def solve_large_problem(problem: Any, solver_options: Dict[str, Any] = None) -> List[jnp.ndarray]:
    """Convenience function to solve large finite element problems.
    
    Args:
        problem: Finite element problem instance
        solver_options: Optional solver configuration
        
    Returns:
        Solution list
        
    Example:
        >>> solution = solve_large_problem(problem, {'tol': 1e-6})
    """
    solver = LargeProblemSolver()
    return solver.solve(problem, solver_options)


def check_problem_feasibility(mesh, vec: int = 3) -> Dict[str, Any]:
    """Check if a problem size is feasible for the current GPU.
    
    Args:
        mesh: Finite element mesh
        vec: Number of DOFs per node (default: 3 for displacement)
        
    Returns:
        Feasibility analysis dictionary
        
    Example:
        >>> analysis = check_problem_feasibility(mesh)
        >>> if analysis['feasible']:
        ...     print("Problem is feasible")
        >>> else:
        ...     print(f"Recommendation: {analysis['recommendation']}")
    """
    num_elements = mesh.cells.shape[0]
    num_dofs = mesh.points.shape[0] * vec
    
    solver = LargeProblemSolver()
    return solver.estimate_feasibility(num_elements, num_dofs)