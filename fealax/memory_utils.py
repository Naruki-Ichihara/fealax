"""Memory management utilities for large finite element problems."""

import jax
import jax.numpy as jnp
import gc
from typing import Optional, Dict, Any


def get_gpu_memory_info():
    """Get GPU memory information."""
    try:
        devices = jax.devices('gpu')
        if devices:
            device = devices[0]
            # Get memory stats if available
            if hasattr(device, 'memory_stats'):
                memory_info = device.memory_stats()
                return {
                    'total': memory_info.get('bytes_limit', 0),
                    'used': memory_info.get('bytes_in_use', 0),
                    'available': memory_info.get('bytes_limit', 0) - memory_info.get('bytes_in_use', 0)
                }
    except:
        pass
    return {'total': 0, 'used': 0, 'available': 0}


def estimate_memory_requirements(num_dofs: int, num_elements: int) -> Dict[str, int]:
    """Estimate memory requirements for a given problem size."""
    # Rough estimates in bytes
    # Sparse matrix: assume ~100 non-zeros per row (for 3D hex elements)
    sparse_matrix_memory = num_dofs * 100 * 8 * 3  # indices + data, safety factor
    
    # Element matrices during assembly (dense)
    element_dofs = 24  # HEX8 elements have 8 nodes * 3 DOFs
    element_matrix_memory = num_elements * element_dofs * element_dofs * 8
    
    # Solution vectors
    solution_memory = num_dofs * 8 * 10  # Multiple copies during solving
    
    total_estimated = sparse_matrix_memory + element_matrix_memory + solution_memory
    
    return {
        'sparse_matrix': sparse_matrix_memory,
        'element_assembly': element_matrix_memory,
        'solution_vectors': solution_memory,
        'total_estimated': total_estimated
    }


def compute_adaptive_chunk_size(num_elements: int, available_memory: int, safety_factor: float = 0.7) -> int:
    """Compute adaptive chunk size based on available memory."""
    # Target memory per chunk (in bytes)
    target_memory = int(available_memory * safety_factor)
    
    # Memory per element during assembly (rough estimate)
    element_dofs = 24
    memory_per_element = element_dofs * element_dofs * 8  # Dense element matrix
    
    # Number of elements that fit in target memory
    elements_per_chunk = max(1, target_memory // memory_per_element)
    
    # Don't exceed total number of elements
    chunk_size = min(elements_per_chunk, num_elements)
    
    return chunk_size


def clear_jax_memory():
    """Clear JAX memory and caches."""
    # Clear compilation cache
    jax.clear_caches()
    
    # Force garbage collection
    gc.collect()
    
    # Try to trigger JAX memory cleanup
    try:
        # Create and delete a small array to trigger cleanup
        dummy = jnp.ones(1)
        del dummy
    except:
        pass


class MemoryManager:
    """Memory manager for large finite element problems."""
    
    def __init__(self, target_memory_fraction: float = 0.8):
        """Initialize memory manager.
        
        Args:
            target_memory_fraction: Fraction of available GPU memory to target
        """
        self.target_memory_fraction = target_memory_fraction
        self.initial_memory = get_gpu_memory_info()
        
    def get_available_memory(self) -> int:
        """Get currently available GPU memory."""
        current_memory = get_gpu_memory_info()
        return current_memory['available']
    
    def should_use_chunking(self, num_elements: int, num_dofs: int) -> bool:
        """Determine if chunking should be used."""
        memory_req = estimate_memory_requirements(num_dofs, num_elements)
        available = self.get_available_memory()
        
        return memory_req['total_estimated'] > available * self.target_memory_fraction
    
    def get_optimal_chunk_size(self, num_elements: int) -> int:
        """Get optimal chunk size for current memory conditions."""
        available = self.get_available_memory()
        return compute_adaptive_chunk_size(num_elements, available)
    
    def cleanup(self):
        """Perform memory cleanup."""
        clear_jax_memory()


def create_memory_efficient_solver_options(problem_size: int) -> Dict[str, Any]:
    """Create solver options optimized for memory usage."""
    # Base options for memory efficiency
    base_options = {
        'method': 'cg',  # CG is more memory efficient than BiCGSTAB
        'precond': True,
        'tol': 1e-5,  # Slightly relaxed for large problems
        'rel_tol': 1e-6,
    }
    
    if problem_size < 100000:  # Small problems
        base_options.update({
            'use_jit': True,
            'max_iter': 20
        })
    elif problem_size < 500000:  # Medium problems
        base_options.update({
            'use_jit': False,  # Reduce compilation memory overhead
            'max_iter': 15,
            'line_search_flag': True  # Help convergence with fewer iterations
        })
    else:  # Large problems
        base_options.update({
            'use_jit': False,
            'max_iter': 10,
            'line_search_flag': True,
            'tol': 1e-4,  # More relaxed for very large problems
            'rel_tol': 1e-5
        })
    
    return base_options


def monitor_memory_usage(func):
    """Decorator to monitor memory usage of functions."""
    def wrapper(*args, **kwargs):
        memory_before = get_gpu_memory_info()
        print(f"Memory before {func.__name__}: {memory_before['used']/1e9:.2f} GB")
        
        try:
            result = func(*args, **kwargs)
            memory_after = get_gpu_memory_info()
            print(f"Memory after {func.__name__}: {memory_after['used']/1e9:.2f} GB")
            print(f"Memory delta: {(memory_after['used'] - memory_before['used'])/1e9:.2f} GB")
            return result
        except Exception as e:
            memory_error = get_gpu_memory_info()
            print(f"Memory at error: {memory_error['used']/1e9:.2f} GB")
            raise e
    
    return wrapper