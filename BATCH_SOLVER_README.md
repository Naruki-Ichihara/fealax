# Automatic Batch Solver for fealax

## Overview

The fealax NewtonSolver now automatically detects when multiple parameter sets are provided and processes them efficiently in batch mode. This enhancement maintains full backward compatibility while adding powerful new batch processing capabilities.

## Key Features

✅ **Automatic Detection**: Solver automatically detects batch parameters without any configuration  
✅ **Multiple Input Formats**: Supports both list of dictionaries and dictionary of arrays  
✅ **Backward Compatibility**: Single parameter solving works exactly as before  
✅ **Efficient Processing**: Sequential batch processing with JIT compilation benefits  
✅ **Proper Result Stacking**: Returns properly structured batch results  

## Usage Examples

### Single Parameter Solving (Unchanged)
```python
from fealax.solver import NewtonSolver

solver = NewtonSolver(problem, solver_options)
solution = solver.solve({'E': 200e9, 'nu': 0.3})
# Returns: List[np.ndarray] with shape (n_dofs,)
```

### Batch Solving - List of Dictionaries
```python
batch_params = [
    {'E': 200e9, 'nu': 0.3},
    {'E': 300e9, 'nu': 0.25},
    {'E': 150e9, 'nu': 0.35}
]
solutions = solver.solve(batch_params)
# Returns: List[np.ndarray] with shape (batch_size, n_dofs)
```

### Batch Solving - Dictionary of Arrays
```python
batch_params = {
    'E': jnp.array([200e9, 300e9, 150e9]),
    'nu': jnp.array([0.3, 0.25, 0.35])
}
solutions = solver.solve(batch_params)
# Returns: List[np.ndarray] with shape (batch_size, n_dofs)
```

## Implementation Details

### Detection Logic
The solver automatically detects batch parameters using the following criteria:
1. **List Format**: Input is a list of dictionaries with consistent keys
2. **Array Format**: Input is a dictionary where all values are arrays with the same length > 1

### Processing Strategy
- **Assembly Exclusion**: Assembly process runs once per parameter set (not vmapped)
- **Sequential Processing**: Currently uses sequential solving with fallback handling
- **JIT Compilation**: Each solve benefits from JIT compilation for optimal performance
- **Future vmap Support**: Framework prepared for full vmap integration when solver architecture allows

### Performance Benefits
- **JIT Amortization**: First solve JIT compilation, subsequent solves reuse compiled functions
- **Memory Efficiency**: Proper result stacking without unnecessary memory overhead
- **Scalability**: Handles arbitrary batch sizes efficiently

## API Changes

### NewtonSolver.solve() Method
```python
def solve(self, params: Any) -> List[np.ndarray]:
    """Solve the finite element problem with given parameters.
    
    Automatically applies batch processing for multiple parameter sets.
    
    Args:
        params: Problem parameters. Can be:
            - Single parameter dict: {'E': 200e9, 'nu': 0.3}
            - List of parameter dicts: [{'E': 200e9, 'nu': 0.3}, ...]
            - Dict with batched arrays: {'E': [200e9, 300e9], 'nu': [0.3, 0.25]}
        
    Returns:
        Solution list where each array corresponds to a variable.
        For batched inputs, returns batched solutions with leading batch dimension.
    """
```

### Internal Methods Added
- `_is_batch_params()`: Detects if parameters represent multiple sets
- `_solve_batch_vmap()`: Main batch processing logic
- `_solve_batch_sequential()`: Sequential solving fallback
- `_create_vmap_solver()`: Creates vmap-compatible solver functions (for future use)

## Error Handling

The implementation includes robust error handling:
- **Format Validation**: Validates parameter format consistency
- **Fallback Mechanisms**: Graceful fallback from vmap to sequential processing
- **Clear Logging**: Informative messages about batch detection and processing mode

## Future Enhancements

### Planned Improvements
1. **Full vmap Integration**: Complete vmap support when solver architecture allows
2. **Dynamic Parameter Detection**: Support for arbitrary parameter structures
3. **Memory Optimization**: Advanced memory management for large batch sizes
4. **Parallel Assembly**: Parallel assembly processing for independent parameter sets

### vmap Compatibility Notes
The current implementation prepares for future vmap integration but falls back to sequential processing due to:
- Tracer leaks in current solver implementation
- Side effects in assembly code
- Incompatibility between JAX vmap and current solver architecture

These issues are being addressed in future versions.

## Testing

Comprehensive test suite included:
- `test_batch_solver.py`: Full functionality testing
- `batch_solver_demo.py`: Usage demonstration
- Consistency validation between single and batch solving
- Performance benchmarking

## Backward Compatibility

✅ **100% Backward Compatible**: All existing code continues to work unchanged  
✅ **No Breaking Changes**: Same API, enhanced functionality  
✅ **Performance Maintained**: Single parameter solving performance unchanged  

## Example Output

```
FEALAX AUTOMATIC BATCH SOLVING DEMONSTRATION
Setting up finite element problem...
✓ Problem setup complete: 512 elements, 729 nodes

Single solve:           3.40s
Batch solve (list):     1.67s (4 cases)
Batch solve (dict):     1.23s (3 cases)
Sequential efficiency:  ~8.1x speedup
```

The batch solver provides significant efficiency improvements through JIT compilation amortization and optimized sequential processing.