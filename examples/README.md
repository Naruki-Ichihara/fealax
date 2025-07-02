# Fealax Examples

This directory contains example problems demonstrating how to use the fealax finite element analysis library.

## Simple Linear Elasticity Example

**File**: `simple_elasticity.py`

### Problem Description

This example solves a 3D linear elasticity problem on a unit cube (1×1×1) under uniaxial compression:

- **Domain**: 1×1×1 unit cube
- **Element Type**: HEX8 (8-node hexahedral elements)
- **Material**: Linear elastic isotropic material (steel-like properties)
  - Young's modulus: E = 200 GPa
  - Poisson's ratio: ν = 0.3
- **Boundary Conditions**:
  - Bottom face (z=0): Fixed in all directions (u_x = u_y = u_z = 0)
  - Top face (z=1): Applied displacement u_z = -0.1 (10% compression)
  - Other faces: Free

### Physics

The problem solves the linear elasticity equations:

```
∇ · σ = 0
σ = C : ε
ε = ½(∇u + ∇u^T)
```

Where:
- σ is the stress tensor
- ε is the strain tensor  
- u is the displacement field
- C is the fourth-order elasticity tensor

For isotropic materials, the stress-strain relationship is:
```
σ = 2μ*ε + λ*tr(ε)*I
```

Where μ and λ are the Lamé parameters:
- μ = E/(2(1+ν)) (shear modulus)
- λ = Eν/((1+ν)(1-2ν)) (first Lamé parameter)

### Running the Example

#### Prerequisites

Make sure you have the required dependencies installed:

```bash
# Install fealax in development mode
pip install -e .

# Install required dependencies
pip install jax[cuda] numpy scipy matplotlib meshio nlopt fenics-basix
```

#### Execution

```bash
cd examples
python simple_elasticity.py
```

### Expected Output

The example will output:

1. Mesh information (number of nodes and elements)
2. Material properties
3. Boundary condition setup
4. Solver progress and convergence
5. Solution summary including:
   - Maximum displacement magnitude
   - Range of z-displacements
   - Displacement at the center of the top face

### Understanding the Code Structure

The example demonstrates the standard fealax workflow:

1. **Mesh Creation**: Using `box_mesh()` to create a structured hexahedral mesh
2. **Boundary Conditions**: Defining `DirichletBC` objects for essential boundary conditions
3. **Problem Definition**: Subclassing `Problem` and implementing `get_tensor_map()` for the elasticity formulation
4. **Solution**: Using the `solver()` function with appropriate solver options
5. **Post-processing**: Extracting and analyzing the displacement field

### Key Features Demonstrated

- **JAX Integration**: The example uses JAX for automatic differentiation and GPU acceleration
- **Tensor-based Weak Forms**: Implementation of elasticity using the tensor map approach
- **Boundary Condition Specification**: Using lambda functions to define geometric regions
- **Material Parameter Handling**: Proper initialization of material properties in the problem class
- **Solver Configuration**: Setting up iterative linear solvers with preconditioning

### Customization

You can modify the example to explore different scenarios:

- **Mesh Resolution**: Change `nx, ny, nz` parameters in `create_mesh()`
- **Material Properties**: Modify the `material_params` dictionary
- **Loading Conditions**: Adjust the applied displacement or add different boundary conditions
- **Solver Settings**: Tune solver tolerances and iteration limits

This example serves as a foundation for more complex finite element problems in fealax.