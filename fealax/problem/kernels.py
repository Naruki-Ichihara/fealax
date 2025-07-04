"""Kernel generation functionality for finite element weak forms.

This module provides the KernelGenerator class which contains methods for generating
JAX-compiled computational kernels for finite element weak forms. These kernels handle
the evaluation of integrals in weak form computations for different types of terms.

The module supports:
    - Gradient-based (Laplace-type) weak forms for elasticity, diffusion, etc.
    - Mass-type weak forms for reaction terms and time derivatives
    - Surface integral weak forms for boundary conditions and interface terms

Key Classes:
    KernelGenerator: Main class containing kernel generation methods

Example:
    Basic usage for creating kernels:

    >>> from fealax.problem.kernels import KernelGenerator
    >>> generator = KernelGenerator(fes, unflatten_fn_dof)
    >>> laplace_kernel = generator.get_laplace_kernel(tensor_map)
    >>> mass_kernel = generator.get_mass_kernel(mass_map)
    >>> surface_kernel = generator.get_surface_kernel(surface_map)
"""

import jax
import jax.numpy as np
import jax.flatten_util
from typing import Callable, List
from fealax.fe import FiniteElement


class KernelGenerator:
    """Generator class for finite element weak form computational kernels.

    This class provides methods to generate JAX-compiled functions for computing
    element-level contributions to finite element weak forms. The kernels handle
    different types of integral terms common in finite element methods.

    Attributes:
        fes (List[FiniteElement]): List of finite element spaces for each variable.
        unflatten_fn_dof (Callable): Function to unflatten degree of freedom arrays.
        dim (int): Spatial dimension of the problem.

    Note:
        The kernels are designed to work with JAX transformations (jit, vmap, grad)
        for optimal performance and automatic differentiation capabilities.
    """

    def __init__(self, fes: List[FiniteElement], unflatten_fn_dof: Callable, dim: int):
        """Initialize the kernel generator.

        Args:
            fes (List[FiniteElement]): List of finite element spaces.
            unflatten_fn_dof (Callable): Function to unflatten DOF arrays.
            dim (int): Spatial dimension of the problem.
        """
        self.fes = fes
        self.unflatten_fn_dof = unflatten_fn_dof
        self.dim = dim

    def get_laplace_kernel(self, tensor_map: Callable) -> Callable:
        """Create a kernel function for Laplace-type (gradient-based) weak forms.

        Generates a function that computes element-level contributions to the weak form
        involving solution gradients, such as diffusion or elasticity terms. This kernel
        implements the gradient-based weak form integral:

        ∫_Ω ∇v · σ(∇u) dΩ

        where σ is the constitutive relationship (e.g., stress-strain for elasticity).

        Args:
            tensor_map (Callable): Function that maps solution gradients to flux/stress.
                Signature: tensor_map(u_grads, *internal_vars) -> flux
                where u_grads has shape (num_quads, vec, dim) and represents gradients
                of the solution at quadrature points.

        Returns:
            Callable: Element kernel function with signature:
                kernel(cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *internal_vars)
                -> element_residual
                
                Args:
                    cell_sol_flat: Flattened cell solution vector (num_nodes*vec + ...)
                    cell_shape_grads: Shape function gradients (num_quads, num_nodes+..., dim)
                    cell_v_grads_JxW: Test function gradients with Jacobian weights
                        (num_quads, num_nodes+..., 1, dim)
                    *internal_vars: Additional internal variables for the tensor_map
                
                Returns:
                    np.ndarray: Element residual vector (num_nodes*vec + ...)

        Note:
            This kernel is typically used for:
            - Linear elasticity problems (Hooke's law)
            - Diffusion/heat conduction (Fourier's law)
            - Fluid mechanics (viscous stress terms)
            - General elliptic PDEs with gradient-dependent physics
        """

        def laplace_kernel(
            cell_sol_flat, cell_shape_grads, cell_v_grads_JxW, *cell_internal_vars
        ):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # cell_shape_grads: (num_quads, num_nodes + ..., dim)
            # cell_v_grads_JxW: (num_quads, num_nodes + ..., 1, dim)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_shape_grads = cell_shape_grads[:, : self.fes[0].num_nodes, :]
            cell_sol = cell_sol_list[0]
            cell_v_grads_JxW = cell_v_grads_JxW[:, : self.fes[0].num_nodes, :, :]
            vec = self.fes[0].vec

            # Compute solution gradients at quadrature points
            # (1, num_nodes, vec, 1) * (num_quads, num_nodes, 1, dim) -> (num_quads, num_nodes, vec, dim)
            u_grads = cell_sol[None, :, :, None] * cell_shape_grads[:, :, None, :]
            u_grads = np.sum(u_grads, axis=1)  # (num_quads, vec, dim)
            u_grads_reshape = u_grads.reshape(
                -1, vec, self.dim
            )  # (num_quads, vec, dim)
            
            # Apply constitutive relationship (tensor_map) at each quadrature point
            # (num_quads, vec, dim)
            u_physics = jax.vmap(tensor_map)(
                u_grads_reshape, *cell_internal_vars
            ).reshape(u_grads.shape)
            
            # Integrate with test functions to get element residual
            # (num_quads, num_nodes, vec, dim) -> (num_nodes, vec)
            val = np.sum(u_physics[:, None, :, :] * cell_v_grads_JxW, axis=(0, -1))
            val = jax.flatten_util.ravel_pytree(val)[0]  # (num_nodes*vec + ...,)
            return val

        return laplace_kernel

    def get_mass_kernel(self, mass_map: Callable) -> Callable:
        """Create a kernel function for mass-type (solution-based) weak forms.

        Generates a function that computes element-level contributions to the weak form
        involving solution values (not gradients), such as reaction terms or time
        derivatives. This kernel implements the mass-type weak form integral:

        ∫_Ω v · f(u, x) dΩ

        where f represents source terms, reaction terms, or time derivatives.

        Args:
            mass_map (Callable): Function that maps solution values to source terms.
                Signature: mass_map(u, x, *internal_vars) -> source
                where u has shape (num_quads, vec) representing solution values at
                quadrature points, and x has shape (num_quads, dim) representing
                spatial coordinates.

        Returns:
            Callable: Element kernel function with signature:
                kernel(cell_sol_flat, x, cell_JxW, *internal_vars) -> element_residual
                
                Args:
                    cell_sol_flat: Flattened cell solution vector (num_nodes*vec + ...)
                    x: Physical coordinates at quadrature points (num_quads, dim)
                    cell_JxW: Jacobian determinant times quadrature weights (num_vars, num_quads)
                    *internal_vars: Additional internal variables for the mass_map
                
                Returns:
                    np.ndarray: Element residual vector (num_nodes*vec + ...)

        Note:
            This kernel is typically used for:
            - Time-dependent problems (mass matrix terms)
            - Reaction terms in reaction-diffusion equations
            - Source terms and body forces
            - Density-dependent or concentration-dependent terms
        """

        def mass_kernel(cell_sol_flat, x, cell_JxW, *cell_internal_vars):
            # cell_sol_flat: (num_nodes*vec + ...,)
            # cell_sol_list: [(num_nodes, vec), ...]
            # x: (num_quads, dim)
            # cell_JxW: (num_vars, num_quads)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            cell_JxW = cell_JxW[0]
            vec = self.fes[0].vec
            
            # Interpolate solution to quadrature points
            # (1, num_nodes, vec) * (num_quads, num_nodes, 1) -> (num_quads, num_nodes, vec) -> (num_quads, vec)
            u = np.sum(
                cell_sol[None, :, :] * self.fes[0].shape_vals[:, :, None], axis=1
            )
            
            # Apply physics function (mass_map) at each quadrature point
            u_physics = jax.vmap(mass_map)(
                u, x, *cell_internal_vars
            )  # (num_quads, vec)
            
            # Integrate with test functions and quadrature weights
            # (num_quads, 1, vec) * (num_quads, num_nodes, 1) * (num_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(
                u_physics[:, None, :]
                * self.fes[0].shape_vals[:, :, None]
                * cell_JxW[:, None, None],
                axis=0,
            )
            val = jax.flatten_util.ravel_pytree(val)[0]  # (num_nodes*vec + ...,)
            return val

        return mass_kernel

    def get_surface_kernel(self, surface_map: Callable) -> Callable:
        """Create a kernel function for surface integral weak forms.

        Generates a function that computes face-level contributions to the weak form,
        such as Neumann boundary conditions or interface terms. This kernel implements
        the surface integral weak form:

        ∫_Γ v · g(u, x) dΓ

        where g represents surface fluxes, tractions, or interface conditions.

        Args:
            surface_map (Callable): Function that maps surface solution values to fluxes.
                Signature: surface_map(u, x, *internal_vars) -> flux
                where u has shape (num_face_quads, vec) representing solution values at
                face quadrature points, and x has shape (num_face_quads, dim) representing
                spatial coordinates on the surface.

        Returns:
            Callable: Surface kernel function with signature:
                kernel(cell_sol_flat, x, face_shape_vals, face_shape_grads,
                       face_nanson_scale, *internal_vars) -> face_residual
                
                Args:
                    cell_sol_flat: Flattened cell solution vector (num_nodes*vec + ...)
                    x: Physical coordinates at face quadrature points (num_face_quads, dim)
                    face_shape_vals: Face shape function values (num_face_quads, num_nodes+...)
                    face_shape_grads: Face shape function gradients (num_face_quads, num_nodes+..., dim)
                    face_nanson_scale: Surface area scaling factor (num_vars, num_face_quads)
                    *internal_vars: Additional internal variables for the surface_map
                
                Returns:
                    np.ndarray: Face residual vector (num_nodes*vec + ...)

        Note:
            This kernel is typically used for:
            - Neumann boundary conditions (prescribed tractions/fluxes)
            - Robin/mixed boundary conditions
            - Interface conditions between materials
            - Contact mechanics and surface interactions
            - Natural boundary conditions in variational formulations
        """

        def surface_kernel(
            cell_sol_flat,
            x,
            face_shape_vals,
            face_shape_grads,
            face_nanson_scale,
            *cell_internal_vars_surface,
        ):
            # face_shape_vals: (num_face_quads, num_nodes + ...)
            # face_shape_grads: (num_face_quads, num_nodes + ..., dim)
            # x: (num_face_quads, dim)
            # face_nanson_scale: (num_vars, num_face_quads)

            cell_sol_list = self.unflatten_fn_dof(cell_sol_flat)
            cell_sol = cell_sol_list[0]
            face_shape_vals = face_shape_vals[:, : self.fes[0].num_nodes]
            face_nanson_scale = face_nanson_scale[0]

            # Interpolate solution to face quadrature points
            # (1, num_nodes, vec) * (num_face_quads, num_nodes, 1) -> (num_face_quads, vec)
            u = np.sum(cell_sol[None, :, :] * face_shape_vals[:, :, None], axis=1)
            
            # Apply surface physics function (surface_map) at each face quadrature point
            u_physics = jax.vmap(surface_map)(
                u, x, *cell_internal_vars_surface
            )  # (num_face_quads, vec)
            
            # Integrate with test functions and surface area scaling
            # (num_face_quads, 1, vec) * (num_face_quads, num_nodes, 1) * (num_face_quads, 1, 1) -> (num_nodes, vec)
            val = np.sum(
                u_physics[:, None, :]
                * face_shape_vals[:, :, None]
                * face_nanson_scale[:, None, None],
                axis=0,
            )

            return jax.flatten_util.ravel_pytree(val)[0]

        return surface_kernel