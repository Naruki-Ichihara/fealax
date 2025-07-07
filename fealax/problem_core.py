"""Finite element problem definition and solution framework.

This module provides the core classes and infrastructure for defining and solving
finite element problems. It includes boundary condition specifications, problem
setup, weak form computation, and numerical integration routines.

The module supports:
    - Multi-variable finite element problems
    - Dirichlet and Neumann boundary conditions
    - Volume and surface integral computation
    - Automatic differentiation for Jacobian assembly
    - Sparse matrix assembly and memory-efficient computation
    - Integration with JAX for GPU acceleration

Key Classes:
    Problem: Main finite element problem class with assembly and solution methods
    
Note:
    DirichletBC is now located in fealax.problem.boundary_conditions

Example:
    Basic usage for defining a finite element problem:

    >>> from fealax.problem import Problem
    >>> from fealax.problem.boundary_conditions import DirichletBC
    >>> from fealax.mesh import box_mesh
    >>>
    >>> mesh = box_mesh(10, 10, 10, 1.0, 1.0, 1.0)
    >>> dirichlet_bcs = [DirichletBC(subdomain=lambda x: x[0] < 1e-6,
    ...                              vec=0,
    ...                              eval=lambda x: 0.0)]
    >>> problem = Problem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=dirichlet_bcs)
"""

import jax
import jax.numpy as np
import jax.flatten_util
from dataclasses import dataclass
from typing import Any, Callable, Optional, List, Iterable, Union, Tuple
import functools
from jax.experimental.sparse import BCOO
from fealax.mesh import Mesh
from fealax.fe import FiniteElement
from fealax import logger
from fealax.problem.boundary_conditions import (
    DirichletBC,
    BoundaryConditionManager,
    process_dirichlet_bcs,
    validate_dirichlet_bcs,
)
from fealax.problem.kernels import KernelGenerator
from fealax.problem.assembly import AssemblyManager
import gc


# DirichletBC is now imported from fealax.problem.boundary_conditions


@dataclass
class Problem:
    """Main finite element problem class for multi-variable systems.

    This class provides the core infrastructure for setting up and solving finite element
    problems. It handles mesh management, boundary condition specification, weak form
    assembly, and numerical integration. Supports multi-variable problems with different
    element types and automatic differentiation for Jacobian computation.

    Attributes:
        mesh (Mesh): The computational mesh for the problem. Can be a single mesh
            or list of meshes for multi-variable problems.
        vec (int): Number of vector components for the primary variable.
            For example, vec=3 for 3D displacement problems (ux, uy, uz).
        dim (int): Spatial dimension of the problem (1, 2, or 3).
        ele_type (str, optional): Finite element type identifier. Defaults to 'HEX8'.
            Supported types include 'TET4', 'TET10', 'HEX8', 'HEX20', 'HEX27',
            'TRI3', 'TRI6', 'QUAD4', 'QUAD8'.
        gauss_order (int, optional): Gaussian quadrature order for numerical integration.
            If None, uses default order based on element type.
        dirichlet_bcs (Optional[Iterable[DirichletBC]], optional): Collection of
            Dirichlet boundary conditions to apply to the problem.
        neumann_subdomains (Optional[List[Callable]], optional): List of functions
            defining Neumann boundary subdomains for natural boundary conditions.
        additional_info (Any, optional): Additional data passed to custom_init().
            Used for problem-specific initialization parameters.
        prolongation_matrix (Optional[np.ndarray], optional): Prolongation matrix
            for constraint enforcement or multigrid methods.
        macro_term (Optional[np.ndarray], optional): Macroscopic displacement field,
            typically an affine function defined through macroscopic strain.

    Note:
        The class automatically handles conversion to multi-variable format internally,
        so single-variable problems can be specified with scalar parameters.

    Example:
        >>> # Setup a 3D elasticity problem
        >>> mesh = box_mesh(10, 10, 10, 1.0, 1.0, 1.0)
        >>> bcs = [DirichletBC(lambda x: x[0] < 1e-6, 0, lambda x: 0.0)]
        >>> problem = Problem(mesh=mesh, vec=3, dim=3, dirichlet_bcs=bcs)
    """

    mesh: Union[Mesh, List[Mesh]]
    vec: Union[int, List[int]]
    dim: int
    ele_type: Union[str, List[str]] = "HEX8"
    gauss_order: Union[int, List[int], None] = None
    dirichlet_bcs: Optional[Iterable[DirichletBC]] = None
    neumann_subdomains: Optional[List[Callable[[np.ndarray], bool]]] = None
    additional_info: Any = ()
    prolongation_matrix: Optional[BCOO] = None
    macro_term: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        """Initialize the finite element problem after dataclass construction.

        This method performs the heavy computational setup including:
        - Converting single-variable inputs to multi-variable format
        - Setting up finite element spaces and boundary conditions
        - Computing shape functions, quadrature points, and integration weights
        - Preparing data structures for efficient assembly operations
        - JIT-compiling kernel functions for volume and surface integrals

        The initialization is automatically called after dataclass construction
        and prepares all necessary data structures for subsequent computations.

        Note:
            This method can be computationally expensive for large problems as it
            performs shape function computation and JIT compilation.
        """

        if self.prolongation_matrix is not None:
            logger.debug("Using provided prolongation matrix.")

        if self.macro_term is not None:
            logger.debug(
                f"Using provided perturbation. Size is: {len(self.macro_term)}"
            )

        if type(self.mesh) != type([]):
            self.mesh = [self.mesh]
            self.vec = [self.vec]
            self.ele_type = [self.ele_type]
            self.gauss_order = [self.gauss_order]

        # Process Dirichlet boundary conditions
        self.dirichlet_bc_info = process_dirichlet_bcs(self.dirichlet_bcs)
        
        # Validate boundary conditions if present
        if self.dirichlet_bcs is not None:
            # Convert to list format for validation
            vec_list = self.vec if isinstance(self.vec, list) else [self.vec]
            max_vec = max(vec_list)
            validate_dirichlet_bcs(self.dirichlet_bcs, max_vec)

        self.num_vars = len(self.mesh)

        self.fes = [
            FiniteElement(
                mesh=self.mesh[i],
                vec=self.vec[i],
                dim=self.dim,
                ele_type=self.ele_type[i],
                gauss_order=(
                    self.gauss_order[i]
                    if type(self.gauss_order) == type([])
                    else self.gauss_order
                ),
                dirichlet_bc_info=self.dirichlet_bc_info,
            )
            for i in range(self.num_vars)
        ]
        self.fe = self.fes[0]  # For convenience, use the first FE as the default one

        self.cells_list = [fe.cells for fe in self.fes]
        # Assume all fes have the same number of cells, same dimension
        self.num_cells = self.fes[0].num_cells
        self.num_nodes = self.fes[0].num_nodes
        self.num_quads = self.fes[0].num_quads
        self.boundary_inds_list = self.fes[0].get_boundary_conditions_inds(
            self.neumann_subdomains
        )

        self.offset = [0]
        for i in range(len(self.fes) - 1):
            self.offset.append(self.offset[i] + self.fes[i].num_total_dofs)

        def find_ind(*x):
            inds = []
            for i in range(len(x)):
                x[i].reshape(-1)
                crt_ind = (
                    self.fes[i].vec * x[i][:, None]
                    + np.arange(self.fes[i].vec)[None, :]
                    + self.offset[i]
                )
                inds.append(crt_ind.reshape(-1))

            return np.hstack(inds)

        # (num_cells, num_nodes*vec + ...)
        # Force computation on CPU to avoid GPU memory issues for large meshes
        with jax.default_device(jax.devices("cpu")[0]):
            inds = np.array(jax.vmap(find_ind)(*self.cells_list))
            self.I = np.repeat(inds[:, :, None], inds.shape[1], axis=2).reshape(-1)
            self.J = np.repeat(inds[:, None, :], inds.shape[1], axis=1).reshape(-1)
        self.cells_list_face_list = []

        for i, boundary_inds in enumerate(self.boundary_inds_list):
            cells_list_face = [
                cells[boundary_inds[:, 0]] for cells in self.cells_list
            ]  # [(num_selected_faces, num_nodes), ...]
            inds_face = np.array(
                jax.vmap(find_ind)(*cells_list_face)
            )  # (num_selected_faces, num_nodes*vec + ...)
            I_face = np.repeat(
                inds_face[:, :, None], inds_face.shape[1], axis=2
            ).reshape(-1)
            J_face = np.repeat(
                inds_face[:, None, :], inds_face.shape[1], axis=1
            ).reshape(-1)
            self.I = np.hstack((self.I, I_face))
            self.J = np.hstack((self.J, J_face))
            self.cells_list_face_list.append(cells_list_face)

        self.cells_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(
            *self.cells_list
        )  # (num_cells, num_nodes + ...)

        dumb_array_dof = [np.zeros((fe.num_nodes, fe.vec)) for fe in self.fes]
        dumb_array_node = [np.zeros(fe.num_nodes) for fe in self.fes]
        # _, unflatten_fn_node = jax.flatten_util.ravel_pytree(dumb_array_node)
        _, self.unflatten_fn_dof = jax.flatten_util.ravel_pytree(dumb_array_dof)

        dumb_sol_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in self.fes]
        dumb_dofs, self.unflatten_fn_sol_list = jax.flatten_util.ravel_pytree(
            dumb_sol_list
        )
        self.num_total_dofs_all_vars = len(dumb_dofs)

        self.num_nodes_cumsum = np.cumsum(np.array([0] + [fe.num_nodes for fe in self.fes]))
        # (num_cells, num_vars, num_quads)
        self.JxW = np.transpose(np.stack([fe.JxW for fe in self.fes]), axes=(1, 0, 2))
        # (num_cells, num_quads, num_nodes +..., dim)
        self.shape_grads = np.concatenate([fe.shape_grads for fe in self.fes], axis=2)
        # (num_cells, num_quads, num_nodes + ..., 1, dim)
        self.v_grads_JxW = np.concatenate([fe.v_grads_JxW for fe in self.fes], axis=2)

        # Verify all variables have the same quadrature points
        if len(self.fes) > 1:
            ref_quad_points = self.fes[0].get_physical_quad_points()
            for i, fe in enumerate(self.fes[1:], 1):
                quad_points = fe.get_physical_quad_points()
                assert np.allclose(ref_quad_points, quad_points), \
                    f"Finite element space {i} has different quadrature points"
        # (num_cells, num_quads, dim)
        self.physical_quad_points = self.fes[0].get_physical_quad_points()

        self.selected_face_shape_grads = []
        self.nanson_scale = []
        self.selected_face_shape_vals = []
        self.physical_surface_quad_points = []
        for boundary_inds in self.boundary_inds_list:
            s_shape_grads = []
            n_scale = []
            s_shape_vals = []
            for fe in self.fes:
                # (num_selected_faces, num_face_quads, num_nodes, dim), (num_selected_faces, num_face_quads)
                face_shape_grads_physical, nanson_scale = fe.get_face_shape_grads(
                    boundary_inds
                )
                selected_face_shape_vals = fe.face_shape_vals[
                    boundary_inds[:, 1]
                ]  # (num_selected_faces, num_face_quads, num_nodes)
                s_shape_grads.append(face_shape_grads_physical)
                n_scale.append(nanson_scale)
                s_shape_vals.append(selected_face_shape_vals)

            # (num_selected_faces, num_face_quads, num_nodes + ..., dim)
            s_shape_grads = np.concatenate(s_shape_grads, axis=2)
            # (num_selected_faces, num_vars, num_face_quads)
            n_scale = np.transpose(np.stack(n_scale), axes=(1, 0, 2))
            # (num_selected_faces, num_face_quads, num_nodes + ...)
            s_shape_vals = np.concatenate(s_shape_vals, axis=2)
            # (num_selected_faces, num_face_quads, dim)
            physical_surface_quad_points = self.fes[0].get_physical_surface_quad_points(
                boundary_inds
            )

            self.selected_face_shape_grads.append(s_shape_grads)
            self.nanson_scale.append(n_scale)
            self.selected_face_shape_vals.append(s_shape_vals)
            # Verify all variables have the same face quadrature points
            if len(self.fes) > 1:
                ref_face_quad_points = self.fes[0].get_physical_quad_points_face(self.boundary_inds_list[ind])
                for i, fe in enumerate(self.fes[1:], 1):
                    other_quad_points = fe.get_physical_quad_points_face(self.boundary_inds_list[ind])
                    assert np.allclose(ref_face_quad_points, other_quad_points), \
                        f"Finite element space {i} has different face quadrature points for boundary {ind}"
            self.physical_surface_quad_points.append(physical_surface_quad_points)

        self.internal_vars = ()
        self.internal_vars_surfaces = [() for _ in range(len(self.boundary_inds_list))]
        
        # Initialize kernel generator
        self.kernel_generator = KernelGenerator(self.fes, self.unflatten_fn_dof, self.dim)
        
        self.custom_init(*self.additional_info)
        self.pre_jit_fns()

    def custom_init(self, *args: Any) -> None:
        """Custom initialization hook for subclasses.

        This method is called during __post_init__ and can be overridden by
        subclasses to perform problem-specific initialization tasks such as
        setting up material parameters, internal variables, or custom data structures.

        Args:
            *args: Variable arguments passed from additional_info attribute.

        Note:
            The default implementation does nothing. Subclasses should override
            this method to implement their specific initialization requirements.
        """
        pass

    def get_laplace_kernel(self, tensor_map: Callable) -> Callable:
        """Create a kernel function for Laplace-type (gradient-based) weak forms.

        Delegates to KernelGenerator for kernel creation.

        Args:
            tensor_map (Callable): Function that maps solution gradients to flux/stress.

        Returns:
            Callable: Element kernel function for gradient-based weak forms.
        """
        return self.kernel_generator.get_laplace_kernel(tensor_map)

    def get_mass_kernel(self, mass_map: Callable) -> Callable:
        """Create a kernel function for mass-type (solution-based) weak forms.

        Delegates to KernelGenerator for kernel creation.

        Args:
            mass_map (Callable): Function that maps solution values to source terms.

        Returns:
            Callable: Element kernel function for mass-type weak forms.
        """
        return self.kernel_generator.get_mass_kernel(mass_map)

    def get_surface_kernel(self, surface_map: Callable) -> Callable:
        """Create a kernel function for surface integral weak forms.

        Delegates to KernelGenerator for kernel creation.

        Args:
            surface_map (Callable): Function that maps surface solution values to fluxes.

        Returns:
            Callable: Surface kernel function for boundary/interface terms.
        """
        return self.kernel_generator.get_surface_kernel(surface_map)

    def pre_jit_fns(self) -> None:
        """Prepare and JIT-compile kernel functions for efficient computation.

        This method sets up the computational kernels for volume and surface integrals,
        applies JAX transformations for automatic differentiation, and JIT-compiles
        the resulting functions for optimal performance during assembly.

        The method creates:
        - Volume integral kernels (self.kernel, self.kernel_jac)
        - Surface integral kernels (self.kernel_face, self.kernel_jac_face)
        - Forward and reverse mode automatic differentiation wrappers

        Note:
            This method is computationally expensive due to JIT compilation but only
            needs to be called once during problem setup.
        """

        def value_and_jacfwd(f, x):
            pushfwd = functools.partial(jax.jvp, f, (x,))
            basis = np.eye(len(x.reshape(-1)), dtype=x.dtype).reshape(-1, *x.shape)
            y, jac = jax.vmap(pushfwd, out_axes=(None, -1))((basis,))
            return y, jac

        def get_kernel_fn_cell():
            def kernel(
                cell_sol_flat,
                physical_quad_points,
                cell_shape_grads,
                cell_JxW,
                cell_v_grads_JxW,
                *cell_internal_vars,
            ):
                """
                universal_kernel should be able to cover all situations (including mass_kernel and laplace_kernel).
                mass_kernel and laplace_kernel are from legacy JAX-FEM. They can still be used, but not mandatory.
                """

                # TODO: If there is no kernel map, returning 0. is not a good choice.
                # Return a zero array with proper shape will be better.
                if hasattr(self, "get_mass_map"):
                    mass_map = self.get_mass_map()
                    if mass_map is not None:
                        mass_kernel = self.get_mass_kernel(mass_map)
                        mass_val = mass_kernel(
                            cell_sol_flat,
                            physical_quad_points,
                            cell_JxW,
                            *cell_internal_vars,
                        )
                    else:
                        mass_val = 0.0
                else:
                    mass_val = 0.0

                if hasattr(self, "get_tensor_map"):
                    tensor_map = self.get_tensor_map()
                    if tensor_map is not None:
                        laplace_kernel = self.get_laplace_kernel(tensor_map)
                        laplace_val = laplace_kernel(
                            cell_sol_flat,
                            cell_shape_grads,
                            cell_v_grads_JxW,
                            *cell_internal_vars,
                        )
                    else:
                        laplace_val = 0.0
                else:
                    laplace_val = 0.0

                if hasattr(self, "get_universal_kernel"):
                    universal_kernel = self.get_universal_kernel()
                    if universal_kernel is not None:
                        universal_val = universal_kernel(
                            cell_sol_flat,
                            physical_quad_points,
                            cell_shape_grads,
                            cell_JxW,
                            cell_v_grads_JxW,
                            *cell_internal_vars,
                        )
                    else:
                        universal_val = 0.0
                else:
                    universal_val = 0.0

                return laplace_val + mass_val + universal_val

            def kernel_jac(cell_sol_flat, *args):
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return value_and_jacfwd(
                    kernel_partial, cell_sol_flat
                )  # kernel(cell_sol_flat, *args), jax.jacfwd(kernel)(cell_sol_flat, *args)

            return kernel, kernel_jac

        def get_kernel_fn_face(ind):
            def kernel(
                cell_sol_flat,
                physical_surface_quad_points,
                face_shape_vals,
                face_shape_grads,
                face_nanson_scale,
                *cell_internal_vars_surface,
            ):
                """
                universal_kernel should be able to cover all situations (including surface_kernel).
                surface_kernel is from legacy JAX-FEM. It can still be used, but not mandatory.
                """
                if hasattr(self, "get_surface_maps"):
                    surface_maps = self.get_surface_maps()
                    if surface_maps is not None and ind < len(surface_maps) and surface_maps[ind] is not None:
                        surface_kernel = self.get_surface_kernel(surface_maps[ind])
                        surface_val = surface_kernel(
                            cell_sol_flat,
                            physical_surface_quad_points,
                            face_shape_vals,
                            face_shape_grads,
                            face_nanson_scale,
                            *cell_internal_vars_surface,
                        )
                    else:
                        surface_val = 0.0
                else:
                    surface_val = 0.0

                if hasattr(self, "get_universal_kernels_surface"):
                    universal_kernels_surface = self.get_universal_kernels_surface()
                    if universal_kernels_surface is not None and ind < len(universal_kernels_surface) and universal_kernels_surface[ind] is not None:
                        universal_kernel = universal_kernels_surface[ind]
                        universal_val = universal_kernel(
                            cell_sol_flat,
                            physical_surface_quad_points,
                            face_shape_vals,
                            face_shape_grads,
                            face_nanson_scale,
                            *cell_internal_vars_surface,
                        )
                    else:
                        universal_val = 0.0
                else:
                    universal_val = 0.0

                return surface_val + universal_val

            def kernel_jac(cell_sol_flat, *args):
                # return jax.jacfwd(kernel)(cell_sol_flat, *args)
                kernel_partial = lambda cell_sol_flat: kernel(cell_sol_flat, *args)
                return value_and_jacfwd(
                    kernel_partial, cell_sol_flat
                )  # kernel(cell_sol_flat, *args), jax.jacfwd(kernel)(cell_sol_flat, *args)

            return kernel, kernel_jac

        kernel, kernel_jac = get_kernel_fn_cell()
        kernel = jax.jit(jax.vmap(kernel))
        kernel_jac = jax.jit(jax.vmap(kernel_jac))
        self.kernel = kernel
        self.kernel_jac = kernel_jac

        num_surfaces = len(self.boundary_inds_list)
        if hasattr(self, "get_surface_maps"):
            assert num_surfaces == len(
                self.get_surface_maps()
            ), f"Mismatched number of surfaces: {num_surfaces} != {len(self.get_surface_maps())}"
        elif hasattr(self, "get_universal_kernels_surface"):
            assert num_surfaces == len(
                self.get_universal_kernels_surface()
            ), f"Mismatched number of surfaces: {num_surfaces} != {len(self.get_universal_kernels_surface())}"
        else:
            assert num_surfaces == 0, "Missing definitions for surface integral"

        self.kernel_face = []
        self.kernel_jac_face = []
        for i in range(len(self.boundary_inds_list)):
            kernel_face, kernel_jac_face = get_kernel_fn_face(i)
            kernel_face = jax.jit(jax.vmap(kernel_face))
            kernel_jac_face = jax.jit(jax.vmap(kernel_jac_face))
            self.kernel_face.append(kernel_face)
            self.kernel_jac_face.append(kernel_jac_face)

    def split_and_compute_cell(
        self, cells_sol_flat: np.ndarray, np_version: bool, jac_flag: bool, internal_vars: List[Any]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[Any]]:
        """Compute volume integrals in the weak form with memory-efficient batching.

        Evaluates element-level residuals and optionally Jacobians for all cells in the mesh.
        Uses batching to manage memory usage for large problems.

        Args:
            cells_sol_flat (np.ndarray): Flattened cell solution data with shape
                (num_cells, num_nodes*vec + ...).
            np_version: NumPy backend (np for JAX).
            jac_flag (bool): Whether to compute Jacobians in addition to residuals.
            internal_vars (tuple): Additional internal variables for the computation.

        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]: Element residuals with shape
                (num_cells, num_nodes*vec + ...). If jac_flag=True, also returns
                Jacobians with shape (num_cells, num_nodes*vec + ..., num_nodes*vec + ...).
        """
        kernel = self.kernel_jac if jac_flag else self.kernel
        kernel_jac = self.kernel_jac if jac_flag else None
        
        return AssemblyManager.split_and_compute_cell(
            cells_sol_flat=cells_sol_flat,
            kernel=kernel,
            kernel_jac=kernel_jac,
            physical_quad_points=self.physical_quad_points,
            shape_grads=self.shape_grads,
            JxW=self.JxW,
            v_grads_JxW=self.v_grads_JxW,
            internal_vars=internal_vars,
            jac_flag=jac_flag,
            num_cells=self.num_cells,
            np_version=np_version
        )

    def compute_face(
        self, cells_sol_flat, np_version, jac_flag, internal_vars_surfaces
    ):
        """Compute surface integrals in the weak form for all boundary subdomains.

        Evaluates face-level residuals and optionally Jacobians for boundary faces.
        Handles multiple boundary subdomains with different boundary conditions.

        Args:
            cells_sol_flat (np.ndarray): Flattened cell solution data with shape
                (num_cells, num_nodes*vec + ...).
            np_version: NumPy backend (np for JAX).
            jac_flag (bool): Whether to compute Jacobians in addition to residuals.
            internal_vars_surfaces (List[tuple]): Internal variables for each surface subdomain.

        Returns:
            List[np.ndarray] or Tuple[List[np.ndarray], List[np.ndarray]]:
                List of face residuals for each boundary subdomain. If jac_flag=True,
                also returns list of face Jacobians.
        """
        return AssemblyManager.compute_face(
            cells_sol_flat=cells_sol_flat,
            kernel_face=self.kernel_face,
            kernel_jac_face=self.kernel_jac_face,
            boundary_inds_list=self.boundary_inds_list,
            physical_surface_quad_points=self.physical_surface_quad_points,
            selected_face_shape_vals=self.selected_face_shape_vals,
            selected_face_shape_grads=self.selected_face_shape_grads,
            nanson_scale=self.nanson_scale,
            internal_vars_surfaces=internal_vars_surfaces,
            jac_flag=jac_flag,
            np_version=np_version
        )

    def compute_residual_vars_helper(self, weak_form_flat: List[np.ndarray], weak_form_face_flat: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Assemble global residual from element and face contributions.

        Accumulates element-level weak form contributions into global residual vectors
        for each variable using scatter-add operations.

        Args:
            weak_form_flat (np.ndarray): Element weak form contributions with shape
                (num_cells, num_nodes*vec + ...).
            weak_form_face_flat (List[np.ndarray]): Face weak form contributions
                for each boundary subdomain.

        Returns:
            List[np.ndarray]: Global residual vectors for each variable.
        """
        return AssemblyManager.compute_residual_vars_helper(
            weak_form_flat=weak_form_flat,
            weak_form_face_flat=weak_form_face_flat,
            fes=self.fes,
            unflatten_fn_dof=self.unflatten_fn_dof,
            cells_list=self.cells_list,
            cells_list_face_list=self.cells_list_face_list
        )

    def compute_residual_vars(self, sol_list: List[np.ndarray], internal_vars: List[Any], internal_vars_surfaces: List[Any]) -> Tuple[List[np.ndarray], List[Any], List[Any]]:
        """Compute residual vectors with specified internal variables.

        Lower-level interface for residual computation that allows specifying
        custom internal variables for advanced use cases.

        Args:
            sol_list (List[np.ndarray]): Solution arrays for each variable.
            internal_vars (tuple): Internal variables for volume integrals.
            internal_vars_surfaces (List[tuple]): Internal variables for surface integrals.

        Returns:
            List[np.ndarray]: Residual vectors for each variable.
        """
        logger.debug(f"Computing cell residual...")
        cells_sol_list = [
            sol[cells] for cells, sol in zip(self.cells_list, sol_list)
        ]  # [(num_cells, num_nodes, vec), ...]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(
            *cells_sol_list
        )  # (num_cells, num_nodes*vec + ...)
        weak_form_flat = self.split_and_compute_cell(
            cells_sol_flat, np, False, internal_vars
        )  # (num_cells, num_nodes*vec + ...)
        weak_form_face_flat = self.compute_face(
            cells_sol_flat, np, False, internal_vars_surfaces
        )  # [(num_selected_faces, num_nodes*vec + ...), ...]
        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

    def compute_newton_vars(self, sol_list: List[np.ndarray], internal_vars: List[Any], internal_vars_surfaces: List[Any]) -> Tuple[List[np.ndarray], List[Any], List[Any]]:
        """Compute residual and Jacobian with specified internal variables.

        Lower-level interface for Newton step computation that allows specifying
        custom internal variables for advanced use cases.

        Args:
            sol_list (List[np.ndarray]): Solution arrays for each variable.
            internal_vars (tuple): Internal variables for volume integrals.
            internal_vars_surfaces (List[tuple]): Internal variables for surface integrals.

        Returns:
            List[np.ndarray]: Residual vectors for each variable. Jacobian data is
                stored in self.V for subsequent sparse matrix assembly.
        """
        logger.debug(f"Computing cell Jacobian and cell residual...")
        cells_sol_list = [
            sol[cells] for cells, sol in zip(self.cells_list, sol_list)
        ]  # [(num_cells, num_nodes, vec), ...]
        cells_sol_flat = jax.vmap(lambda *x: jax.flatten_util.ravel_pytree(x)[0])(
            *cells_sol_list
        )  # (num_cells, num_nodes*vec + ...)
        # (num_cells, num_nodes*vec + ...),  (num_cells, num_nodes*vec + ..., num_nodes*vec + ...)
        weak_form_flat, cells_jac_flat = self.split_and_compute_cell(
            cells_sol_flat, np, True, internal_vars
        )
        # Handle traced arrays during vmap/autodiff
        if hasattr(cells_jac_flat, '_trace'):
            self.V = cells_jac_flat.reshape(-1)  # Keep as JAX array for traced computation
        else:
            self.V = np.array(cells_jac_flat.reshape(-1))

        # [(num_selected_faces, num_nodes*vec + ...,), ...], [(num_selected_faces, num_nodes*vec + ..., num_nodes*vec + ...,), ...]
        weak_form_face_flat, cells_jac_face_flat = self.compute_face(
            cells_sol_flat, np, True, internal_vars_surfaces
        )
        for cells_jac_f_flat in cells_jac_face_flat:
            self.V = np.hstack((self.V, np.array(cells_jac_f_flat.reshape(-1))))

        return self.compute_residual_vars_helper(weak_form_flat, weak_form_face_flat)

    def compute_residual(self, sol_list: List[np.ndarray]) -> List[np.ndarray]:
        """Compute the residual vector for the current solution.

        Evaluates the weak form residual R(u) = 0 for the given solution.
        This is the main interface for residual computation used by nonlinear solvers.

        Args:
            sol_list (List[np.ndarray]): List of solution arrays for each variable.
                Each array has shape (num_nodes, vec).

        Returns:
            List[np.ndarray]: List of residual arrays for each variable with the
                same structure as sol_list.
        """
        return self.compute_residual_vars(
            sol_list, self.internal_vars, self.internal_vars_surfaces
        )

    def newton_update(self, sol_list: List[np.ndarray]) -> List[np.ndarray]:
        """Compute residual and Jacobian for Newton-Raphson iteration.

        Performs the core computation for Newton's method by evaluating both the
        residual vector and its Jacobian matrix at the current solution state.

        Args:
            sol_list (List[np.ndarray]): List of solution arrays for each variable.
                Each array has shape (num_nodes, vec).

        Returns:
            List[np.ndarray]: List of residual arrays for each variable. The Jacobian
                data is stored internally in self.V for subsequent sparse matrix assembly.

        Note:
            After calling this method, use compute_csr() to assemble the global
            sparse matrix from the computed Jacobian data.
        """
        return self.compute_newton_vars(
            sol_list, self.internal_vars, self.internal_vars_surfaces
        )

    def set_params(self, params: Any) -> None:
        """Set problem parameters for inverse problems and optimization.

        This method updates problem parameters (e.g., material properties, geometry)
        during inverse problem solving or parameter optimization.

        Args:
            params: Problem parameters to update. Format depends on specific implementation.

        Raises:
            NotImplementedError: This method must be implemented by subclasses that
                support parameter updates.

        Note:
            Used primarily in parameter identification, shape optimization, and
            material property estimation problems.
        """
        raise NotImplementedError("Child class must implement this function!")

    def compute_csr(self, chunk_size: Optional[int] = None):
        """Assemble the global sparse matrix in CSR format.

        Constructs the global system matrix from element-level Jacobian contributions.
        Supports memory-efficient assembly using chunking for large problems.

        Args:
            chunk_size (Optional[int], optional): Size of chunks for memory-efficient
                assembly. If None, assembles the entire matrix at once. Useful for
                large problems to control memory usage.

        Raises:
            ValueError: If newton_update() has not been called first to compute element
                Jacobians, or if chunk_size is not positive.

        Note:
            Must call newton_update() before this method to populate the self.V array
            with element Jacobian data. The resulting sparse matrix is stored in
            self.csr_array.
        """
        if not hasattr(self, "V"):
            raise ValueError(
                "You must call newton_update() before computing the CSR matrix."
            )

        self.csr_array = AssemblyManager.compute_csr(
            V=self.V,
            I=self.I,
            J=self.J,
            num_total_dofs_all_vars=self.num_total_dofs_all_vars,
            chunk_size=chunk_size
        )

    def precompute_bc_data(self) -> dict:
        """Pre-compute boundary condition data for efficient JIT compilation.
        
        This method extracts all boundary condition information from the finite element
        spaces and converts it to JAX-compatible arrays. This separation allows the
        boundary condition processing (which uses non-JIT operations like np.argwhere)
        to be done once during setup, while the actual application can be JIT-compiled.
        
        Returns:
            dict: Dictionary containing boundary condition data.
                
        Note:
            This method delegates to BoundaryConditionManager.precompute_bc_data().
        """
        return BoundaryConditionManager.precompute_bc_data(self.fes, self.offset)

    def assemble(self, dofs: np.ndarray, bc_data: dict = None, prolongation_matrix: BCOO = None, 
                macro_term: np.ndarray = None, apply_bcs: bool = True) -> tuple:
        """Assemble the finite element system matrix and residual vector.
        
        This method performs the complete assembly of the finite element system
        including residual computation, Jacobian matrix assembly, boundary condition
        application, and handling of prolongation matrices and macro terms.
        
        Args:
            dofs (np.ndarray): Current solution degrees of freedom vector.
            bc_data (dict, optional): Boundary condition data from precompute_bc_data().
                If None, uses problem's BC data. If no BCs exist, pass empty dict.
            prolongation_matrix (BCOO, optional): Prolongation matrix for constraints.
                If None, uses problem's prolongation matrix.
            macro_term (np.ndarray, optional): Macro displacement term.
                If None, uses problem's macro term.
            apply_bcs (bool, optional): Whether to apply boundary conditions. Defaults to True.
            
        Returns:
            tuple: (A, b) where:
                - A: System matrix (BCOO sparse format) with BCs applied if requested
                - b: Right-hand side vector with BCs applied if requested
                
        Note:
            This method handles the complete assembly including non-JIT-compatible 
            parts like sparse matrix construction and boundary condition application.
            The resulting (A, b) can be passed directly to solver.solve().
            
        Example:
            >>> bc_data = problem.precompute_bc_data()
            >>> A, b = problem.assemble(dofs, bc_data)
            >>> solution = solver.solve(A, b, solver_options)
        """
        from fealax.solver import get_A
        
        return AssemblyManager.assemble(
            dofs=dofs,
            fes=self.fes,
            unflatten_fn_sol_list=self.unflatten_fn_sol_list,
            newton_update_fn=self.newton_update,
            compute_csr_fn=self.compute_csr,
            get_A_fn=lambda: get_A(self),
            precompute_bc_data_fn=self.precompute_bc_data,
            apply_bcs_fn=self.apply_bcs_to_assembled_system,
            bc_data=bc_data,
            prolongation_matrix=prolongation_matrix if prolongation_matrix is not None else self.prolongation_matrix,
            macro_term=macro_term if macro_term is not None else self.macro_term,
            apply_bcs=apply_bcs
        )


    def apply_bcs_to_assembled_system(self, A: BCOO, b: np.ndarray, bc_data: dict) -> tuple:
        """Apply boundary conditions to an assembled finite element system.
        
        This method applies Dirichlet boundary conditions to the assembled system
        matrix and right-hand side vector using the row elimination method. It uses
        the precomputed boundary condition data to avoid non-JIT-compatible operations.
        
        Args:
            A (BCOO): System matrix in JAX sparse format.
            b (np.ndarray): Right-hand side vector.
            bc_data (dict): Precomputed boundary condition data from precompute_bc_data().
            
        Returns:
            tuple: (A_bc, b_bc) with boundary conditions applied.
                
        Note:
            This method delegates to BoundaryConditionManager.apply_bcs_to_assembled_system().
        """
        return BoundaryConditionManager.apply_bcs_to_assembled_system(A, b, bc_data)
