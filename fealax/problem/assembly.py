"""Finite element system assembly functionality.

This module provides classes and utilities for assembling finite element systems
including volume integrals, surface integrals, global residual assembly, and
sparse matrix construction. The assembly operations are core to finite element
computation and handle both residual and Jacobian assembly for nonlinear problems.

The module supports:
    - Memory-efficient volume integral computation with batching
    - Surface integral computation for boundary conditions
    - Global residual assembly using scatter-add operations
    - Sparse matrix assembly in JAX BCOO format
    - Complete finite element system assembly with boundary conditions

Key Classes:
    AssemblyManager: Main class containing all assembly operations

Example:
    Basic usage for finite element assembly:

    >>> from fealax.problem.assembly import AssemblyManager
    >>> assembler = AssemblyManager()
    >>> 
    >>> # Compute volume integrals
    >>> weak_form_flat = assembler.split_and_compute_cell(
    ...     cells_sol_flat, kernel, kernel_jac, physical_quad_points,
    ...     shape_grads, JxW, v_grads_JxW, internal_vars, jac_flag=False
    ... )
    >>> 
    >>> # Compute surface integrals
    >>> weak_form_face_flat = assembler.compute_face(
    ...     cells_sol_flat, kernel_face, kernel_jac_face, boundary_inds_list,
    ...     physical_surface_quad_points, selected_face_shape_vals,
    ...     selected_face_shape_grads, nanson_scale, internal_vars_surfaces,
    ...     jac_flag=False
    ... )
    >>> 
    >>> # Assemble global residual
    >>> res_list = assembler.compute_residual_vars_helper(
    ...     weak_form_flat, weak_form_face_flat, fes, unflatten_fn_dof,
    ...     cells_list, cells_list_face_list
    ... )
"""

import jax
import jax.numpy as np
import jax.flatten_util
from typing import Any, Callable, Optional, List, Tuple, Union
import functools
from jax.experimental.sparse import BCOO
from fealax.fe import FiniteElement
from fealax import logger
from fealax.problem.boundary_conditions import BoundaryConditionManager
import gc


class AssemblyManager:
    """Manager class for finite element system assembly operations.

    This class provides methods for assembling finite element systems including
    volume and surface integral computation, global residual assembly, and sparse
    matrix construction. The methods are designed to work with JAX transformations
    for optimal performance and automatic differentiation.

    The class handles:
        - Element-level weak form computation with memory-efficient batching
        - Surface integral evaluation for boundary conditions
        - Global assembly using scatter-add operations
        - Sparse matrix assembly in JAX BCOO format
        - Complete system assembly with boundary condition application

    Note:
        This class contains static methods that can be used independently or
        as part of a larger finite element framework. The methods are designed
        to be composable and work with JAX's functional programming paradigm.
    """

    @staticmethod
    def split_and_compute_cell(
        cells_sol_flat: np.ndarray,
        kernel: Callable,
        kernel_jac: Callable,
        physical_quad_points: np.ndarray,
        shape_grads: np.ndarray,
        JxW: np.ndarray,
        v_grads_JxW: np.ndarray,
        internal_vars: List[Any],
        jac_flag: bool,
        num_cells: int,
        np_version: Any = np
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Compute volume integrals in the weak form with memory-efficient batching.

        Evaluates element-level residuals and optionally Jacobians for all cells in the mesh.
        Uses batching to manage memory usage for large problems by processing cells in chunks.

        Args:
            cells_sol_flat (np.ndarray): Flattened cell solution data with shape
                (num_cells, num_nodes*vec + ...).
            kernel (Callable): JIT-compiled kernel function for residual computation.
            kernel_jac (Callable): JIT-compiled kernel function for Jacobian computation.
            physical_quad_points (np.ndarray): Physical coordinates of quadrature points
                with shape (num_cells, num_quads, dim).
            shape_grads (np.ndarray): Shape function gradients with shape
                (num_cells, num_quads, num_nodes + ..., dim).
            JxW (np.ndarray): Jacobian determinant times quadrature weights with shape
                (num_cells, num_vars, num_quads).
            v_grads_JxW (np.ndarray): Test function gradients times JxW with shape
                (num_cells, num_quads, num_nodes + ..., 1, dim).
            internal_vars (List[Any]): Additional internal variables for the computation.
            jac_flag (bool): Whether to compute Jacobians in addition to residuals.
            num_cells (int): Total number of cells in the mesh.
            np_version (Any, optional): NumPy backend (np for JAX). Defaults to np.

        Returns:
            Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]: Element residuals with shape
                (num_cells, num_nodes*vec + ...). If jac_flag=True, also returns
                Jacobians with shape (num_cells, num_nodes*vec + ..., num_nodes*vec + ...).

        Note:
            The batching strategy uses 20 chunks by default, which can be adjusted based
            on available memory. For very large problems, consider using fewer chunks
            to reduce memory overhead.
        """
        vmap_fn = kernel_jac if jac_flag else kernel
        num_cuts = 20
        if num_cuts > num_cells:
            num_cuts = num_cells
        batch_size = num_cells // num_cuts
        input_collection = [
            cells_sol_flat,
            physical_quad_points,
            shape_grads,
            JxW,
            v_grads_JxW,
            *internal_vars,
        ]

        if jac_flag:
            values = []
            jacs = []
            for i in range(num_cuts):
                if i < num_cuts - 1:
                    input_col = jax.tree_map(
                        lambda x: x[i * batch_size : (i + 1) * batch_size],
                        input_collection,
                    )
                else:
                    input_col = jax.tree_map(
                        lambda x: x[i * batch_size :], input_collection
                    )

                val, jac = vmap_fn(*input_col)
                values.append(val)
                jacs.append(jac)
            # Handle traced arrays during vmap/autodiff by using JAX operations
            if values and hasattr(values[0], '_trace'):
                values = np.vstack(values)
                jacs = np.vstack(jacs) if jacs else jacs
            else:
                values = np_version.vstack(values)
                jacs = np_version.vstack(jacs)

            return values, jacs
        else:
            values = []
            for i in range(num_cuts):
                if i < num_cuts - 1:
                    input_col = jax.tree_map(
                        lambda x: x[i * batch_size : (i + 1) * batch_size],
                        input_collection,
                    )
                else:
                    input_col = jax.tree_map(
                        lambda x: x[i * batch_size :], input_collection
                    )

                val = vmap_fn(*input_col)
                values.append(val)
            # Handle traced arrays during vmap/autodiff by using JAX operations
            if values and hasattr(values[0], '_trace'):
                values = np.vstack(values)
            else:
                values = np_version.vstack(values)
            return values

    @staticmethod
    def compute_face(
        cells_sol_flat: np.ndarray,
        kernel_face: List[Callable],
        kernel_jac_face: List[Callable],
        boundary_inds_list: List[np.ndarray],
        physical_surface_quad_points: List[np.ndarray],
        selected_face_shape_vals: List[np.ndarray],
        selected_face_shape_grads: List[np.ndarray],
        nanson_scale: List[np.ndarray],
        internal_vars_surfaces: List[List[Any]],
        jac_flag: bool,
        np_version: Any = np
    ) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
        """Compute surface integrals in the weak form for all boundary subdomains.

        Evaluates face-level residuals and optionally Jacobians for boundary faces.
        Handles multiple boundary subdomains with different boundary conditions and
        supports both Neumann and Robin-type boundary conditions.

        Args:
            cells_sol_flat (np.ndarray): Flattened cell solution data with shape
                (num_cells, num_nodes*vec + ...).
            kernel_face (List[Callable]): List of JIT-compiled surface kernel functions
                for residual computation, one for each boundary subdomain.
            kernel_jac_face (List[Callable]): List of JIT-compiled surface kernel functions
                for Jacobian computation, one for each boundary subdomain.
            boundary_inds_list (List[np.ndarray]): List of boundary face indices for each
                subdomain. Each array has shape (num_selected_faces, 2) with [cell_id, face_id].
            physical_surface_quad_points (List[np.ndarray]): List of physical coordinates
                of surface quadrature points for each subdomain.
            selected_face_shape_vals (List[np.ndarray]): List of face shape function values
                for each subdomain.
            selected_face_shape_grads (List[np.ndarray]): List of face shape function gradients
                for each subdomain.
            nanson_scale (List[np.ndarray]): List of surface measure scaling factors
                for each subdomain.
            internal_vars_surfaces (List[List[Any]]): List of internal variables for each
                surface subdomain.
            jac_flag (bool): Whether to compute Jacobians in addition to residuals.
            np_version (Any, optional): NumPy backend (np for JAX). Defaults to np.

        Returns:
            Union[List[np.ndarray], Tuple[List[np.ndarray], List[np.ndarray]]]:
                List of face residuals for each boundary subdomain. If jac_flag=True,
                also returns list of face Jacobians.

        Note:
            The surface integrals are computed independently for each boundary subdomain,
            allowing for different boundary condition types and material properties
            on different parts of the boundary.
        """
        if jac_flag:
            values = []
            jacs = []
            for i, boundary_inds in enumerate(boundary_inds_list):
                vmap_fn = kernel_jac_face[i]
                selected_cell_sols_flat = cells_sol_flat[
                    boundary_inds[:, 0]
                ]  # (num_selected_faces, num_nodes*vec + ...))
                input_collection = [
                    selected_cell_sols_flat,
                    physical_surface_quad_points[i],
                    selected_face_shape_vals[i],
                    selected_face_shape_grads[i],
                    nanson_scale[i],
                    *internal_vars_surfaces[i],
                ]

                val, jac = vmap_fn(*input_collection)
                values.append(val)
                jacs.append(jac)
            return values, jacs
        else:
            values = []
            for i, boundary_inds in enumerate(boundary_inds_list):
                vmap_fn = kernel_face[i]
                selected_cell_sols_flat = cells_sol_flat[
                    boundary_inds[:, 0]
                ]  # (num_selected_faces, num_nodes*vec + ...))
                input_collection = [
                    selected_cell_sols_flat,
                    physical_surface_quad_points[i],
                    selected_face_shape_vals[i],
                    selected_face_shape_grads[i],
                    nanson_scale[i],
                    *internal_vars_surfaces[i],
                ]
                val = vmap_fn(*input_collection)
                values.append(val)
            return values

    @staticmethod
    def compute_residual_vars_helper(
        weak_form_flat: np.ndarray,
        weak_form_face_flat: List[np.ndarray],
        fes: List[FiniteElement],
        unflatten_fn_dof: Callable,
        cells_list: List[np.ndarray],
        cells_list_face_list: List[List[np.ndarray]]
    ) -> List[np.ndarray]:
        """Assemble global residual from element and face contributions.

        Accumulates element-level weak form contributions into global residual vectors
        for each variable using scatter-add operations. This method handles the transition
        from element-level computations to the global finite element system.

        Args:
            weak_form_flat (np.ndarray): Element weak form contributions with shape
                (num_cells, num_nodes*vec + ...). These are the computed element
                residuals from volume integral evaluation.
            weak_form_face_flat (List[np.ndarray]): Face weak form contributions
                for each boundary subdomain. Each array has shape
                (num_selected_faces, num_nodes*vec + ...).
            fes (List[FiniteElement]): List of finite element spaces for each variable.
            unflatten_fn_dof (Callable): Function to unflatten degree of freedom arrays
                from flat representation back to structured format.
            cells_list (List[np.ndarray]): List of cell connectivity arrays for each variable.
                Each array has shape (num_cells, num_nodes_per_element).
            cells_list_face_list (List[List[np.ndarray]]): List of face connectivity arrays
                for each boundary subdomain and variable.

        Returns:
            List[np.ndarray]: Global residual vectors for each variable with shape
                (num_total_nodes, vec) where vec is the number of components per node.

        Note:
            This method uses JAX's scatter-add functionality (.at[indices].add(values))
            to efficiently accumulate contributions from overlapping elements. The
            operation is differentiable and compatible with JAX transformations.
        """
        res_list = [np.zeros((fe.num_total_nodes, fe.vec)) for fe in fes]
        weak_form_list = jax.vmap(lambda x: unflatten_fn_dof(x))(
            weak_form_flat
        )  # [(num_cells, num_nodes, vec), ...]
        res_list = [
            res_list[i]
            .at[cells_list[i].reshape(-1)]
            .add(weak_form_list[i].reshape(-1, fes[i].vec))
            for i in range(len(fes))
        ]

        for ind, cells_list_face in enumerate(cells_list_face_list):
            weak_form_face_list = jax.vmap(lambda x: unflatten_fn_dof(x))(
                weak_form_face_flat[ind]
            )  # [(num_selected_faces, num_nodes, vec), ...]
            res_list = [
                res_list[i]
                .at[cells_list_face[i].reshape(-1)]
                .add(weak_form_face_list[i].reshape(-1, fes[i].vec))
                for i in range(len(fes))
            ]

        return res_list

    @staticmethod
    def compute_csr(
        V: np.ndarray,
        I: np.ndarray,
        J: np.ndarray,
        num_total_dofs_all_vars: int,
        chunk_size: Optional[int] = None
    ) -> BCOO:
        """Assemble the global sparse matrix in CSR format.

        Constructs the global system matrix from element-level Jacobian contributions
        stored in coordinate (COO) format. Supports memory-efficient assembly using
        chunking for large problems that exceed available memory.

        Args:
            V (np.ndarray): Flattened array of all Jacobian values from element and
                face contributions. Shape: (total_nnz,).
            I (np.ndarray): Row indices for sparse matrix assembly. Shape: (total_nnz,).
            J (np.ndarray): Column indices for sparse matrix assembly. Shape: (total_nnz,).
            num_total_dofs_all_vars (int): Total number of degrees of freedom across
                all variables in the system.
            chunk_size (Optional[int], optional): Size of chunks for memory-efficient
                assembly. If None, assembles the entire matrix at once. Useful for
                large problems to control memory usage.

        Returns:
            BCOO: Sparse matrix in JAX BCOO (Block Coordinate) format with shape
                (num_total_dofs_all_vars, num_total_dofs_all_vars).

        Raises:
            ValueError: If chunk_size is not positive when provided.

        Note:
            The BCOO format is used because it's compatible with JAX transformations
            and GPU acceleration. For very large systems, chunking helps manage
            memory by processing the sparse matrix construction in smaller pieces.

        Example:
            >>> # Assemble without chunking
            >>> csr_matrix = AssemblyManager.compute_csr(V, I, J, total_dofs)
            >>> 
            >>> # Assemble with memory-efficient chunking
            >>> csr_matrix = AssemblyManager.compute_csr(V, I, J, total_dofs, chunk_size=100000)
        """
        logger.debug(f"Creating sparse matrix with JAX BCOO...")

        if chunk_size is not None:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be a positive integer.")

            num_chunks = (V.shape[0] + chunk_size - 1) // chunk_size
            csr_shape = (num_total_dofs_all_vars, num_total_dofs_all_vars)
            # Initialize empty lists to accumulate chunks
            all_data = []
            all_indices = []

            for i in range(num_chunks):
                V_chunk = V[i * chunk_size : (i + 1) * chunk_size]
                I_chunk = I[i * chunk_size : (i + 1) * chunk_size]
                J_chunk = J[i * chunk_size : (i + 1) * chunk_size]
                logger.debug(f"Building chunk {i+1}/{num_chunks}, size={len(V_chunk)}")

                # Collect data and indices
                all_data.append(V_chunk)
                indices_chunk = np.column_stack([I_chunk, J_chunk])
                all_indices.append(indices_chunk)
                
                del V_chunk
                del I_chunk
                del J_chunk
                gc.collect()

            # Combine all chunks
            combined_data = np.concatenate(all_data)
            combined_indices = np.concatenate(all_indices, axis=0)
            
            # Create BCOO matrix
            csr_array = BCOO((combined_data, combined_indices), shape=csr_shape)
        else:
            # Create indices array from I and J
            indices = np.column_stack([I, J])
            
            # Create BCOO matrix
            csr_array = BCOO(
                (np.array(V), indices),
                shape=(num_total_dofs_all_vars, num_total_dofs_all_vars),
            )

        return csr_array

    @staticmethod
    def assemble(
        dofs: np.ndarray,
        fes: List[FiniteElement],
        unflatten_fn_sol_list: Callable,
        newton_update_fn: Callable,
        compute_csr_fn: Callable,
        get_A_fn: Callable,
        precompute_bc_data_fn: Callable,
        apply_bcs_fn: Callable,
        bc_data: Optional[dict] = None,
        prolongation_matrix: Optional[BCOO] = None,
        macro_term: Optional[np.ndarray] = None,
        apply_bcs: bool = True
    ) -> Tuple[BCOO, np.ndarray]:
        """Assemble the finite element system matrix and residual vector.

        This method performs the complete assembly of the finite element system
        including residual computation, Jacobian matrix assembly, boundary condition
        application, and handling of prolongation matrices and macro terms.

        Args:
            dofs (np.ndarray): Current solution degrees of freedom vector.
            fes (List[FiniteElement]): List of finite element spaces.
            unflatten_fn_sol_list (Callable): Function to unflatten solution list.
            newton_update_fn (Callable): Function to compute residual and Jacobian.
            compute_csr_fn (Callable): Function to assemble sparse matrix.
            get_A_fn (Callable): Function to get system matrix.
            precompute_bc_data_fn (Callable): Function to precompute BC data.
            apply_bcs_fn (Callable): Function to apply boundary conditions.
            bc_data (Optional[dict], optional): Boundary condition data from precompute_bc_data().
                If None, uses problem's BC data. If no BCs exist, pass empty dict.
            prolongation_matrix (Optional[BCOO], optional): Prolongation matrix for constraints.
                If None, no constraint handling is applied.
            macro_term (Optional[np.ndarray], optional): Macro displacement term.
                If None, no macro term is applied.
            apply_bcs (bool, optional): Whether to apply boundary conditions. Defaults to True.

        Returns:
            Tuple[BCOO, np.ndarray]: (A, b) where:
                - A: System matrix (BCOO sparse format) with BCs applied if requested
                - b: Right-hand side vector with BCs applied if requested

        Note:
            This method handles the complete assembly including non-JIT-compatible
            parts like sparse matrix construction and boundary condition application.
            The resulting (A, b) can be passed directly to solver.solve().

        Example:
            >>> # Complete assembly with boundary conditions
            >>> A, b = AssemblyManager.assemble(
            ...     dofs, fes, unflatten_fn_sol_list, newton_update_fn,
            ...     compute_csr_fn, get_A_fn, precompute_bc_data_fn, apply_bcs_fn
            ... )
            >>> solution = solver.solve(A, b, solver_options)
        """
        # 1. Compute element residuals and Jacobians
        sol_list = unflatten_fn_sol_list(dofs)
        res_list = newton_update_fn(sol_list)
        res_vec = jax.flatten_util.ravel_pytree(res_list)[0]
        
        # 2. Assemble sparse matrix using existing infrastructure
        compute_csr_fn()  # This calls compute_csr() without chunk_size
        
        # 3. Get system matrix with boundary condition structure
        A_result = get_A_fn()
        
        # 4. Handle prolongation matrices for constrained systems
        A_original = None
        
        if prolongation_matrix is not None:
            if isinstance(A_result, tuple):
                A_original, A = A_result
                # Apply prolongation to residual
                res_vec = prolongation_matrix.T @ res_vec
            else:
                A = A_result
                A_original = A
        else:
            A = A_result
        
        # 5. Handle macro terms (affine displacements)
        if macro_term is not None and A_original is not None:
            # Convert macro term to JAX array with correct size
            macro_term_jax = np.array(macro_term)
            if macro_term_jax.shape[0] == A_original.shape[0]:
                # Compute affine force contribution
                K_affine_vec = A_original @ macro_term_jax
                if prolongation_matrix is not None:
                    affine_force = prolongation_matrix.T @ K_affine_vec
                else:
                    affine_force = K_affine_vec
                res_vec += affine_force
        
        # 6. Convert to Newton system (b = -residual for Ax = -r)
        b = -res_vec
        
        # 7. Apply boundary conditions if requested
        if apply_bcs:
            if bc_data is None:
                bc_data = precompute_bc_data_fn()
            
            A, b = apply_bcs_fn(A, b, bc_data)
        
        return A, b

    @staticmethod
    def assemble_system(
        dofs: np.ndarray,
        assemble_fn: Callable,
        precompute_bc_data_fn: Callable,
        has_prolongation: bool = False
    ) -> dict:
        """Legacy method for backward compatibility.

        This method maintains the old interface for existing code while
        using the new assemble() method internally. It provides the same
        dictionary-based return format as the original implementation.

        Args:
            dofs (np.ndarray): Current solution degrees of freedom vector.
            assemble_fn (Callable): Function to perform assembly.
            precompute_bc_data_fn (Callable): Function to precompute BC data.
            has_prolongation (bool, optional): Whether the problem has prolongation
                matrix constraints. Defaults to False.

        Returns:
            dict: Dictionary containing assembled system data with keys:
                - 'A': System matrix
                - 'b': Right-hand side vector (negative residual)
                - 'residual': Original residual vector
                - 'has_prolongation': Boolean indicating constraint presence
                - 'A_original': Original matrix (if no prolongation) or None

        Note:
            This method is provided for backward compatibility with existing
            code that expects the dictionary interface. New code should prefer
            the direct assemble() method.
        """
        # Get system without BCs applied first
        A, b = assemble_fn(dofs, apply_bcs=False)
        
        # Get BC data
        bc_data = precompute_bc_data_fn()
        
        # Return data in old format for compatibility
        return {
            'A': A,
            'b': b,  # This is already -residual
            'residual': -b,  # Original residual
            'has_prolongation': has_prolongation,
            'A_original': A if not has_prolongation else None
        }