"""Finite element mesh generation and management utilities.

This module provides classes and functions for creating, manipulating, and managing
finite element meshes. It supports various element types commonly used in computational
mechanics and provides interfaces to external mesh libraries like meshio.

The module includes:
    - Mesh class for storing and manipulating mesh data
    - Structured mesh generation functions (box_mesh)
    - Utilities for converting between different mesh formats
    - Quality assessment functions for mesh validation

Example:
    Basic usage for creating a structured mesh:

    >>> from fealax.mesh import box_mesh
    >>> mesh = box_mesh(10, 10, 10, 1.0, 1.0, 1.0)
    >>> print(f"Created mesh with {mesh.points.shape[0]} nodes")
"""

from typing import Callable, Dict, Tuple, Union

import numpy as onp
import meshio

from fealax.basis import get_elements
from fealax.basis import get_face_shape_vals_and_grads

import jax
import jax.numpy as np


class Mesh:
    """A finite element mesh class for storing mesh data and performing mesh operations.

    This class provides a lightweight interface for managing mesh data including node
    coordinates and element connectivity. It supports various element types commonly
    used in finite element analysis and provides utilities for boundary face operations.

    Attributes:
        points (np.ndarray): Node coordinates with shape (num_nodes, spatial_dim).
        cells (np.ndarray): Element connectivity matrix with shape (num_elements, nodes_per_element).
        ele_type (str): Element type identifier (e.g., 'TET4', 'HEX8', 'QUAD4').

    Note:
        A more robust mesh manager using a third-party library like meshio might be
        preferable for complex mesh operations.
    """

    def __init__(self, points: onp.ndarray, cells: onp.ndarray, ele_type: str = "TET4") -> None:
        """Initialize a finite element mesh.

        Args:
            points (np.ndarray): Node coordinates with shape (num_nodes, spatial_dim).
                Each row represents a node's coordinates in space.
            cells (np.ndarray): Element connectivity matrix with shape (num_elements, nodes_per_element).
                Each row contains the node indices that form an element.
            ele_type (str, optional): Element type identifier. Defaults to "TET4".
                Supported types include 'TET4', 'TET10', 'HEX8', 'HEX20', 'HEX27',
                'TRI3', 'TRI6', 'QUAD4', 'QUAD8'.

        Todo:
            Add validation to assert that cells have correct node ordering for the
            specified element type (important for debugging).
        """
        # TODO (Very important for debugging purpose!): Assert that cells must have correct orders
        self.points: onp.ndarray = points
        self.cells: onp.ndarray = cells
        self.ele_type: str = ele_type

    def count_selected_faces(self, location_fn: Callable[[onp.ndarray], bool]) -> int:
        """Given location functions, compute the count of faces that satisfy the location function.

        Useful for setting up distributed load conditions.

        Args:
            location_fn (Callable): A function that inputs a point and returns a boolean
                value describing whether the boundary condition should be applied.

        Returns:
            int: The count of faces that satisfy the location function.
        """
        _, _, _, _, face_inds = get_face_shape_vals_and_grads(self.ele_type)
        cell_points = onp.take(self.points, self.cells, axis=0)
        cell_face_points = onp.take(cell_points, face_inds, axis=1)

        vmap_location_fn = jax.vmap(location_fn)

        def on_boundary(cell_points):
            boundary_flag = vmap_location_fn(cell_points)
            return onp.all(boundary_flag)

        vvmap_on_boundary = jax.vmap(jax.vmap(on_boundary))
        boundary_flags = vvmap_on_boundary(cell_face_points)
        boundary_inds = onp.argwhere(boundary_flags)
        return boundary_inds.shape[0]


def get_meshio_cell_type(ele_type: str) -> str:
    """Convert element type identifier to meshio cell type string.

    This function maps internal element type identifiers used in the finite element
    framework to the corresponding cell type strings used by the meshio library.

    Args:
        ele_type (str): Internal element type identifier. Supported types:
            - 'TET4': 4-node tetrahedral element
            - 'TET10': 10-node tetrahedral element
            - 'HEX8': 8-node hexahedral element
            - 'HEX20': 20-node hexahedral element
            - 'HEX27': 27-node hexahedral element
            - 'TRI3': 3-node triangular element
            - 'TRI6': 6-node triangular element
            - 'QUAD4': 4-node quadrilateral element
            - 'QUAD8': 8-node quadrilateral element

    Returns:
        str: Corresponding meshio cell type string.

    Raises:
        NotImplementedError: If the element type is not supported.

    Reference:
        https://github.com/nschloe/meshio/blob/9dc6b0b05c9606cad73ef11b8b7785dd9b9ea325/src/meshio/xdmf/common.py#L36
    """
    if ele_type == "TET4":
        cell_type = "tetra"
    elif ele_type == "TET10":
        cell_type = "tetra10"
    elif ele_type == "HEX8":
        cell_type = "hexahedron"
    elif ele_type == "HEX27":
        cell_type = "hexahedron27"
    elif ele_type == "HEX20":
        cell_type = "hexahedron20"
    elif ele_type == "TRI3":
        cell_type = "triangle"
    elif ele_type == "TRI6":
        cell_type = "triangle6"
    elif ele_type == "QUAD4":
        cell_type = "quad"
    elif ele_type == "QUAD8":
        cell_type = "quad8"
    else:
        raise NotImplementedError
    return cell_type


def box_mesh(Nx: int, Ny: int, Nz: int, domain_x: float, domain_y: float, domain_z: float, ele_type: str = "HEX8") -> 'Mesh':
    """Generate a structured 3D rectangular mesh using hexahedral elements.

    Creates a regular grid of hexahedral elements within a rectangular domain.
    The mesh is structured with uniform element spacing in each coordinate direction.

    Args:
        Nx (int): Number of elements in the x-direction.
        Ny (int): Number of elements in the y-direction.
        Nz (int): Number of elements in the z-direction.
        domain_x (float): Domain extent in the x-direction.
        domain_y (float): Domain extent in the y-direction.
        domain_z (float): Domain extent in the z-direction.
        ele_type (str, optional): Element type for the mesh. Defaults to "HEX8".
            Currently supports hexahedral element types.

    Returns:
        Mesh: A Mesh object containing the generated structured mesh with:
            - points: Node coordinates with shape ((Nx+1)*(Ny+1)*(Nz+1), 3)
            - cells: Element connectivity with shape (Nx*Ny*Nz, 8)
            - ele_type: Element type identifier

    Note:
        The mesh domain spans from (0,0,0) to (domain_x, domain_y, domain_z).
        Node numbering follows a structured grid pattern with i-j-k indexing.
        Element connectivity follows the standard hexahedral node ordering.

    Example:
        >>> mesh = box_mesh(10, 10, 10, 1.0, 1.0, 1.0)
        >>> print(f"Generated mesh with {mesh.points.shape[0]} nodes and {mesh.cells.shape[0]} elements")
    """
    dim = 3
    x = onp.linspace(0, domain_x, Nx + 1)
    y = onp.linspace(0, domain_y, Ny + 1)
    z = onp.linspace(0, domain_z, Nz + 1)
    xv, yv, zv = onp.meshgrid(x, y, z, indexing="ij")
    points_xyz = onp.stack((xv, yv, zv), axis=dim)
    points = points_xyz.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xyz = points_inds.reshape(Nx + 1, Ny + 1, Nz + 1)
    inds1 = points_inds_xyz[:-1, :-1, :-1]
    inds2 = points_inds_xyz[1:, :-1, :-1]
    inds3 = points_inds_xyz[1:, 1:, :-1]
    inds4 = points_inds_xyz[:-1, 1:, :-1]
    inds5 = points_inds_xyz[:-1, :-1, 1:]
    inds6 = points_inds_xyz[1:, :-1, 1:]
    inds7 = points_inds_xyz[1:, 1:, 1:]
    inds8 = points_inds_xyz[:-1, 1:, 1:]
    cells = onp.stack(
        (inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8), axis=dim
    ).reshape(-1, 8)
    meshio_mesh = meshio.Mesh(points=points, cells={"hexahedron": cells})
    cell_type = get_meshio_cell_type(ele_type)
    out_mesh = Mesh(
        meshio_mesh.points[:, :3], meshio_mesh.cells_dict[cell_type], ele_type=ele_type
    )
    return out_mesh


