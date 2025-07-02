"""Finite element basis functions and shape function utilities.

This module provides utilities for creating finite element basis functions,
shape functions, and their gradients using the Basix library. It supports
various element types including hexahedral, tetrahedral, quadrilateral,
and triangular elements.
"""
import basix
import numpy as onp
from typing import Tuple, Optional, List

from fealax import logger


def get_elements(ele_type: str) -> Tuple[basix.ElementFamily, basix.CellType, basix.CellType, Tuple[int, int, List[int]]]:
    """Get element configuration for specified element type.
    
    Returns the element family, cell types, and ordering parameters
    needed to create finite element basis functions.
    
    Args:
        ele_type (str): Element type identifier. Supported types:
            - 'HEX8': 8-node hexahedral element
            - 'HEX20': 20-node hexahedral element (serendipity)
            - 'TET4': 4-node tetrahedral element
            - 'TET10': 10-node tetrahedral element
            - 'QUAD4': 4-node quadrilateral element
            - 'QUAD8': 8-node quadrilateral element (serendipity)
            - 'TRI3': 3-node triangular element
            - 'TRI6': 6-node triangular element
    
    Returns:
        tuple: A 4-tuple containing:
            - element_family (basix.ElementFamily): Basix element family
            - basix_ele (basix.CellType): Basix cell type for the element
            - basix_face_ele (basix.CellType): Basix cell type for faces
            - orders (tuple): Gauss order, degree, and reordering array
    
    Raises:
        NotImplementedError: If element type is not supported.
    
    Example:
        >>> family, ele, face_ele, orders = get_elements('HEX8')
        >>> gauss_order, degree, re_order = orders
    """
    element_family = basix.ElementFamily.P
    if ele_type == 'HEX8':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6]
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2
        degree = 1
    elif ele_type == 'HEX20':
        re_order = [0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13,
                    9, 16, 18, 19, 17, 10, 12, 15, 14]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.hexahedron
        basix_face_ele = basix.CellType.quadrilateral
        gauss_order = 2
        degree = 2
    elif ele_type == 'TET4':
        re_order = [0, 1, 2, 3]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 0
        degree = 1
    elif ele_type == 'TET10':
        re_order = [0, 1, 2, 3, 9, 6, 8, 7, 5, 4]
        basix_ele = basix.CellType.tetrahedron
        basix_face_ele = basix.CellType.triangle
        gauss_order = 2
        degree = 2
    # TODO: Check if this is correct.
    elif ele_type == 'QUAD4':
        re_order = [0, 1, 3, 2]
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 1
    elif ele_type == 'QUAD8':
        re_order = [0, 1, 3, 2, 4, 6, 7, 5]
        element_family = basix.ElementFamily.serendipity
        basix_ele = basix.CellType.quadrilateral
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 2
    elif ele_type == 'TRI3':
        re_order = [0, 1, 2]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 0
        degree = 1
    elif ele_type == 'TRI6':
        re_order = [0, 1, 2, 5, 3, 4]
        basix_ele = basix.CellType.triangle
        basix_face_ele = basix.CellType.interval
        gauss_order = 2
        degree = 2
    else:
        raise NotImplementedError
    
    orders = (gauss_order, degree, re_order)

    return element_family, basix_ele, basix_face_ele, orders


def reorder_inds(inds: onp.ndarray, re_order: List[int]) -> onp.ndarray:
    """Reorder node indices according to element-specific ordering.
    
    Converts indices from one ordering convention to another using
    a reordering array. This is needed to match different finite
    element software conventions.
    
    Args:
        inds (numpy.ndarray): Array of node indices to reorder.
        re_order (numpy.ndarray): Reordering array mapping old to new indices.
    
    Returns:
        numpy.ndarray: Reordered indices with same shape as input.
    
    Example:
        >>> inds = np.array([0, 1, 2, 3])
        >>> re_order = np.array([0, 1, 3, 2])
        >>> new_inds = reorder_inds(inds, re_order)
    """
    new_inds = []
    for ind in inds.reshape(-1):
        new_inds.append(onp.argwhere(re_order == ind))
    new_inds = onp.array(new_inds).reshape(inds.shape)
    return new_inds


def get_shape_vals_and_grads(ele_type: str, gauss_order: Optional[int] = None) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray]:
    """Compute shape function values and gradients at quadrature points.
    
    Calculates the shape function values and their gradients in the
    reference element coordinate system at Gauss quadrature points.
    
    Args:
        ele_type (str): Element type identifier (see get_elements for options).
        gauss_order (int, optional): Quadrature order. If None, uses default
            order for the element type.
    
    Returns:
        tuple: A 3-tuple containing:
            - shape_values (numpy.ndarray): Shape function values at quad points.
                Shape: (num_quad_points, num_nodes)
            - shape_grads_ref (numpy.ndarray): Shape function gradients in 
                reference coordinates. Shape: (num_quad_points, num_nodes, dim)
            - weights (numpy.ndarray): Quadrature weights.
                Shape: (num_quad_points,)
    
    Example:
        >>> vals, grads, weights = get_shape_vals_and_grads('HEX8')
        >>> print(f"Shape values: {vals.shape}")
        >>> print(f"Gradients: {grads.shape}")
    """
    element_family, basix_ele, basix_face_ele, orders = get_elements(ele_type)
    gauss_order_default, degree, re_order = orders

    if gauss_order is None:
        gauss_order = gauss_order_default

    quad_points, weights = basix.make_quadrature(basix_ele, gauss_order)
    element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, quad_points)[:, :, re_order, :]
    shape_values = vals_and_grads[0, :, :, 0]
    shape_grads_ref = onp.transpose(vals_and_grads[1:, :, :, 0],
                                    axes=(1, 2, 0))
    return shape_values, shape_grads_ref, weights


def get_face_shape_vals_and_grads(ele_type: str, gauss_order: Optional[int] = None) -> Tuple[onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray, onp.ndarray]:
    """Compute shape function values and gradients at face quadrature points.
    
    Calculates shape function values and gradients evaluated at quadrature
    points on element faces. This is essential for boundary integrals and
    inter-element continuity enforcement.
    
    Args:
        ele_type (str): Element type identifier (see get_elements for options).
        gauss_order (int, optional): Quadrature order for faces. If None, 
            uses default order for the element type.
    
    Returns:
        tuple: A 5-tuple containing:
            - face_shape_vals (numpy.ndarray): Shape function values at face 
                quad points. Shape: (num_faces, num_face_quads, num_nodes)
            - face_shape_grads_ref (numpy.ndarray): Shape function gradients 
                in reference coordinates. 
                Shape: (num_faces, num_face_quads, num_nodes, dim)
            - face_weights (numpy.ndarray): Face quadrature weights including
                Jacobian scaling. Shape: (num_faces, num_face_quads)
            - face_normals (numpy.ndarray): Outward normal vectors for each face.
                Shape: (num_faces, dim)
            - face_inds (numpy.ndarray): Node indices for each face.
                Shape: (num_faces, nodes_per_face)
    
    Notes:
        The face quadrature weights include the Jacobian determinant scaling
        to account for the mapping from reference to physical coordinates.
        
    Example:
        >>> vals, grads, weights, normals, inds = get_face_shape_vals_and_grads('HEX8')
        >>> print(f"Face shape values: {vals.shape}")
        >>> print(f"Face normals: {normals.shape}")
    """
    element_family, basix_ele, basix_face_ele, orders = get_elements(ele_type)
    gauss_order_default, degree, re_order = orders

    if gauss_order is None:
        gauss_order = gauss_order_default

    points, weights = basix.make_quadrature(basix_face_ele, gauss_order)

    map_degree = 1
    lagrange_map = basix.create_element(basix.ElementFamily.P,
                                        basix_face_ele, map_degree)
    values = lagrange_map.tabulate(0, points)[0, :, :, 0]
    vertices = basix.geometry(basix_ele)
    dim = len(vertices[0])
    facets = basix.cell.sub_entity_connectivity(basix_ele)[dim - 1]
    face_quad_points = []
    face_inds = []
    face_weights = []
    for f, facet in enumerate(facets):
        mapped_points = []
        for i in range(len(points)):
            vals = values[i]
            mapped_point = onp.sum(vertices[facet[0]] * vals[:, None], axis=0)
            mapped_points.append(mapped_point)
        face_quad_points.append(mapped_points)
        face_inds.append(facet[0])
        jacobian = basix.cell.facet_jacobians(basix_ele)[f]
        if dim == 2:
            size_jacobian = onp.linalg.norm(jacobian)
        else:
            size_jacobian = onp.linalg.norm(
                onp.cross(jacobian[:, 0], jacobian[:, 1]))
        face_weights.append(weights*size_jacobian)
    face_quad_points = onp.stack(face_quad_points)
    face_weights = onp.stack(face_weights)

    face_normals = basix.cell.facet_outward_normals(basix_ele)
    face_inds = onp.array(face_inds)
    face_inds = reorder_inds(face_inds, re_order)
    num_faces, num_face_quads, dim = face_quad_points.shape
    element = basix.create_element(element_family, basix_ele, degree)
    vals_and_grads = element.tabulate(1, face_quad_points.reshape(-1, dim))[:, :, re_order, :]
    face_shape_vals = vals_and_grads[0, :, :, 0].reshape(num_faces, num_face_quads, -1)
    face_shape_grads_ref = vals_and_grads[1:, :, :, 0].reshape(dim, num_faces, num_face_quads, -1)
    face_shape_grads_ref = onp.transpose(face_shape_grads_ref, axes=(1, 2, 3, 0))
    logger.debug(f"face_quad_points.shape = (num_faces, num_face_quads, dim) = {face_quad_points.shape}")
    return face_shape_vals, face_shape_grads_ref, face_weights, face_normals, face_inds
