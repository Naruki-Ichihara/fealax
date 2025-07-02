"""
Tests for fealax.basis module.

This module tests the finite element basis function utilities including
element configuration, shape functions, and face quadrature computations.
"""
import pytest
import numpy as np
import basix
from unittest.mock import patch

from fealax.basis import (
    get_elements,
    reorder_inds,
    get_shape_vals_and_grads,
    get_face_shape_vals_and_grads
)

# Mark integration tests that require environment validation to pass first
pytestmark = pytest.mark.integration


class TestGetElements:
    """Test the get_elements function for various element types."""
    
    @pytest.mark.parametrize("ele_type,expected_degree,expected_gauss_order", [
        ("HEX8", 1, 2),
        ("HEX20", 2, 2),
        ("TET4", 1, 0),
        ("TET10", 2, 2),
        ("QUAD4", 1, 2),
        ("QUAD8", 2, 2),
        ("TRI3", 1, 0),
        ("TRI6", 2, 2),
    ])
    def test_supported_element_types(self, ele_type, expected_degree, expected_gauss_order):
        """Test that all supported element types return correct configuration."""
        element_family, basix_ele, basix_face_ele, orders = get_elements(ele_type)
        gauss_order, degree, re_order = orders
        
        # Verify basic properties
        assert isinstance(element_family, basix.ElementFamily)
        assert isinstance(basix_ele, basix.CellType)
        assert isinstance(basix_face_ele, basix.CellType)
        assert degree == expected_degree
        assert gauss_order == expected_gauss_order
        assert isinstance(re_order, list)
        assert len(re_order) > 0

    def test_hex8_configuration(self):
        """Test specific configuration for HEX8 element."""
        element_family, basix_ele, basix_face_ele, orders = get_elements("HEX8")
        gauss_order, degree, re_order = orders
        
        assert element_family == basix.ElementFamily.P
        assert basix_ele == basix.CellType.hexahedron
        assert basix_face_ele == basix.CellType.quadrilateral
        assert degree == 1
        assert gauss_order == 2
        assert re_order == [0, 1, 3, 2, 4, 5, 7, 6]

    def test_hex20_serendipity_family(self):
        """Test that HEX20 uses serendipity element family."""
        element_family, _, _, _ = get_elements("HEX20")
        assert element_family == basix.ElementFamily.serendipity

    def test_quad8_serendipity_family(self):
        """Test that QUAD8 uses serendipity element family."""
        element_family, _, _, _ = get_elements("QUAD8")
        assert element_family == basix.ElementFamily.serendipity

    def test_tet4_configuration(self):
        """Test specific configuration for TET4 element."""
        element_family, basix_ele, basix_face_ele, orders = get_elements("TET4")
        gauss_order, degree, re_order = orders
        
        assert element_family == basix.ElementFamily.P
        assert basix_ele == basix.CellType.tetrahedron
        assert basix_face_ele == basix.CellType.triangle
        assert degree == 1
        assert gauss_order == 0
        assert re_order == [0, 1, 2, 3]

    def test_tri3_configuration(self):
        """Test specific configuration for TRI3 element."""
        element_family, basix_ele, basix_face_ele, orders = get_elements("TRI3")
        gauss_order, degree, re_order = orders
        
        assert element_family == basix.ElementFamily.P
        assert basix_ele == basix.CellType.triangle
        assert basix_face_ele == basix.CellType.interval
        assert degree == 1
        assert gauss_order == 0
        assert re_order == [0, 1, 2]

    def test_unsupported_element_type(self):
        """Test that unsupported element types raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            get_elements("UNSUPPORTED_ELEMENT")

    def test_reorder_arrays_have_correct_lengths(self):
        """Test that reorder arrays have expected lengths for each element type."""
        expected_lengths = {
            "HEX8": 8, "HEX20": 20, "TET4": 4, "TET10": 10,
            "QUAD4": 4, "QUAD8": 8, "TRI3": 3, "TRI6": 6
        }
        
        for ele_type, expected_length in expected_lengths.items():
            _, _, _, orders = get_elements(ele_type)
            _, _, re_order = orders
            assert len(re_order) == expected_length, f"Wrong reorder length for {ele_type}"


class TestReorderInds:
    """Test the reorder_inds function."""
    
    def test_simple_reordering(self):
        """Test basic index reordering functionality."""
        inds = np.array([0, 1, 2, 3])
        re_order = np.array([0, 1, 3, 2])
        result = reorder_inds(inds, re_order)
        
        expected = np.array([0, 1, 3, 2])
        np.testing.assert_array_equal(result, expected)

    def test_2d_array_reordering(self):
        """Test reordering with 2D input arrays."""
        inds = np.array([[0, 1], [2, 3]])
        re_order = np.array([0, 1, 3, 2])
        result = reorder_inds(inds, re_order)
        
        expected = np.array([[0, 1], [3, 2]])
        np.testing.assert_array_equal(result, expected)

    def test_preserve_input_shape(self):
        """Test that output shape matches input shape."""
        shapes_to_test = [(4,), (2, 2), (2, 1, 2)]
        re_order = np.array([0, 1, 3, 2])
        
        for shape in shapes_to_test:
            inds = np.arange(4).reshape(shape)
            result = reorder_inds(inds, re_order)
            assert result.shape == shape

    def test_hex8_reordering(self):
        """Test reordering specifically for HEX8 element."""
        _, _, _, orders = get_elements("HEX8")
        _, _, re_order = orders
        re_order = np.array(re_order)
        
        # Original ordering
        inds = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        result = reorder_inds(inds, re_order)
        
        # Should reorder according to HEX8 convention
        expected = np.array([0, 1, 3, 2, 4, 5, 7, 6])
        np.testing.assert_array_equal(result, expected)


class TestGetShapeValsAndGrads:
    """Test the get_shape_vals_and_grads function."""
    
    @pytest.mark.parametrize("ele_type", [
        "HEX8", "HEX20", "TET4", "TET10", "QUAD4", "QUAD8", "TRI3", "TRI6"
    ])
    def test_all_element_types(self, ele_type):
        """Test that all element types return valid shape functions."""
        shape_values, shape_grads_ref, weights = get_shape_vals_and_grads(ele_type)
        
        # Check return types
        assert isinstance(shape_values, np.ndarray)
        assert isinstance(shape_grads_ref, np.ndarray)
        assert isinstance(weights, np.ndarray)
        
        # Check shapes are consistent
        num_quad_points = shape_values.shape[0]
        num_nodes = shape_values.shape[1]
        
        assert shape_grads_ref.shape[0] == num_quad_points
        assert shape_grads_ref.shape[1] == num_nodes
        assert weights.shape[0] == num_quad_points
        
        # Check that weights are positive
        assert np.all(weights > 0)

    def test_hex8_specific_properties(self):
        """Test specific properties of HEX8 shape functions."""
        shape_values, shape_grads_ref, weights = get_shape_vals_and_grads("HEX8")
        
        # HEX8 should have 8 nodes and 8 quadrature points (2x2x2)
        assert shape_values.shape == (8, 8)
        assert shape_grads_ref.shape == (8, 8, 3)  # 3D gradients
        assert weights.shape == (8,)
        
        # Partition of unity: shape functions should sum to 1 at each quad point
        row_sums = np.sum(shape_values, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(8), decimal=10)

    def test_tri3_specific_properties(self):
        """Test specific properties of TRI3 shape functions."""
        shape_values, shape_grads_ref, weights = get_shape_vals_and_grads("TRI3")
        
        # TRI3 should have 3 nodes and 1 quadrature point (gauss_order=0)
        assert shape_values.shape == (1, 3)
        assert shape_grads_ref.shape == (1, 3, 2)  # 2D gradients
        assert weights.shape == (1,)
        
        # Partition of unity
        row_sums = np.sum(shape_values, axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(1), decimal=10)

    def test_custom_gauss_order(self):
        """Test using custom Gauss quadrature order."""
        # Test with different gauss orders
        vals_default, _, weights_default = get_shape_vals_and_grads("HEX8")
        vals_custom, _, weights_custom = get_shape_vals_and_grads("HEX8", gauss_order=1)
        
        # Different gauss orders should give different numbers of quad points
        assert vals_default.shape[0] != vals_custom.shape[0] or not np.array_equal(weights_default, weights_custom)

    def test_gradients_have_correct_dimension(self):
        """Test that gradients have correct spatial dimension."""
        # 3D elements
        for ele_type in ["HEX8", "HEX20", "TET4", "TET10"]:
            _, grads, _ = get_shape_vals_and_grads(ele_type)
            assert grads.shape[2] == 3, f"3D element {ele_type} should have 3D gradients"
        
        # 2D elements  
        for ele_type in ["QUAD4", "QUAD8", "TRI3", "TRI6"]:
            _, grads, _ = get_shape_vals_and_grads(ele_type)
            assert grads.shape[2] == 2, f"2D element {ele_type} should have 2D gradients"


class TestGetFaceShapeValsAndGrads:
    """Test the get_face_shape_vals_and_grads function."""
    
    @pytest.mark.parametrize("ele_type", [
        "HEX8", "TET4", "QUAD4", "TRI3"
    ])
    def test_basic_functionality(self, ele_type):
        """Test basic functionality for various element types."""
        face_vals, face_grads, face_weights, face_normals, face_inds = get_face_shape_vals_and_grads(ele_type)
        
        # Check return types
        assert isinstance(face_vals, np.ndarray)
        assert isinstance(face_grads, np.ndarray)
        assert isinstance(face_weights, np.ndarray)
        assert isinstance(face_normals, np.ndarray)
        assert isinstance(face_inds, np.ndarray)
        
        # Check basic shape consistency
        num_faces = face_vals.shape[0]
        num_face_quads = face_vals.shape[1]
        num_nodes = face_vals.shape[2]
        
        assert face_grads.shape[0] == num_faces
        assert face_grads.shape[1] == num_face_quads
        assert face_grads.shape[2] == num_nodes
        assert face_weights.shape == (num_faces, num_face_quads)
        assert face_normals.shape[0] == num_faces
        assert face_inds.shape[0] == num_faces

    def test_hex8_face_properties(self):
        """Test specific properties for HEX8 faces."""
        face_vals, face_grads, face_weights, face_normals, face_inds = get_face_shape_vals_and_grads("HEX8")
        
        # HEX8 has 6 faces, each face should have quadrature points
        assert face_vals.shape[0] == 6  # 6 faces
        assert face_normals.shape == (6, 3)  # 6 faces, 3D normals
        assert face_inds.shape[0] == 6  # 6 faces
        
        # Each face should have 4 quadrature points (2x2 for gauss_order=2)
        assert face_vals.shape[1] == 4
        
        # All weights should be positive
        assert np.all(face_weights > 0)

    def test_tet4_face_properties(self):
        """Test specific properties for TET4 faces."""
        face_vals, face_grads, face_weights, face_normals, face_inds = get_face_shape_vals_and_grads("TET4")
        
        # TET4 has 4 faces
        assert face_vals.shape[0] == 4
        assert face_normals.shape == (4, 3)  # 4 faces, 3D normals
        assert face_inds.shape[0] == 4
        
        # All weights should be positive
        assert np.all(face_weights > 0)

    def test_tri3_face_properties(self):
        """Test specific properties for TRI3 faces (edges in 2D)."""
        face_vals, face_grads, face_weights, face_normals, face_inds = get_face_shape_vals_and_grads("TRI3")
        
        # TRI3 has 3 edges (faces in 2D)
        assert face_vals.shape[0] == 3
        assert face_normals.shape == (3, 2)  # 2D normals
        assert face_inds.shape[0] == 3
        
        # All weights should be positive
        assert np.all(face_weights > 0)

    def test_face_normals_are_unit_vectors(self):
        """Test that face normals are unit vectors."""
        for ele_type in ["HEX8", "TET4", "TRI3"]:
            _, _, _, face_normals, _ = get_face_shape_vals_and_grads(ele_type)
            
            # Compute norms of normal vectors
            norms = np.linalg.norm(face_normals, axis=1)
            
            # All normals should be unit vectors (norm = 1)
            np.testing.assert_array_almost_equal(norms, np.ones(len(norms)), decimal=10)

    def test_partition_of_unity_on_faces(self):
        """Test partition of unity property for face shape functions."""
        for ele_type in ["HEX8", "TET4", "TRI3"]:
            face_vals, _, _, _, _ = get_face_shape_vals_and_grads(ele_type)
            
            # Sum of shape functions should be 1 at each face quadrature point
            for face_idx in range(face_vals.shape[0]):
                face_shape_sums = np.sum(face_vals[face_idx], axis=1)
                np.testing.assert_array_almost_equal(
                    face_shape_sums, 
                    np.ones(face_vals.shape[1]), 
                    decimal=10,
                    err_msg=f"Partition of unity failed for {ele_type} face {face_idx}"
                )

    def test_custom_gauss_order_faces(self):
        """Test using custom Gauss order for face quadrature."""
        # Test with different gauss orders
        face_vals_default, _, _, _, _ = get_face_shape_vals_and_grads("HEX8")
        face_vals_custom, _, _, _, _ = get_face_shape_vals_and_grads("HEX8", gauss_order=1)
        
        # Different gauss orders should potentially give different numbers of face quad points
        # (though not always, depending on the basix implementation)
        assert face_vals_default.shape[0] == face_vals_custom.shape[0]  # Same number of faces


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_element_consistency(self):
        """Test that element configuration is consistent across functions."""
        for ele_type in ["HEX8", "TET4", "TRI3"]:
            # Get element configuration
            element_family, basix_ele, basix_face_ele, orders = get_elements(ele_type)
            gauss_order, degree, re_order = orders
            
            # Get shape functions
            shape_vals, shape_grads, weights = get_shape_vals_and_grads(ele_type)
            
            # Get face shape functions
            face_vals, face_grads, face_weights, face_normals, face_inds = get_face_shape_vals_and_grads(ele_type)
            
            # Number of nodes should be consistent
            num_nodes_from_shape = shape_vals.shape[1]
            num_nodes_from_face = face_vals.shape[2]
            num_nodes_from_reorder = len(re_order)
            
            assert num_nodes_from_shape == num_nodes_from_face == num_nodes_from_reorder

    def test_reorder_consistency_with_face_inds(self):
        """Test that reordering is applied consistently to face indices."""
        for ele_type in ["HEX8", "TET4"]:
            _, _, _, orders = get_elements(ele_type)
            _, _, re_order = orders
            re_order = np.array(re_order)
            
            _, _, _, _, face_inds = get_face_shape_vals_and_grads(ele_type)
            
            # Face indices should be within valid range after reordering
            assert np.all(face_inds >= 0)
            assert np.all(face_inds < len(re_order))


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])