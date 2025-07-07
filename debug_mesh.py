#!/usr/bin/env python3
"""Debug mesh creation."""

from fealax.mesh import box_mesh

# Create a simple mesh
mesh = box_mesh(2, 2, 2, 1.0, 1.0, 1.0, ele_type="HEX8")

print(f"Mesh type: {type(mesh)}")
print(f"Mesh attributes: {dir(mesh)}")
if hasattr(mesh, 'points'):
    print(f"Points shape: {mesh.points.shape}")
else:
    print("No 'points' attribute found")