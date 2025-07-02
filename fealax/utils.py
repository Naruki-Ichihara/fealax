import numpy as onp
from fealax.mesh import get_meshio_cell_type
import meshio
import os

def save_as_vtk(fe, sol_file, cell_infos=None, point_infos=None):
    if cell_infos is None and point_infos is None:
        raise ValueError("At least one of cell_infos or point_infos must be provided.")
    cell_type = get_meshio_cell_type(fe.ele_type)
    sol_dir = os.path.dirname(sol_file)
    os.makedirs(sol_dir, exist_ok=True)

    out_mesh = meshio.Mesh(points=fe.points, cells={cell_type: fe.cells})

    if cell_infos is not None:
        out_mesh.cell_data = {}
        for cell_info in cell_infos:
            name, data = cell_info
            assert data.shape[0] == fe.num_cells, (
                f"cell data wrong shape, got {data.shape}, expected first dim = {fe.num_cells}"
            )
            data = onp.array(data, dtype=onp.float32)
            if data.ndim == 3:
                # Tensor (num_cells, 3, 3) -> flatten to (num_cells, 9)
                data = data.reshape(fe.num_cells, -1)
            elif data.ndim == 2:
                # Vector (num_cells, n) is OK
                pass
            else:
                # Scalar (num_cells,)
                data = data.reshape(fe.num_cells, 1)
            out_mesh.cell_data[name] = [data]

    if point_infos is not None:
        for point_info in point_infos:
            name, data = point_info
            out_mesh.point_data[name] = onp.array(data, dtype=onp.float32)

    out_mesh.write(sol_file)