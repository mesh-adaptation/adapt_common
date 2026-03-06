"""Test the utilities from the mesh module."""

import firedrake as fd
import numpy as np
import pytest

from adapt_common.mesh import cell_volume, patch_volume


@pytest.fixture
def n():
    """Set number of cells in each direction."""
    return 2


@pytest.fixture(params=[1, 2, 3], ids=["1D", "2D", "3D"])
def topological_dimension(request):
    """Set the topological dimension."""
    return request.param


@pytest.fixture
def uniform_mesh(n, topological_dimension):
    """Create a uniform unit hypercube mesh with n cells in each direction."""
    return {
        1: fd.UnitIntervalMesh(n),
        2: fd.UnitSquareMesh(n, n, reorder=False),
        3: fd.UnitCubeMesh(n, n, n, reorder=False),
    }[topological_dimension]


expected_coords_1d = np.array([0.0, 0.5, 1.0])

expected_coords_2d = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.5],
        [0.5, 0.0],
        [0.5, 0.5],
        [0.0, 1.0],
        [0.5, 1.0],
        [1.0, 0.0],
        [1.0, 0.5],
        [1.0, 1.0],
    ],
)

expected_coords_3d = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [1.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 0.5, 0.5],
        [1.0, 0.0, 0.5],
        [0.5, 1.0, 0.0],
        [0.5, 1.0, 0.5],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.5],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.5],
        [0.5, 0.5, 1.0],
        [0.5, 0.0, 1.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.5, 1.0],
        [1.0, 0.5, 1.0],
        [1.0, 0.0, 1.0],
        [0.5, 1.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
)


def test_mesh_coordinates(topological_dimension, uniform_mesh):
    """Test that the mesh coordinates take expected values under reorder=False."""
    expected_coords = {
        1: expected_coords_1d,
        2: expected_coords_2d,
        3: expected_coords_3d,
    }[topological_dimension]
    np.testing.assert_almost_equal(uniform_mesh.coordinates.dat.data, expected_coords)


def test_cell_volume_uniform_mesh(uniform_mesh):
    """Test that the cell volume of a uniform mesh is indeed uniform."""
    volume = cell_volume(uniform_mesh)
    expected = 1.0 / uniform_mesh.num_cells()
    np.testing.assert_almost_equal(volume.dat.data, expected)


def test_patch_volume_uniform_mesh(topological_dimension, uniform_mesh):
    """Test that the patch volumes of a uniform mesh takes expected values."""
    patch_vol = patch_volume(uniform_mesh)
    cell_vol = 1.0 / uniform_mesh.num_cells()
    # fmt: off
    num_neighbouring_cells = {
        1: [1, 2, 1],
        2: [1, 3, 3, 6, 2, 3, 2, 3, 1],
        3: [6, 8, 12, 24, 12, 8, 8, 12, 2, 4, 12, 4, 4, 12, 2, 4, 2, 8, 12, 4, 2, 4, 8, 2, 8, 2, 6],  # noqa
    }[topological_dimension]
    # fmt: on
    expected = np.array(num_neighbouring_cells) * cell_vol
    np.testing.assert_almost_equal(patch_vol.dat.data, expected)
