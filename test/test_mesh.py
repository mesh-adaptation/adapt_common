"""Test the utilities from the mesh module."""

import firedrake as fd
import numpy as np
import pytest

from adapt_common.mesh import cell_volume


@pytest.fixture
def n():
    """Set number of elements in each direction."""
    return 5


@pytest.fixture(params=[1, 2, 3], ids=["1D", "2D", "3D"])
def topological_dimension(request):
    """Set the topological dimension."""
    return request.param


@pytest.fixture
def uniform_mesh(n, topological_dimension):
    """Create a uniform unit simplex mesh with n elements in each direction."""
    return {
        1: fd.UnitIntervalMesh(n),
        2: fd.UnitSquareMesh(n, n),
        3: fd.UnitCubeMesh(n, n, n),
    }[topological_dimension]


def test_cell_volume_uniform_mesh(uniform_mesh):
    """Test that the cell volume of a uniform mesh is indeed uniform."""
    volume = cell_volume(uniform_mesh)
    expected = 1.0 / uniform_mesh.num_cells()
    np.testing.assert_almost_equal(volume.dat.data, expected)
