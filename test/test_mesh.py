"""Test the utilities from the mesh module."""

import firedrake as fd
import numpy as np
import pytest
import ufl

from adapt_common.mesh import boundary_patch_volume, cell_volume, patch_volume


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


expected_coords_1d = np.array([[0.0], [0.5], [1.0]])

expected_coords_2d = np.array(
    [
        [0.5, 0.0],
        [0.0, 0.5],
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 0.0],
        [1.0, 0.5],
        [0.0, 1.0],
        [0.5, 1.0],
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
    data = np.asarray(uniform_mesh.coordinates.dat.data)
    expected = np.asarray(expected_coords).reshape(data.shape)
    np.testing.assert_almost_equal(data, expected)


def test_cell_volume_uniform_mesh(uniform_mesh):
    """Test that the cell volume of a uniform mesh is indeed uniform."""
    volume = cell_volume(uniform_mesh)
    expected = 1.0 / uniform_mesh.num_cells()
    np.testing.assert_almost_equal(volume.dat.data, expected)


def test_patch_volume_uniform_mesh(uniform_mesh):
    """Test that patch volume at each vertex equals the sum of adjacent cell volumes."""
    mesh = uniform_mesh
    patch_vol = patch_volume(mesh)
    P1 = fd.FunctionSpace(mesh, "Lagrange", 1)
    # Count the number of cells adjacent to each vertex via an independent loop
    count = fd.Function(P1)
    domain = "{[i]: 0 <= i < cnt.dofs}"
    instructions = "cnt[i, 0] = cnt[i, 0] + 1"
    fd.par_loop((domain, instructions), ufl.dx(domain=mesh), {"cnt": (count, fd.RW)})
    expected = count.dat.data * (1.0 / mesh.num_cells())
    np.testing.assert_almost_equal(patch_vol.dat.data, expected)


def test_boundary_patch_volume_uniform_mesh(n, uniform_mesh):
    """Test that boundary patch volume at each vertex equals the sum of adjacent
    boundary facet areas."""
    mesh = uniform_mesh
    dim = mesh.topological_dimension
    patch_vol = boundary_patch_volume(mesh)
    P1 = fd.FunctionSpace(mesh, "Lagrange", 1)
    indicator = fd.Function(P1)
    fd.DirichletBC(P1, 1, "on_boundary").apply(indicator)
    # Each boundary facet of a uniform unit simplex mesh has area n**(1 - dim),
    # halved in 3D where facets are right triangles
    facet_area = n ** (1 - dim) / (2 if dim == 3 else 1)
    # Count boundary facets adjacent to each vertex via an independent loop
    count = fd.Function(P1)
    domain = "{[i]: 0 <= i < cnt.dofs}"
    instructions = "cnt[i, 0] = cnt[i, 0] + indicator[i, 0]"
    keys = {"indicator": (indicator, fd.READ), "cnt": (count, fd.RW)}
    fd.par_loop((domain, instructions), ufl.ds(domain=mesh), keys)
    expected = count.dat.data * facet_area
    np.testing.assert_almost_equal(patch_vol.dat.data, expected)
