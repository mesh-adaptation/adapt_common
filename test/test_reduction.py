"""Unit tests for MPI reduction operators."""

import numpy as np
import pytest
import firedrake as fd
import ufl
from adapt_common.reduction import (
    function_data_min,
    function_data_max,
    function_data_sum,
)


@pytest.fixture(params=[2, 4, 8])
def n(request):
    """Number of elements in each dimension for test meshes."""
    return request.param


def uniform_simplex_mesh(dim, extent=2):
    """Create a uniform simplex Firedrake mesh for testing in 1D, 2D, or 3D."""
    return {
        1: fd.UnitIntervalMesh(extent),
        2: fd.UnitSquareMesh(extent, extent),
        3: fd.UnitCubeMesh(extent, extent, extent),
    }[dim]


def test_interval_mesh(n):
    """Test reduction operators over an interval mesh."""
    mesh = uniform_simplex_mesh(1, n)
    f = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    f.interpolate(ufl.SpatialCoordinate(mesh)[0])
    assert np.isclose(function_data_min(f), 0.0)
    assert np.isclose(function_data_max(f), 1.0)
    assert np.isclose(function_data_sum(f), 0.5 * (n + 1))


@pytest.mark.parallel(nprocs=2)
def test_interval_mesh_np2(n):
    """Test reduction operators over an interval mesh with 2 MPI ranks."""
    test_interval_mesh(n)
