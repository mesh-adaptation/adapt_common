"""Unit tests for MPI reduction operators."""

import firedrake as fd
import numpy as np
import pytest
import ufl

from adapt_common.reduction import (
    function_data_max,
    function_data_min,
    function_data_sum,
)


@pytest.fixture(params=[4, 8])
def n(request):
    """Set number of mesh elements in each dimension."""
    return request.param


def uniform_simplex_mesh(dim, extent=2):
    """Create a uniform simplex Firedrake mesh for testing in 1D, 2D, or 3D."""
    return {
        1: fd.UnitIntervalMesh(extent),
        2: fd.UnitSquareMesh(extent, extent),
        3: fd.UnitCubeMesh(extent, extent, extent),
    }[dim]


def test_line_interval_mesh(n):
    """Test reduction operators over a line on an interval mesh."""
    mesh = uniform_simplex_mesh(1, n)
    f = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    f.interpolate(ufl.SpatialCoordinate(mesh)[0])
    assert np.isclose(function_data_min(f), 0.0)
    assert np.isclose(function_data_max(f), 1.0)
    assert np.isclose(function_data_sum(f), 0.5 * (n + 1))


@pytest.mark.parallel(nprocs=[2, 3])
def test_line_interval_mesh_parallel():
    """Run test_line_interval_mesh with MPI parallelism."""
    test_line_interval_mesh(12)


def test_quadratic_interval_mesh(n):
    """Test reduction operators over a quadratic on an interval mesh."""
    mesh = uniform_simplex_mesh(1, n)
    f = fd.Function(fd.FunctionSpace(mesh, "CG", 1))
    x = ufl.SpatialCoordinate(mesh)[0]
    f.interpolate((x - 0.5) ** 2 + 1)
    assert np.isclose(function_data_min(f), 1.0)
    assert np.isclose(function_data_max(f), 1.25)


@pytest.mark.parallel(nprocs=[2, 3])
def test_quadratic_interval_mesh_parallel():
    """Run test_quadratic_interval_mesh with MPI parallelism."""
    test_quadratic_interval_mesh(12)
