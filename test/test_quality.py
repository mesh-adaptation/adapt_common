"""Tests for the QualityMeasure class."""

import firedrake as fd
import numpy as np
import pytest

from adapt_common.norms import errornorm
from adapt_common.quality import QualityMeasure
from adapt_common.reduction import function_data_sum


@pytest.fixture(params=[2, 3])
def dim(request):
    """Fixture for the mesh dimension."""
    return request.param


def uniform_mesh(dim, n):
    """Create a uniform mesh of the given dimension and number of subdivisions."""
    if dim == 2:
        return fd.UnitSquareMesh(n, n)
    return fd.UnitCubeMesh(n, n, n)


def test_facet_areas(dim):
    """Check that the computation of facet areas sums to the expected value for a
    uniform (isotropic) simplex mesh.
    """
    mesh = uniform_mesh(dim, 1)
    qm = QualityMeasure(mesh)
    f = qm("facet_area")
    expected = 5.414214 if dim == 2 else 10.242641
    assert np.isclose(sum(f.dat.data), expected)


@pytest.mark.parametrize(
    "measure, expected",
    [
        ("area", 0.005),
        ("aspect_ratio", 1.207107),
        ("scaled_jacobian", 0.707107),
    ],
)
def test_uniform_quality_2d(measure, expected):
    """Check quality measures on a uniform 2D mesh against known values."""
    mesh = uniform_mesh(2, 10)
    q = QualityMeasure(mesh)(measure)
    truth = fd.Function(q.function_space()).assign(expected)
    assert np.isclose(errornorm(truth, q), 0.0, atol=1e-6)
    if measure == "area":
        assert np.isclose(function_data_sum(q), 1.0)


@pytest.mark.parametrize("measure, expected", [("volume", 0.00260417)])
def test_uniform_quality_3d(measure, expected):
    """Check the volume quality measure on a uniform 3D mesh."""
    mesh = uniform_mesh(3, 4)
    q = QualityMeasure(mesh)(measure)
    truth = fd.Function(q.function_space()).assign(expected)
    assert np.isclose(errornorm(truth, q), 0.0)
    if measure == "volume":
        assert np.isclose(function_data_sum(q), 1.0)


@pytest.mark.parametrize("measure, dim", [("aspect_ratio", 3), ("scaled_jacobian", 3)])
def test_notimplemented(measure, dim):
    """Check that 3D aspect ratio and scaled Jacobian raise NotImplementedError."""
    mesh = uniform_mesh(dim, 1)
    msg = f"Quality measure '{measure}' not implemented in the {dim}D case."
    with pytest.raises(NotImplementedError, match=msg):
        QualityMeasure(mesh)(measure)


def test_unrecognised_error():
    """Check that an unrecognised quality measure raises a ValueError."""
    mesh = uniform_mesh(2, 1)
    msg = "Quality measure 'invalid' not recognised."
    with pytest.raises(ValueError, match=msg):
        QualityMeasure(mesh)("invalid")
