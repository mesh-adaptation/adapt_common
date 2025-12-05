"""Unit tests for mesh quality measures."""

import firedrake as fd
import numpy as np
import pytest
import ufl

from adapt_common.quality import QualityMeasure
from adapt_common.reduction import function_data_sum


def uniform_mesh(dim, n=5, length=1, recentre=False, **kwargs):
    """Create a uniform mesh of a specified dimension and size."""
    try:
        mesh = {
            1: fd.IntervalMesh,
            2: fd.SquareMesh,
            3: fd.CubeMesh,
        }[dim](*(dim * [n]), length, **kwargs)
    except KeyError as ke:
        val_err = f"Can only adapt in 2D or 3D, not {dim}D"
        raise ValueError(val_err) from ke
    if recentre:
        coords = fd.fd.Function(mesh.coordinates)
        coords.interpolate(2 * (coords - ufl.as_vector([0.5 * length] * dim)))
        return fd.Mesh(coords)
    return mesh


@pytest.fixture(params=[2, 3])
def dim(request):
    """Return test dimension."""
    return request.param


def quality(name, mesh, **kwargs):
    """Compute a quality measure on a mesh."""
    if name == "metric":
        P1_ten = fd.TensorFunctionSpace(mesh, "CG", 1)
        M = fd.Function(P1_ten).interpolate(ufl.Identity(mesh.topological_dimension))
        kwargs["metric"] = M
    return QualityMeasure(mesh, **kwargs)(name)


@pytest.mark.parametrize(
    "measure, expected",
    [
        ("min_angle", np.pi / 4),
        ("area", 0.005),
        ("eskew", 1.070796),
        ("aspect_ratio", 1.207107),
        ("scaled_jacobian", 0.707107),
        ("skewness", 0.463648),
        ("metric", 6.928203),
    ],
)
def test_uniform_quality_2d(measure, expected):
    """Test quality measures on a uniform 2D mesh."""
    mesh = uniform_mesh(2, 10)
    q = quality(measure, mesh)
    truth = fd.Function(q.function_space()).assign(expected)
    assert fd.errornorm(truth, q) == pytest.approx(0.0, abs=1e-6)
    if measure == "area":
        s = function_data_sum(q)
        assert s == pytest.approx(1.0)


@pytest.mark.parametrize(
    "measure, expected",
    [
        ("min_angle", 0.61547971),
        ("volume", 0.00260417),
        ("eskew", 0.41226017),
        ("aspect_ratio", 1.39384685),
        ("scaled_jacobian", 0.40824829),
        ("metric", 1.25),
    ],
)
def test_uniform_quality_3d(measure, expected):
    """Test quality measures on a uniform 3D mesh."""
    mesh = uniform_mesh(3, 4)
    q = quality(measure, mesh)
    truth = fd.Function(q.function_space()).assign(expected)
    assert fd.errornorm(truth, q) == pytest.approx(0.0)
    if measure == "volume":
        s = function_data_sum(q)
        assert s == pytest.approx(1.0)


@pytest.mark.parametrize(
    "measure, dim",
    [
        ("area", 2),
        ("aspect_ratio", 2),
        ("scaled_jacobian", 2),
        ("volume", 3),
    ],
)
def test_consistency(measure, dim):
    """Test consistency between C++ and Python implementations."""
    rng = np.random.default_rng()
    mesh = uniform_mesh(dim, 4)
    mesh.coordinates.dat.data[:] += rng.random(*mesh.coordinates.dat.data.shape)
    quality_cpp = quality(measure, mesh, python=False)
    quality_py = quality(measure, mesh, python=True)
    assert fd.errornorm(quality_cpp, quality_py) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "measure, dim",
    [
        ("facet_area", 2),
        ("facet_area", 3),
        ("skewness", 3),
    ],
)
def test_cxx_notimplemented(measure, dim):
    """Test NotImplementedError for missing C++ implementations."""
    mesh = uniform_mesh(dim, 1)
    not_impl_err = (f"Quality measure '{measure}' {dim}D case not implemented in C++.",)
    with pytest.raises(NotImplementedError, match=not_impl_err):
        quality(measure, mesh, python=False)


@pytest.mark.parametrize(
    "measure, dim",
    [
        ("min_angle", 2),
        ("min_angle", 3),
        ("aspect_ratio", 3),
        ("eskew", 2),
        ("eskew", 3),
        ("skewness", 2),
        ("skewness", 3),
        ("scaled_jacobian", 3),
        ("metric", 2),
        ("metric", 3),
    ],
)
def test_python_notimplemented(measure, dim):
    """Test NotImplementedError for missing Python implementations."""
    mesh = uniform_mesh(dim, 1)
    not_impl_err = (
        f"Quality measure '{measure}' not implemented in the {dim}D case in Python.",
    )
    with pytest.raises(NotImplementedError, match=not_impl_err):
        quality(measure, mesh, python=True)


def test_unrecognised_error():
    """Test ValueError for unrecognised quality measure."""
    mesh = uniform_mesh(2, 1)
    val_err = "Quality measure 'invalid' not recognised."
    with pytest.raises(ValueError, match=val_err):
        quality("invalid", mesh, python=False)
