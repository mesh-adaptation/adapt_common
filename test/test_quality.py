import firedrake as fd
import numpy as np
import pytest
import ufl

from adapt_common.quality import QualityMeasure
from adapt_common.reduction import function_data_sum


def uniform_mesh(dim, n=5, length=1, recentre=False, **kwargs):
    """Create a uniform mesh of a specified dimension and size."""
    if dim == 1:
        mesh = fd.IntervalMesh(n, length, **kwargs)
    elif dim == 2:
        mesh = fd.SquareMesh(n, n, length, **kwargs)
    elif dim == 3:
        mesh = fd.CubeMesh(n, n, n, length, **kwargs)
    else:
        raise ValueError(f"Can only adapt in 2D or 3D, not {dim}D")
    if recentre:
        coords = fd.fd.Function(mesh.coordinates)
        coords.interpolate(2 * (coords - ufl.as_vector([0.5 * length] * dim)))
        return fd.Mesh(coords)
    return mesh


@pytest.fixture(params=[2, 3])
def dim(request):
    return request.param


def quality(name, mesh, **kwargs):
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
    np.random.seed(0)
    mesh = uniform_mesh(dim, 4)
    mesh.coordinates.dat.data[:] += np.random.rand(*mesh.coordinates.dat.data.shape)
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
    mesh = uniform_mesh(dim, 1)
    with pytest.raises(
        NotImplementedError,
        match=f"Quality measure '{measure}' not implemented in the {dim}D case in C++.",
    ):
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
    mesh = uniform_mesh(dim, 1)
    with pytest.raises(
        NotImplementedError,
        match=f"Quality measure '{measure}' not implemented in the {dim}D case in Python.",
    ):
        quality(measure, mesh, python=True)


def test_unrecognised_error():
    mesh = uniform_mesh(2, 1)
    with pytest.raises(ValueError, match="Quality measure 'invalid' not recognised."):
        quality("invalid", mesh, python=False)
