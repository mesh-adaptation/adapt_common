"""Test the Clement interpolation functionality from the transfer module."""

import firedrake as fd
import numpy as np
import pytest
import ufl

from adapt_common.transfer import clement_interpolant
from adapt_common.utility import cofunction2function, get_function_space


@pytest.fixture
def n():
    """Set number of elements in each direction."""
    return 5


@pytest.fixture(params=[1, 2, 3], ids=["1D", "2D", "3D"])
def topological_dimension(request):
    """Set the topological dimension."""
    return request.param


@pytest.fixture(
    params=[(), (1,), (2,), (3,), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)],
    ids=[
        "scalar",
        "1vector",
        "2vector",
        "3vector",
        "1x1tensor",
        "1x2tensor",
        "2x2tensor",
        "2x3tensor",
        "3x3tensor",
    ],
)
def shape(request):
    """Set the tensor shape."""
    return request.param


@pytest.fixture
def uniform_mesh(n, topological_dimension):
    """Create a uniform unit simplex mesh with n elements in each direction."""
    return {
        1: fd.UnitIntervalMesh(n),
        2: fd.UnitSquareMesh(n, n),
        3: fd.UnitCubeMesh(n, n, n),
    }[topological_dimension]


@pytest.fixture
def P0(uniform_mesh, shape):
    """Create a P0 function space of a given shape on the uniform mesh."""
    return get_function_space(uniform_mesh, "DG", 0, shape)


@pytest.fixture
def P1(uniform_mesh, shape):
    """Create a P1 function space of a given shape on the uniform mesh."""
    return get_function_space(uniform_mesh, "CG", 1, shape)


@pytest.fixture
def expression(uniform_mesh, topological_dimension, shape, P1):
    """Expression function for testing based on tensor shape."""
    x = fd.SpatialCoordinate(uniform_mesh)
    if len(shape) == 0:
        return sum(x)
    elif len(shape) == 1:
        dim = shape[0]
        return ufl.as_vector(
            x if dim == topological_dimension else [x[0] for _ in range(dim)]
        )
    rows = [
        fd.Constant(tuple(range(i + 1, i + 1 + topological_dimension)))
        for i in range(P1.block_size)
    ]
    return ufl.as_tensor(np.reshape([ufl.dot(row, x) for row in rows], shape))


def test_source_type_error():
    """Test that providing an invalid source type raises a TypeError."""
    type_err = (
        "Expected Cofunction or Function, got '<class 'firedrake.constant.Constant'>'."
    )
    with pytest.raises(TypeError, match=type_err):
        clement_interpolant(fd.Constant(0.0))


def test_source_space_error(uniform_mesh):
    """Test that providing a non-P0 source function raises a ValueError."""
    shape = ()
    fs = get_function_space(uniform_mesh, "CG", 1, shape)
    val_err = "Source function provided must be from a P0 space."
    with pytest.raises(ValueError, match=val_err):
        clement_interpolant(fd.Function(fs))


def test_target_function_space_error(uniform_mesh):
    """Test that providing a non-P1 target space raises a ValueError."""
    shape = ()
    fs = get_function_space(uniform_mesh, "DG", 0, shape)
    val_err = "Target space provided must be P1."
    with pytest.raises(ValueError, match=val_err):
        clement_interpolant(fd.Function(fs), target=fs)


def test_cofunction_dual_target_function_space(P0, P1):
    """Test that Clement interpolation works with dual spaces."""
    source = fd.Cofunction(P0.dual())
    source.dat.data[:] = 1.0
    target = clement_interpolant(source, target=P1.dual())
    assert isinstance(target, fd.Cofunction)
    target_function = cofunction2function(target)

    # Account for the fact that the Clement interpolant breaks down at domain boundaries
    expected = fd.Function(P1).assign(1.0)
    fd.DirichletBC(P1, expected, "on_boundary").apply(target_function)

    np.testing.assert_almost_equal(target_function.dat.data, expected.dat.data)


def test_cofunction_primal_target_function_space(P0, P1):
    """Test that Clement interpolation works with primal target spaces."""
    source = fd.Cofunction(P0.dual())
    source.dat.data[:] = 1.0
    target = clement_interpolant(source, target=P1)
    assert isinstance(target, fd.Function)

    # Account for the fact that the Clement interpolant breaks down at domain boundaries
    expected = fd.Function(P1).assign(1.0)
    fd.DirichletBC(P1, expected, "on_boundary").apply(target)

    np.testing.assert_almost_equal(target.dat.data, expected.dat.data)


def test_volume_average(P0, P1, expression):
    """Test Clement interpolation in the interior of a 2D domain."""
    exact = expression
    source_space = P0
    source = fd.Function(source_space).project(exact)
    target = clement_interpolant(source)
    target_function_space = P1
    expected = fd.Function(target_function_space).interpolate(exact)

    # Account for the fact that the Clement interpolant breaks down at domain boundaries
    bc = fd.DirichletBC(target_function_space, expected, "on_boundary")
    bc.apply(target)

    np.testing.assert_almost_equal(target.dat.data, expected.dat.data)
