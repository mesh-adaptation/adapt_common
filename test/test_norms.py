"""Unit tests for overloaded norm and errornorm functions."""

import firedrake as fd
import numpy as np
import pytest
import ufl

from adapt_common.norms import errornorm, norm

integral_scalar_norm_types = ("L1", "L2", "L4", "H1", "HCurl")
scalar_norm_types = ("l1", "l2", "linf", *integral_scalar_norm_types)


@pytest.fixture
def mesh():
    """Create a simple unit square mesh for testing."""
    return fd.UnitSquareMesh(4, 4)


@pytest.fixture
def scalar_function(mesh):
    """Create a scalar function on the mesh for testing."""
    x, y = ufl.SpatialCoordinate(mesh)
    V = fd.FunctionSpace(mesh, "CG", 1)
    return fd.Function(V).interpolate(x**2 + y)


@pytest.fixture
def vector_function(mesh):
    """Create a vector function on the mesh for testing."""
    x, y = ufl.SpatialCoordinate(mesh)
    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    return fd.Function(V).interpolate(ufl.as_vector([y * y, -x * x]))


def test_boundary_error(scalar_function):
    """Test that boundary error raises NotImplementedError under lp norm."""
    not_impl_err = "lp errors on the boundary not yet implemented."
    with pytest.raises(NotImplementedError, match=not_impl_err):
        norm(scalar_function, norm_type="l1", boundary=True)


def test_l1(scalar_function):
    """Test l1 norm computation."""
    expected = np.sum(np.abs(scalar_function.dat.data))
    got = norm(scalar_function, norm_type="l1")
    assert np.isclose(expected, got)


def test_l2(scalar_function):
    """Test l2 norm computation."""
    expected = np.sqrt(np.sum(scalar_function.dat.data**2))
    got = norm(scalar_function, norm_type="l2")
    assert np.isclose(expected, got)


def test_linf(scalar_function):
    """Test linf norm computation."""
    expected = np.max(scalar_function.dat.data)
    got = norm(scalar_function, norm_type="linf")
    assert np.isclose(expected, got)


def test_notimplemented_lp_error(scalar_function):
    """Test that lp norm raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="lp norm of order p not supported."):
        norm(scalar_function, norm_type="lp")


def test_invalid_norm_type_error(scalar_function):
    """Test that invalid norm type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown norm type 'X'."):
        norm(scalar_function, norm_type="X")


@pytest.mark.parametrize("norm_type", integral_scalar_norm_types)
def test_consistency_firedrake(scalar_function, norm_type):
    """Test consistency with Firedrake's norm implementation."""
    expected = fd.norm(scalar_function, norm_type=norm_type)
    got = norm(scalar_function, norm_type=norm_type)
    assert np.isclose(expected, got)


@pytest.mark.parametrize("norm_type", scalar_norm_types)
def test_zero_scalar(scalar_function, norm_type):
    """Test that errornorm returns zero for identical scalar functions."""
    err = errornorm(scalar_function, scalar_function, norm_type=norm_type)
    assert np.isclose(err, 0.0)


@pytest.mark.parametrize("norm_type", integral_scalar_norm_types)
def test_consistency_errornorm(scalar_function, norm_type):
    """Test consistency of errornorm with Firedrake's implementation."""
    g = fd.Function(scalar_function.function_space()).interpolate(scalar_function + 1)
    expected = fd.errornorm(scalar_function, g, norm_type=norm_type)
    got = errornorm(scalar_function, g, norm_type=norm_type)
    assert np.isclose(expected, got)


def test_zero_hdiv(vector_function):
    """Test that errornorm returns zero for identical vector functions in HDiv norm."""
    err = errornorm(vector_function, vector_function, norm_type="HDiv")
    assert np.isclose(err, 0.0)
