"""Unit tests for derivative recovery based on L2-projection."""

import firedrake as fd
import pytest

from adapt_common.recovery import recover_gradient_l2


@pytest.fixture(params=[1, 2, 3])
def dim(request):
    """Set the spatial dimension."""
    return request.param


@pytest.fixture
def mesh(dim):
    """Create a uniform simplex mesh."""
    n = 4
    return {
        1: fd.UnitIntervalMesh(n),
        2: fd.UnitSquareMesh(n, n),
        3: fd.UnitCubeMesh(n, n, n),
    }[dim]


def test_recover_gradient_p2_scalar(mesh):
    """Test gradient recovery for a P2 scalar field."""
    # Define a scalar function in P2 space
    f = fd.Function(fd.FunctionSpace(mesh, "CG", 2))
    x = fd.SpatialCoordinate(mesh)
    f.interpolate(sum([0.5 * xi**2 for xi in x]))

    # Recovery its gradient using L2 projection
    grad_f = recover_gradient_l2(f)

    # Check the function space of the recovered gradient
    grad_space = grad_f.function_space()
    assert grad_space.ufl_element().family() == "Lagrange"
    assert grad_space.ufl_element().degree() == 1
    # TODO: Check grad_space is vector-valued with correct dimension

    # Verify the accuracy of the recovered gradient
    expected = x
    assert fd.errornorm(expected, grad_f, norm_type="L2") == pytest.approx(0.0)


def test_recover_gradient_p2_vector(mesh):
    """Test gradient recovery for a P2 vector field."""
    f = fd.Function(fd.VectorFunctionSpace(mesh, "CG", 2))
    x, y = fd.SpatialCoordinate(mesh)
    f.interpolate(fd.as_vector([0.5 * xi**2 for xi in fd.SpatialCoordinate(mesh)]))

    grad_f = recover_gradient_l2(f)

    grad_space = grad_f.function_space()
    assert grad_space.ufl_element().family() == "Lagrange"
    assert grad_space.ufl_element().degree() == 1
    # TODO: Check grad_space is tensor-valued with correct dimension

    # Verify the accuracy of the recovered gradient
    expected = fd.Function(fd.TensorFunctionSpace(mesh, "CG", 1))
    expected.interpolate(
        fd.as_tensor(
            [
                [xi if i == j else 0 for j in range(mesh.geometric_dimension())]
                for i, xi in enumerate(x)
            ]
        )
    )
    assert fd.errornorm(expected, grad_f, norm_type="L2") == pytest.approx(0.0)


def test_recover_gradient_invalid_input():
    """Test that an error is raised for invalid input."""
    val_err = "If a target space is not provided then the input must be a Function."
    with pytest.raises(ValueError, match=val_err):
        recover_gradient_l2("not_a_function")


def test_recover_gradient_degree_error():
    """Test that an error is raised for degree below 2."""
    val_err = (
        "Input Function must be at least degree 2 to recover gradient in CG space."
    )
    with pytest.raises(ValueError, match=val_err):
        recover_gradient_l2(fd.Function(fd.FunctionSpace(mesh, "CG", 1)))


def test_recover_gradient_rank_error(mesh):
    """Test that an error is raised for unsupported function ranks."""
    f = fd.Function(fd.TensorFunctionSpace(mesh, "CG", 2))
    val_err = (
        "L2 projection can only be used to compute gradients of scalar or vector"
        " Functions, not Functions of rank 2."
    )
    with pytest.raises(ValueError, match=val_err):
        recover_gradient_l2(f)
