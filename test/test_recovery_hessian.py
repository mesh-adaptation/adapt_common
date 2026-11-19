"""Unit tests for Hessian recovery based on Clement interpolation."""

import firedrake as fd
import pytest
import ufl

from adapt_common.norms import errornorm, norm
from adapt_common.recovery import recover_hessian_clement, recover_hessian_l2


@pytest.fixture(params=[1, 2, 3], ids=["1D", "2D", "3D"])
def dim(request):
    """Set the spatial dimension."""
    return request.param


@pytest.fixture
def mesh(dim):
    """Create a uniform simplex mesh of [-1, 1]^dim centred on the origin."""
    n = 20
    length = 2.0
    base = {
        1: fd.IntervalMesh(n, length),
        2: fd.SquareMesh(n, n, length),
        3: fd.CubeMesh(n, n, n, length),
    }[dim]
    coords = fd.Function(base.coordinates)
    coords.interpolate(2 * (coords - fd.as_vector([0.5 * length] * dim)))
    return fd.Mesh(coords)


def bowl(mesh):
    """Quadratic bowl sensor function."""
    return 0.5 * sum([xi**2 for xi in fd.SpatialCoordinate(mesh)])


def relative_error(approx, mesh, ignore_boundary=False):
    """Relative L2 error of the recovered Hessian against the identity."""
    d = mesh.topological_dimension
    identity = fd.Function(approx.function_space()).interpolate(ufl.Identity(d))
    cond = fd.Constant(1.0)
    if ignore_boundary:
        x = fd.SpatialCoordinate(mesh)
        cond = ufl.And(x[0] > -0.8, x[0] < 0.8)
        for i in range(1, d):
            cond = ufl.And(cond, ufl.And(x[i] > -0.8, x[i] < 0.8))
        cond = ufl.conditional(cond, 1, 0)
    err = errornorm(identity, approx, norm_type="L2", condition=cond)
    err /= norm(identity, norm_type="L2", condition=cond)
    return err


def test_recover_hessian_clement_linear(mesh):
    """Test Clement Hessian recovery for a P1 scalar field (linear bowl)."""
    f = fd.Function(fd.FunctionSpace(mesh, "CG", 1)).interpolate(bowl(mesh))

    g, H = recover_hessian_clement(f)

    # Check the function spaces of the recovered gradient and Hessian
    g_element = g.function_space().ufl_element()
    assert g_element.family() == "Lagrange"
    assert g_element.degree() == 1
    assert g_element.num_sub_elements == mesh.geometric_dimension
    H_element = H.function_space().ufl_element()
    assert H_element.family() == "Lagrange"
    assert H_element.degree() == 1
    assert H_element.num_sub_elements == mesh.geometric_dimension**2

    # Verify the accuracy of the recovered Hessian away from the boundary
    assert relative_error(H, mesh, ignore_boundary=True) < 1.0e-05


def test_recover_hessian_clement_function_error(mesh):
    """Test that an error is raised for non-Function input."""
    val_err = (
        "Clement interpolation can only be used to compute gradients of continuous"
        " Lagrange Functions of degree 1."
    )
    with pytest.raises(ValueError, match=val_err):
        recover_hessian_clement(bowl(mesh))


def test_recover_hessian_clement_family_error(mesh):
    """Test that an error is raised for non-Lagrange input."""
    f = fd.Function(fd.VectorFunctionSpace(mesh, "DG", 1)).assign(1.0)
    val_err = (
        "Clement interpolation can only be used to compute gradients of continuous"
        " Lagrange Functions of degree 1."
    )
    with pytest.raises(ValueError, match=val_err):
        recover_hessian_clement(f)


def test_recover_hessian_clement_degree_error(mesh):
    """Test that an error is raised for degree 0 input."""
    f = fd.Function(fd.FunctionSpace(mesh, "CG", 2)).assign(1.0)
    val_err = (
        "Clement interpolation can only be used to compute gradients of continuous"
        " Lagrange Functions of degree 1."
    )
    with pytest.raises(ValueError, match=val_err):
        recover_hessian_clement(f)


@pytest.mark.parametrize("mixed", [False, True], ids=["L2", "mixed_L2"])
def test_recover_hessian_l2_quadratic(mesh, mixed):
    """Test L2 Hessian recovery for a quadratic field."""
    # TODO: parallel version (#33)
    f = fd.Function(fd.FunctionSpace(mesh, "CG", 2)).interpolate(bowl(mesh))
    target_space = fd.TensorFunctionSpace(mesh, "CG", 1)
    g, H = recover_hessian_l2(f, target_space=target_space, mixed=mixed)

    # Check the function spaces of the recovered gradient and Hessian
    g_element = g.function_space().ufl_element()
    assert g_element.family() == "Lagrange"
    assert g_element.degree() == 1
    assert g_element.num_sub_elements == mesh.geometric_dimension
    H_element = H.function_space().ufl_element()
    assert H_element.family() == "Lagrange"
    assert H_element.degree() == 1
    assert H_element.num_sub_elements == mesh.geometric_dimension**2

    # Verify the accuracy of the recovered Hessian
    assert relative_error(H, mesh) < 1.0e-07


def test_recover_hessian_l2_rank_error(mesh):
    """Test that an error is raised for a non-rank-2 target space."""
    f = fd.Function(fd.FunctionSpace(mesh, "CG", 2)).interpolate(bowl(mesh))
    val_err = "Target space must be rank 2, not 1."
    with pytest.raises(ValueError, match=val_err):
        recover_hessian_l2(f, target_space=fd.VectorFunctionSpace(mesh, "CG", 1))
