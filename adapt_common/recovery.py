"""Module containing functions for recovering derivatives of Functions."""

import firedrake as fd
import ufl
from firedrake.petsc import PETSc

from adapt_common.transfer import clement_interpolant

__all__ = ["recover_gradient_l2", "recover_hessian_clement"]


@PETSc.Log.EventDecorator()
def recover_gradient(f, target_space=None, method="l2"):
    r"""Recover the gradient of a scalar or vector field using the specified method.

    :arg f: the scalar or vector field whose derivatives we seek to recover
    :type f: :class:`firedrake.function.Function`
    :kwarg target_space: the vector-valued function space to recover the gradient in
    :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`
    :kwarg method: the recovery method to use
    :type method: str
    :returns: recovered gradient
    :rtype: :class:`firedrake.function.Function`
    """
    method = method.lower()
    if method == "l2":
        return recover_gradient_l2(f, target_space=target_space)
    elif method == "clement":
        return recover_gradient_clement(f, target_space=target_space)
    elif method == "zz":
        raise NotImplementedError("Zienkiewicz-Zhu recovery not yet implemented.")
    raise NotImplementedError(f"Gradient recovery method '{method}' not recognised.")


@PETSc.Log.EventDecorator()
def recover_gradient_l2(f, target_space=None):
    r"""Recover the gradient of a scalar or vector field using :math:`L^2` projection.

    :arg f: the scalar or vector field whose derivatives we seek to recover
    :type f: :class:`firedrake.function.Function`
    :kwarg target_space: the vector-valued function space to recover the gradient in
    :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`
    :returns: recovered gradient
    :rtype: :class:`firedrake.function.Function`
    """
    if target_space is None:
        if not isinstance(f, fd.Function):
            val_err = (
                "If a target space is not provided then the input must be a Function."
            )
            raise ValueError(val_err)
        source_degree = f.ufl_element().degree()
        if source_degree <= 0:
            val_err = "Input Function must be at least degree 1."
            raise ValueError(val_err)
        target_degree = source_degree - 1
        mesh = f.function_space().mesh()
        rank = len(f.function_space().value_shape)
        family = "DG" if target_degree == 0 else "CG"
        if rank == 0:
            target_space = fd.VectorFunctionSpace(mesh, family, target_degree)
        elif rank == 1:
            target_space = fd.TensorFunctionSpace(
                mesh,
                family,
                target_degree,
                shape=(f.function_space().value_size, mesh.geometric_dimension),
            )
        else:
            val_err = (
                "L2 projection can only be used to compute gradients of scalar or"
                f" vector Functions, not Functions of rank {rank}."
            )
            raise ValueError(val_err)

    return fd.project(ufl.grad(f), target_space)


@PETSc.Log.EventDecorator()
def recover_gradient_clement(f, target_space=None):
    r"""
    Recover the gradient of a scalar field using Clement interpolation.

    Note that the field must be :math:`\mathbb P1`.

    :arg f: the scalar field whose derivatives we seek to recover
    :type f: :class:`firedrake.function.Function`
    :kwarg target_space: the vector-valued function space to recover the gradient in
    :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`
    :returns: recovered gradient
    :rtype: :class:`firedrake.function.Function`
    """
    val_err = (
        "Clement interpolation can only be used to compute gradients of Lagrange"
        " Functions of degree 1."
    )
    if not isinstance(f, fd.Function):
        raise ValueError(val_err)
    if f.ufl_element().family() != "Lagrange" or f.ufl_element().degree() != 1:
        raise ValueError(val_err)
    mesh = f.function_space().mesh()
    rank = len(f.ufl_shape)
    if rank > 1:
        raise ValueError(
            "Clement interpolation can only be used to compute gradients of scalar or"
            f" vector Functions, not Functions of rank {rank}."
        )

    # Recover gradient
    if target_space is None:
        target_space = [fd.VectorFunctionSpace, fd.TensorFunctionSpace][rank](
            mesh, "DG", 0
        )
    return clement_interpolant(fd.project(ufl.grad(f), target_space))


@PETSc.Log.EventDecorator()
def recover_hessian_clement(f, target_space=None):
    r"""
    Recover the gradient and Hessian of a scalar field using two applications of
    Clement interpolation.

    Note that the field must be :math:`\mathbb P1`.

    :arg f: the scalar field whose derivatives we seek to recover
    :type f: :class:`firedrake.function.Function`
    :kwarg target_space: the vector-valued function space to recover the Hessian in
    :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`
    :returns: recovered gradient and Hessian
    :rtype: tuple of two :class:`firedrake.function.Function`\s
    """
    g = recover_gradient_clement(f)
    H = recover_gradient_clement(g, target_space=target_space)
    return g, H
