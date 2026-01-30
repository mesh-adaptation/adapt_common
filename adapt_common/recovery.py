"""Module containing functions for recovering derivatives of Functions."""

import firedrake as fd
import ufl
from firedrake.petsc import PETSc

__all__ = ["recover_gradient_l2"]


@PETSc.Log.EventDecorator()
def recover_gradient_l2(f, target_space=None):
    r"""Recover the gradient of a scalar or vector field using :math:`L^2` projection.

    :arg f: the scalar field whose derivatives we seek to recover
    :type f: :class:`firedrake.function.Function`
    :kwarg mesh: the underlying mesh
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
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
        if source_degree <= 1:
            val_err = (
                "Input Function must be at least degree 2 to recover gradient"
                " in CG space."
            )
            raise ValueError(val_err)
        target_degree = max(1, source_degree - 1)
        mesh = f.function_space().mesh()
        rank = len(f.function_space().value_shape)
        if rank == 0:
            target_space = fd.VectorFunctionSpace(mesh, "CG", target_degree)
        elif rank == 1:
            target_space = fd.TensorFunctionSpace(mesh, "CG", target_degree)
        else:
            val_err = (
                "L2 projection can only be used to compute gradients of scalar or"
                f" vector Functions, not Functions of rank {rank}."
            )
            raise ValueError(val_err)

    return fd.project(ufl.grad(f), target_space)
