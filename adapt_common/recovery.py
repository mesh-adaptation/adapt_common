"""Module containing functions for recovering derivatives of Functions."""

import firedrake as fd
import ufl
from firedrake.petsc import PETSc

from adapt_common.transfer import clement_interpolant

__all__ = ["recover_gradient_l2", "recover_hessian_l2", "recover_hessian_clement"]


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
def recover_gradient_l2(f, target_space=None, **kwargs):
    r"""Recover the gradient of a scalar or vector field using :math:`L^2` projection.

    :arg f: the scalar or vector field whose derivatives we seek to recover
    :type f: :class:`firedrake.function.Function`
    :kwarg target_space: the vector- or tensor-valued function space to recover the
        gradient in
    :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`
    :returns: recovered gradient
    :rtype: :class:`firedrake.function.Function`

    Any other keyword arguments are passed to :func:`firedrake.projection.Project`.
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

    return fd.project(ufl.grad(f), target_space, **kwargs)


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
        "Clement interpolation can only be used to compute gradients of continuous"
        " Lagrange Functions of degree 1."
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


def _recover_hessian_l2_mixed(f, solver_parameters=None, target_space=None):
    mesh = f.ufl_domain()
    if target_space is None:
        target_space = fd.TensorFunctionSpace(mesh, "CG", 1)
    rank = len(target_space.value_shape)
    if rank != 2:
        raise ValueError(f"Target space must be rank 2, not {rank}.")

    V = fd.VectorFunctionSpace(mesh, "CG", 1)
    W = V * target_space
    g, H = fd.TrialFunctions(W)
    phi, tau = fd.TestFunctions(W)
    sol = fd.Function(W)
    n = ufl.FacetNormal(mesh)

    a = (
        ufl.inner(tau, H) * ufl.dx
        + ufl.inner(ufl.div(tau), g) * ufl.dx
        - ufl.dot(g, ufl.dot(tau, n)) * ufl.ds
        - ufl.dot(ufl.avg(g), ufl.jump(tau, n)) * ufl.dS
        + ufl.inner(phi, g) * ufl.dx
    )
    L = (
        f * ufl.dot(phi, n) * ufl.ds
        + ufl.avg(f) * ufl.jump(phi, n) * ufl.dS
        - f * ufl.div(phi) * ufl.dx
    )
    if solver_parameters is None:
        solver_parameters = {
            "mat_type": "aij",
            "ksp_type": "gmres",
            "ksp_max_it": 20,
            "pc_type": "fieldsplit",
            "pc_fieldsplit_type": "schur",
            "pc_fieldsplit_0_fields": "1",
            "pc_fieldsplit_1_fields": "0",
            "pc_fieldsplit_schur_precondition": "selfp",
            "fieldsplit_0_ksp_type": "preonly",
            "fieldsplit_1_ksp_type": "preonly",
            "fieldsplit_1_pc_type": "gamg",
            "fieldsplit_1_mg_levels_ksp_max_it": 5,
        }
        if fd.COMM_WORLD.size == 1:
            solver_parameters["fieldsplit_0_pc_type"] = "ilu"
            solver_parameters["fieldsplit_1_mg_levels_pc_type"] = "ilu"
        else:
            solver_parameters["fieldsplit_0_pc_type"] = "bjacobi"
            solver_parameters["fieldsplit_0_sub_ksp_type"] = "preonly"
            solver_parameters["fieldsplit_0_sub_pc_type"] = "ilu"
            solver_parameters["fieldsplit_1_mg_levels_pc_type"] = "bjacobi"
            solver_parameters["fieldsplit_1_mg_levels_sub_ksp_type"] = "preonly"
            solver_parameters["fieldsplit_1_mg_levels_sub_pc_type"] = "ilu"
    fd.solve(a == L, sol, solver_parameters=solver_parameters)
    return sol.subfunctions


@PETSc.Log.EventDecorator()
def recover_hessian_l2(f, target_space=None, mixed=True, solver_parameters=None):
    r"""
    Recover the gradient and Hessian of a scalar field using :math:`L^2` projection.

    :arg f: the scalar field whose derivatives we seek to recover
    :type f: :class:`firedrake.function.Function`
    :kwarg target_space: the tensor-valued function space to recover the gradient in
    :type target_space: :class:`firedrake.functionspaceimpl.FunctionSpace`
    :kwarg mixed: if `True`, solve the projection using a mixed formulation, otherwise
        apply one :math:`L^2` projection for the gradient and another for the Hessian
    :type mixed: bool
    :returns: recovered gradient and Hessian
    :rtype: tuple of two :class:`firedrake.function.Function`\s
    """
    if mixed:
        return _recover_hessian_l2_mixed(
            f, target_space=target_space, solver_parameters=solver_parameters
        )
    g = recover_gradient_l2(f, solver_parameters=solver_parameters)
    H = recover_gradient_l2(
        g, target_space=target_space, solver_parameters=solver_parameters
    )
    return g, H


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
