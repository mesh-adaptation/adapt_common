"""Module containing functions for transferring fields between function spaces."""

import firedrake as fd
import ufl
from firedrake.petsc import PETSc

from adapt_common.mesh import cell_volume, patch_volume
from adapt_common.utility import (
    cofunction2function,
    function2cofunction,
    get_function_space,
)

__all__ = ["clement_interpolant"]


def _process_source(source):
    """Process and validate the source function.

    :arg source: the source function or cofunction
    :type source: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :return: the source function space
    :rtype: :class:`firedrake.functionspace.FunctionSpace`
    """
    if isinstance(source, fd.Cofunction):
        Vs = source.function_space().dual()
    elif isinstance(source, fd.Function):
        Vs = source.function_space()
    else:
        type_err = f"Expected Cofunction or Function, got '{type(source)}'."
        raise TypeError(type_err)

    element = Vs.ufl_element()
    if not (element.family() == "Discontinuous Lagrange" and element.degree() == 0):
        val_err = "Source function provided must be from a P0 space."
        raise ValueError(val_err)
    return Vs


def _process_target(source_space, target):
    """Process and validate the target function or function space.

    When `target` is None, the target's shape is inferred from the source space.

    :arg source_space: the function space used for the source function
    :type source_space: :class:`firedrake.functionspaceimpl.FunctionSpace`
    :arg target: the target function, function space, cofunction, dual space, or None
    :type target: :class:`firedrake.function.Function`,
        :class:`firedrake.functionspaceimpl.FunctionSpace`,
        :class:`firedrake.cofunction.Counction`,
        :class:`firedrake.functionspaceimpl.FiredrakeDualSpace`, or None
    :return: the target function and whether the user requested a cofunction target
    :rtype: tuple[:class:`firedrake.function.Function`, bool]
    """
    is_cofunction = False
    rank = len(source_space.value_shape)
    mesh = source_space.mesh()

    if isinstance(target, fd.Function):
        Vt = target.function_space()
    elif isinstance(target, fd.Cofunction):
        target = cofunction2function(target)
        Vt = target.function_space()
        is_cofunction = True
    elif isinstance(target, fd.functionspaceimpl.FiredrakeDualSpace):
        target = fd.Function(target.dual())
        Vt = target.function_space()
        is_cofunction = True
    elif isinstance(target, fd.functionspaceimpl.WithGeometry):
        target = fd.Function(target)
        Vt = target.function_space()
    elif target is None:
        Vt = get_function_space(mesh, "CG", 1, source_space.value_shape)
        target = fd.Function(Vt)
    else:
        type_err = f"Unexpected target type '{type(target)}'."
        raise TypeError(type_err)

    # Validate that Vt is a P1 Lagrange space
    element = Vt.ufl_element()
    if not (element.family() == "Lagrange" and element.degree() == 1):
        val_err = "Target space provided must be P1."
        raise ValueError(val_err)

    # Validate rank consistency
    value_shape = getattr(Vt, "value_shape", ())
    if rank != len(value_shape):
        val_err = f"Rank-{rank} input inconsistent with target space."
        raise ValueError(val_err)

    # Ensure meshes match
    if Vt.mesh() != mesh:
        val_err = "Target function mesh inconsistent with source function mesh."
        raise ValueError(val_err)

    return target, is_cofunction


@PETSc.Log.EventDecorator()
def clement_interpolant(source, target=None):
    r"""Compute the Clement interpolant of a :math:`\mathbb P0` source field.

    i.e. take the volume average over neighbouring cells at each vertex. See
    :cite:`Clement:1975`.

    When `target` is None, the target's shape is inferred from the source space. If the
    user requested a dual/cofunction target, a Cofunction will be returned; otherwise a
    Function is returned.

    :arg source: the :math:`\mathbb P0` source field or dual equivalent
    :type source: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    :arg target: the target function, function space, cofunction, dual space, or None
    :type target: :class:`firedrake.function.Function`,
        :class:`firedrake.functionspaceimpl.FunctionSpace`,
        :class:`firedrake.cofunction.Counction`,
        :class:`firedrake.functionspaceimpl.FiredrakeDualSpace`, or None
    :return: the interpolated :math:`\mathbb P1` field
    :rtype: :class:`firedrake.function.Function` or
        :class:`firedrake.cofunction.Cofunction`
    """
    if isinstance(source, fd.Cofunction):
        source = cofunction2function(source)
    Vs = _process_source(source)
    mesh = Vs.mesh()
    target, is_cofunction = _process_target(Vs, target)

    # Take the weighted average of the source function over the neighbouring cells
    block_size = getattr(Vs, "block_size", 1)
    domain = f"{{[i, j]: 0 <= i < out.dofs and 0 <= j < {block_size}}}"
    instructions = "out[i, j] = out[i, j] + vol[0] * f[0, j]"
    keys = {
        "f": (source, fd.READ),
        "vol": (cell_volume(mesh), fd.READ),
        "out": (target, fd.RW),
    }
    fd.par_loop((domain, instructions), ufl.dx(domain=mesh), keys)

    # Divide by the volume of the patch of neighbouring cells
    domain = f"{{[j]: 0 <= j < {block_size}}}"
    instructions = "out[0, j] = out[0, j] / patch[0]"
    keys = {
        "patch": (patch_volume(mesh), fd.READ),
        "out": (target, fd.RW),
    }
    fd.par_loop((domain, instructions), fd.parloops.direct, keys)

    # Map back to Cofunction, if one is requested or was passed originally
    if is_cofunction:
        return function2cofunction(target)
    return target
