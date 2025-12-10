"""Module containing overloaded versions of Firedrake's norm and errornorm functions."""

import firedrake as fd
import ufl
from firedrake.petsc import PETSc

from adapt_common.utility import cofunction2function

__all__ = ["errornorm", "norm"]


@PETSc.Log.EventDecorator()
def norm(v, norm_type="L2", condition=None, boundary=False):
    r"""Overload :func:`fd.norms.norm` to allow for :math:`\ell^p` norms.

    Currently supported ``norm_type`` options:
    * ``'l1'``
    * ``'l2'``
    * ``'linf'``
    * ``'L2'``
    * ``'H1'``
    * ``'Hdiv'``
    * ``'Hcurl'``
    * or any ``'Lp'`` with :math:`p >= 1`.

    Note that this version is case sensitive, i.e. ``'l2'`` and ``'L2'`` will give
    different results in general.

    :arg v: the function to take the norm of
    :type v: :class:`fd.function.Function` or
        :class:`fd.cofunction.Cofunction`
    :kwarg norm_type: the type of norm to use
    :type norm_type: :class:`str`
    :kwarg condition: a UFL condition for specifying a subdomain to compute the norm
        over
    :kwarg boundary: if ``True``, the norm is computed over the domain boundary
    :type boundary: :class:`bool`
    :returns: the norm value
    :rtype: :class:`float`
    """
    if isinstance(v, fd.Cofunction):
        v = cofunction2function(v)
    condition = condition or fd.Constant(1.0)
    norm_codes = {
        "l1": PETSc.NormType.NORM_1,
        "l2": PETSc.NormType.NORM_2,
        "linf": PETSc.NormType.NORM_INFINITY,
    }
    p = 2
    if norm_type in norm_codes:
        if boundary:
            not_impl_err = "lp errors on the boundary not yet implemented."
            raise NotImplementedError(not_impl_err)
        v.interpolate(condition * v)
        with v.dat.vec_ro as vv:
            return vv.norm(norm_codes[norm_type])
    elif norm_type[0] == "l":
        not_impl_err = f"lp norm of order {norm_type[1:]} not supported."
        raise NotImplementedError(not_impl_err)
    else:
        dX = ufl.ds if boundary else ufl.dx
        if norm_type.startswith("L"):
            try:
                p = int(norm_type[1:])
            except ValueError as exc:
                val_err = f"Unable to interpret '{norm_type}' norm."
                raise ValueError(val_err) from exc
            if p < 1:
                val_err = f"'{norm_type}' norm does not make sense."
                raise ValueError(val_err)
            integrand = ufl.inner(v, v)
        elif norm_type.lower() in ("h1", "hdiv", "hcurl"):
            integrand = {
                "h1": lambda w: ufl.inner(w, w) + ufl.inner(ufl.grad(w), ufl.grad(w)),
                "hdiv": lambda w: ufl.inner(w, w) + ufl.div(w) * ufl.div(w),
                "hcurl": lambda w: ufl.inner(w, w)
                + ufl.inner(ufl.curl(w), ufl.curl(w)),
            }[norm_type.lower()](v)
        else:
            val_err = f"Unknown norm type '{norm_type}'."
            raise ValueError(val_err)
        return fd.assemble(condition * integrand ** (p / 2) * dX) ** (1 / p)


@PETSc.Log.EventDecorator()
def errornorm(u, uh, norm_type="L2", boundary=False, **kwargs):
    r"""Overload :func:`fd.norms.errornorm` to allow for :math:`\ell^p` norms.

    Currently supported ``norm_type`` options:
    * ``'l1'``
    * ``'l2'``
    * ``'linf'``
    * ``'L2'``
    * ``'H1'``
    * ``'Hdiv'``
    * ``'Hcurl'``
    * or any ``'Lp'`` with :math:`p >= 1`.

    Note that this version is case sensitive, i.e. ``'l2'`` and ``'L2'`` will give
    different results in general.

    :arg u: the 'true' value
    :type u: :class:`fd.function.Function` or
        :class:`fd.cofunction.Cofunction`
    :arg uh: the approximation of the 'truth'
    :type uh: :class:`fd.function.Function` or
        :class:`fd.cofunction.Cofunction`
    :kwarg norm_type: the type of norm to use
    :type norm_type: :class:`str`
    :kwarg boundary: if ``True``, the norm is computed over the domain boundary
    :type boundary: :class:`bool`
    :returns: the error norm value
    :rtype: :class:`float`

    Any other keyword arguments are passed to :func:`fd.norms.errornorm`.
    """
    if isinstance(u, fd.Cofunction):
        u = cofunction2function(u)
    if isinstance(uh, fd.Cofunction):
        uh = cofunction2function(uh)
    if not isinstance(uh, fd.Function):
        type_err = f"uh should be a Function, is a '{type(uh)}'."
        raise TypeError(type_err)
    if norm_type[0] == "l" and not isinstance(u, fd.Function):
        type_err = f"u should be a Function, is a '{type(u)}'."
        raise TypeError(type_err)

    if len(u.ufl_shape) != len(uh.ufl_shape):
        val_err = "Mismatching rank between u and uh."
        raise ValueError(val_err)

    if isinstance(u, fd.Function):
        degree_u = u.function_space().ufl_element().degree()
        degree_uh = uh.function_space().ufl_element().degree()
        if degree_uh > degree_u:
            fd.logging.warning(
                "Degree of exact solution less than approximation degree"
            )

    # Case 1: point-wise norms
    if norm_type[0] == "l":
        v = u
        v -= uh

    # Case 2: UFL norms for mixed function spaces
    elif hasattr(uh.function_space(), "num_sub_spaces"):
        if norm_type == "L2":
            vv = [
                uu - uuh
                for uu, uuh in zip(u.subfunctions, uh.subfunctions, strict=False)
            ]
            dX = ufl.ds if boundary else ufl.dx
            return ufl.sqrt(fd.assemble(sum([ufl.inner(v, v) for v in vv]) * dX))
        else:
            not_impl_err = f"Norm type '{norm_type}' not supported for mixed spaces."
            raise NotImplementedError(not_impl_err)

    # Case 3: UFL norms for non-mixed spaces
    else:
        v = u - uh

    return norm(v, norm_type=norm_type, **kwargs)
