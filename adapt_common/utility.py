"""Module containing utility functions and classes for mesh adaptation."""

import firedrake as fd
import ufl
from firedrake.petsc import PETSc

__all__ = ["VTKFile"]


class VTKFile(fd.output.VTKFile):
    """Overload :class:`firedrake.output.VTKFile` to use ``adaptive`` mode by default.

    Whilst this means that the mesh topology is recomputed at every export, it removes
    any need for the user to reset it manually.
    """

    def __init__(self, *args, **kwargs):
        """Construct the VTKFile.

        All args and kwargs passed to :meth:`firedrake.output.VTKFile.__init__`.
        """
        kwargs.setdefault("adaptive", True)
        super().__init__(*args, **kwargs)

    def _write_vtu(self, *functions):
        """Write functions to VTU file.

        Overload the Firedrake functionality under the blind assumption that the same
        list of functions are outputted each time (albeit on different meshes).

        The arguments and return values are the same as for
        :meth:`firedrake.output.File._write_vtu`.
        """
        if self._fnames is not None:
            if len(self._fnames) != len(functions):
                value_err = (
                    "Writing different number of functions: expected"
                    f" {len(self._fnames)}, got {len(functions)}."
                )
                raise ValueError(value_err)
            for name, f in zip(self._fnames, functions, strict=False):
                if f.name() != name:
                    f.rename(name)
        return super()._write_vtu(*functions)

    @property
    def adaptive(self):
        """Return whether the VTKFile is in adaptive mode.

        :returns: True if the VTKFile is in adaptive mode
        :rtype: :class:`bool`
        """
        return self._adaptive


@PETSc.Log.EventDecorator()
def assemble_mass_matrix(space, norm_type="L2", lumped=False):
    """Assemble a mass matrix associated with some finite element space and norm.

    :arg space: function space to build the mass matrix with
    :type space: :class:`firedrake.functionspaceimpl.FunctionSpace`
    :kwarg norm_type: the type norm to build the mass matrix with
    :type norm_type: :class:`str`
    :kwarg lumped: if `True`, mass lumping is applied
    :type lumped: :class:`bool`
    :returns: the corresponding mass matrix
    :rtype: petsc4py.PETSc.Mat
    """
    trial = fd.TrialFunction(space)
    test = fd.TestFunction(space)
    if norm_type == "L2":
        lhs = ufl.inner(trial, test) * ufl.dx
    elif norm_type == "H1":
        lhs = (
            ufl.inner(trial, test) * ufl.dx
            + ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx
        )
    else:
        value_err = f"Norm type '{norm_type}' not recognised."
        raise ValueError(value_err)
    mass_matrix = fd.assemble(lhs).petscmat
    if not lumped:
        return mass_matrix
    rhs = fd.Function(space).assign(1.0)
    product = fd.Function(space)
    with rhs.dat.vec_ro as b, product.dat.vec as x:
        mass_matrix.mult(b, x)
        return mass_matrix.createDiagonal(x)


def cofunction2function(cofunc):
    """Convert a Cofunction into a Function.

    :arg cofunc: a cofunction
    :type cofunc: :class:`firedrake.cofunction.Cofunction`
    :returns: a function with the same underyling data
    :rtype: :class:`firedrake.function.Function`
    """
    func = fd.Function(cofunc.function_space().dual())
    if isinstance(func.dat.data_with_halos, tuple):
        for i, arr in enumerate(func.dat.data_with_halos):
            arr[:] = cofunc.dat.data_with_halos[i]
    else:
        func.dat.data_with_halos[:] = cofunc.dat.data_with_halos
    return func


def function2cofunction(func):
    """Convert a Cofunction into a Function.

    :arg func: a function
    :type func: :class:`firedrake.function.Function`
    :returns: a cofunction with the same underlying data
    :rtype: :class:`firedrake.cofunction.Cofunction`
    """
    cofunc = fd.Cofunction(func.function_space().dual())
    if isinstance(cofunc.dat.data_with_halos, tuple):
        for i, arr in enumerate(cofunc.dat.data_with_halos):
            arr[:] = func.dat.data_with_halos[i]
    else:
        cofunc.dat.data_with_halos[:] = func.dat.data_with_halos
    return cofunc
