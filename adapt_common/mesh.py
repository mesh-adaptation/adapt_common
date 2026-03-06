"""Module containing mesh volume properties."""

import firedrake as fd
import ufl

__all__ = ["cell_volume", "patch_volume"]


# TODO: Upstream as a cached_property of MeshGeometry in Firedrake
# TODO: Write unit tests
def cell_volume(mesh):
    """Interpolate a mesh's cell volume in :math:`P^0` space.

    This is computed by interpolating the UFL :class:`~.CellVolume` for the mesh.

    :arg mesh: the mesh to compute cell volumes for
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :return: the cell volume Function
    :rtype: :class:`firedrake.function.Function`
    """
    volume = fd.Function(fd.FunctionSpace(mesh, "Discontinuous Lagrange", 0))
    return volume.interpolate(ufl.CellVolume(mesh))


# TODO: Upstream as a cached_property of MeshGeometry in Firedrake
# TODO: Write unit tests
def patch_volume(mesh):
    """Sum the volumes of cells neighbouring a vertex in :math:`P^1` space.

    :arg mesh: the mesh to compute sums of cell volumes for
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :return: the patch volume Function
    :rtype: :class:`firedrake.function.Function`
    """
    patch_vol = fd.Function(fd.FunctionSpace(mesh, "Lagrange", 1))
    domain = "{[i]: 0 <= i < patch.dofs}"
    instructions = "patch[i] = patch[i] + vol[0]"
    keys = {"vol": (cell_volume(mesh), fd.READ), "patch": (patch_vol, fd.RW)}
    fd.par_loop((domain, instructions), ufl.dx(domain=mesh), keys)
    return patch_vol
