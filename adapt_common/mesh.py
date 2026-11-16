"""Module containing mesh volume properties."""

import firedrake as fd
import ufl

__all__ = [
    "cell_volume",
    "patch_volume",
]


# TODO: Upstream as a cached_property of MeshGeometry in Firedrake (#17)
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


# TODO: Upstream as a cached_property of MeshGeometry in Firedrake (#17)
def patch_volume(mesh):
    """Sum the volumes of cells neighbouring a vertex in :math:`P^1` space.

    :arg mesh: the mesh to compute sums of cell volumes for
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :return: the patch volume Function
    :rtype: :class:`firedrake.function.Function`
    """
    patch_vol = fd.Function(fd.FunctionSpace(mesh, "Lagrange", 1))
    return patch_vol.interpolate(ufl.CellVolume(mesh), access=fd.INC)


def _boundary_submesh(mesh, surface_id):
    """Construct a submesh of the boundary facets of a mesh.

    :arg mesh: the mesh to take the boundary submesh of
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :arg surface_id: the boundary facet marker to submesh, or None for all
        exterior facets ("on_boundary")
    :type surface_id: str or tuple[int]
    :return: the boundary submesh
    :rtype: :class:`firedrake.mesh.MeshGeometry`
    """
    if mesh.topological_dimension < 2:
        raise ValueError(
            "Boundary Clement interpolation is not supported in 1D: "
            "boundary facets are points and Submesh is not defined for them."
        )
    subdomain_id = "on_boundary" if surface_id is None else surface_id
    return fd.Submesh(mesh, subdomain_id=subdomain_id)
