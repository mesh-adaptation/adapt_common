"""Module containing mesh volume properties."""

import firedrake as fd
import ufl

__all__ = ["boundary_facet_area", "boundary_patch_volume", "cell_volume", "patch_volume"]


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
    domain = "{[i]: 0 <= i < patch.dofs}"
    instructions = "patch[i, 0] = patch[i, 0] + vol[0, 0]"
    keys = {"vol": (cell_volume(mesh), fd.READ), "patch": (patch_vol, fd.RW)}
    fd.par_loop((domain, instructions), ufl.dx(domain=mesh), keys)
    return patch_vol


# TODO: Upstream as a cached_property of MeshGeometry in Firedrake (#17)
def boundary_facet_area(mesh):
    """Interpolate a mesh's boundary facet areas in :math:`P^0` space.

    :arg mesh: the mesh to compute boundary facet areas for
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :return: the boundary facet area Function
    :rtype: :class:`firedrake.function.Function`
    """
    P0 = fd.FunctionSpace(mesh, "Discontinuous Lagrange", 0)
    v = fd.TestFunction(P0)
    u = fd.TrialFunction(P0)
    facet_area = fd.Function(P0)
    sp = {"snes_type": "ksponly", "ksp_type": "preonly", "pc_type": "jacobi"}
    fd.solve(
        v * u * ufl.ds(domain=mesh) == v * ufl.FacetArea(mesh) * ufl.ds(domain=mesh),
        facet_area,
        solver_parameters=sp,
    )
    return facet_area


# TODO: Upstream as a cached_property of MeshGeometry in Firedrake (#17)
def boundary_patch_volume(mesh):
    """Sum the facet areas of boundary facets neighbouring a vertex in :math:`P^1` space.

    :arg mesh: the mesh to compute sums of boundary facet areas for
    :type mesh: :class:`firedrake.mesh.MeshGeometry`
    :return: the boundary patch volume Function
    :rtype: :class:`firedrake.function.Function`
    """
    P1 = fd.FunctionSpace(mesh, "Lagrange", 1)

    # Indicate which vertices lie on the domain boundary
    indicator = fd.Function(P1)
    fd.DirichletBC(P1, 1, "on_boundary").apply(indicator)

    # Sum the facet areas of boundary facets neighbouring each vertex
    patch_vol = fd.Function(P1)
    domain = "{[i]: 0 <= i < patch.dofs}"
    instructions = "patch[i, 0] = patch[i, 0] + indicator[i, 0] * area[0, 0]"
    keys = {
        "indicator": (indicator, fd.READ),
        "area": (boundary_facet_area(mesh), fd.READ),
        "patch": (patch_vol, fd.RW),
    }
    fd.par_loop((domain, instructions), ufl.ds(domain=mesh), keys)
    return patch_vol
