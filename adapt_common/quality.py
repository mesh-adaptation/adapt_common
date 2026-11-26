"""Class for computing mesh quality measures."""

import firedrake as fd
import numpy as np
import ufl
from firedrake.petsc import PETSc

__all__ = ["QualityMeasure"]


class QualityMeasure:
    """
    Class for computing quality measures associated with a given mesh.

    Choices of quality measure:
      * ``min_angle``: the minimum angle of each cell
      * ``eskew``: the equiangle skew of each cell
      * ``skewness``: the skewness of each cell in a 2D triangular mesh
      * ``area``: the area of each cell in a 2D mesh
      * ``volume``: the volume of each cell in a 3D mesh
      * ``facet_area``: the area of each *facet*.
      * ``aspect_ratio``: the aspect ratio of each cell in a 2D triangular or 3D
        tetrahedral mesh
      * ``scaled_jacobian``: the scaled Jacobian of each cell in a 2D triangular or
        3D tetrahedral mesh
    """

    _measures = (
        "min_angle",
        "eskew",
        "skewness",
        "area",
        "volume",
        "facet_area",
        "aspect_ratio",
        "scaled_jacobian",
    )

    @PETSc.Log.EventDecorator()
    def __init__(self, mesh):
        """
        :arg mesh: the input mesh to do computations on
        :type mesh: :class:`firedrake.mesh.MeshGeometry`
        """
        self.mesh = mesh
        self.dim = mesh.topological_dimension
        self.coords = mesh.coordinates
        self.P0 = fd.FunctionSpace(mesh, "DG", 0)

    @PETSc.Log.EventDecorator()
    def __call__(self, name):
        """
        :arg name: the quality measure name
        :type name: :class:`str`
        :returns: the quality measure applied to the provided mesh
        :rtype: :class:`firedrake.function.Function`
        """
        if name not in QualityMeasure._measures:
            raise ValueError(f"Quality measure '{name}' not recognised.")

        self._check_simplicial(name)

        match (name, self.dim):
            case ("area" | "volume", _):
                return fd.Function(self.P0).interpolate(ufl.CellVolume(self.mesh))

            case ("facet_area", _):
                return self._facet_area()

            case ("min_angle" | "eskew" | "aspect_ratio" | "scaled_jacobian", 3):
                J = self._jacobian()
                expr = {
                    "min_angle": self._min_angle_3d,
                    "eskew": self._eskew_3d,
                    "aspect_ratio": self._aspect_ratio_3d,
                    "scaled_jacobian": self._scaled_jacobian_3d,
                }[name](J)
                return fd.Function(self.P0).interpolate(expr)

            case ("min_angle" | "eskew", 2):
                edge1, edge2, edge3, a, b, c = self._edges_2d()
                expr = {
                    "min_angle": self._min_angle_2d,
                    "eskew": self._eskew_2d,
                }[name](edge1, edge2, edge3, a, b, c)
                return fd.Function(self.P0).interpolate(expr)

            case ("skewness", 2):
                edge1, edge2 = self._edges_2d()[:2]
                return fd.Function(self.P0).interpolate(self._skewness_2d(edge1, edge2))

            case ("aspect_ratio", 2):
                _, _, _, a, b, c = self._edges_2d()
                ar = fd.Function(self.P0)
                ar.interpolate(
                    a * b * c / ((a + b - c) * (b + c - a) * (c + a - b))
                )
                return ar

            case ("scaled_jacobian", 2):
                _, _, _, a, b, c = self._edges_2d()
                detJ = ufl.JacobianDeterminant(self.mesh)
                jacobian_sign = ufl.sign(detJ)
                max_product = ufl.max_value(
                    ufl.max_value(
                        ufl.max_value(a * b, a * c), ufl.max_value(b * c, b * a)
                    ),
                    ufl.max_value(c * a, c * b),
                )
                return fd.Function(self.P0).interpolate(
                    detJ / max_product * jacobian_sign
                )

            case _:
                raise NotImplementedError(
                    f"Quality measure '{name}' not implemented in the {self.dim}D case."
                )

    def _check_simplicial(self, name):
        """Check the mesh is triangular/tetrahedral where a measure requires it."""
        if name in ("area", "volume", "facet_area"):
            return
        expected = "triangle" if self.dim == 2 else "tetrahedron"
        cellname = self.mesh.ufl_cell().cellname
        if cellname != expected:
            raise ValueError(
                f"Quality measure '{name}' requires a {expected} mesh, "
                f"but the mesh has '{cellname}' cells."
            )

    def _facet_area(self):
        """Area of each facet, computed via a mass-matrix solve."""
        HDivTrace = fd.FunctionSpace(self.mesh, "HDiv Trace", 0)
        v = fd.TestFunction(HDivTrace)
        u = fd.TrialFunction(HDivTrace)
        facet_area = fd.Function(HDivTrace, name="Facet areas")
        mass_term = v("+") * u("+") * ufl.dS + v * u * ufl.ds
        rhs = (
            v("+") * ufl.FacetArea(self.mesh) * ufl.dS
            + v * ufl.FacetArea(self.mesh) * ufl.ds
        )
        sp = {
            "snes_type": "ksponly",
            "ksp_type": "preonly",
            "pc_type": "jacobi",
        }
        fd.solve(mass_term == rhs, facet_area, solver_parameters=sp)
        return facet_area

    def _jacobian(self):
        """Cell-wise Jacobian of the mesh as a DG0 tensor function."""
        P0_ten = fd.TensorFunctionSpace(self.mesh, "DG", 0)
        return fd.Function(P0_ten).interpolate(ufl.Jacobian(self.mesh))

    def _edges_2d(self):
        """Edge vectors and lengths of a triangular cell from its Jacobian."""
        J = self._jacobian()
        edge1 = ufl.as_vector([J[0, 0], J[1, 0]])
        edge2 = ufl.as_vector([J[0, 1], J[1, 1]])
        edge3 = edge1 - edge2
        a = ufl.sqrt(ufl.dot(edge1, edge1))
        b = ufl.sqrt(ufl.dot(edge2, edge2))
        c = ufl.sqrt(ufl.dot(edge3, edge3))
        return edge1, edge2, edge3, a, b, c

    def _angles_2d(self, edge1, edge2, edge3, a, b, c):
        """The three interior angles of a triangular cell."""
        return [
            ufl.acos(ufl.dot(edge1, edge2) / (a * b)),
            ufl.acos(ufl.dot(edge1, edge3) / (a * c)),
            ufl.acos(-ufl.dot(edge2, edge3) / (b * c)),
        ]

    def _min_angle_2d(self, edge1, edge2, edge3, a, b, c):
        """Minimum angle of a triangular cell from its edge vectors and lengths."""
        a1, a2, a3 = self._angles_2d(edge1, edge2, edge3, a, b, c)
        return ufl.min_value(ufl.min_value(a1, a2), a3)

    def _eskew_2d(self, edge1, edge2, edge3, a, b, c):
        """Equiangle skew of a triangular cell from its edge vectors and lengths."""
        a1, a2, a3 = self._angles_2d(edge1, edge2, edge3, a, b, c)
        a_min = ufl.min_value(ufl.min_value(a1, a2), a3)
        a_max = ufl.max_value(ufl.max_value(a1, a2), a3)
        a_ideal = np.pi / 3
        return ufl.max_value(
            a_max - a_ideal / (np.pi - a_ideal),
            (a_ideal - a_min) / a_ideal,
        )

    def _skewness_2d(self, edge1, edge2):
        """Skewness of a triangular cell from its edge vectors."""
        half = 1 / 2
        mid1 = (edge1 + edge2) * half  # midpoint of edge p2p3, from p1
        mid2 = edge2 * half  # midpoint of edge p3p1, from p1
        mid3 = edge1 * half  # midpoint of edge p1p2, from p1
        orth1 = mid3 - mid2
        orth2 = mid1 - mid3
        orth3 = mid2 - mid1

        def deviation(normal, orth):
            cos_t = ufl.dot(normal, orth) / (
                ufl.sqrt(ufl.dot(normal, normal)) * ufl.sqrt(ufl.dot(orth, orth))
            )
            t = ufl.acos(cos_t)
            return ufl.min_value(t, np.pi - t)

        t_min = ufl.min_value(
            ufl.min_value(deviation(mid1, orth1), deviation(mid2 - edge1, orth2)),
            deviation(mid3 - edge2, orth3),
        )
        return np.pi / 2 - t_min

    def _angles_3d(self, J):
        """The 12 planar angles of a tetrahedral cell from its Jacobian."""
        v12 = ufl.as_vector([J[i, 0] for i in range(self.dim)])
        v13 = ufl.as_vector([J[i, 1] for i in range(self.dim)])
        v14 = ufl.as_vector([J[i, 2] for i in range(self.dim)])
        v23, v24, v34 = v13 - v12, v14 - v12, v14 - v13

        def length(v):
            return ufl.sqrt(ufl.dot(v, v))

        def angle(u, v):
            return ufl.acos(ufl.dot(u, v) / (length(u) * length(v)))

        d12, d13 = length(v12), length(v13)
        d23, d24, d34 = length(v23), length(v24), length(v34)
        return [
            angle(v13, v14),
            angle(v12, v14),
            angle(v13, v12),
            angle(v23, v24),
            ufl.acos(-ufl.dot(v12, v24) / (d12 * d24)),
            ufl.acos(-ufl.dot(v12, v23) / (d12 * d23)),
            ufl.acos(-ufl.dot(v23, v34) / (d23 * d34)),
            ufl.acos(-ufl.dot(v13, v34) / (d13 * d34)),
            angle(v13, v23),
            angle(v24, v34),
            angle(v14, v34),
            angle(v14, v24),
        ]

    def _min_angle_3d(self, J):
        """Minimum planar angle of a tetrahedral cell from its Jacobian."""
        a_min = self._angles_3d(J)[0]
        for ang in self._angles_3d(J)[1:]:
            a_min = ufl.min_value(a_min, ang)
        return a_min

    def _aspect_ratio_3d(self, J):
        """Aspect ratio of a tetrahedral cell from its Jacobian."""
        v12 = ufl.as_vector([J[i, 0] for i in range(self.dim)])
        v13 = ufl.as_vector([J[i, 1] for i in range(self.dim)])
        v14 = ufl.as_vector([J[i, 2] for i in range(self.dim)])
        v23, v24, v34 = v13 - v12, v14 - v12, v14 - v13

        def length(v):
            return ufl.sqrt(ufl.dot(v, v))

        d12, d13, d14 = length(v12), length(v13), length(v14)
        d23, d24, d34 = length(v23), length(v24), length(v34)

        vol_matrix = ufl.as_matrix([list(v) for v in (v12, v13, v14)])
        volume = ufl.algebra.Abs(ufl.det(vol_matrix)) / 6

        # Circumradius via the Cayley-Menger-style formula
        p = d12 * d34
        q = d13 * d24
        r = d14 * d23
        cir_radius = ufl.sqrt(
            (p + q + r) * (p + q - r) * (p - q + r) * (-p + q + r)
        ) / (24 * volume)

        # Face areas via Heron's formula
        def face_area(x, y, z):
            s = (x + y + z) / 2
            return ufl.sqrt(s * (s - x) * (s - y) * (s - z))

        in_radius = 3 * volume / (
            face_area(d23, d24, d34)
            + face_area(d13, d14, d34)
            + face_area(d12, d14, d24)
            + face_area(d12, d13, d23)
        )
        return cir_radius / (3 * in_radius)

    def _scaled_jacobian_3d(self, J):
        """Scaled Jacobian of a tetrahedral cell from its Jacobian."""
        v12 = ufl.as_vector([J[i, 0] for i in range(self.dim)])
        v13 = ufl.as_vector([J[i, 1] for i in range(self.dim)])
        v14 = ufl.as_vector([J[i, 2] for i in range(self.dim)])
        v23, v24, v34 = v13 - v12, v14 - v12, v14 - v13

        def length(v):
            return ufl.sqrt(ufl.dot(v, v))

        d12, d13, d14 = length(v12), length(v13), length(v14)
        d23, d24, d34 = length(v23), length(v24), length(v34)

        def corner_det(u, v, w):
            return ufl.algebra.Abs(ufl.det(ufl.as_matrix([list(u), list(v), list(w)])))

        sj = [
            corner_det(v12, v13, v14) / (d12 * d13 * d14),
            corner_det(-v12, v23, v24) / (d12 * d23 * d24),
            corner_det(-v13, -v23, v34) / (d13 * d23 * d34),
            corner_det(-v14, -v24, -v34) / (d14 * d24 * d34),
        ]
        return ufl.min_value(ufl.min_value(sj[0], sj[1]), ufl.min_value(sj[2], sj[3]))

    def _eskew_3d(self, J):
        """Equiangle skew of a tetrahedral cell from its Jacobian."""
        angles = self._angles_3d(J)
        a_min, a_max = angles[0], angles[0]
        for ang in angles[1:]:
            a_min = ufl.min_value(a_min, ang)
            a_max = ufl.max_value(a_max, ang)
        a_ideal = np.pi / 3
        return ufl.max_value(
            (a_max - a_ideal) / (np.pi - a_ideal),
            (a_ideal - a_min) / a_ideal,
        )
