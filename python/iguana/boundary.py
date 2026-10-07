# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Boundaries, the NURBS faces of a domain where conditions are imposed

A boundary gathers faces of the boundary of a domain as the CAD model gives
them: curves bounding a region of the plane, or surfaces bounding a volume.
The sign of each face tells whether the normal of its parametrization
points out of the domain. A boundary need not hold every face of the
domain, only those where a condition is imposed, and it gives the closest
point of its faces to any point of the plane.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp
from iguana.patch import CurvePatch, SurfacePatch


class Projection(NamedTuple):
    """Closest points of a boundary to some points, one row or entry per
    point, unpackable as `positions, normals, faces`.

    Attributes:
        positions: The closest points, of shape `(num_points, 2)`.
        normals: Unit normals of the faces at the closest points, pointing
            out of the domain, of shape `(num_points, 2)`.
        faces: Face holding each closest point, of shape `(num_points,)`.
    """

    positions: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64]
    faces: npt.NDArray[np.int32]


class Boundary:
    """Faces of the boundary of a domain, each with the side its normal
    points to."""

    _cpp_object: _cpp.Boundary2d | _cpp.Boundary3d

    def __init__(self, faces: Sequence[CurvePatch] | Sequence[SurfacePatch],
                 signs: Sequence[int] | None = None) -> None:
        """Initialize the boundary from its faces.

        Args:
            faces: Faces of the boundary, all curves in the plane or all
                surfaces in space. B-spline faces are taken as NURBS faces
                of unit weights.
            signs: 1 for each face whose normal points out of the domain and
                -1 for each whose normal points into it, as a CAD model
                marks reversed faces. Without them, every normal points out.

        Raises:
            TypeError: If the faces are neither all curves in the plane nor
                all surfaces in space.
            ValueError: If there is not one sign per face, or if a sign is
                neither 1 nor -1.
        """
        faces = list(faces)

        if all(isinstance(face, CurvePatch)
               and face.control_points.shape[1] == 2 for face in faces):
            boundary = _cpp.Boundary2d
        elif all(isinstance(face, SurfacePatch) for face in faces):
            boundary = _cpp.Boundary3d
        else:
            raise TypeError('the faces must be all curves in the plane or '
                            'all surfaces in space')

        if signs is None:
            signs = [1] * len(faces)

        self._cpp_object = boundary(faces=[_nurbs(face) for face in faces],
                                    signs=list(signs))

    @property
    def faces(self) -> list[CurvePatch] | list[SurfacePatch]:
        """Faces of the boundary, as NURBS patches."""
        if isinstance(self._cpp_object, _cpp.Boundary2d):
            return [CurvePatch(face) for face in self._cpp_object.faces]

        return [SurfacePatch(face) for face in self._cpp_object.faces]

    @property
    def signs(self) -> npt.NDArray[np.int_]:
        """Sign of each face, of shape `(num_faces,)`: 1 where its normal
        points out of the domain and -1 where it points into it."""
        return np.array(self._cpp_object.signs)

    def project_points(self, points: npt.ArrayLike) -> Projection:
        """Closest points of the boundary to points of the plane.

        Every element of every face is searched, so that each closest
        point is the global one, found with no initial guess. Of points at
        one distance the first face wins, so that a vertex takes the normal
        of the face it is found on first.

        Args:
            points: The points, of shape `(num_points, 2)`.

        Returns:
            The closest point to each, with the outward normal of its face
            there and the face.

        Raises:
            TypeError: If the boundary is made of surfaces.
            ValueError: If the points are not of shape `(num_points, 2)`,
                if the boundary has no face, or if the knot vector of a
                face is not clamped.
        """
        if not isinstance(self._cpp_object, _cpp.Boundary2d):
            raise TypeError('only a boundary of curves in the plane projects '
                            'points')

        points = np.asarray(points, dtype=np.float64)

        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('the points must be of shape (num_points, 2)')

        return Projection(*self._cpp_object.project_points(points))

    def __repr__(self) -> str:
        return f'Boundary(num_faces={self._cpp_object.num_faces})'


def _nurbs(face: CurvePatch | SurfacePatch
           ) -> _cpp.NURBSPlanarCurvePatch | _cpp.NURBSSurfacePatch:
    """C++ NURBS patch of a face, of unit weights if it is a B-spline."""
    patch = face._cpp_object

    if isinstance(patch, (_cpp.NURBSPlanarCurvePatch,
                          _cpp.NURBSSurfacePatch)):
        return patch

    weights = np.ones(len(patch.coefficients))

    if isinstance(patch, _cpp.PlanarCurvePatch):
        basis = _cpp.UnivariateNURBS(bspline=patch.basis, weights=weights)
        return _cpp.NURBSPlanarCurvePatch(basis=basis,
                                          coefficients=patch.coefficients)

    basis = _cpp.BivariateNURBS(bspline=patch.basis, weights=weights)
    return _cpp.NURBSSurfacePatch(basis=basis,
                                  coefficients=patch.coefficients)
