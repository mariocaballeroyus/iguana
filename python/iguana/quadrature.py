# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Quadratures, the points integrating over a patch or along a boundary

A domain quadrature is filled one cell type at a time, each with a rule of
its own, such as Gauss-Legendre on the inside cells. A boundary quadrature
places one rule on every piece of a boundary that the knot lines of the
patch divide, or on every face of the surrogate boundary of the inside
cells, whose points it can shift towards the true boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp
from iguana.boundary import Boundary
from iguana.grid import HierarchicalGrid
from iguana.embedding import CellType, SurrogateBoundary
from iguana.patch import PlanarPatch, SurfacePatch, VolumePatch


class DomainQuadrature:
    """The quadrature points over the cells of a patch."""

    _cpp_object: _cpp.DomainQuadrature2d | _cpp.DomainQuadrature3d

    def __init__(self, patch: PlanarPatch | SurfacePatch | VolumePatch,
                 cell_types: Sequence[CellType] | None = None,
                 grid: HierarchicalGrid | None = None) -> None:
        """Initialize an empty quadrature over the cells of a patch.

        Args:
            patch: The background patch, a planar, surface or volume
                patch, which places the points in physical space.
            cell_types: The type of each cell, in the numbering of the
                grid. Without them, every cell lies inside.
            grid: A hierarchical grid on the knots of the patch, whose
                active elements are the cells. Without it, the cells are
                the elements of the patch, with the first direction running
                fastest.

        Raises:
            TypeError: If the patch is not a planar, surface or volume
                patch, or if the grid is not a hierarchical grid.
            ValueError: If the grid does not lie on the knots of the
                patch, or if there is not one cell type per cell.
        """
        if isinstance(patch, VolumePatch):
            self._cpp_object = _cpp.DomainQuadrature3d()
            classification = _cpp.CellClassification3d
            self._dimension = 3
        elif isinstance(patch, (PlanarPatch, SurfacePatch)):
            self._cpp_object = _cpp.DomainQuadrature2d()
            classification = _cpp.CellClassification2d
            self._dimension = 2
        else:
            raise TypeError('the patch must be a planar, surface or volume '
                            'patch')

        if grid is None:
            # The elements of the patch, as a grid that is not refined
            grid = HierarchicalGrid(patch.degrees, patch.knots)
        elif not isinstance(grid, HierarchicalGrid):
            raise TypeError('the grid must be a hierarchical grid')
        elif grid._dimension != self._dimension:
            raise ValueError('the grid must lie on the knots of the '
                             'patch')
        else:
            # Positions of the empty quadrature only check that the grid
            # lies on the knots of the patch
            self._cpp_object.positions(patch._cpp_object,
                                       grid._cpp_object)

        num_elements = grid.num_elements

        if cell_types is None:
            cell_types = [CellType.inside] * num_elements

        if len(cell_types) != num_elements:
            raise ValueError('there must be one cell type per cell')

        self._patch = patch
        self._grid = grid
        self._cell_types = list(cell_types)
        self._classification = classification(self._cell_types)

    @property
    def patch(self) -> PlanarPatch | SurfacePatch | VolumePatch:
        """The background patch, which places the points in space."""
        return self._patch

    @property
    def grid(self) -> HierarchicalGrid:
        """The grid whose active elements are the cells."""
        return self._grid

    @property
    def cell_types(self) -> list[CellType]:
        """Type of each cell, in the numbering of the grid."""
        return list(self._cell_types)

    @property
    def num_elements(self) -> int:
        """Number of filled elements."""
        return self._cpp_object.num_elements

    @property
    def num_points(self) -> int:
        """Number of points over all filled elements."""
        return self._cpp_object.num_points

    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """Positions of the points in physical space, of shape
        `(num_points, 2)` on a planar patch and `(num_points, 3)`
        otherwise."""
        return self._cpp_object.positions(self._patch._cpp_object,
                                          self._grid._cpp_object)

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Weights of the points in physical space, of shape
        `(num_points,)`.

        Each is the weight in parameter space times the measure of the
        patch at its point, so that they integrate over the cells as they
        lie in space: by area on planar and surface patches and by volume
        on volume patches.
        """
        return self._cpp_object.physical_weights(self._patch._cpp_object,
                                                 self._grid._cpp_object)

    def fill_gauss_legendre(self, cell_type: CellType,
                            num_points: int | Sequence[int]) -> None:
        """Fill the cells of one type with a Gauss-Legendre rule.

        Args:
            cell_type: The type of the cells to fill.
            num_points: Number of points of each direction, or a single
                number for all of them.

        Raises:
            ValueError: If there is not one number per direction, if a
                number lies outside [1, 8], or if the cells of this type
                are already filled.
        """
        if isinstance(num_points, int):
            num_points = [num_points] * self._dimension

        if len(num_points) != self._dimension:
            raise ValueError('there must be one number of points per '
                             'direction')

        self._cpp_object.fill_gauss_legendre(
            self._grid._cpp_object, self._classification, cell_type,
            list(num_points))

    def fill_moment_fitting(self, cell_type: CellType,
                            vertices: npt.ArrayLike,
                            facets: npt.ArrayLike,
                            order: int = 2) -> None:
        """Fill the cells of one type with rules fitted to a domain.

        The rule of each cell integrates the Legendre polynomials up to the
        order in each direction over its part inside the domain, with
        positive weights at points inside it.

        On a surface patch the domain is the region a closed polygon
        trims, given in parameter space, where the trimming curves of a
        trimmed surface lie. On a planar patch it is a region a closed
        polygon bounds in the plane, and on a volume patch a solid given
        in space. The planar or volume patch must then have an affine map,
        such as that of a rectangle or block from create_rectangle or
        create_box, through which the vertices map back to parameters.

        Args:
            cell_type: The type of the cells to fill, usually the cut cells.
            vertices: On a surface patch, the vertices of the polygon in
                parameter space, and on a planar patch in the plane, of
                shape `(num_vertices, 2)`. On a volume patch, the
                vertices of a closed triangle mesh of the solid in space,
                of shape `(num_vertices, 3)`.
            facets: On a surface or planar patch, the vertices of each
                segment of the polygon, with the domain on their left, of
                shape `(num_segments, 2)`. On a volume patch, the
                vertices of each triangle, counterclockwise seen from
                outside, of shape `(num_triangles, 3)`.
            order: Highest polynomial degree of each direction.

        Raises:
            ValueError: If the polygon or mesh is malformed, if the map of
                a planar or volume patch is not affine, if the order lies
                outside [0, 7] on a surface or planar patch or [0, 4] on a
                volume patch, or if the cells of this type are already
                filled.
        """
        # Trimming curves lie in parameter space already
        if isinstance(self._patch, SurfacePatch):
            parameters = np.asarray(vertices, dtype=float)
        else:
            parameters = self._patch.invert_points(vertices)

        self._cpp_object.fill_moment_fitting(
            self._grid._cpp_object, self._classification, cell_type,
            parameters, np.asarray(facets, dtype=np.int32), order)

    def __repr__(self) -> str:
        return (f'DomainQuadrature(num_elements={self.num_elements}, '
                f'num_points={self.num_points})')


class BoundaryQuadrature:
    """The quadrature points along a boundary in the plane of a patch."""

    _cpp_object: _cpp.BoundaryQuadrature2d

    def __init__(self, patch: PlanarPatch,
                 boundary: Boundary | SurrogateBoundary,
                 num_points: int, shift: Boundary | None = None,
                 order: int | None = None) -> None:
        """Initialize the quadrature of a boundary in the plane of a patch.

        The faces of a boundary of curves are divided exactly where they
        cross the knot lines of the patch, and each piece carries a
        Gauss-Legendre rule of its own along its face, so that the points
        follow the curves as they are. The faces of a surrogate boundary
        are whole sides of cells, each carrying the rule.

        A quadrature of a surrogate boundary can be shifted towards the
        true boundary, as the shifted boundary method does. Each point is
        mapped to its closest point on the true boundary, where the
        conditions take their data, and the trace of the field is expanded
        from the point to it by a Taylor series. The points, weights and
        normals stay those of the surrogate boundary.

        Args:
            patch: The background patch, a planar patch. For a boundary of
                curves or a shift its map must be affine, such as that of
                a rectangle from `create_rectangle`.
            boundary: The boundary, made of curves in the plane of the
                patch, or the surrogate boundary of its inside cells.
            num_points: Number of points on each piece or face.
            shift: The true boundary, made of curves in the plane of the
                patch, to shift the points of a surrogate boundary towards.
                Without it, the quadrature is not shifted.
            order: Total order of the Taylor expansion of a shifted
                quadrature, nonnegative. Without it, the highest degree of
                the patch.

        Raises:
            TypeError: If the patch is not a planar patch, if the boundary
                is neither made of curves in the plane nor a surrogate
                boundary, or if the shift is not made of curves in the
                plane.
            ValueError: If a surrogate boundary lies on another patch, if
                the map of the patch is not affine for a boundary of
                curves or a shift, if the knot vector of a face is not
                clamped, if the number of points lies outside [1, 8], if
                a boundary of curves is shifted, if an order is given
                without a shift, or if the order is negative.
        """
        if not isinstance(patch, PlanarPatch):
            raise TypeError('the patch must be a planar patch')

        if isinstance(boundary, SurrogateBoundary):
            if boundary.patch is not patch:
                raise ValueError('the surrogate boundary must lie on the '
                                 'patch')

            divided = boundary._cpp_object
        elif _in_the_plane(boundary):
            # The boundary divided over the elements, which the quadrature
            # needs only to be built
            divided = _cpp.EmbeddedBoundary2d(patch=patch._cpp_object,
                                              boundary=boundary._cpp_object)
        else:
            raise TypeError('the boundary must be made of curves in the '
                            'plane, or be a surrogate boundary')

        if shift is not None:
            if not _in_the_plane(shift):
                raise TypeError('the shift must be made of curves in the '
                                'plane')

            if not isinstance(boundary, SurrogateBoundary):
                raise ValueError('only a surrogate boundary is shifted')
        elif order is not None:
            raise ValueError('an order needs a shift')

        self._cpp_object = _cpp.BoundaryQuadrature2d(boundary=divided,
                                                     num_points=num_points)
        self._patch = patch
        self._boundary = boundary

        if shift is not None:
            if order is None:
                order = max(patch.degrees)

            self._cpp_object.shift(patch=patch._cpp_object,
                                   boundary=shift._cpp_object, order=order)

    @property
    def patch(self) -> PlanarPatch:
        """The background patch, which places the points in the plane."""
        return self._patch

    @property
    def boundary(self) -> Boundary | SurrogateBoundary:
        """The boundary the points lie on."""
        return self._boundary

    @property
    def num_elements(self) -> int:
        """Number of elements of the patch holding a piece or face of the
        boundary."""
        return self._cpp_object.num_elements

    @property
    def num_points(self) -> int:
        """Number of points over all elements."""
        return self._cpp_object.num_points

    @property
    def faces(self) -> npt.NDArray[np.int32]:
        """Face of the boundary holding each point, of shape
        `(num_points,)`."""
        return np.array(self._cpp_object.faces)

    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """Positions of the points in the plane, of shape
        `(num_points, 2)`."""
        return self._cpp_object.positions(self._patch._cpp_object)

    @property
    def projections(self) -> npt.NDArray[np.float64]:
        """Closest points of the points on the true boundary, where the
        conditions take their data, of shape `(num_points, 2)`.

        On a quadrature that is not shifted, they are the positions
        themselves.
        """
        return self._cpp_object.projections(self._patch._cpp_object)

    @property
    def order(self) -> int | None:
        """Total order of the Taylor expansion of a shifted quadrature, or
        `None` if it is not shifted."""
        if not self._cpp_object.is_shifted:
            return None

        return self._cpp_object.order

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Weights of the points in the plane, of shape `(num_points,)`.

        They integrate along the boundary by length, so that the weights
        of a face add up to the length of its part over the patch.
        """
        return self._cpp_object.physical_weights(self._patch._cpp_object)

    @property
    def normals(self) -> npt.NDArray[np.float64]:
        """Unit normals at the points, pointing out of the domain, of shape
        `(num_points, 2)`."""
        return self._cpp_object.physical_normals(self._patch._cpp_object)

    def __repr__(self) -> str:
        return (f'BoundaryQuadrature(num_elements={self.num_elements}, '
                f'num_points={self.num_points})')


def _in_the_plane(boundary: object) -> bool:
    """Whether a boundary is made of curves in the plane."""
    return (isinstance(boundary, Boundary)
            and isinstance(boundary._cpp_object, _cpp.Boundary2d))
