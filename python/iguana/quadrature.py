# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Quadratures, the points integrating over the cells of a patch

A quadrature is filled one cell type at a time, each with a rule of its
own, such as Gauss-Legendre on the inside cells.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp
from iguana.domain import HierarchicalDomain
from iguana.embedding import CellType
from iguana.patch import SurfacePatch, VolumePatch


class DomainQuadrature:
    """The quadrature points over the cells of a patch."""

    _cpp_object: _cpp.SurfaceQuadrature | _cpp.VolumeQuadrature

    def __init__(self, patch: SurfacePatch | VolumePatch,
                 cell_types: Sequence[CellType] | None = None,
                 domain: HierarchicalDomain | None = None) -> None:
        """Initialize an empty quadrature over the cells of a patch.

        Args:
            patch: The background patch, a surface or volume patch, which
                places the points in space.
            cell_types: The type of each cell, in the numbering of the
                domain. Without them, every cell lies inside.
            domain: A hierarchical domain on the knots of the patch, whose
                active elements are the cells. Without it, the cells are
                the elements of the patch, with the first direction running
                fastest.

        Raises:
            TypeError: If the patch is neither a surface nor a volume patch,
                or if the domain is not a hierarchical domain.
            ValueError: If the domain does not lie on the knots of the
                patch, or if there is not one cell type per cell.
        """
        if isinstance(patch, VolumePatch):
            self._cpp_object = _cpp.VolumeQuadrature()
            embedding = _cpp.VolumeEmbedding
            self._dimension = 3
        elif isinstance(patch, SurfacePatch):
            self._cpp_object = _cpp.SurfaceQuadrature()
            embedding = _cpp.SurfaceEmbedding
            self._dimension = 2
        else:
            raise TypeError('the patch must be a surface or volume patch')

        if domain is None:
            # The elements of the patch, as a domain that is not refined
            domain = HierarchicalDomain(patch.degrees, patch.knots)
        elif not isinstance(domain, HierarchicalDomain):
            raise TypeError('the domain must be a hierarchical domain')
        elif domain._dimension != self._dimension:
            raise ValueError('the domain must lie on the knots of the '
                             'patch')
        else:
            # Positions of the empty quadrature only check that the domain
            # lies on the knots of the patch
            self._cpp_object.positions(patch._cpp_object,
                                       domain._cpp_object)

        num_elements = domain.num_elements

        if cell_types is None:
            cell_types = [CellType.inside] * num_elements

        if len(cell_types) != num_elements:
            raise ValueError('there must be one cell type per cell')

        self._patch = patch
        self._domain = domain
        self._cell_types = list(cell_types)
        self._embedding = embedding(self._cell_types)

    @property
    def patch(self) -> SurfacePatch | VolumePatch:
        """The background patch, which places the points in space."""
        return self._patch

    @property
    def domain(self) -> HierarchicalDomain:
        """The domain whose active elements are the cells."""
        return self._domain

    @property
    def cell_types(self) -> list[CellType]:
        """Type of each cell, in the numbering of the domain."""
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
        """Positions of the points in space, of shape ``(num_points, 3)``."""
        return self._cpp_object.positions(self._patch._cpp_object,
                                          self._domain._cpp_object)

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Weights of the points in parameter space, of shape
        ``(num_points,)``."""
        return self._cpp_object.weights

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
            self._domain._cpp_object, self._embedding, cell_type,
            list(num_points))

    def fill_moment_fitting(self, cell_type: CellType,
                            vertices: npt.ArrayLike,
                            triangles: npt.ArrayLike,
                            order: int = 2) -> None:
        """Fill the cells of one type with rules fitted to a solid.

        The rule of each cell integrates the Legendre polynomials up to the
        order in each direction over its part inside the solid, with
        positive weights at points inside it. The patch must map its
        parameter box onto an axis-aligned block, as create_box builds it,
        through which the vertices map back to parameters.

        Args:
            cell_type: The type of the cells to fill, usually the cut cells.
            vertices: Vertices of a closed triangle mesh of the solid in
                space, of shape ``(num_vertices, 3)``.
            triangles: Vertices of each triangle, counterclockwise seen from
                outside, of shape ``(num_triangles, 3)``.
            order: Highest polynomial degree of each direction.

        Raises:
            TypeError: If the patch is not a volume patch.
            ValueError: If the mesh is malformed, if the order lies outside
                [0, 4], or if the cells of this type are already filled.
        """
        if self._dimension != 3:
            raise TypeError('moment fitting needs a volume patch')

        parameters = _to_parameters(self._patch, vertices)

        self._cpp_object.fill_moment_fitting(
            self._domain._cpp_object, self._embedding, cell_type,
            parameters, np.asarray(triangles, dtype=np.int32), order)

    def __repr__(self) -> str:
        return (f'DomainQuadrature(num_elements={self.num_elements}, '
                f'num_points={self.num_points})')


def _to_parameters(patch: VolumePatch,
                   points: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Parameters of points in space under the map of an axis-aligned block.

    The corners of such a block are its extreme control points, so each
    coordinate scales back onto the knot range of its direction.
    """
    points = np.asarray(points, dtype=float)
    net = patch.control_points
    low, high = net.min(axis=0), net.max(axis=0)

    first = np.array([knots[0] for knots in patch.knots])
    last = np.array([knots[-1] for knots in patch.knots])

    return first + (points - low) / (high - low) * (last - first)
