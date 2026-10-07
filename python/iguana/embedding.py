# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Embedded domains, how the physical domain lies on the elements of a patch

CellType tells where an element lies with respect to the physical domain:
CellType.inside, CellType.outside or CellType.cut. The type is geometric
only; how each type of cell is integrated is up to the method built on
the domain. A surrogate boundary bounds the inside cells, where a shifted
boundary method imposes the conditions of the physical boundary.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp
from iguana.grid import HierarchicalGrid
from iguana.patch import PlanarPatch

CellType = _cpp.CellType


class SurrogateBoundary:
    """The boundary of the inside cells of a patch, where a shifted boundary
    method imposes the conditions of the physical boundary."""

    _cpp_object: _cpp.SurrogateBoundary2d

    def __init__(self, patch: PlanarPatch,
                 cell_types: Sequence[CellType]) -> None:
        """Initialize the surrogate boundary of the inside cells of a patch.

        The inside cells form the surrogate domain, and its boundary runs
        along knot lines, over the faces between an inside cell and a cell
        that is not inside or lies past the edge of the patch, holes
        included. The space and the domain quadrature of the surrogate
        domain take the same cells, with every cut cell outside: a cut cell
        left in the space gives degrees of freedom that nothing integrates.

        Args:
            patch: The background patch, a planar patch whose elements are
                the cells.
            cell_types: The type of each element of the patch, with the
                first direction running fastest.

        Raises:
            TypeError: If the patch is not a planar patch.
            ValueError: If there is not one cell type per element.
        """
        if not isinstance(patch, PlanarPatch):
            raise TypeError('the patch must be a planar patch')

        cell_types = list(cell_types)
        num_elements = HierarchicalGrid(patch.degrees,
                                        patch.knots).num_elements

        if len(cell_types) != num_elements:
            raise ValueError('there must be one cell type per cell')

        self._patch = patch
        self._cpp_object = _cpp.SurrogateBoundary2d(
            patch=patch._cpp_object, domain=_cpp.EmbeddedDomain2d(cell_types))

    @property
    def patch(self) -> PlanarPatch:
        """The background patch, whose inside cells the boundary bounds."""
        return self._patch

    @property
    def num_faces(self) -> int:
        """Number of faces, the sides of the inside cells along the
        boundary."""
        return self._cpp_object.num_faces

    @property
    def elements(self) -> npt.NDArray[np.int32]:
        """Element of the patch holding each face, an inside cell, of shape
        `(num_faces,)`."""
        return np.array(self._cpp_object.elements)

    def __repr__(self) -> str:
        return f'SurrogateBoundary(num_faces={self.num_faces})'
