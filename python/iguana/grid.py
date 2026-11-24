# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Hierarchical grids, elements refined level by level

A grid holds the elements of a spline space, the non-empty knot spans of
the knot vector of each direction. Level 0 holds these elements, and each
level halves every element of the one before in each direction. Refining
an element replaces it by its children on the next level, so the active
elements always tile the parameter box. They are numbered level by level
and, within a level, with the first direction running fastest, as the
elements of a patch on the same knots.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp


class HierarchicalGrid:
    """Elements of a spline space, refined level by level."""

    _cpp_object: _cpp.HierarchicalGrid2d | _cpp.HierarchicalGrid3d

    def __init__(self, degrees: Sequence[int],
                 knots: Sequence[npt.ArrayLike]) -> None:
        """Initialize the grid with every element on level 0.

        Args:
            degrees: Polynomial degree of each parametric direction.
            knots: Non-decreasing knot vector of each parametric direction.

        Raises:
            ValueError: If there are not two or three directions, if there
                is not one degree per knot vector, or if a knot vector is
                invalid for its degree.
        """
        if len(knots) != len(degrees):
            raise ValueError('there must be one degree per knot vector')

        if len(knots) == 3:
            self._cpp_object = _cpp.HierarchicalGrid3d(degrees, knots)
        elif len(knots) == 2:
            self._cpp_object = _cpp.HierarchicalGrid2d(degrees, knots)
        else:
            raise ValueError('there must be two or three directions')

        self._dimension = len(knots)

    @property
    def num_levels(self) -> int:
        """Number of levels, the finest one possibly without elements."""
        return self._cpp_object.num_levels

    @property
    def num_elements(self) -> int:
        """Number of active elements over all levels."""
        return self._cpp_object.num_elements

    @property
    def levels(self) -> npt.NDArray[np.int32]:
        """Level of each active element, of shape ``(num_elements,)``."""
        return self._cpp_object.levels

    @property
    def bounds(self) -> npt.NDArray[np.float64]:
        """Parameter box of each active element, of shape
        ``(num_elements, 2, dimension)``.

        ``bounds[:, 0]`` holds the parameters at which each element starts
        and ``bounds[:, 1]`` those at which it ends.
        """
        return self._cpp_object.bounds.reshape(-1, 2, self._dimension)

    def refine(self, elements: npt.ArrayLike) -> HierarchicalGrid:
        """Refine the grid, replacing elements by their children.

        Each marked element is replaced by its children on the next level,
        which halve it in each direction. The other elements stay active,
        although their indices may change. This grid is left unchanged.

        Args:
            elements: Indices of the active elements to refine, in any
                order and possibly repeated.

        Returns:
            The refined grid.

        Raises:
            ValueError: If an index lies outside [0, num_elements), or if a
                new level would have more elements than can be counted.
        """
        return _wrap(self._cpp_object.refine(np.asarray(elements).tolist()))

    def _refine_admissible(self, elements: npt.ArrayLike,
                           boxes: bool) -> HierarchicalGrid:
        """Refine the grid at elements, graded for the truncated
        hierarchical B-splines on it.

        Each marked element is replaced by its children, or by those of
        the aligned box of degree + 2 elements of its level along each
        direction that holds it when refining by boxes, the last box along
        a direction cut short where its count is not a multiple. Coarser
        elements, or boxes, are then refined until the functions non-zero
        on any element come from at most two consecutive levels.

        Args:
            elements: Indices of the active elements to refine, in any
                order and possibly repeated.
            boxes: Whether to refine the boxes holding the elements rather
                than the elements alone.

        Returns:
            The refined grid.

        Raises:
            ValueError: If an index lies outside [0, num_elements), or if a
                new level would have more elements than can be counted.
        """
        return _wrap(self._cpp_object.refine_admissible(
            np.asarray(elements).tolist(), boxes))

    def __eq__(self, other: object) -> bool:
        """Whether two grids have the same knots and the same active
        elements, level by level."""
        if not isinstance(other, HierarchicalGrid):
            return NotImplemented

        return (self._dimension == other._dimension
                and self._cpp_object == other._cpp_object)

    def __repr__(self) -> str:
        return (f'HierarchicalGrid(num_levels={self.num_levels}, '
                f'num_elements={self.num_elements})')


def _wrap(
    grid: _cpp.HierarchicalGrid2d | _cpp.HierarchicalGrid3d,
) -> HierarchicalGrid:
    """The Python grid holding a C++ grid."""
    wrapped = object.__new__(HierarchicalGrid)
    wrapped._cpp_object = grid
    wrapped._dimension = (3 if isinstance(grid, _cpp.HierarchicalGrid3d)
                          else 2)

    return wrapped
