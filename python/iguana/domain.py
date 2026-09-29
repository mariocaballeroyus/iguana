# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Hierarchical domains, elements refined level by level

A domain holds the elements of a spline space, the non-empty knot spans of
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


class HierarchicalDomain:
    """Elements of a spline space, refined level by level."""

    _cpp_object: _cpp.SurfaceHierarchicalDomain | _cpp.VolumeHierarchicalDomain

    def __init__(self, degrees: Sequence[int],
                 knots: Sequence[npt.ArrayLike]) -> None:
        """Initialize the domain with every element on level 0.

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
            self._cpp_object = _cpp.VolumeHierarchicalDomain(degrees, knots)
        elif len(knots) == 2:
            self._cpp_object = _cpp.SurfaceHierarchicalDomain(degrees, knots)
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

    def refine(self, elements: npt.ArrayLike) -> HierarchicalDomain:
        """Refine the domain, replacing elements by their children.

        Each marked element is replaced by its children on the next level,
        which halve it in each direction. The other elements stay active,
        although their indices may change. This domain is left unchanged.

        Args:
            elements: Indices of the active elements to refine, in any
                order and possibly repeated.

        Returns:
            The refined domain.

        Raises:
            ValueError: If an index lies outside [0, num_elements), or if a
                new level would have more elements than can be counted.
        """
        refined = object.__new__(HierarchicalDomain)
        refined._cpp_object = self._cpp_object.refine(
            np.asarray(elements).tolist())
        refined._dimension = self._dimension

        return refined

    def __repr__(self) -> str:
        return (f'HierarchicalDomain(num_levels={self.num_levels}, '
                f'num_elements={self.num_elements})')
