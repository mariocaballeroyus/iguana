# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Domains, the physical domain discretized over the elements of a patch

CellType tells where an element lies with respect to the physical domain:
CellType.inside, CellType.outside or CellType.cut. The type is geometric
only; how each type of cell is integrated is up to the method built on
the domain.
"""

from __future__ import annotations

from collections.abc import Sequence

from iguana import cpp as _cpp
from iguana.patch import SurfacePatch, VolumePatch

CellType = _cpp.CellType


class TensorDomain:
    """The physical domain over the elements of a tensor-product patch."""

    _cpp_object: _cpp.SurfaceDomain | _cpp.VolumeDomain

    def __init__(
        self,
        patch: SurfacePatch | VolumePatch,
        cell_types: Sequence[CellType] | None = None,
    ) -> None:
        """Initialize the domain over the elements of a patch.

        Args:
            patch: The background patch, a surface or volume patch.
            cell_types: The type of each element, with the first direction
                running fastest. Without them, every element lies inside.

        Raises:
            TypeError: If the patch is neither a surface nor a volume patch.
            ValueError: If there is not one cell type per element.
        """
        if isinstance(patch, VolumePatch):
            domain = _cpp.VolumeDomain
        elif isinstance(patch, SurfacePatch):
            domain = _cpp.SurfaceDomain
        else:
            raise TypeError('the patch must be a surface or volume patch')

        self._patch = patch

        if cell_types is None:
            self._cpp_object = domain(patch._cpp_object)
        else:
            self._cpp_object = domain(patch._cpp_object, list(cell_types))

    @property
    def patch(self) -> SurfacePatch | VolumePatch:
        """The background patch."""
        return self._patch

    @property
    def num_elements(self) -> int:
        """Number of elements of the patch."""
        return self._cpp_object.num_elements

    @property
    def cell_types(self) -> list[CellType]:
        """Type of each element, with the first direction running fastest."""
        return [self._cpp_object.cell_type(element)
                for element in range(self.num_elements)]

    def __repr__(self) -> str:
        types = self.cell_types
        counts = ', '.join(f'{name}={types.count(getattr(CellType, name))}'
                           for name in ('inside', 'cut', 'outside'))

        return f'TensorDomain(num_elements={self.num_elements}, {counts})'
