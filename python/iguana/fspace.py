# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Function spaces, the functions of a patch that carry unknowns

A function space holds the basis functions of a patch that a field is
built from. The standard space gives one degree of freedom to each function
active on a cell that is not outside the physical domain, so that a
function supported on outside cells alone carries none.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp
from iguana.grid import HierarchicalGrid
from iguana.embedding import CellType
from iguana.patch import PlanarPatch, SurfacePatch, VolumePatch


class FunctionSpace:
    """The standard space of a patch, one degree of freedom per function
    active on the physical domain."""

    _cpp_object: (_cpp.FunctionSpace2d | _cpp.FunctionSpace3d
                  | _cpp.NURBSFunctionSpace2d)

    def __init__(self, patch: PlanarPatch | SurfacePatch | VolumePatch,
                 cell_types: Sequence[CellType] | None = None) -> None:
        """Initialize the space of the basis of a patch.

        Args:
            patch: The background patch, a planar, surface or volume
                patch, whose basis spans the space.
            cell_types: The type of each element of the patch, with the
                first direction running fastest. Without them, every cell
                lies inside and every function gets a degree of freedom.

        Raises:
            TypeError: If the patch is not a planar, surface or volume
                patch.
            ValueError: If there is not one cell type per element.
        """
        if isinstance(patch, VolumePatch):
            space = _cpp.FunctionSpace3d
            classification = _cpp.CellClassification3d
        elif isinstance(patch, (PlanarPatch, SurfacePatch)):
            classification = _cpp.CellClassification2d

            # A NURBS surface spans its space with its rational functions
            if isinstance(patch._cpp_object, _cpp.NURBSSurfacePatch):
                space = _cpp.NURBSFunctionSpace2d
            else:
                space = _cpp.FunctionSpace2d
        else:
            raise TypeError('the patch must be a planar, surface or volume '
                            'patch')

        # The elements of the patch, as a grid that is not refined
        num_elements = HierarchicalGrid(patch.degrees,
                                          patch.knots).num_elements

        if cell_types is None:
            cell_types = [CellType.inside] * num_elements

        if len(cell_types) != num_elements:
            raise ValueError('there must be one cell type per cell')

        self._patch = patch
        self._cpp_object = space(patch._cpp_object.basis,
                                 classification(list(cell_types)))

    @property
    def patch(self) -> PlanarPatch | SurfacePatch | VolumePatch:
        """The background patch, whose basis spans the space."""
        return self._patch

    @property
    def num_dofs(self) -> int:
        """Number of degrees of freedom."""
        return self._cpp_object.num_dofs

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control point of the function of each degree of freedom, of
        shape `(num_dofs, 2)` on a planar patch and `(num_dofs, 3)`
        otherwise."""
        return self._patch.control_points[self._cpp_object.functions]

    @property
    def boundary_dofs(self) -> npt.NDArray[np.int64]:
        """Degrees of freedom of the functions that do not vanish on the
        boundary of the patch, in increasing order. With open knot vectors,
        as those of `create_rectangle` and `create_box`, they are the
        functions first or last along some direction."""
        counts = [len(knots) - degree - 1
                  for degree, knots in zip(self._patch.degrees,
                                           self._patch.knots)]

        # Index of the function of each degree of freedom along each
        # direction, the first running fastest
        indices = np.unravel_index(self._cpp_object.functions, counts,
                                   order='F')
        on_boundary = np.zeros(self.num_dofs, dtype=bool)

        for index, count in zip(indices, counts):
            on_boundary |= (index == 0) | (index == count - 1)

        return np.flatnonzero(on_boundary)

    def __repr__(self) -> str:
        return f'FunctionSpace(num_dofs={self.num_dofs})'
