# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Function spaces, the functions of a patch that carry unknowns

A function space holds the basis functions that a field is built from,
those of a patch or B-splines of other degrees on its knots. The standard
space gives one degree of freedom to each function active on a cell that is
not outside the physical domain, so that a function supported on outside
cells alone carries none. A field of several components, such as the
velocity and the pressure of a fluid, gives each degree of freedom a
coefficient per component, as many as the block size of the space. A mixed
space holds several fields instead, each a function space of its own
degrees, such as the velocity and the pressure of a divergence-conforming
space.
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
    """The standard space of a basis on the elements of a patch, one degree
    of freedom per function active on the physical domain."""

    _cpp_object: (_cpp.FunctionSpace2d | _cpp.FunctionSpace3d
                  | _cpp.NURBSFunctionSpace2d)

    def __init__(
        self,
        patch: PlanarPatch | SurfacePatch | VolumePatch,
        cell_types: Sequence[CellType] | None = None,
        *,
        block_size: int = 1,
        degrees: Sequence[int] | None = None,
    ) -> None:
        """Initialize the space of a basis on the elements of a patch.

        Without degrees, the basis of the patch spans the space. With them,
        the B-splines of those degrees on the knots of the patch do: the
        interior knots keep their multiplicities and the end knots repeat
        once per degree plus one, so that each degree raised above the
        patch adds an order of continuity across its knots, as the
        components of the velocity of a divergence-conforming space take
        them.

        Args:
            patch: The background patch, a planar, surface or volume
                patch, whose elements carry the space.
            cell_types: The type of each element of the patch, with the
                first direction running fastest. Without them, every cell
                lies inside and every function gets a degree of freedom.
            block_size: The number of components of the field, positive,
                such as the three of `StokesElement`.
            degrees: The degree of the basis along each direction, on the
                knots of the patch, which must be open. Without them, the
                degrees of the patch.

        Raises:
            TypeError: If the patch is not a planar, surface or volume
                patch, or if degrees are given for a NURBS patch, whose
                weights belong to its own basis.
            ValueError: If there is not one cell type per element, if the
                block size is not positive, or if the degrees are not one
                non-negative degree per direction on open knots.
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

        if block_size < 1:
            raise ValueError('the block size must be positive')

        if degrees is None:
            basis = patch._cpp_object.basis
            self._degrees = tuple(patch.degrees)
            self._knots = tuple(patch.knots)
        elif space is _cpp.NURBSFunctionSpace2d:
            raise TypeError('a NURBS patch spans its space with its own '
                            'degrees')
        else:
            basis, self._degrees, self._knots = _bspline_basis(patch,
                                                               degrees)

        self._patch = patch
        self._cpp_object = space(basis, classification(list(cell_types)),
                                 int(block_size))

    @property
    def patch(self) -> PlanarPatch | SurfacePatch | VolumePatch:
        """The background patch, whose basis spans the space."""
        return self._patch

    @property
    def degrees(self) -> tuple[int, ...]:
        """Degree of the basis of the space along each direction."""
        return self._degrees

    @property
    def knots(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Knots of the basis of the space along each direction."""
        return tuple(knots.copy() for knots in self._knots)

    @property
    def block_size(self) -> int:
        """Number of components of the field, the coefficients of each
        degree of freedom."""
        return self._cpp_object.block_size

    @property
    def num_dofs(self) -> int:
        """Number of degrees of freedom, one per function the space keeps,
        each with a coefficient per component of the field."""
        return self._cpp_object.num_dofs

    @property
    def functions(self) -> npt.NDArray[np.int64]:
        """Basis function of each degree of freedom, its index in the basis
        of the space, with the first direction running fastest."""
        return np.asarray(self._cpp_object.functions, dtype=np.int64)

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control point of the function of each degree of freedom, of
        shape `(num_dofs, 2)` on a planar patch and `(num_dofs, 3)`
        otherwise.

        Raises:
            ValueError: If the space has other degrees than its patch,
                whose control points belong to the basis of the patch.
        """
        if self._degrees != tuple(self._patch.degrees):
            raise ValueError('a space of other degrees than its patch has no '
                             'control points')

        return self._patch.control_points[self._cpp_object.functions]

    @property
    def boundary_dofs(self) -> npt.NDArray[np.int64]:
        """Degrees of freedom of the functions that do not vanish on the
        boundary of the patch, in increasing order. With open knot vectors,
        as those of `create_rectangle` and `create_box`, they are the
        functions first or last along some direction."""
        counts = [len(knots) - degree - 1
                  for degree, knots in zip(self._degrees, self._knots)]

        # Index of the function of each degree of freedom along each
        # direction, the first running fastest
        indices = np.unravel_index(self._cpp_object.functions, counts,
                                   order='F')
        on_boundary = np.zeros(self.num_dofs, dtype=bool)

        for index, count in zip(indices, counts):
            on_boundary |= (index == 0) | (index == count - 1)

        return np.flatnonzero(on_boundary)

    def __repr__(self) -> str:
        details = [f'num_dofs={self.num_dofs}']

        if self._degrees != tuple(self._patch.degrees):
            details.append(f'degrees={self._degrees}')

        if self.block_size != 1:
            details.append(f'block_size={self.block_size}')

        return 'FunctionSpace(' + ', '.join(details) + ')'


class MixedFunctionSpace:
    """Space of several fields on the elements of one planar patch, each a
    function space of its own degrees and block size, their degrees of
    freedom one field after another."""

    _cpp_object: _cpp.MixedFunctionSpace2d

    def __init__(self, fields: Sequence[FunctionSpace]) -> None:
        """Initialize the space of several fields.

        Each field takes a contiguous range of the degrees of freedom, from
        its offset, in the order of the fields.

        Args:
            fields: The spaces of the fields, in the order of their degrees
                of freedom, on one planar patch.

        Raises:
            TypeError: If a field is not a function space, or if the fields
                do not lie on a planar patch.
            ValueError: If there is no field, or if the fields do not lie on
                one patch.
        """
        fields = tuple(fields)

        if not fields:
            raise ValueError('a mixed space needs at least one field')

        if not all(isinstance(field, FunctionSpace) for field in fields):
            raise TypeError('the fields must be function spaces')

        patch = fields[0].patch

        if any(field.patch is not patch for field in fields):
            raise ValueError('the fields must lie on one patch')

        # The compiled mixed spaces are planar, as the problems are so far
        if not isinstance(patch, PlanarPatch):
            raise TypeError('the fields must lie on a planar patch')

        self._fields = fields
        self._cpp_object = _cpp.MixedFunctionSpace2d(
            [field._cpp_object for field in fields])

    @classmethod
    def divergence_conforming(
        cls,
        patch: PlanarPatch,
        cell_types: Sequence[CellType] | None = None,
    ) -> MixedFunctionSpace:
        """The divergence-conforming space of a fluid on a planar patch.

        Its fields are the components of the velocity and the pressure, in
        that order. The pressure takes the basis of the patch, of degrees
        (p, q), and each component of the velocity one degree and one order
        of continuity more along its own direction, (p + 1, q) and
        (p, q + 1). The divergence then maps the velocities onto the
        pressures, so that a velocity whose divergence is orthogonal to
        every pressure is divergence-free at every point.

        Args:
            patch: The background patch, a planar patch with open knots.
            cell_types: The type of each element of the patch, with the
                first direction running fastest. Without them, every cell
                lies inside and every function gets a degree of freedom.

        Returns:
            The space of the velocity and the pressure of the Galerkin
            `StokesElement`.

        Raises:
            TypeError: If the patch is not a planar patch.
            ValueError: If there is not one cell type per element, or if
                the knots of the patch are not open.
        """
        if not isinstance(patch, PlanarPatch):
            raise TypeError('the patch must be a planar patch')

        p, q = patch.degrees

        return cls([FunctionSpace(patch, cell_types, degrees=(p + 1, q)),
                    FunctionSpace(patch, cell_types, degrees=(p, q + 1)),
                    FunctionSpace(patch, cell_types)])

    @property
    def fields(self) -> tuple[FunctionSpace, ...]:
        """Spaces of the fields, in the order of their degrees of freedom."""
        return self._fields

    @property
    def patch(self) -> PlanarPatch:
        """The background patch, whose elements carry the fields."""
        return self._fields[0].patch

    @property
    def num_dofs(self) -> int:
        """Number of degrees of freedom of all the fields."""
        return self._cpp_object.num_dofs

    @property
    def offsets(self) -> tuple[int, ...]:
        """First degree of freedom of each field."""
        return tuple(self._cpp_object.offset(field)
                     for field in range(len(self._fields)))

    def __repr__(self) -> str:
        return (f'MixedFunctionSpace(num_dofs={self.num_dofs}, '
                f'num_fields={len(self._fields)})')


def _bspline_basis(
    patch: PlanarPatch | SurfacePatch | VolumePatch,
    degrees: Sequence[int],
) -> tuple[_cpp.BivariateBSpline | _cpp.TrivariateBSpline, tuple[int, ...],
           tuple[npt.NDArray[np.float64], ...]]:
    """B-splines of some degrees on the knots of a patch, with its interior
    knots and its ends repeated once per degree plus one, and their degrees
    and knots."""
    degrees = tuple(int(degree) for degree in degrees)

    if len(degrees) != len(patch.degrees) or min(degrees) < 0:
        raise ValueError('the degrees must be one non-negative degree per '
                         'direction')

    axes = []
    all_knots = []

    for degree, own, knots in zip(degrees, patch.degrees, patch.knots):
        # Open knots repeat each end once per degree plus one
        start, end = knots[:own + 1], knots[len(knots) - own - 1:]

        if np.any(start != knots[0]) or np.any(end != knots[-1]):
            raise ValueError('the knots of the patch must be open')

        interior = knots[own + 1:len(knots) - own - 1]
        knots = np.concatenate([np.full(degree + 1, knots[0]), interior,
                                np.full(degree + 1, knots[-1])])

        axes.append(_cpp.BSpline(degree, knots.tolist()))
        all_knots.append(knots)

    basis = (_cpp.BivariateBSpline(axes) if len(axes) == 2
             else _cpp.TrivariateBSpline(axes))

    return basis, degrees, tuple(all_knots)
