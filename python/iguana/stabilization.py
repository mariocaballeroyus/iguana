# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Stabilizations, terms that keep a discrete problem well posed

A stabilization prescribes no data: it vanishes on the exact solution and
controls the field where the problem alone leaves it loose. The ghost
penalty extends the field smoothly across the faces of the cells beyond the
physical domain, so that a space keeping every function of the background
patch stays determined. The solve assembles it with the element.
"""

from __future__ import annotations

from collections.abc import Sequence

from iguana import cpp as _cpp
from iguana.element import PoissonElement
from iguana.embedding import CellType
from iguana.grid import HierarchicalGrid
from iguana.patch import PlanarPatch


class GhostPenalty:
    """Penalty on the jumps of a trace of an element across the faces of
    the cells beyond the physical domain."""

    def __init__(
        self,
        trace: PoissonElement.U,
        patch: PlanarPatch,
        cell_types: Sequence[CellType],
        penalty: float,
    ) -> None:
        """Initialize the ghost penalty on a trace.

        The penalty acts on the faces between the cells of a patch whose
        volume fractions are not both one, nor one and zero: the faces of
        the cut cells and those between outside cells. On each it
        penalizes the jump of the derivative of the trace along the normal,
        of the order p of the degree across the face, the only one that
        jumps for splines of maximal continuity, scaled by the size h of
        the cells as h^(2p - 1). It vanishes on any field that is one
        polynomial across the faces, and extends the field from the
        physical domain over every function the cells beyond it keep.

        Args:
            trace: The trace whose jumps are penalized, such as
                `PoissonElement().u`.
            patch: The background patch, a planar patch with an affine map,
                whose elements are the cells.
            cell_types: The type of each element of the patch, with the
                first direction running fastest.
            penalty: The penalty, positive, which the size of the cells
                scales.

        Raises:
            TypeError: If the trace is not the trace of an element, or if
                the patch is not a planar patch.
            ValueError: If there is not one cell type per element, or if
                the penalty is not positive.
        """
        if not isinstance(trace, PoissonElement.U):
            raise TypeError('the trace must be the trace of an element')

        if not isinstance(patch, PlanarPatch):
            raise TypeError('the patch must be a planar patch')

        cell_types = list(cell_types)
        num_elements = HierarchicalGrid(patch.degrees,
                                        patch.knots).num_elements

        if len(cell_types) != num_elements:
            raise ValueError('there must be one cell type per cell')

        if not penalty > 0.:
            raise ValueError('the penalty must be positive')

        # Along a face, the squared jumps are polynomials of twice the degree
        # along it, which one point more than the degree integrates exactly
        # on an affine map
        faces = _cpp.GhostFaces2d(
            patch=patch._cpp_object,
            classification=_cpp.CellClassification2d(cell_types))

        self._trace = trace
        self._patch = patch
        self._penalty = float(penalty)
        self._quadrature = _cpp.FaceQuadrature2d(
            faces=faces, num_points=max(patch.degrees) + 1)

    @property
    def trace(self) -> PoissonElement.U:
        """The trace whose jumps are penalized."""
        return self._trace

    @property
    def patch(self) -> PlanarPatch:
        """The background patch, whose cells the faces separate."""
        return self._patch

    @property
    def num_faces(self) -> int:
        """Number of faces the penalty acts on."""
        return self._quadrature.num_faces

    @property
    def penalty(self) -> float:
        """The penalty."""
        return self._penalty

    def _assemble(self, assembler: _cpp.Assembler2d) -> None:
        """Add the stiffness of the penalty into the compiled assembler of
        a planar space, built with its faces coupled."""
        penalty = _cpp.GhostPenalty2d(self._trace._cpp_trace(),
                                      self._penalty)

        assembler.assemble_stiffness(penalty, self._quadrature)

    def __repr__(self) -> str:
        return f'GhostPenalty({self._trace!r}, penalty={self._penalty})'
