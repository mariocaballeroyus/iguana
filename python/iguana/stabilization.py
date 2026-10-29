# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Stabilizations, terms that keep a discrete problem well posed

A stabilization prescribes no data: it vanishes on the exact solution and
controls the field where the problem alone leaves it loose. The ghost
penalty extends the field smoothly across the faces of the cells beyond the
physical domain, so that a space keeping every function of the background
patch stays determined, on the jumps of the derivatives across the faces or
on the polynomials of their cells. A problem assembles it with the element.
"""

from __future__ import annotations

from collections.abc import Sequence

from iguana import cpp as _cpp
from iguana.element import PoissonElement, StokesElement
from iguana.embedding import CellType
from iguana.grid import HierarchicalGrid
from iguana.patch import PlanarPatch
from iguana.quadrature import QuadratureDomain


# The traces a ghost penalty extends
_TRACES = (PoissonElement.U, StokesElement.U, StokesElement.P)


def _ghost_faces(
    trace: PoissonElement.U | StokesElement.U | StokesElement.P,
    patch: PlanarPatch,
    cell_types: Sequence[CellType],
    penalty: float,
    faces: str,
) -> list[int]:
    """The faces of a ghost penalty, checked as its initialization
    documents, in the numbering of the grid of the patch."""
    if not isinstance(trace, _TRACES):
        raise TypeError('the trace must be the trace of an element')

    if not isinstance(patch, PlanarPatch):
        raise TypeError('the patch must be a planar patch')

    cell_types = list(cell_types)
    num_elements = HierarchicalGrid(patch.degrees, patch.knots).num_elements

    if len(cell_types) != num_elements:
        raise ValueError('there must be one cell type per cell')

    if not penalty > 0.:
        raise ValueError('the penalty must be positive')

    if faces not in ('all', 'cut', 'outside'):
        raise ValueError('the faces must be all, cut or outside')

    return _cpp.ghost_faces(
        patch=patch._cpp_object,
        classification=_cpp.CellClassification2d(cell_types),
        faces=getattr(_cpp.GhostFaces, faces))


class GhostPenalty:
    """Penalty on the jumps of a trace of an element across the faces of
    the cells beyond the physical domain."""

    def __init__(
        self,
        trace: PoissonElement.U | StokesElement.U | StokesElement.P,
        patch: PlanarPatch,
        cell_types: Sequence[CellType],
        penalty: float,
        faces: str = 'all',
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
        physical domain over every function the cells beyond it keep. The
        patch must have simple interior knots, or lower derivatives would
        jump too, which its problem rejects when it solves.

        The penalty may instead act on either kind of face alone, so that
        two penalties of different strengths share the faces, such as a
        weaker one deep in the void. Without one on the faces between
        outside cells, the functions of those cells alone are left free.

        The traces of a fluid scale the penalty as their fields need: the
        velocity by the viscosity, on each of its components, and the
        pressure by minus h^2 over the viscosity, on the side of the
        pressure block of the symmetric Stokes form, negative with a
        stabilization and zero without. The penalty given stays positive
        for every trace.

        Args:
            trace: The trace whose jumps are penalized, such as the field u
                of `PoissonElement`, or the velocity u or the pressure p of
                `StokesElement`.
            patch: The background patch, a planar patch with an affine map,
                whose elements are the cells.
            cell_types: The type of each element of the patch, with the
                first direction running fastest.
            penalty: The penalty, positive, which the size of the cells
                scales.
            faces: The faces it acts on: `'all'`, `'cut'`, those of the cut
                cells, or `'outside'`, those between two outside cells.

        Raises:
            TypeError: If the trace is not the trace of an element, or if
                the patch is not a planar patch.
            ValueError: If there is not one cell type per element, if the
                penalty is not positive, or if the faces are not all, cut
                or outside.
        """
        selected = _ghost_faces(trace, patch, cell_types, penalty, faces)

        self._trace = trace
        self._patch = patch
        self._penalty = float(penalty)
        self._faces = faces

        # Along a face, the squared jumps are polynomials of twice the degree
        # along it, which one point more than the degree integrates exactly
        # on an affine map, for fields up to one degree above the patch, as
        # those of a divergence-conforming space
        self._quadrature = _cpp.QuadratureSkeleton2d(
            patch=patch._cpp_object, faces=selected,
            num_points=max(patch.degrees) + 2)

    @property
    def trace(self) -> PoissonElement.U | StokesElement.U | StokesElement.P:
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

    @property
    def faces(self) -> str:
        """The faces it acts on, `'all'`, `'cut'` or `'outside'`."""
        return self._faces

    def _add_to(self, form: _cpp.Form2d) -> None:
        """Add the integral of the penalty to the compiled form of a planar
        problem, assembled with its faces coupled."""
        factor, scaling = self._trace._ghost_scaling()

        for trace in self._trace._cpp_traces():
            penalty = _cpp.SkeletonGhostPenalty2d(trace,
                                                  factor * self._penalty,
                                                  scaling)
            form.add_skeleton_integral(penalty, self._quadrature)

    def __repr__(self) -> str:
        faces = '' if self._faces == 'all' else f', faces={self._faces!r}'
        return (f'GhostPenalty({self._trace!r}, '
                f'penalty={self._penalty}{faces})')


class VolumeGhostPenalty:
    """Penalty on the difference between the polynomials of a trace of an
    element on the two cells of each face beyond the physical domain."""

    def __init__(
        self,
        trace: PoissonElement.U | StokesElement.U | StokesElement.P,
        patch: PlanarPatch,
        cell_types: Sequence[CellType],
        penalty: float,
        faces: str = 'all',
    ) -> None:
        """Initialize the volume ghost penalty on a trace.

        The penalty acts on the faces `GhostPenalty` acts on. On each it
        penalizes, over both of its cells, the difference between the
        polynomial of the trace on a cell and that of the other cell
        extended into it, scaled by the size h of the cells as h^-2. It
        vanishes on any field that is one polynomial across the faces, and
        needs no derivatives across them, so it holds on patches whose knots
        repeat or whose map is not affine. Integrated, the polynomials give
        the scaling with the degree p the face penalty takes: with simple
        interior knots on equal cells, it is `GhostPenalty` with the penalty
        2 / ((p!)^2 (2p + 1)) times this one.

        The traces of a fluid scale the penalty as for `GhostPenalty`, and
        it may act on either kind of face alone in the same way.

        Args:
            trace: The trace whose polynomials are compared, such as the
                field u of `PoissonElement`, or the velocity u or the
                pressure p of `StokesElement`.
            patch: The background patch, a planar patch, whose elements are
                the cells.
            cell_types: The type of each element of the patch, with the
                first direction running fastest.
            penalty: The penalty, positive, which the size of the cells
                scales.
            faces: The faces it acts on: `'all'`, `'cut'`, those of the cut
                cells, or `'outside'`, those between two outside cells.

        Raises:
            TypeError: If the trace is not the trace of an element, or if
                the patch is not a planar patch.
            ValueError: If there is not one cell type per element, if the
                penalty is not positive, or if the faces are not all, cut
                or outside.
        """
        selected = _ghost_faces(trace, patch, cell_types, penalty, faces)

        self._trace = trace
        self._patch = patch
        self._penalty = float(penalty)
        self._faces = faces

        # Only the cells of the faces matter, so one point per face
        self._faces_quadrature = _cpp.QuadratureSkeleton2d(
            patch=patch._cpp_object, faces=selected, num_points=1)

        # On a cell, the squared difference of two polynomials has twice
        # their degree along each direction, which one point more than the
        # degree integrates exactly on an affine map, for fields up to one
        # degree above the patch, as those of a divergence-conforming space
        self._cells = QuadratureDomain(patch)
        self._cells.fill_gauss_legendre(CellType.inside,
                                        max(patch.degrees) + 2)

    @property
    def trace(self) -> PoissonElement.U | StokesElement.U | StokesElement.P:
        """The trace whose polynomials are compared."""
        return self._trace

    @property
    def patch(self) -> PlanarPatch:
        """The background patch, whose cells the faces separate."""
        return self._patch

    @property
    def num_faces(self) -> int:
        """Number of faces the penalty acts on."""
        return self._faces_quadrature.num_faces

    @property
    def penalty(self) -> float:
        """The penalty."""
        return self._penalty

    @property
    def faces(self) -> str:
        """The faces it acts on, `'all'`, `'cut'` or `'outside'`."""
        return self._faces

    def _add_to(self, form: _cpp.Form2d) -> None:
        """Add the integral of the penalty to the compiled form of a planar
        problem, assembled with its faces coupled."""
        factor, scaling = self._trace._ghost_scaling()

        for trace in self._trace._cpp_traces():
            penalty = _cpp.VolumeGhostPenalty2d(trace,
                                                factor * self._penalty,
                                                scaling)
            form.add_skeleton_integral(penalty, self._faces_quadrature,
                                       self._cells._cpp_object)

    def __repr__(self) -> str:
        faces = '' if self._faces == 'all' else f', faces={self._faces!r}'
        return (f'VolumeGhostPenalty({self._trace!r}, '
                f'penalty={self._penalty}{faces})')
