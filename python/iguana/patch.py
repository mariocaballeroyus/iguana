# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Spline patches, the maps from a parameter box into physical space"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp


class CurvePatch:
    """A spline map from an interval into the plane or into space."""

    _cpp_object: _cpp.PlanarCurvePatch | _cpp.CurvePatch

    def __init__(self,
                 curve: _cpp.PlanarCurvePatch | _cpp.CurvePatch) -> None:
        """Initialize the curve from a C++ curve.

        Args:
            curve: A C++ curve patch.
        """
        self._cpp_object = curve

    @property
    def degree(self) -> int:
        """Polynomial degree of the curve."""
        return self._cpp_object.basis.axis(0).degree

    @property
    def knots(self) -> npt.NDArray[np.float64]:
        """Knot vector of the curve."""
        return np.asarray(self._cpp_object.basis.axis(0).knots, float)

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control points, of shape `(num_control_points, 2)` in the plane
        and `(num_control_points, 3)` in space."""
        return self._cpp_object.coefficients

    def __repr__(self) -> str:
        return (f'CurvePatch(degree={self.degree}, '
                f'num_control_points={len(self.control_points)})')


class PlanarPatch:
    """A spline map from a two-dimensional parameter box onto a planar
    region."""

    _cpp_object: _cpp.PlanarPatch

    def __init__(self, patch: _cpp.PlanarPatch) -> None:
        """Initialize the patch from a C++ patch.

        Args:
            patch: A C++ planar patch object.
        """
        self._cpp_object = patch

    @property
    def degrees(self) -> tuple[int, ...]:
        """Polynomial degree of each parametric direction."""
        basis = self._cpp_object.basis

        return tuple(basis.axis(direction).degree
                     for direction in range(2))

    @property
    def knots(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Knot vector of each parametric direction."""
        basis = self._cpp_object.basis

        return tuple(np.asarray(basis.axis(direction).knots, float)
                     for direction in range(2))

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control points, of shape `(num_control_points, 2)`.

        They are numbered with the first direction running fastest.
        """
        return self._cpp_object.coefficients

    def isocurves(self) -> list[CurvePatch]:
        """Isocurves of the patch along all of its knot lines, in the plane.

        The curves pinned in the first direction come first, then those
        pinned in the second, each group in increasing order of knot line.
        """
        return [CurvePatch(curve) for curve in self._cpp_object.isocurves()]

    def __repr__(self) -> str:
        return (f'PlanarPatch(degrees={self.degrees}, '
                f'num_control_points={len(self.control_points)})')


class SurfacePatch:
    """A spline map from a two-dimensional parameter box into space."""

    _cpp_object: _cpp.SurfacePatch

    def __init__(self, patch: _cpp.SurfacePatch) -> None:
        """Initialize the patch from a C++ patch.

        Args:
            patch: A C++ patch object.
        """
        self._cpp_object = patch

    @property
    def degrees(self) -> tuple[int, ...]:
        """Polynomial degree of each parametric direction."""
        basis = self._cpp_object.basis

        return tuple(basis.axis(direction).degree
                     for direction in range(2))

    @property
    def knots(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Knot vector of each parametric direction."""
        basis = self._cpp_object.basis

        return tuple(np.asarray(basis.axis(direction).knots, float)
                     for direction in range(2))

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control points, of shape `(num_control_points, 3)`.

        They are numbered with the first direction running fastest.
        """
        return self._cpp_object.coefficients

    def isocurves(self) -> list[CurvePatch]:
        """Isocurves of the patch along all of its knot lines.

        The curves pinned in the first direction come first, then those
        pinned in the second, each group in increasing order of knot line.
        """
        return [CurvePatch(curve) for curve in self._cpp_object.isocurves()]

    def __repr__(self) -> str:
        return (f'SurfacePatch(degrees={self.degrees}, '
                f'num_control_points={len(self.control_points)})')


class VolumePatch:
    """A spline map from a three-dimensional parameter box into space."""

    _cpp_object: _cpp.VolumePatch

    def __init__(self, patch: _cpp.VolumePatch) -> None:
        """Initialize the patch from a C++ patch.

        Args:
            patch: A C++ patch object.
        """
        self._cpp_object = patch

    @property
    def degrees(self) -> tuple[int, ...]:
        """Polynomial degree of each parametric direction."""
        basis = self._cpp_object.basis

        return tuple(basis.axis(direction).degree
                     for direction in range(3))

    @property
    def knots(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Knot vector of each parametric direction."""
        basis = self._cpp_object.basis

        return tuple(np.asarray(basis.axis(direction).knots, float)
                     for direction in range(3))

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control points, of shape `(num_control_points, 3)`.

        A read-only view of the patch, not a copy. It stays valid for as
        long as the patch does.
        """
        return self._cpp_object.coefficients

    def isosurfaces(self) -> list[SurfacePatch]:
        """Isosurfaces of the patch along all of its knot planes.

        The surfaces pinned in the first direction come first, then those
        pinned in the second and third, each group in increasing order of
        knot plane. Each surface keeps the remaining two directions, in
        increasing order.
        """
        return [SurfacePatch(surface)
                for surface in self._cpp_object.isosurfaces()]

    def __repr__(self) -> str:
        return (f'VolumePatch(degrees={self.degrees}, '
                f'num_control_points={len(self.control_points)})')


def _uniform_knots(degree: int, elements: int) -> npt.NDArray[np.float64]:
    """Clamped knot vector of a uniform basis on the unit interval."""
    interior = np.linspace(0., 1., elements + 1)[1:-1]

    return np.concatenate([np.zeros(degree + 1), interior,
                           np.ones(degree + 1)])


def _greville(degree: int,
              knots: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Greville abscissa of every function of a univariate basis."""
    return np.array([knots[i + 1:i + degree + 1].mean()
                     for i in range(len(knots) - degree - 1)])


def _block(
    dimension: int,
    lengths: Sequence[float],
    elements: Sequence[int],
    degrees: Sequence[int],
    origin: Sequence[float],
) -> tuple[list[_cpp.BSpline], npt.NDArray[np.float64]]:
    """Axes and control points of the affine map of an axis-aligned block.

    Sampling an affine map at the Greville abscissae gives control points
    that reproduce it exactly.

    Raises:
        ValueError: If an argument does not hold one entry per direction.
    """
    given = (lengths, elements, degrees, origin)

    if any(len(argument) != dimension for argument in given):
        raise ValueError(f'every argument must hold {dimension} entries, '
                         'one per direction')

    knots = [_uniform_knots(degree, span)
             for degree, span in zip(degrees, elements)]

    nodes = np.meshgrid(*(_greville(degree, knot)
                          for degree, knot in zip(degrees, knots)),
                        indexing='ij')

    box = np.column_stack([node.ravel(order='F') for node in nodes])
    points = np.asarray(origin, float) + box * np.asarray(lengths, float)

    axes = [_cpp.BSpline(degree=degree, knots=list(knot))
            for degree, knot in zip(degrees, knots)]

    return axes, points


def create_rectangle(
    lengths: Sequence[float],
    elements: Sequence[int],
    degrees: Sequence[int] = (2, 2),
    origin: Sequence[float] = (0., 0.),
) -> PlanarPatch:
    """Create a patch mapping the parameter box onto a rectangle.

    The rectangle is axis-aligned and its basis uniform. The map is affine,
    so the patch reproduces the rectangle exactly rather than approximating
    it.

    Args:
        lengths: Side of the rectangle along each direction.
        elements: Number of knot spans of each direction.
        degrees: Polynomial degree of each direction.
        origin: Corner of the rectangle the parameter box maps its own
            origin to.

    Returns:
        The patch of the rectangle.

    Raises:
        ValueError: If an argument does not hold two entries.
    """
    axes, points = _block(2, lengths, elements, degrees, origin)
    basis = _cpp.BivariateBSpline(axes=axes)

    return PlanarPatch(_cpp.PlanarPatch(basis=basis, coefficients=points))


def create_box(
    lengths: Sequence[float],
    elements: Sequence[int],
    degrees: Sequence[int] = (2, 2, 2),
    origin: Sequence[float] = (0., 0., 0.),
) -> VolumePatch:
    """Create a patch mapping the parameter box onto a block.

    The block is axis-aligned and its basis uniform. The map is affine,
    so sampling it at the Greville abscissae reproduces the block
    exactly rather than approximating it.

    Args:
        lengths: Side of the block along each direction.
        elements: Number of knot spans of each direction.
        degrees: Polynomial degree of each direction.
        origin: Corner of the block the parameter box maps its own
            origin to.

    Returns:
        The patch of the block.

    Raises:
        ValueError: If an argument does not hold three entries.
    """
    axes, points = _block(3, lengths, elements, degrees, origin)
    basis = _cpp.TrivariateBSpline(axes=axes)

    return VolumePatch(_cpp.VolumePatch(basis=basis, coefficients=points))


def create_curve(
    degree: int,
    knots: npt.ArrayLike,
    control_points: npt.ArrayLike,
) -> CurvePatch:
    """Create the patch of a B-spline curve from its control points.

    The curve lies in the plane or in space, as its control points have two
    or three coordinates.

    Args:
        degree: Polynomial degree of the curve.
        knots: Full, non-decreasing knot vector.
        control_points: Control points, of shape `(num_control_points, 2)`
            in the plane or `(num_control_points, 3)` in space.

    Returns:
        The patch of the curve.

    Raises:
        ValueError: If the control points do not have two or three
            coordinates, if the knot vector is invalid for the degree, or if
            there is not one control point per basis function.
    """
    points = np.asarray(control_points, float)

    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError('the control points must have two or three '
                         'coordinates')

    basis = _cpp.UnivariateBSpline(
        axes=[_cpp.BSpline(degree=degree,
                           knots=np.asarray(knots, float).tolist())])

    if points.shape[1] == 2:
        curve = _cpp.PlanarCurvePatch(basis=basis, coefficients=points)
    else:
        curve = _cpp.CurvePatch(basis=basis, coefficients=points)

    return CurvePatch(curve)


def create_surface(
    degrees: Sequence[int],
    knots: Sequence[npt.ArrayLike],
    control_points: npt.ArrayLike,
) -> SurfacePatch:
    """Create the patch of a B-spline surface from its control net.

    Args:
        degrees: Polynomial degree of each parametric direction.
        knots: Full, non-decreasing knot vector of each direction.
        control_points: Control points, of shape
            `(num_control_points, 3)`, numbered with the first direction
            running fastest.

    Returns:
        The patch of the surface.

    Raises:
        ValueError: If there are not two degrees and two knot vectors, if
            a knot vector is invalid for its degree, or if there is not
            one control point per basis function.
    """
    if len(degrees) != 2 or len(knots) != 2:
        raise ValueError('there must be two degrees and two knot vectors, '
                         'one per direction')

    basis = _cpp.BivariateBSpline(
        axes=[_cpp.BSpline(degree=degree,
                           knots=np.asarray(knot, float).tolist())
              for degree, knot in zip(degrees, knots)])

    points = np.asarray(control_points, float).reshape(-1, 3)

    return SurfacePatch(_cpp.SurfacePatch(basis=basis, coefficients=points))
