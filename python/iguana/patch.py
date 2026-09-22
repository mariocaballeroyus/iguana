# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Spline patches, the maps from a parameter box into physical space"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp


class CurvePatch:
    """A spline map from an interval into space."""

    _cpp_object: _cpp.CurvePatch

    def __init__(
        self,
        curve: _cpp.CurvePatch,
        degree: int,
        knots: npt.NDArray[np.float64],
    ) -> None:
        """Initialize the curve from a C++ curve.

        Args:
            curve: A C++ curve patch.
            degree: Polynomial degree of the curve.
            knots: Knot vector of the curve.
        """
        self._cpp_object = curve
        self._degree = degree
        self._knots = knots

    @property
    def degree(self) -> int:
        """Polynomial degree of the curve."""
        return self._degree

    @property
    def knots(self) -> npt.NDArray[np.float64]:
        """Knot vector of the curve."""
        return self._knots

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control points, of shape ``(num_control_points, 3)``."""
        return self._cpp_object.coefficients

    def __repr__(self) -> str:
        return (f'CurvePatch(degree={self._degree}, '
                f'num_control_points={len(self.control_points)})')


class VolumePatch:
    """A spline map from a three-dimensional parameter box into space."""

    _cpp_object: _cpp.VolumePatch

    def __init__(
        self,
        patch: _cpp.VolumePatch,
        degrees: Sequence[int],
        knots: Sequence[Sequence[float]],
    ) -> None:
        """Initialize the patch from a C++ patch.

        Args:
            patch: A C++ patch object.
            degrees: Polynomial degree of each parametric direction.
            knots: Knot vector of each parametric direction.
        """
        self._cpp_object = patch
        self._degrees = tuple(degrees)
        self._knots = tuple(np.asarray(k, float) for k in knots)

    @property
    def degrees(self) -> tuple[int, ...]:
        """Polynomial degree of each parametric direction."""
        return self._degrees

    @property
    def knots(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Knot vector of each parametric direction."""
        return self._knots

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control points, of shape ``(num_control_points, 3)``.

        A read-only view of the patch, not a copy. It stays valid for as
        long as the patch does.
        """
        return self._cpp_object.coefficients

    def isocurves(self) -> list[CurvePatch]:
        """Isocurves of the patch along every knot line."""
        curves = []

        for direction, group in enumerate(self._cpp_object.isocurves()):
            degree = self._degrees[direction]
            knots = self._knots[direction]

            curves.extend(CurvePatch(curve, degree, knots)
                          for curve in group)

        return curves

    def __repr__(self) -> str:
        return (f'VolumePatch(degrees={self._degrees}, '
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
    given = (lengths, elements, degrees, origin)

    if any(len(argument) != 3 for argument in given):
        raise ValueError('every argument must hold three entries, one '
                         'per direction')

    knots = [_uniform_knots(degree, span)
             for degree, span in zip(degrees, elements)]

    nodes = np.meshgrid(*(_greville(degree, knot)
                          for degree, knot in zip(degrees, knots)),
                        indexing='ij')

    box = np.column_stack([node.ravel(order='F') for node in nodes])
    points = np.asarray(origin, float) + box * np.asarray(lengths, float)

    basis = _cpp.TrivariateBSpline(
        axes=[_cpp.BSpline(degree=degree, knots=list(knot))
              for degree, knot in zip(degrees, knots)])

    return VolumePatch(_cpp.VolumePatch(basis=basis, coefficients=points),
                       degrees=degrees, knots=knots)
