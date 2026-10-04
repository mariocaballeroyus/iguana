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

    _cpp_object: (_cpp.PlanarCurvePatch | _cpp.CurvePatch
                  | _cpp.NURBSPlanarCurvePatch | _cpp.NURBSCurvePatch)

    def __init__(self,
                 curve: (_cpp.PlanarCurvePatch | _cpp.CurvePatch
                         | _cpp.NURBSPlanarCurvePatch
                         | _cpp.NURBSCurvePatch)) -> None:
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

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Weight of each control point, of shape `(num_control_points,)`.

        A B-spline curve has unit weights.
        """
        basis = self._cpp_object.basis

        if isinstance(basis, _cpp.UnivariateNURBS):
            return basis.weights

        return np.ones(len(self.control_points))

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

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Weight of each control point, of shape `(num_control_points,)`.

        Planar patches are B-spline patches, so their weights are ones.
        """
        return np.ones(len(self.control_points))

    def isocurves(self) -> list[CurvePatch]:
        """Isocurves of the patch along all of its knot lines, in the plane.

        The curves pinned in the first direction come first, then those
        pinned in the second, each group in increasing order of knot line.
        """
        return [CurvePatch(curve) for curve in self._cpp_object.isocurves()]

    def invert_points(self,
                      points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Parameters of points in the plane, by inverting the map.

        Only affine maps are inverted so far, such as those of an
        axis-aligned, rotated or sheared rectangle. Points outside the patch
        get parameters outside its box.

        Args:
            points: The points, of shape `(num_points, 2)`.

        Returns:
            Their parameters, of shape `(num_points, 2)`.

        Raises:
            ValueError: If the map of the patch is not affine.
        """
        return self._cpp_object.invert_points(np.asarray(points, float))

    def __repr__(self) -> str:
        return (f'PlanarPatch(degrees={self.degrees}, '
                f'num_control_points={len(self.control_points)})')


class SurfacePatch:
    """A spline map from a two-dimensional parameter box into space."""

    _cpp_object: _cpp.SurfacePatch | _cpp.NURBSSurfacePatch

    def __init__(self,
                 patch: _cpp.SurfacePatch | _cpp.NURBSSurfacePatch) -> None:
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

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Weight of each control point, of shape `(num_control_points,)`.

        A B-spline surface has unit weights.
        """
        basis = self._cpp_object.basis

        if isinstance(basis, _cpp.BivariateNURBS):
            return basis.weights

        return np.ones(len(self.control_points))

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

    @property
    def weights(self) -> npt.NDArray[np.float64]:
        """Weight of each control point, of shape `(num_control_points,)`.

        Volume patches are B-spline patches, so their weights are ones.
        """
        return np.ones(len(self.control_points))

    def isosurfaces(self) -> list[SurfacePatch]:
        """Isosurfaces of the patch along all of its knot planes.

        The surfaces pinned in the first direction come first, then those
        pinned in the second and third, each group in increasing order of
        knot plane. Each surface keeps the remaining two directions, in
        increasing order.
        """
        return [SurfacePatch(surface)
                for surface in self._cpp_object.isosurfaces()]

    def invert_points(self,
                      points: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Parameters of points in space, by inverting the map.

        Only affine maps are inverted so far, such as those of an
        axis-aligned, rotated or sheared box. Points outside the patch get
        parameters outside its box.

        Args:
            points: The points, of shape `(num_points, 3)`.

        Returns:
            Their parameters, of shape `(num_points, 3)`.

        Raises:
            ValueError: If the map of the patch is not affine.
        """
        return self._cpp_object.invert_points(np.asarray(points, float))

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
    weights: npt.ArrayLike | None = None,
) -> CurvePatch:
    """Create the patch of a B-spline or NURBS curve from its control
    points.

    The curve lies in the plane or in space, as its control points have two
    or three coordinates.

    Args:
        degree: Polynomial degree of the curve.
        knots: Full, non-decreasing knot vector.
        control_points: Control points, of shape `(num_control_points, 2)`
            in the plane or `(num_control_points, 3)` in space.
        weights: Positive weight of each control point, which makes the
            curve a NURBS curve. Without them it is a B-spline curve.

    Returns:
        The patch of the curve.

    Raises:
        ValueError: If the control points do not have two or three
            coordinates, if the knot vector is invalid for the degree, or if
            there is not one control point, and one positive weight when
            given, per basis function.
    """
    points = np.asarray(control_points, float)

    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError('the control points must have two or three '
                         'coordinates')

    basis = _cpp.UnivariateBSpline(
        axes=[_cpp.BSpline(degree=degree,
                           knots=np.asarray(knots, float).tolist())])

    if weights is None:
        planar, spatial = _cpp.PlanarCurvePatch, _cpp.CurvePatch
    else:
        basis = _cpp.UnivariateNURBS(bspline=basis,
                                     weights=np.asarray(weights, float))
        planar, spatial = _cpp.NURBSPlanarCurvePatch, _cpp.NURBSCurvePatch

    curve = planar if points.shape[1] == 2 else spatial

    return CurvePatch(curve(basis=basis, coefficients=points))


def create_surface(
    degrees: Sequence[int],
    knots: Sequence[npt.ArrayLike],
    control_points: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
) -> SurfacePatch:
    """Create the patch of a B-spline or NURBS surface from its control net.

    Args:
        degrees: Polynomial degree of each parametric direction.
        knots: Full, non-decreasing knot vector of each direction.
        control_points: Control points, of shape
            `(num_control_points, 3)`, numbered with the first direction
            running fastest.
        weights: Positive weight of each control point, in the same
            numbering, which makes the surface a NURBS surface. Without
            them it is a B-spline surface.

    Returns:
        The patch of the surface.

    Raises:
        ValueError: If there are not two degrees and two knot vectors, if
            a knot vector is invalid for its degree, or if there is not
            one control point, and one positive weight when given, per
            basis function.
    """
    if len(degrees) != 2 or len(knots) != 2:
        raise ValueError('there must be two degrees and two knot vectors, '
                         'one per direction')

    basis = _cpp.BivariateBSpline(
        axes=[_cpp.BSpline(degree=degree,
                           knots=np.asarray(knot, float).tolist())
              for degree, knot in zip(degrees, knots)])

    points = np.asarray(control_points, float).reshape(-1, 3)

    if weights is None:
        surface = _cpp.SurfacePatch
    else:
        basis = _cpp.BivariateNURBS(bspline=basis,
                                    weights=np.asarray(weights, float))
        surface = _cpp.NURBSSurfacePatch

    return SurfacePatch(surface(basis=basis, coefficients=points))
