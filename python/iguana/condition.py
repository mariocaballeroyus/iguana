# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Conditions, what a problem prescribes on a boundary

A condition prescribes data along a boundary through a trace of an element:
a penalty imposes values on the trace, Nitsche's method imposes them
consistently through the flux conjugate to the trace, and a Neumann
condition loads the trace of the test functions with natural data, such as
a flux. Nitsche's method on faces imposes them across the faces of the cells
where their weights jump, as the generalized shifted boundary method does.
A trace of several components, such as the velocity of `StokesElement`,
takes as many values per point, each acting on its component. A condition
holds its data and where it applies, and a problem assembles it with the
element. Its values can change between solves, as in a time loop.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp
from iguana.boundary import Boundary
from iguana.element import PoissonElement, StokesElement
from iguana.embedding import CellType
from iguana.grid import HierarchicalGrid
from iguana.patch import PlanarPatch
from iguana.quadrature import QuadratureBoundary

# The traces conditions act on, of one value per point or of several
_TRACES = (PoissonElement.U, StokesElement.U)


class PenaltyCondition:
    """Values imposed on a trace of an element along a boundary through a
    penalty."""

    def __init__(
        self,
        trace: PoissonElement.U | StokesElement.U,
        quadrature: QuadratureBoundary,
        values: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
        penalty: float,
    ) -> None:
        """Initialize the condition imposing values on a trace.

        The penalty weighs the imposed values against the rest of the
        problem. The error decays like one over the penalty, which a larger
        one buys with a worse conditioning.

        On a shifted quadrature, the values are imposed on the Taylor
        expansion of the trace from each point to its closest point, the
        shifted penalty. Alone it is not consistent: nothing balances the
        flux of the field across the surrogate boundary, so the error stays
        of the order of the cells, whatever the penalty. `NitscheCondition`
        imposes the values consistently.

        Args:
            trace: The trace whose values are imposed, such as the field u
                of `PoissonElement` or the velocity u of `StokesElement`.
            quadrature: The quadrature of the boundary the values are
                imposed along.
            values: The values imposed at each point of the quadrature, in
                its order, one per point or two for the velocity, or a
                function that takes the points where they apply,
                `quadrature.projections` of shape `(num_points, 2)`, and
                returns those values.
            penalty: The penalty, positive.

        Raises:
            TypeError: If the trace is not the trace of an element, or if
                the quadrature is not a boundary quadrature.
            ValueError: If the values are not those of the trace at each
                point, or if the penalty is not positive.
        """
        if not isinstance(trace, _TRACES):
            raise TypeError('the trace must be the trace of an element')

        if not isinstance(quadrature, QuadratureBoundary):
            raise TypeError('the quadrature must be a boundary quadrature')

        values = _values(quadrature.projections, values, trace._shape)

        if not penalty > 0.:
            raise ValueError('the penalty must be positive')

        self._trace = trace
        self._quadrature = quadrature
        self._values = values
        self._penalty = float(penalty)

    @property
    def trace(self) -> PoissonElement.U | StokesElement.U:
        """The trace whose values are imposed."""
        return self._trace

    @property
    def quadrature(self) -> QuadratureBoundary:
        """The quadrature of the boundary the values are imposed along."""
        return self._quadrature

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Values imposed at each point of the quadrature, of shape
        `(num_points,)`, or `(num_points, 2)` for the velocity. They are set
        as at initialization, by their values or by a function of the points
        where they apply."""
        return _given(self._values, self._trace._shape)

    @values.setter
    def values(
        self,
        values: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
    ) -> None:
        self._values[...] = _values(self._quadrature.projections, values,
                                    self._trace._shape)

    @property
    def penalty(self) -> float:
        """The penalty."""
        return self._penalty

    def _add_to(self, form: _cpp.Form2d) -> None:
        """Add the integral of the condition to the compiled form of a
        planar problem, which reads its values where they lie."""
        for trace, values in zip(self._trace._cpp_traces(), self._values):
            condition = _cpp.PenaltyCondition2d(trace, self._penalty)
            _add_along(form, condition, self._quadrature, values)

    def __repr__(self) -> str:
        return f'PenaltyCondition({self._trace!r}, penalty={self._penalty})'


class NitscheCondition:
    """Values imposed on a trace of an element along a boundary by
    Nitsche's method."""

    def __init__(
        self,
        trace: PoissonElement.U | StokesElement.U,
        quadrature: QuadratureBoundary,
        values: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
        penalty: float,
    ) -> None:
        """Initialize the condition imposing values on a trace.

        Nitsche's method keeps the boundary term of the weak form, which
        pairs the trace with its flux, such as the normal derivative of the
        field u of the Poisson element, and adds its adjoint and a penalty
        scaled by the size of the elements. Unlike a penalty alone, it is
        consistent: the error decays with the order of the space for any
        penalty large enough to keep the method stable, about ten times the
        square of the degree of the patch, and times the viscosity for the
        velocity of a fluid, whose flux is the traction.

        On a shifted quadrature, the adjoint and the penalty act on the
        Taylor expansion of the trace from each point to its closest point,
        while the boundary term stays at the points, as the shifted boundary
        method needs.

        Args:
            trace: The trace whose values are imposed, such as the field u
                of `PoissonElement` or the velocity u of `StokesElement`.
            quadrature: The quadrature of the boundary the values are
                imposed along.
            values: The values imposed at each point of the quadrature, in
                its order, one per point or two for the velocity, or a
                function that takes the points where they apply,
                `quadrature.projections` of shape `(num_points, 2)`, and
                returns those values.
            penalty: The penalty, positive, which the size of the elements
                scales.

        Raises:
            TypeError: If the trace is not the trace of an element, or if
                the quadrature is not a boundary quadrature.
            ValueError: If the values are not those of the trace at each
                point, or if the penalty is not positive.
        """
        if not isinstance(trace, _TRACES):
            raise TypeError('the trace must be the trace of an element')

        if not isinstance(quadrature, QuadratureBoundary):
            raise TypeError('the quadrature must be a boundary quadrature')

        values = _values(quadrature.projections, values, trace._shape)

        if not penalty > 0.:
            raise ValueError('the penalty must be positive')

        self._trace = trace
        self._quadrature = quadrature
        self._values = values
        self._penalty = float(penalty)

    @property
    def trace(self) -> PoissonElement.U | StokesElement.U:
        """The trace whose values are imposed."""
        return self._trace

    @property
    def quadrature(self) -> QuadratureBoundary:
        """The quadrature of the boundary the values are imposed along."""
        return self._quadrature

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Values imposed at each point of the quadrature, of shape
        `(num_points,)`, or `(num_points, 2)` for the velocity. They are set
        as at initialization, by their values or by a function of the points
        where they apply."""
        return _given(self._values, self._trace._shape)

    @values.setter
    def values(
        self,
        values: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
    ) -> None:
        self._values[...] = _values(self._quadrature.projections, values,
                                    self._trace._shape)

    @property
    def penalty(self) -> float:
        """The penalty."""
        return self._penalty

    def _add_to(self, form: _cpp.Form2d) -> None:
        """Add the integral of the condition to the compiled form of a
        planar problem, which reads its values where they lie."""
        for trace, flux, values in zip(self._trace._cpp_traces(),
                                       self._trace._cpp_fluxes(),
                                       self._values):
            condition = _cpp.NitscheCondition2d(trace, flux, self._penalty)
            _add_along(form, condition, self._quadrature, values)

    def __repr__(self) -> str:
        return f'NitscheCondition({self._trace!r}, penalty={self._penalty})'


class NeumannCondition:
    """Natural data, such as a flux, loading a trace of an element along a
    boundary."""

    def __init__(
        self,
        trace: PoissonElement.U | StokesElement.U,
        quadrature: QuadratureBoundary,
        values: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
    ) -> None:
        """Initialize the condition loading a trace with natural data.

        The data is the boundary term of the weak form where it is known,
        so it adds a load and no stiffness. For the field u of the Poisson
        element it is the outward flux ∂u/∂n, and for the velocity of the
        Stokes element the traction. Data that depends on the normal is
        given at the points, from the normals of the quadrature, such as
        `quadrature.normals @ gradient`.

        Args:
            trace: The trace of the test functions the data does work on,
                such as the field u of `PoissonElement` for a flux or the
                velocity u of `StokesElement` for a traction.
            quadrature: The quadrature of the boundary the data is given
                along, not shifted.
            values: The data at each point of the quadrature, in its order,
                one value per point or two for the velocity, or a function
                that takes their positions, of shape `(num_points, 2)`, and
                returns it.

        Raises:
            TypeError: If the trace is not the trace of an element, or if
                the quadrature is not a boundary quadrature.
            ValueError: If the quadrature is shifted, whose flux would need
                the gradients at the closest points, or if the values are
                not those of the trace at each point.
        """
        if not isinstance(trace, _TRACES):
            raise TypeError('the trace must be the trace of an element')

        if not isinstance(quadrature, QuadratureBoundary):
            raise TypeError('the quadrature must be a boundary quadrature')

        if quadrature.order is not None:
            raise ValueError('the quadrature must not be shifted')

        self._trace = trace
        self._quadrature = quadrature
        self._values = _values(quadrature.positions, values, trace._shape)

    @property
    def trace(self) -> PoissonElement.U | StokesElement.U:
        """The trace of the test functions the data does work on."""
        return self._trace

    @property
    def quadrature(self) -> QuadratureBoundary:
        """The quadrature of the boundary the data is given along."""
        return self._quadrature

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Data at each point of the quadrature, of shape `(num_points,)`,
        or `(num_points, 2)` for the velocity. It is set as at
        initialization, by its values or by a function of the positions of
        the points."""
        return _given(self._values, self._trace._shape)

    @values.setter
    def values(
        self,
        values: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
    ) -> None:
        self._values[...] = _values(self._quadrature.positions, values,
                                    self._trace._shape)

    def _add_to(self, form: _cpp.Form2d) -> None:
        """Add the integral of the condition, a load, to the compiled form of
        a planar problem, which reads its values where they lie."""
        for trace, values in zip(self._trace._cpp_traces(), self._values):
            condition = _cpp.NeumannCondition2d(trace)
            form.add_boundary_integral(condition,
                                       self._quadrature._cpp_object, values)

    def __repr__(self) -> str:
        return f'NeumannCondition({self._trace!r})'


class SkeletonNitscheCondition:
    """Values imposed on a trace of an element by Nitsche's method across the
    faces of a patch where the weights of the cells jump, as the
    generalized shifted boundary method does."""

    def __init__(
        self,
        trace: PoissonElement.U | StokesElement.U,
        patch: PlanarPatch,
        cell_types: Sequence[CellType],
        boundary: Boundary,
        values: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
        penalty: float,
        order: int | None = None,
        source: Callable[[npt.NDArray[np.float64]], npt.ArrayLike]
        | None = None,
    ) -> None:
        """Initialize the condition imposing values on a trace.

        The condition acts on the faces between cells whose weights, the
        volume fractions a problem takes as `cell_weights`, may differ:
        those between cells of different types or between two cut cells.
        On each, the jump of the weights scales Nitsche's terms, over the
        means of the two cells weighted by them, and each cell expands its
        own functions to the closest point on the boundary of every face
        point, where the values apply. With weights of one inside and zero
        elsewhere, it is Nitsche's condition on the surrogate boundary.

        The expansion is a Taylor series of the functions of each cell.
        With the source of the Poisson equation it is instead the series of
        the solution along the normal of the boundary from the closest
        point, whose normal derivatives from the second on the equation
        gives from the values, the source and the curvature of the
        boundary. It keeps the condition coercive at high degree. The
        boundary must then be smooth at the closest points, and the values
        and the source functions of the points around them, on both sides
        of the boundary.

        Only faces between two cells carry it, so the cells of non-zero
        weight must stay off the edge of the patch.

        Args:
            trace: The trace whose values are imposed, such as the field u
                of `PoissonElement` or the velocity u of `StokesElement`.
            patch: The background patch, a planar patch with an affine map,
                whose elements are the cells.
            cell_types: The type of each element of the patch, with the
                first direction running fastest.
            boundary: The true boundary, made of curves in the plane of the
                patch, where the values apply.
            values: The values imposed at the closest point of each face
                point, one per point or two for the velocity, or a function
                that takes those closest points, of shape `(num_points, 2)`,
                and returns their values.
            penalty: The penalty, positive, which the size of the cells
                scales, as for `NitscheCondition`.
            order: Total order of the Taylor expansion towards the
                boundary, or of the normal series with a source,
                nonnegative. Without it, the highest degree of the patch.
            source: The source f of the Poisson equation −Δu = f, a
                function of points of shape `(num_points, 2)` returning its
                values, with which the expansion is the normal series of the
                solution. Without it, the Taylor series.

        Raises:
            TypeError: If the trace is not the trace of an element, if the
                patch is not a planar patch, if the boundary is not a
                boundary, or, with a source, if the source or the values are
                not functions of the points.
            ValueError: If there is not one cell type per element, if the
                map of the patch is not affine, if the values are not those
                of the trace at each point, if the penalty is not positive,
                if the order is negative, or, with a source, if the trace is
                not that of a Poisson element or if the boundary is not
                smooth at a closest point.
        """
        if not isinstance(trace, _TRACES):
            raise TypeError('the trace must be the trace of an element')

        if not isinstance(patch, PlanarPatch):
            raise TypeError('the patch must be a planar patch')

        if not isinstance(boundary, Boundary):
            raise TypeError('the boundary must be a boundary')

        cell_types = list(cell_types)
        num_elements = HierarchicalGrid(patch.degrees,
                                        patch.knots).num_elements

        if len(cell_types) != num_elements:
            raise ValueError('there must be one cell type per cell')

        if not penalty > 0.:
            raise ValueError('the penalty must be positive')

        if order is None:
            order = max(patch.degrees)

        if order < 0:
            raise ValueError('the order must be non-negative')

        if source is not None:
            if not isinstance(trace, PoissonElement.U):
                raise ValueError('only the trace of a Poisson element takes '
                                 'a source')

            if not callable(source):
                raise TypeError('the source must be a function of the points')

            if not callable(values):
                raise TypeError('the values must be a function of the points '
                                'with a source')

        # The faces between cells whose weights may jump, with as many
        # points as on those of a ghost penalty, projected onto the boundary
        faces = _cpp.jump_faces(
            patch=patch._cpp_object,
            classification=_cpp.CellClassification2d(cell_types))
        quadrature = _cpp.QuadratureSkeleton2d(
            patch=patch._cpp_object, faces=faces,
            num_points=max(patch.degrees) + 1)
        projection = _cpp.project_closest_points(
            patch=patch._cpp_object, boundary=boundary._cpp_object,
            points=quadrature.points)
        positions = np.array(projection.positions)

        self._trace = trace
        self._patch = patch
        self._quadrature = quadrature
        self._projection = projection
        self._order = order
        self._values = _values(positions, values, trace._shape)
        self._penalty = float(penalty)
        self._source = source

        # Rewritten with the equation, the assembly combines derivatives of
        # its own order at the closest points, and the data imposed at the
        # points are those of the series, from the derivatives of the values
        # and of the source there
        if source is None:
            self._imposed = self._values
        else:
            shift = _cpp.poisson_shift(
                patch=patch._cpp_object, boundary=boundary._cpp_object,
                projection=projection, order=order)
            projection.shift_weights = shift.weights
            distances = np.abs(np.array(shift.distances))
            radius = distances.max() if distances.max() > 0. else 1.

            self._order = shift.order
            self._series = order
            self._radius = radius
            self._value_weights = np.array(shift.value_weights)
            self._source_part = np.sum(
                np.array(shift.source_weights)
                * _derivatives(source, positions, order - 2, radius), axis=1)
            self._imposed = np.empty_like(self._values)
            self._impose(values)

    @property
    def trace(self) -> PoissonElement.U | StokesElement.U:
        """The trace whose values are imposed."""
        return self._trace

    @property
    def patch(self) -> PlanarPatch:
        """The background patch, whose cells the faces separate."""
        return self._patch

    @property
    def num_faces(self) -> int:
        """Number of faces between cells whose weights may jump."""
        return self._quadrature.num_faces

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Values imposed at the closest point of each face point, of shape
        `(num_points,)`, or `(num_points, 2)` for the velocity. They are set
        as at initialization, by their values or by a function of the
        closest points."""
        return _given(self._values, self._trace._shape)

    @values.setter
    def values(
        self,
        values: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
    ) -> None:
        positions = np.array(self._projection.positions)

        if self._source is not None and not callable(values):
            raise TypeError('the values must be a function of the points '
                            'with a source')

        self._values[...] = _values(positions, values, self._trace._shape)

        if self._source is not None:
            self._impose(values)

    @property
    def penalty(self) -> float:
        """The penalty."""
        return self._penalty

    def _impose(
        self,
        values: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    ) -> None:
        """Data imposed at the points by the series rewritten with the
        equation, in place, from the derivatives of the values at the
        closest points, their values themselves taken exactly."""
        positions = np.array(self._projection.positions)
        derivatives = _derivatives(values, positions, self._series,
                                   self._radius)
        derivatives[:, 0] = self._values[0]

        self._imposed[0] = (np.sum(self._value_weights * derivatives, axis=1)
                            + self._source_part)

    def _add_to(self, form: _cpp.Form2d,
                cell_weights: npt.NDArray[np.float64]) -> None:
        """Add the integral of the condition to the compiled form of a planar
        problem, assembled with its faces coupled, with the weights of the
        cells. The form reads the imposed data and weights where they
        lie."""
        for trace, flux, values in zip(self._trace._cpp_traces(),
                                       self._trace._cpp_fluxes(),
                                       self._imposed):
            condition = _cpp.SkeletonNitscheCondition2d(trace, flux,
                                                        self._penalty)
            form.add_skeleton_integral(condition, self._quadrature,
                                       self._projection, self._order, values,
                                       cell_weights)

    def __repr__(self) -> str:
        return (f'SkeletonNitscheCondition({self._trace!r}, '
                f'penalty={self._penalty})')


def _add_along(
    form: _cpp.Form2d,
    condition: _cpp.PenaltyCondition2d | _cpp.NitscheCondition2d,
    quadrature: QuadratureBoundary,
    values: npt.NDArray[np.float64],
) -> None:
    """Add the integral of a compiled condition along a boundary quadrature
    to a compiled form, shifted along the projection of a shifted
    quadrature."""
    if quadrature._projection is None:
        form.add_boundary_integral(condition, quadrature._cpp_object, values)
    else:
        form.add_boundary_integral(condition, quadrature._cpp_object,
                                   quadrature._projection, quadrature.order,
                                   values)


def _values(
    points: npt.NDArray[np.float64],
    values: (npt.ArrayLike
             | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
    shape: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Values of a condition at the points where they apply, such as the
    closest points of a shifted quadrature, given at them or as a function
    of them, checked to be of the shape of the trace at each point. They are
    stored a component per row, each contiguous, as a compiled form reads
    them where they lie."""
    if callable(values):
        values = values(points)

    values = np.asarray(values, dtype=np.float64)

    if values.shape != (len(points), *shape):
        raise ValueError('the values must be those of the trace at each '
                         'point')

    return np.ascontiguousarray(values.reshape(len(points), -1).T)


def _derivatives(
    function: Callable[[npt.NDArray[np.float64]], npt.ArrayLike],
    points: npt.NDArray[np.float64],
    order: int,
    radius: float,
) -> npt.NDArray[np.float64]:
    """Cartesian derivatives d_x^a d_y^b of a function of points up to a
    total order at each point, of shape `(num_points, num_terms)` with a
    running fastest, from a polynomial of two degrees more fitted to its
    values at Chebyshev points of a square of the radius around each. The fit
    is exact for polynomials of that degree, and the same for every point."""
    if order < 0:
        return np.zeros((len(points), 0))

    degree = order + 2
    nodes = np.cos(np.pi * (np.arange(degree + 2) + .5) / (degree + 2))
    stencil = np.stack(np.meshgrid(nodes, nodes), axis=-1).reshape(-1, 2)
    fitted = [(a, b) for b in range(degree + 1) for a in range(degree + 1 - b)]
    vandermonde = np.column_stack([stencil[:, 0] ** a * stencil[:, 1] ** b
                                   for a, b in fitted])
    inverse = np.linalg.pinv(vandermonde)

    # The values at the stencil of every point at once
    around = points[:, None, :] + radius * stencil[None, :, :]
    values = np.asarray(function(around.reshape(-1, 2)), dtype=np.float64)
    coefficients = values.reshape(len(points), -1) @ inverse.T

    # d^alpha at the center is alpha! c_alpha / radius^|alpha|
    terms = [(a, b) for b in range(order + 1) for a in range(order + 1 - b)]
    scales = np.array([math.factorial(a) * math.factorial(b)
                       / radius ** (a + b) for a, b in terms])
    columns = [fitted.index(term) for term in terms]

    return coefficients[:, columns] * scales


def _given(
    values: npt.NDArray[np.float64],
    shape: tuple[int, ...],
) -> npt.NDArray[np.float64]:
    """Copy of the stored values of a condition, of shape
    `(num_points, *shape)` as they were given."""
    return values.T.reshape(-1, *shape).copy()
