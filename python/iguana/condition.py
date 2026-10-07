# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Conditions, what a problem prescribes on a boundary

A condition prescribes data along a boundary through a trace of an element:
a penalty imposes values on the trace, and a Neumann condition loads the
trace of the test functions with natural data, such as a flux. It holds its
data and where it applies, and the solve assembles it with the element.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp
from iguana.element import PoissonElement
from iguana.quadrature import BoundaryQuadrature


class PenaltyCondition:
    """Values imposed on a trace of an element along a boundary through a
    penalty."""

    def __init__(
        self,
        trace: PoissonElement.U,
        quadrature: BoundaryQuadrature,
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
        of the order of the cells, whatever the penalty.

        Args:
            trace: The trace whose values are imposed, such as
                `PoissonElement().u`.
            quadrature: The quadrature of the boundary the values are
                imposed along.
            values: The value imposed at each point of the quadrature, in
                its order, or a function that takes the points where they
                apply, `quadrature.projections` of shape `(num_points, 2)`,
                and returns those values.
            penalty: The penalty, positive.

        Raises:
            TypeError: If the trace is not the trace of an element, or if
                the quadrature is not a boundary quadrature.
            ValueError: If the values are not one per point, or if the
                penalty is not positive.
        """
        if not isinstance(trace, PoissonElement.U):
            raise TypeError('the trace must be the trace of an element')

        if not isinstance(quadrature, BoundaryQuadrature):
            raise TypeError('the quadrature must be a boundary quadrature')

        values = _values(quadrature, values)

        if not penalty > 0.:
            raise ValueError('the penalty must be positive')

        self._trace = trace
        self._quadrature = quadrature
        self._values = values
        self._penalty = float(penalty)

    @property
    def trace(self) -> PoissonElement.U:
        """The trace whose values are imposed."""
        return self._trace

    @property
    def quadrature(self) -> BoundaryQuadrature:
        """The quadrature of the boundary the values are imposed along."""
        return self._quadrature

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Value imposed at each point of the quadrature, of shape
        `(num_points,)`."""
        return self._values.copy()

    @property
    def penalty(self) -> float:
        """The penalty."""
        return self._penalty

    def _assemble(self, assembler: _cpp.Assembler2d) -> None:
        """Add the stiffness and load of the condition into the compiled
        assembler of a planar space."""
        condition = _cpp.PenaltyCondition2d(self._trace._cpp_trace(),
                                            self._penalty)
        quadrature = self._quadrature._cpp_object

        assembler.assemble_stiffness(condition, quadrature)
        assembler.assemble_load(condition, quadrature, self._values)

    def __repr__(self) -> str:
        return f'PenaltyCondition({self._trace!r}, penalty={self._penalty})'


class NeumannCondition:
    """Natural data, such as a flux, loading a trace of an element along a
    boundary."""

    def __init__(
        self,
        trace: PoissonElement.U,
        quadrature: BoundaryQuadrature,
        values: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
    ) -> None:
        """Initialize the condition loading a trace with natural data.

        The data is the boundary term of the weak form where it is known,
        so it adds a load and no stiffness. For the field u of the Poisson
        element it is the outward flux ∂u/∂n. Data that depends on the
        normal is given at the points, from the normals of the quadrature,
        such as `quadrature.normals @ gradient`.

        Args:
            trace: The trace of the test functions the data does work on,
                such as `PoissonElement().u` for a flux.
            quadrature: The quadrature of the boundary the data is given
                along, not shifted.
            values: The data at each point of the quadrature, in its order,
                or a function that takes their positions, of shape
                `(num_points, 2)`, and returns it.

        Raises:
            TypeError: If the trace is not the trace of an element, or if
                the quadrature is not a boundary quadrature.
            ValueError: If the quadrature is shifted, whose flux would need
                the gradients at the closest points, or if the values are
                not one per point.
        """
        if not isinstance(trace, PoissonElement.U):
            raise TypeError('the trace must be the trace of an element')

        if not isinstance(quadrature, BoundaryQuadrature):
            raise TypeError('the quadrature must be a boundary quadrature')

        if quadrature.order is not None:
            raise ValueError('the quadrature must not be shifted')

        self._trace = trace
        self._quadrature = quadrature
        self._values = _values(quadrature, values)

    @property
    def trace(self) -> PoissonElement.U:
        """The trace of the test functions the data does work on."""
        return self._trace

    @property
    def quadrature(self) -> BoundaryQuadrature:
        """The quadrature of the boundary the data is given along."""
        return self._quadrature

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """Data at each point of the quadrature, of shape
        `(num_points,)`."""
        return self._values.copy()

    def _assemble(self, assembler: _cpp.Assembler2d) -> None:
        """Add the load of the condition into the compiled assembler of a
        planar space, its stiffness being zero."""
        condition = _cpp.NeumannCondition2d(self._trace._cpp_trace())
        assembler.assemble_load(condition, self._quadrature._cpp_object,
                                self._values)

    def __repr__(self) -> str:
        return f'NeumannCondition({self._trace!r})'


def _values(
    quadrature: BoundaryQuadrature,
    values: (npt.ArrayLike
             | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
) -> npt.NDArray[np.float64]:
    """Values of a condition at the points of a boundary quadrature, given
    at them or as a function of the points where they apply, their closest
    points on a shifted quadrature, checked to be one per point."""
    if callable(values):
        values = values(quadrature.projections)

    values = np.asarray(values, dtype=np.float64)

    if values.shape != (quadrature.num_points,):
        raise ValueError('the values must be one per point')

    return values
