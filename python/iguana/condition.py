# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Conditions, the values a problem takes on a boundary

A condition imposes values on a trace of an element along a boundary. It
holds what it imposes and where, and the solve assembles it with the
element.
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

        Args:
            trace: The trace whose values are imposed, such as
                `PoissonElement().u`.
            quadrature: The quadrature of the boundary the values are
                imposed along.
            values: The value imposed at each point of the quadrature, in
                its order, or a function that takes their positions, of
                shape `(num_points, 2)`, and returns those values.
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

        if callable(values):
            values = values(quadrature.positions)

        values = np.asarray(values, dtype=np.float64)

        if values.shape != (quadrature.num_points,):
            raise ValueError('the values must be one per point')

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
