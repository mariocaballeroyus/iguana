# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Elements, the physics of a problem on the elements of a patch

An element states the problem to solve. It holds the physics alone, so that
one element serves any space, and the solve builds the compiled element that
matches the patch of the space.
"""

from __future__ import annotations

from iguana import cpp as _cpp


class PoissonElement:
    """Element of the Poisson problem, -Δu = f, on a planar or volume
    patch. Its only trace is the field u, the flux being natural."""

    class U:
        """Trace of the field u of the Poisson problem, on which a condition
        imposes values."""

        def _cpp_trace(self) -> _cpp.PoissonElement2d.U:
            """The compiled trace, on a planar patch, the only one boundaries
            lie in so far."""
            return _cpp.PoissonElement2d.U()

        def _cpp_flux(self) -> _cpp.PoissonElement2d.Q:
            """The compiled flux conjugate to the trace, the normal
            derivative of u, which Nitsche's method pairs with it."""
            return _cpp.PoissonElement2d.Q()

        def __repr__(self) -> str:
            return 'PoissonElement.U()'

    @property
    def u(self) -> PoissonElement.U:
        """Trace of the field u, on which a condition imposes values."""
        return PoissonElement.U()

    def _cpp_element(self, dimension: int) -> (_cpp.PoissonElement2d
                                               | _cpp.PoissonElement3d):
        """The compiled element for a patch of two or three directions."""
        if dimension == 2:
            return _cpp.PoissonElement2d()

        return _cpp.PoissonElement3d()

    def __repr__(self) -> str:
        return 'PoissonElement()'
