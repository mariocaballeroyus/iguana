# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Elements, the physics of a problem on the elements of a patch

An element states the problem to solve. It holds the physics alone, so that
one element serves any space, and a problem builds the compiled element that
matches the patch of the space. Its unknowns set the block size of the
space, one for a scalar field, and its traces give one value or several per
point, as many as the components they hold.
"""

from __future__ import annotations

from iguana import cpp as _cpp


class PoissonElement:
    """Element of the Poisson problem, −Δu = f, on a planar patch. Its only
    trace is the field u, the flux being natural."""

    # A scalar field, with a source of one value per point, and no pressure
    _block_size = 1
    _source_shape: tuple[int, ...] = ()
    _pressure_component: int | None = None

    class U:
        """Trace of the field u of the Poisson problem, on which a condition
        imposes values, one per point."""

        _shape: tuple[int, ...] = ()

        def _cpp_traces(self) -> list[_cpp.PoissonElement2d.U]:
            """The compiled trace, on a planar patch, the only one boundaries
            lie in so far, the one component of the field."""
            return [_cpp.PoissonElement2d.U()]

        def _cpp_fluxes(self) -> list[_cpp.PoissonElement2d.Q]:
            """The compiled flux conjugate to the trace, the normal
            derivative of u, which Nitsche's method pairs with it."""
            return [_cpp.PoissonElement2d.Q()]

        def _ghost_scaling(self) -> tuple[float, int]:
            """Factor of the penalty and extra powers of the size of the
            cells of a ghost penalty on the trace, none for the field."""
            return 1., 0

        def __repr__(self) -> str:
            return 'PoissonElement.U()'

    @property
    def u(self) -> PoissonElement.U:
        """Trace of the field u, on which a condition imposes values."""
        return PoissonElement.U()

    def _cpp_element(self) -> _cpp.PoissonElement2d:
        """The compiled element, on a planar patch, the only one problems
        lie on so far."""
        return _cpp.PoissonElement2d()

    def __repr__(self) -> str:
        return 'PoissonElement()'


class StokesElement:
    """Element of the Stokes problem of an incompressible viscous fluid on a
    planar patch, −div(2μ ε(u)) + ∇p = f and div u = 0, stabilized so that
    the velocity and the pressure share the functions of one space. Its
    unknowns are the two components of the velocity followed by the
    pressure, so its space has a block size of three, its source is the
    body force, of two values per point, and its traces the velocity and
    the pressure."""

    # The velocity and the pressure, its last component, with a body force
    # of two values per point
    _block_size = 3
    _source_shape: tuple[int, ...] = (2,)
    _pressure_component: int | None = 2

    class U:
        """Trace of the velocity u of the Stokes problem, on which a
        condition imposes values such as no slip, two per point."""

        _shape: tuple[int, ...] = (2,)

        def __init__(self, element: StokesElement) -> None:
            """Initialize the trace of the velocity of an element.

            Args:
                element: The element of the fluid, whose viscosity the
                    traction conjugate to the trace takes.
            """
            self._element = element

        def _cpp_traces(self) -> list[_cpp.StokesElement2d.U]:
            """The compiled traces of the components of the velocity, on a
            planar patch."""
            return [_cpp.StokesElement2d.U(component) for component in (0, 1)]

        def _cpp_fluxes(self) -> list[_cpp.StokesElement2d.T]:
            """The compiled fluxes conjugate to the traces, the components of
            the traction, which Nitsche's method pairs with them."""
            viscosity = self._element.viscosity

            return [_cpp.StokesElement2d.T(viscosity, component)
                    for component in (0, 1)]

        def _ghost_scaling(self) -> tuple[float, int]:
            """Factor of the penalty and extra powers of the size of the
            cells of a ghost penalty on the trace, the viscosity for the
            velocity."""
            return self._element.viscosity, 0

        def __repr__(self) -> str:
            return 'StokesElement.U()'

    class P:
        """Trace of the pressure p of the Stokes problem, which a ghost
        penalty extends beyond the fluid, one value per point."""

        _shape: tuple[int, ...] = ()

        def __init__(self, element: StokesElement) -> None:
            """Initialize the trace of the pressure of an element.

            Args:
                element: The element of the fluid, whose viscosity scales a
                    ghost penalty on the trace.
            """
            self._element = element

        def _cpp_traces(self) -> list[_cpp.StokesElement2d.P]:
            """The compiled trace of the pressure, on a planar patch."""
            return [_cpp.StokesElement2d.P()]

        def _ghost_scaling(self) -> tuple[float, int]:
            """Factor of the penalty and extra powers of the size of the
            cells of a ghost penalty on the trace: negative over the
            viscosity, as the pressure block of the symmetric form is, and
            two more powers than for the velocity."""
            return -1. / self._element.viscosity, 2

        def __repr__(self) -> str:
            return 'StokesElement.P()'

    def __init__(self, viscosity: float, stabilization: float) -> None:
        """Initialize the element of a fluid.

        The method stabilizes the momentum equation by τ_M = h² / (c μ) and
        the continuity equation by τ_C = μ, with h the size of the elements,
        c the stabilization and μ the viscosity, so a larger stabilization
        stabilizes less.

        With the velocity imposed on the whole boundary, the pressure is
        only determined up to a constant, which the system leaves free and
        a problem fixes by its `pressure_reference`.

        Args:
            viscosity: The dynamic viscosity μ of the fluid, positive.
            stabilization: The constant c of the stabilization, positive.

        Raises:
            ValueError: If the viscosity or the stabilization is not
                positive.
        """
        if not viscosity > 0.:
            raise ValueError('the viscosity must be positive')

        if not stabilization > 0.:
            raise ValueError('the stabilization must be positive')

        self._viscosity = float(viscosity)
        self._stabilization = float(stabilization)

    @property
    def viscosity(self) -> float:
        """The dynamic viscosity of the fluid."""
        return self._viscosity

    @property
    def stabilization(self) -> float:
        """The constant of the stabilization."""
        return self._stabilization

    @property
    def u(self) -> StokesElement.U:
        """Trace of the velocity u, on which a condition imposes values."""
        return StokesElement.U(self)

    @property
    def p(self) -> StokesElement.P:
        """Trace of the pressure p, which a ghost penalty extends."""
        return StokesElement.P(self)

    def _cpp_element(self) -> _cpp.StokesStabilizedElement2d:
        """The compiled element, on a planar patch, the only one problems
        lie on so far."""
        return _cpp.StokesStabilizedElement2d(self._viscosity,
                                              self._stabilization)

    def __repr__(self) -> str:
        return (f'StokesElement(viscosity={self._viscosity}, '
                f'stabilization={self._stabilization})')
