# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Elements, the physics of a problem on the elements of a patch

An element states the problem to solve. It holds the physics alone, so that
one element serves any space, and a problem builds the compiled element that
matches the patch of the space. Its unknowns take the components of one
field, as many as the block size of the space, or a field each of a mixed
space, and its traces give one value or several per point, as many as the
components they hold.
"""

from __future__ import annotations

from iguana import cpp as _cpp


class PoissonElement:
    """Element of the Poisson problem, −Δu = f, on a planar patch. Its only
    trace is the field u, the flux being natural."""

    # A scalar field, with a source of one value per point, and no pressure
    _num_unknowns = 1
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
    planar patch, −div(2μ ε(u)) + ∇p = f and div u = 0. Its unknowns are
    the two components of the velocity followed by the pressure, the three
    components of one field or a field each of a mixed space, its source is
    the body force, of two values per point, and its traces the velocity
    and the pressure. Without a stabilization it is the Galerkin element,
    whose velocity and pressure need a stable pair of spaces, such as the
    divergence-conforming space of `MixedFunctionSpace`. With one, they may
    share the functions of one space."""

    # The velocity and the pressure, its last unknown, with a body force of
    # two values per point
    _num_unknowns = 3
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

        def _cpp_traces(
            self,
            pressure: bool = True,
        ) -> list[_cpp.StokesElement2d.U]:
            """The compiled traces of the components of the velocity, on a
            planar patch, without the pressure gradient of a shift with the
            equation unless `pressure`."""
            return [_cpp.StokesElement2d.U(component, pressure)
                    for component in (0, 1)]

        def _cpp_fluxes(
            self,
            pressure: bool = True,
        ) -> list[_cpp.StokesElement2d.T]:
            """The compiled fluxes conjugate to the traces, the components of
            the traction, which Nitsche's method pairs with them, without
            the pressure unless `pressure`."""
            viscosity = self._element.viscosity

            return [_cpp.StokesElement2d.T(viscosity, component, pressure)
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
            viscosity, on the side of the pressure block of the symmetric
            form, and two more powers than for the velocity."""
            return -1. / self._element.viscosity, 2

        def __repr__(self) -> str:
            return 'StokesElement.P()'

    def __init__(self, viscosity: float,
                 stabilization: float | None = None) -> None:
        """Initialize the element of a fluid.

        A stabilization adds the residuals of the momentum equation, by
        τ_M = h² / (c μ), and of the continuity equation, by τ_C = μ, with h
        the size of the elements, c the stabilization and μ the viscosity,
        so a larger stabilization stabilizes less. Without one, the element
        keeps the Galerkin form alone.

        With the velocity imposed on the whole boundary, the pressure is
        only determined up to a constant, which the system leaves free and
        a problem fixes by its `pressure_reference`.

        Args:
            viscosity: The dynamic viscosity μ of the fluid, positive.
            stabilization: The constant c of the stabilization, positive.
                Without it, the element is the Galerkin one.

        Raises:
            ValueError: If the viscosity or a stabilization given is not
                positive.
        """
        if not viscosity > 0.:
            raise ValueError('the viscosity must be positive')

        if stabilization is not None and not stabilization > 0.:
            raise ValueError('the stabilization must be positive')

        self._viscosity = float(viscosity)
        self._stabilization = (None if stabilization is None
                               else float(stabilization))

    @property
    def viscosity(self) -> float:
        """The dynamic viscosity of the fluid."""
        return self._viscosity

    @property
    def stabilization(self) -> float | None:
        """The constant of the stabilization, none for the Galerkin
        element."""
        return self._stabilization

    @property
    def u(self) -> StokesElement.U:
        """Trace of the velocity u, on which a condition imposes values."""
        return StokesElement.U(self)

    @property
    def p(self) -> StokesElement.P:
        """Trace of the pressure p, which a ghost penalty extends."""
        return StokesElement.P(self)

    def _cpp_element(
        self,
    ) -> _cpp.StokesElement2d | _cpp.StokesStabilizedElement2d:
        """The compiled element, on a planar patch, the only one problems
        lie on so far."""
        if self._stabilization is None:
            return _cpp.StokesElement2d(self._viscosity)

        return _cpp.StokesStabilizedElement2d(self._viscosity,
                                              self._stabilization)

    def __repr__(self) -> str:
        if self._stabilization is None:
            return f'StokesElement(viscosity={self._viscosity})'

        return (f'StokesElement(viscosity={self._viscosity}, '
                f'stabilization={self._stabilization})')


class ElasticityElement:
    """Element of an isotropic elastic solid on a planar patch, which maps its
    reference configuration. The problem chooses the theory. A `LinearProblem`
    solves linear elasticity, for small displacements, −div σ(u) = f with
    σ(u) = 2μ ε(u) + λ (div u) I. A `NonlinearProblem` solves the
    St. Venant-Kirchhoff problem, for large displacements and moderate
    strains, −div(F S) = f with F = I + ∇u, S = 2μ E + λ tr(E) I and
    E = (FᵀF − I) / 2, its body force and the data of its Neumann conditions
    per unit of reference volume and length. Its unknowns are the two
    components of the displacement, those of one field, its source is the body
    force, of two values per point, and its trace the displacement."""

    # The two components of the displacement, with a body force of two values
    # per point, and no pressure
    _num_unknowns = 2
    _source_shape: tuple[int, ...] = (2,)
    _pressure_component: int | None = None

    class U:
        """Trace of the displacement u of a solid, on which a condition
        imposes values such as a clamped side, two per point."""

        _shape: tuple[int, ...] = (2,)

        def __init__(self, element: ElasticityElement) -> None:
            """Initialize the trace of the displacement of an element.

            Args:
                element: The element of the solid, whose moduli the traction
                    conjugate to the trace takes.
            """
            self._element = element

        def _cpp_traces(self) -> list[_cpp.ElasticityElement2d.U]:
            """The compiled traces of the components of the displacement, on
            a planar patch."""
            return [_cpp.ElasticityElement2d.U(component)
                    for component in (0, 1)]

        def _cpp_fluxes(self) -> list[_cpp.ElasticityElement2d.T]:
            """The compiled fluxes conjugate to the traces, the components of
            the traction under small displacements, which Nitsche's method
            pairs with them."""
            element = self._element

            return [_cpp.ElasticityElement2d.T(element.lame,
                                               element.shear_modulus,
                                               component)
                    for component in (0, 1)]

        def __repr__(self) -> str:
            return 'ElasticityElement.U()'

    def __init__(self, young_modulus: float, poisson_ratio: float,
                 plane_stress: bool = False) -> None:
        """Initialize the element of a solid.

        The problem lies in the plane, in plane strain by default, the state
        of a solid long across the plane, or in plane stress, that of a thin
        plate. The moduli give the Lamé parameters μ = E / (2 (1 + ν)) and
        λ = E ν / ((1 + ν) (1 − 2 ν)), or λ = E ν / (1 − ν²) in plane stress.

        The mass of the element has unit density and is the matrix of a first
        derivative in time. A time step of a problem therefore does not give
        elastodynamics, whose second derivative has no time scheme yet.

        Args:
            young_modulus: Young's modulus E of the solid, positive.
            poisson_ratio: Poisson's ratio ν of the solid, above −1 and below
                1/2.
            plane_stress: Whether the solid is in plane stress, rather than
                plane strain.

        Raises:
            ValueError: If Young's modulus is not positive, or if Poisson's
                ratio is not above −1 and below 1/2.
        """
        if not young_modulus > 0.:
            raise ValueError('the Young modulus must be positive')

        if not -1. < poisson_ratio < .5:
            raise ValueError('the Poisson ratio must lie above -1 and below '
                             '1/2')

        self._young_modulus = float(young_modulus)
        self._poisson_ratio = float(poisson_ratio)
        self._plane_stress = bool(plane_stress)

    @property
    def young_modulus(self) -> float:
        """Young's modulus E of the solid."""
        return self._young_modulus

    @property
    def poisson_ratio(self) -> float:
        """Poisson's ratio ν of the solid."""
        return self._poisson_ratio

    @property
    def plane_stress(self) -> bool:
        """Whether the solid is in plane stress, rather than plane strain."""
        return self._plane_stress

    @property
    def lame(self) -> float:
        """First Lamé parameter λ of the solid, in its state in the plane."""
        young, nu = self._young_modulus, self._poisson_ratio

        if self._plane_stress:
            return young * nu / (1. - nu * nu)

        return young * nu / ((1. + nu) * (1. - 2. * nu))

    @property
    def shear_modulus(self) -> float:
        """Shear modulus μ of the solid, the second Lamé parameter."""
        return self._young_modulus / (2. * (1. + self._poisson_ratio))

    @property
    def u(self) -> ElasticityElement.U:
        """Trace of the displacement u, on which a condition imposes
        values."""
        return ElasticityElement.U(self)

    def _cpp_element(self) -> _cpp.ElasticityElement2d:
        """The compiled element, on a planar patch, the only one problems
        lie on so far."""
        return _cpp.ElasticityElement2d(self.lame, self.shear_modulus)

    def __repr__(self) -> str:
        plane_stress = ', plane_stress=True' if self._plane_stress else ''
        return (f'ElasticityElement('
                f'young_modulus={self._young_modulus}, '
                f'poisson_ratio={self._poisson_ratio}{plane_stress})')
