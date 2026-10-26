# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""PETSc problems, the solution of a problem on a function space

A problem compiles the element, quadrature, conditions and stabilizations
of a problem on a space into one form, assembles the form into PETSc
matrices and vectors, each pass over its integrals in one compiled call,
and solves the system with a PETSc solver set up from options. Its data,
the source, the values of its conditions and the previous solution of a
transient problem, change in place between solves, where the form reads
them, so that a time loop keeps one problem. It needs a build of iguana
with PETSc.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from petsc4py import PETSc

from iguana import cpp as _cpp
from iguana.condition import (NeumannCondition, NitscheCondition,
                              PenaltyCondition, SkeletonNitscheCondition)
from iguana.element import PoissonElement, StokesElement
from iguana.fspace import FunctionSpace, MixedFunctionSpace
from iguana.patch import PlanarPatch
from iguana.quadrature import QuadratureDomain
from iguana.stabilization import GhostPenalty


class _Problem:
    """Problem of an element on a space, its form assembled into PETSc, the
    setup shared by the problems of this module."""

    def __init__(
        self,
        element: PoissonElement | StokesElement,
        space: FunctionSpace | MixedFunctionSpace,
        quadrature: QuadratureDomain,
        source: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
        *,
        cell_weights: npt.ArrayLike | None,
        conditions: Sequence[PenaltyCondition | NitscheCondition
                             | NeumannCondition | SkeletonNitscheCondition],
        stabilizations: Sequence[GhostPenalty],
        pressure_reference: npt.ArrayLike | None,
    ) -> None:
        """Initialize the form and the assembler of the problem, checking
        its terms as the problems document."""
        if not isinstance(element, (PoissonElement, StokesElement)):
            raise TypeError('the element must be a Poisson or a Stokes '
                            'element')

        if not isinstance(space, (FunctionSpace, MixedFunctionSpace)):
            raise TypeError('the space must be a function space or a mixed '
                            'space')

        # A coefficient per unknown of the element, the components of one
        # field or a field each
        if not _holds_unknowns(space, element._num_unknowns):
            raise ValueError('the space must give each unknown of the '
                             'element a component of one field or a field '
                             'of its own')

        # Boundaries and faces lie in the plane, and so do the problems so far
        patch = space.patch

        if not isinstance(patch, PlanarPatch):
            raise TypeError('the space must lie on a planar patch')

        if not isinstance(quadrature, QuadratureDomain):
            raise TypeError('the quadrature must be a domain quadrature')

        # The system numbers the cells as the elements of the patch, which
        # another patch or a refined grid would not
        refined = np.any(quadrature.grid.levels != 0)

        if quadrature.patch is not patch or refined:
            raise ValueError('the quadrature must lie on the elements of the '
                             'patch of the space')

        conditions = list(conditions)
        stabilizations = list(stabilizations)
        _check_terms(element, patch, conditions, stabilizations)

        on_faces = any(isinstance(condition, SkeletonNitscheCondition)
                       for condition in conditions)

        # A column per component of the source, contiguous, as the form
        # reads each where it lies
        self._quadrature = quadrature
        self._source_shape = element._source_shape
        self._source = np.empty((quadrature.num_points,
                                 math.prod(self._source_shape)), order='F')
        self.source = source
        self._cell_weights = _cell_weights(quadrature, cell_weights, on_faces)

        self._pinned = _pinned(element, space, pressure_reference)

        # The form reads the data where it lies, so a source or values set
        # later reach it without building it again
        self._form = _cpp.Form2d()
        self._element = element._cpp_element()
        self._form.add_domain_integral(self._element, quadrature._cpp_object,
                                       self._source, self._cell_weights)

        for condition in conditions:
            if isinstance(condition, SkeletonNitscheCondition):
                condition._add_to(self._form, self._cell_weights)
            else:
                condition._add_to(self._form)

        for stabilization in stabilizations:
            stabilization._add_to(self._form)

        # Terms across faces couple the cells on either side
        couple_faces = on_faces or bool(stabilizations)
        self._assembler = _cpp.Assembler2d(space._cpp_object,
                                           patch._cpp_object, couple_faces)
        self._tangent = self._assembler.tangent
        self._mass = self._assembler.mass
        self._residual = self._assembler.residual

        # The solution has a coefficient per degree of freedom and
        # component, field after field on a mixed space
        self._space = space
        self._x = self._residual.duplicate()

    @property
    def source(self) -> npt.NDArray[np.float64]:
        """Source at each point of the quadrature, of shape `(num_points,)`
        for the Poisson element and `(num_points, 2)` for the Stokes
        element. It is set as at initialization, by its values or by a
        function of the positions of the points."""
        return self._source.reshape(-1, *self._source_shape).copy()

    @source.setter
    def source(
        self,
        source: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
    ) -> None:
        if callable(source):
            source = source(self._quadrature.positions)

        source = np.asarray(source, dtype=np.float64)

        if source.shape != (self._quadrature.num_points,
                            *self._source_shape):
            raise ValueError('the source must give each point the values the '
                             'element takes')

        # In place, where the form reads it
        self._source[...] = source.reshape(len(self._source), -1)


class LinearProblem(_Problem):
    """Linear problem of an element on a space, assembled into PETSc and
    solved by a PETSc linear solver."""

    def __init__(
        self,
        element: PoissonElement | StokesElement,
        space: FunctionSpace | MixedFunctionSpace,
        quadrature: QuadratureDomain,
        source: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
        *,
        petsc_options_prefix: str,
        cell_weights: npt.ArrayLike | None = None,
        conditions: Sequence[PenaltyCondition | NitscheCondition
                             | NeumannCondition
                             | SkeletonNitscheCondition] = (),
        stabilizations: Sequence[GhostPenalty] = (),
        step: float | None = None,
        pressure_reference: npt.ArrayLike | None = None,
        petsc_options: dict[str, object] | None = None,
    ) -> None:
        """Initialize the problem of an element on a space.

        The solution solves K u = F, with the stiffness K and load F
        assembled over the quadrature, along the boundaries of the
        conditions and across the faces of the stabilizations. Given a time
        step, each solve takes one step of backward Euler of the transient
        problem instead, from the previous solution u_p: K becomes
        K + M / step and F becomes F + M u_p / step, with the mass M of the
        element weighted by the cells like its other terms.

        With the velocity of a fluid imposed on the whole boundary, the
        pressure is only determined up to a constant, which a pressure
        reference fixes: the pressure of the function whose control point
        lies nearest the reference is set to zero, by its row and column of
        the system after each assembly. Any point serves, the constant
        being the same everywhere.

        Args:
            element: The element stating the problem, a Poisson or a Stokes
                element.
            space: The space of the solution, on a planar patch: a function
                space with a component per unknown of the element, or a
                mixed space with a field of one component each, such as
                the divergence-conforming space of the Galerkin Stokes
                element.
            quadrature: A quadrature over the elements of the patch of the
                space, not over a refined grid.
            source: The source at each point of the quadrature, in its
                order, one value per point for the Poisson element and the
                two components of the body force for the Stokes element, or
                a function that takes their positions, of shape
                `(num_points, 2)`, and returns those values.
            petsc_options_prefix: The prefix of the options of the PETSc
                solver, unique to the problem, such as `'poisson_'`.
            cell_weights: The weight of the test functions of the element on
                each cell, in the order of the elements of the patch, such as
                the volume fractions of the generalized shifted boundary
                method. Cells of weight zero add nothing. Without them, every
                cell weighs one.
            conditions: Conditions on traces of the element along
                boundaries in the patch of the space, penalties or Nitsche
                conditions imposing values, or Neumann conditions loading
                fluxes, or Nitsche conditions on the faces where the cell
                weights jump, which need them.
            stabilizations: Ghost penalties on traces of the element across
                the faces of the cells of the patch of the space, which need
                an affine map. They extend the field beyond the physical
                domain, as a space keeping the functions of every cell
                needs, the velocity and the pressure of a fluid each by its
                own.
            step: The time step of backward Euler, positive. Without it, the
                problem is steady.
            pressure_reference: The point near which the pressure is zero,
                of shape `(2,)`, for an element with a pressure. Without it,
                nothing fixes the constant pressure.
            petsc_options: Options of the PETSc linear solver, such as
                `{'ksp_type': 'gmres', 'pc_type': 'ilu'}`. Without them, the
                solver factorizes the system, with MUMPS when PETSc has it.

        Raises:
            TypeError: If the element is neither a Poisson nor a Stokes
                element, if the space is neither a function space nor a
                mixed space on a planar patch, if the quadrature is not a
                domain quadrature, if a condition is neither a penalty, a
                Nitsche, a Neumann nor a skeleton Nitsche condition, or if a
                stabilization is not a ghost penalty.
            ValueError: If the space does not give each unknown of the
                element a component of one field or a field of its own, if
                the quadrature, the boundary of a condition or
                the cells of a stabilization do not lie on the elements of
                the patch of the space, if a condition or a stabilization
                acts on a trace of another element, if the source does not
                give each point the values the element takes, if the cell
                weights are not one per cell, if a skeleton Nitsche
                condition is given without them, if the time step is not
                positive or is given with a skeleton Nitsche condition
                shifted with the equation, or if a pressure reference is
                given for an element without a pressure or is not a point of
                the plane.
        """
        conditions = list(conditions)
        super().__init__(element, space, quadrature, source,
                         cell_weights=cell_weights, conditions=conditions,
                         stabilizations=stabilizations,
                         pressure_reference=pressure_reference)

        if step is not None and not step > 0.:
            raise ValueError('the time step must be positive')

        # A shift rewritten with the equation holds −Δu = f, which a time
        # step changes
        if step is not None and any(
                isinstance(condition, SkeletonNitscheCondition)
                and condition._source is not None
                for condition in conditions):
            raise ValueError('a condition shifted with the equation needs a '
                             'steady problem')

        self._step = None if step is None else float(step)

        # The time derivative is weighted by the cells like the other terms
        if self._step is not None:
            self._form.add_mass_integral(self._element,
                                         quadrature._cpp_object,
                                         self._cell_weights)

        # A steady system is the stiffness and load themselves, a transient
        # one their sums with the mass, on the pattern of the stiffness
        if self._step is None:
            self._A = self._tangent
            self._b = self._residual
            self._previous = None
        else:
            self._A = self._tangent.duplicate()
            self._b = self._residual.duplicate()
            self._previous = self._residual.duplicate()
            self._previous.zeroEntries()

        self._solver = _linear_solver(self._A, petsc_options_prefix,
                                      petsc_options)

    @property
    def A(self) -> PETSc.Mat:
        """The operator of the system, the stiffness of a steady problem."""
        return self._A

    @property
    def b(self) -> PETSc.Vec:
        """The right-hand side of the system, the load of a steady
        problem."""
        return self._b

    @property
    def solver(self) -> PETSc.KSP:
        """The PETSc linear solver of the system."""
        return self._solver

    @property
    def previous(
        self,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...] | None:
        """Coefficients of each degree of freedom at the previous time, of
        shape `(num_dofs,)`, or `(num_dofs, block_size)` for a space of
        several components, or those of each field of a mixed space, zero
        at first, from which a transient problem takes its next step. A
        steady problem has none. On a space that keeps the functions of
        every cell, as the generalized shifted boundary method does, the
        degrees of freedom stay the same however the boundary moves."""
        if self._previous is None:
            return None

        return _coefficients(self._space, self._previous.getArray())

    @previous.setter
    def previous(
        self,
        previous: npt.ArrayLike | Sequence[npt.ArrayLike],
    ) -> None:
        if self._previous is None:
            raise ValueError('a steady problem has no previous solution')

        self._previous.setArray(_entries(self._space, previous))

    def solve(
        self,
    ) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...]:
        """Assemble the system with the current data and solve it.

        Returns:
            The coefficients of each degree of freedom, of shape
            `(num_dofs,)`, or `(num_dofs, block_size)` for a space of
            several components, the velocity and the pressure of a Stokes
            element in that order. On a mixed space, a tuple of those of
            each field, in its order.

        Raises:
            ValueError: If a stabilization is given on a patch whose map is
                not affine, or that repeats an interior knot.
            RuntimeError: If the linear solver does not converge.
        """
        # Anew from the current data, each pass over every integral in one
        # call. At zero coefficients the tangent is the stiffness K and the
        # residual K 0 - F, so that its negative is the load
        self._assembler.clear()
        self._x.zeroEntries()
        self._form.assemble_tangent(self._assembler, self._x)
        self._form.assemble_residual(self._assembler, self._x)
        self._residual.scale(-1.)

        # Backward Euler: (K + M / step) u = F + M u_p / step
        if self._step is not None:
            self._form.assemble_mass(self._assembler)
            same = PETSc.Mat.Structure.SAME_NONZERO_PATTERN
            self._tangent.copy(self._A, same)
            self._A.axpy(1. / self._step, self._mass, same)
            self._mass.mult(self._previous, self._b)
            self._b.aypx(1. / self._step, self._residual)

        # The pinned pressure, zero as the coefficients still are, left out
        # of the other equations
        if self._pinned is not None:
            self._A.zeroRowsColumns([self._pinned], diag=1., x=self._x,
                                    b=self._b)

        self._solver.solve(self._b, self._x)

        if self._solver.getConvergedReason() < 0:
            raise RuntimeError('the linear solver did not converge')

        return _coefficients(self._space, self._x.getArray())


def _check_terms(
    element: PoissonElement | StokesElement,
    patch: PlanarPatch,
    conditions: list[PenaltyCondition | NitscheCondition | NeumannCondition
                     | SkeletonNitscheCondition],
    stabilizations: list[GhostPenalty],
) -> None:
    """Check that the conditions and stabilizations of a problem are of the
    kinds it assembles, act on traces of its element and lie on the patch of
    its space."""
    # The pressure of a fluid has a ghost penalty, but no condition
    traces = ((StokesElement.U, StokesElement.P)
              if isinstance(element, StokesElement) else PoissonElement.U)

    for condition in conditions:
        if not isinstance(condition, (PenaltyCondition, NitscheCondition,
                                      NeumannCondition,
                                      SkeletonNitscheCondition)):
            raise TypeError('the conditions must be penalty, Nitsche, '
                            'Neumann or skeleton Nitsche conditions')

        if not isinstance(condition.trace, type(element).U):
            raise ValueError('the trace of a condition must be one of the '
                             'element')

        if isinstance(condition, SkeletonNitscheCondition):
            if condition.patch is not patch:
                raise ValueError('the faces of a condition must lie on the '
                                 'patch of the space')
        elif condition.quadrature.patch is not patch:
            raise ValueError('the boundary of a condition must lie on the '
                             'patch of the space')

    for stabilization in stabilizations:
        if not isinstance(stabilization, GhostPenalty):
            raise TypeError('the stabilizations must be ghost penalties')

        if not isinstance(stabilization.trace, traces):
            raise ValueError('the trace of a stabilization must be one of '
                             'the element')

        if stabilization.patch is not patch:
            raise ValueError('the cells of a stabilization must lie on the '
                             'patch of the space')


def _holds_unknowns(
    space: FunctionSpace | MixedFunctionSpace,
    num_unknowns: int,
) -> bool:
    """Whether a space gives each of a number of unknowns a component of its
    one field, or a field of one component each."""
    if isinstance(space, FunctionSpace):
        return space.block_size == num_unknowns

    return (len(space.fields) == num_unknowns
            and all(field.block_size == 1 for field in space.fields))


def _shape(space: FunctionSpace) -> tuple[int, ...]:
    """Shape of the coefficients of a space, a value per degree of freedom
    and component."""
    if space.block_size == 1:
        return (space.num_dofs,)

    return (space.num_dofs, space.block_size)


def _coefficients(
    space: FunctionSpace | MixedFunctionSpace,
    entries: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...]:
    """Coefficients of each degree of freedom of a space from the entries of
    a vector of its system, or those of each field of a mixed space."""
    if isinstance(space, FunctionSpace):
        return entries.reshape(_shape(space)).copy()

    return tuple(entries[offset:offset + math.prod(_shape(field))]
                 .reshape(_shape(field)).copy()
                 for offset, field in zip(space.offsets, space.fields))


def _entries(
    space: FunctionSpace | MixedFunctionSpace,
    coefficients: npt.ArrayLike | Sequence[npt.ArrayLike],
) -> npt.NDArray[np.float64]:
    """Entries of a vector of the system of a space from the coefficients of
    each degree of freedom, or those of each field of a mixed space,
    checked to give every one."""
    if isinstance(space, FunctionSpace):
        spaces, fields = [space], [coefficients]
    else:
        spaces, fields = space.fields, list(coefficients)

    fields = [np.asarray(field, dtype=np.float64) for field in fields]

    mismatched = len(fields) != len(spaces) or any(
        field.shape != _shape(each) for field, each in zip(fields, spaces))

    if mismatched:
        raise ValueError('the previous solution must give the '
                         'coefficients of every degree of freedom')

    return np.concatenate([field.ravel() for field in fields])


def _pinned(
    element: PoissonElement | StokesElement,
    space: FunctionSpace | MixedFunctionSpace,
    pressure_reference: npt.ArrayLike | None,
) -> int | None:
    """Row of the system of the pressure pinned near a reference point, that
    of the function whose control point lies nearest, or none without a
    reference."""
    if pressure_reference is None:
        return None

    if element._pressure_component is None:
        raise ValueError('a pressure reference needs an element with a '
                         'pressure')

    point = np.asarray(pressure_reference, dtype=np.float64)

    if point.shape != (2,):
        raise ValueError('the pressure reference must be a point of the '
                         'plane')

    # The pressure is a component of the one field, or a field of its own
    component = element._pressure_component
    field, offset = space, 0

    if isinstance(space, MixedFunctionSpace):
        field, offset = space.fields[component], space.offsets[component]
        component = 0

    distances = np.linalg.norm(field.control_points - point, axis=1)
    nearest = int(np.argmin(distances))

    return offset + field.block_size * nearest + component


def _cell_weights(
    quadrature: QuadratureDomain,
    cell_weights: npt.ArrayLike | None,
    on_faces: bool,
) -> npt.NDArray[np.float64]:
    """Weights of the test functions on the cells of a quadrature, one
    without them, checked to be one per cell. They are contiguous, as the
    form of the problem reads them where they lie."""
    # Without weights, every skeleton condition would see no jump
    if cell_weights is None and on_faces:
        raise ValueError('skeleton Nitsche conditions need cell weights')

    # The quadrature lies on the elements of the patch, its cells
    num_cells = quadrature.grid.num_elements

    if cell_weights is None:
        return np.ones(num_cells)

    cell_weights = np.ascontiguousarray(cell_weights, dtype=np.float64)

    if cell_weights.shape != (num_cells,):
        raise ValueError('the cell weights must give one value per cell')

    return cell_weights


def _linear_solver(
    operator: PETSc.Mat,
    prefix: str,
    options: dict[str, object] | None,
) -> PETSc.KSP:
    """PETSc linear solver of an operator, set up from options under a
    prefix."""
    if options is None:
        options = {'ksp_type': 'preonly', 'pc_type': 'lu'}

        # MUMPS also factorizes the indefinite systems of mixed problems
        if PETSc.Sys.hasExternalPackage('mumps'):
            options['pc_factor_mat_solver_type'] = 'mumps'

    solver = PETSc.KSP().create(PETSc.COMM_SELF)
    solver.setOperators(operator)
    _set_from_options(solver, prefix, options)

    return solver


def _set_from_options(
    solver: PETSc.KSP,
    prefix: str,
    options: dict[str, object],
) -> None:
    """Set up a PETSc solver from options under a prefix, which leave the
    options database once applied."""
    solver.setOptionsPrefix(prefix)

    # By their full names, as PETSc prefixes the values it sets but not
    # those it clears
    database = PETSc.Options()

    for key, value in options.items():
        database[prefix + key] = value

    solver.setFromOptions()

    for key in options:
        del database[prefix + key]
