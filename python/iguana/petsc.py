# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""PETSc problems, the solution of a problem on a function space

A problem compiles the element, quadrature, conditions and stabilizations
of a problem on a space into one form, assembles the form in a single
compiled call into PETSc matrices and vectors, and solves the system with a
PETSc solver set up from options. Its data, the source, the values of its
conditions and the previous solution of a transient problem, change in
place between solves, where the form reads them, so that a time loop keeps
one problem. It needs a build of iguana with PETSc.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt
from petsc4py import PETSc

from iguana import cpp as _cpp
from iguana.condition import (NeumannCondition, NitscheCondition,
                              PenaltyCondition, SkeletonNitscheCondition)
from iguana.element import PoissonElement
from iguana.fspace import FunctionSpace
from iguana.patch import PlanarPatch
from iguana.quadrature import QuadratureDomain
from iguana.stabilization import GhostPenalty


class LinearProblem:
    """Linear problem of an element on a space, assembled into PETSc and
    solved by a PETSc linear solver."""

    def __init__(
        self,
        element: PoissonElement,
        space: FunctionSpace,
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

        Args:
            element: The element stating the problem.
            space: The space of the solution, on a planar patch.
            quadrature: A quadrature over the elements of the patch of the
                space, not over a refined grid.
            source: The value of the source at each point of the quadrature,
                in its order, or a function that takes their positions, of
                shape `(num_points, 2)`, and returns those values.
            petsc_options_prefix: The prefix of the options of the PETSc
                solver, unique to the problem, such as `'poisson_'`.
            cell_weights: The weight of the test functions of the element on
                each cell, in the order of the elements of the patch, such as
                the volume fractions of the generalized shifted boundary
                method. Cells of weight zero add nothing. Without them, every
                cell weighs one.
            conditions: Conditions along boundaries in the patch of the
                space, penalties or Nitsche conditions imposing values, or
                Neumann conditions loading fluxes, or Nitsche conditions on
                the faces where the cell weights jump, which need them.
            stabilizations: Ghost penalties across the faces of the cells of
                the patch of the space, which need an affine map. They
                extend the field beyond the physical domain, as a space
                keeping the functions of every cell needs.
            step: The time step of backward Euler, positive. Without it, the
                problem is steady.
            petsc_options: Options of the PETSc linear solver, such as
                `{'ksp_type': 'gmres', 'pc_type': 'ilu'}`. Without them, the
                solver factorizes the system, with MUMPS when PETSc has it.

        Raises:
            TypeError: If the element is not a Poisson element, if the space
                is not a function space on a planar patch, if the quadrature
                is not a domain quadrature, if a condition is neither a
                penalty, a Nitsche, a Neumann nor a skeleton Nitsche
                condition, or if a stabilization is not a ghost penalty.
            ValueError: If the quadrature, the boundary of a condition or the
                cells of a stabilization do not lie on the elements of the
                patch of the space, if the source does not give one value per
                point, if the cell weights are not one per cell, if a
                skeleton Nitsche condition is given without them, or if the
                time step is not positive.
        """
        if not isinstance(element, PoissonElement):
            raise TypeError('the element must be a Poisson element')

        if not isinstance(space, FunctionSpace):
            raise TypeError('the space must be a function space')

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
        _check_terms(patch, conditions, stabilizations)

        on_faces = any(isinstance(condition, SkeletonNitscheCondition)
                       for condition in conditions)

        self._quadrature = quadrature
        self._source = np.empty(quadrature.num_points)
        self.source = source
        self._cell_weights = _cell_weights(quadrature, cell_weights, on_faces)

        if step is not None and not step > 0.:
            raise ValueError('the time step must be positive')

        self._step = None if step is None else float(step)

        # The form reads the data where it lies, so a source or values set
        # later reach it without building it again
        self._form = _cpp.Form2d()
        cpp_element = element._cpp_element()
        self._form.add_domain_integral(cpp_element, quadrature._cpp_object,
                                       self._source, self._cell_weights)

        # The time derivative is weighted by the cells like the other terms
        if self._step is not None:
            self._form.add_mass_integral(cpp_element, quadrature._cpp_object,
                                         self._cell_weights)

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
        self._stiffness = self._assembler.stiffness
        self._mass = self._assembler.mass
        self._load = self._assembler.load

        # A steady system is the stiffness and load themselves, a transient
        # one their sums with the mass, on the pattern of the stiffness
        if self._step is None:
            self._A = self._stiffness
            self._b = self._load
            self._previous = None
        else:
            self._A = self._stiffness.duplicate()
            self._b = self._load.duplicate()
            self._previous = self._load.duplicate()
            self._previous.zeroEntries()

        self._x = self._load.duplicate()
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
    def source(self) -> npt.NDArray[np.float64]:
        """Value of the source at each point of the quadrature, of shape
        `(num_points,)`. It is set as at initialization, by its values or
        by a function of the positions of the points."""
        return self._source.copy()

    @source.setter
    def source(
        self,
        source: (npt.ArrayLike
                 | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
    ) -> None:
        if callable(source):
            source = source(self._quadrature.positions)

        source = np.asarray(source, dtype=np.float64)

        if source.shape != (self._quadrature.num_points,):
            raise ValueError('the source must give one value per point')

        # In place, where the form reads it
        self._source[...] = source

    @property
    def previous(self) -> npt.NDArray[np.float64] | None:
        """Coefficient of each degree of freedom at the previous time, of
        shape `(num_dofs,)`, zero at first, from which a transient problem
        takes its next step. A steady problem has none. On a space that
        keeps the functions of every cell, as the generalized shifted
        boundary method does, the degrees of freedom stay the same however
        the boundary moves."""
        if self._previous is None:
            return None

        return self._previous.getArray().copy()

    @previous.setter
    def previous(self, previous: npt.ArrayLike) -> None:
        if self._previous is None:
            raise ValueError('a steady problem has no previous solution')

        previous = np.asarray(previous, dtype=np.float64)

        if previous.shape != (self._previous.getSize(),):
            raise ValueError('the previous solution must give one '
                             'coefficient per degree of freedom')

        self._previous.setArray(previous)

    def solve(self) -> npt.NDArray[np.float64]:
        """Assemble the system with the current data and solve it.

        Returns:
            The coefficient of each degree of freedom, of shape
            `(num_dofs,)`.

        Raises:
            ValueError: If a stabilization is given on a patch whose map is
                not affine.
            RuntimeError: If the linear solver does not converge.
        """
        # Anew from the current data, every integral in one call
        self._assembler.clear()
        self._form.assemble(self._assembler)

        # Backward Euler: (K + M / step) u = F + M u_p / step
        if self._step is not None:
            same = PETSc.Mat.Structure.SAME_NONZERO_PATTERN
            self._stiffness.copy(self._A, same)
            self._A.axpy(1. / self._step, self._mass, same)
            self._mass.mult(self._previous, self._b)
            self._b.aypx(1. / self._step, self._load)

        self._solver.solve(self._b, self._x)

        if self._solver.getConvergedReason() < 0:
            raise RuntimeError('the linear solver did not converge')

        return self._x.getArray().copy()


def _check_terms(
    patch: PlanarPatch,
    conditions: list[PenaltyCondition | NitscheCondition | NeumannCondition
                     | SkeletonNitscheCondition],
    stabilizations: list[GhostPenalty],
) -> None:
    """Check that the conditions and stabilizations of a problem are of the
    kinds it assembles and lie on the patch of its space."""
    for condition in conditions:
        if not isinstance(condition, (PenaltyCondition, NitscheCondition,
                                      NeumannCondition,
                                      SkeletonNitscheCondition)):
            raise TypeError('the conditions must be penalty, Nitsche, '
                            'Neumann or skeleton Nitsche conditions')

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

        if stabilization.patch is not patch:
            raise ValueError('the cells of a stabilization must lie on the '
                             'patch of the space')


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
    prefix, which leave the options database once applied."""
    if options is None:
        options = {'ksp_type': 'preonly', 'pc_type': 'lu'}

        # MUMPS also factorizes the indefinite systems of mixed problems
        if PETSc.Sys.hasExternalPackage('mumps'):
            options['pc_factor_mat_solver_type'] = 'mumps'

    solver = PETSc.KSP().create(PETSc.COMM_SELF)
    solver.setOperators(operator)
    solver.setOptionsPrefix(prefix)

    database = PETSc.Options()
    database.prefixPush(prefix)

    for key, value in options.items():
        database[key] = value

    database.prefixPop()
    solver.setFromOptions()

    database.prefixPush(prefix)

    for key in options:
        del database[key]

    database.prefixPop()

    return solver
