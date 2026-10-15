# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the linear problems on a function space, solved with PETSc"""

import numpy as np
import pytest

import iguana
from iguana import (CellType, FunctionSpace, HierarchicalGrid,
                    NitscheCondition, PoissonElement, QuadratureBoundary,
                    QuadratureDomain, SurrogateBoundary)

try:
    from iguana.petsc import LinearProblem
except ImportError:
    # A build without PETSc skips every test, which all solve
    LinearProblem = None

pytestmark = pytest.mark.petsc


def rectangle():
    """Rectangle of sides 2 and 3 with quadratic elements, three along x and
    two along y."""
    return iguana.create_rectangle(lengths=(2., 3.), elements=(3, 2))


def gauss_quadrature(patch):
    """Gauss quadrature of a patch whose cells all lie inside, exact for the
    products of quadratic functions through an affine map."""
    quadrature = QuadratureDomain(patch)
    quadrature.fill_gauss_legendre(CellType.inside, 3)

    return quadrature


def edges(patch):
    """Quadrature of the boundary of a patch, the surrogate boundary of its
    cells, all inside."""
    num_cells = HierarchicalGrid(patch.degrees, patch.knots).num_elements
    surrogate = SurrogateBoundary(patch, [CellType.inside] * num_cells)

    return QuadratureBoundary(patch, surrogate, 3)


def coefficients(knots):
    """Coefficients of t (1 - t) in a quadratic B-spline basis, the polar
    form (t_{i+1} + t_{i+2}) / 2 - t_{i+1} t_{i+2}."""
    first, second = knots[1:-2], knots[2:-1]

    return (first + second) / 2. - first * second


def test_quadratic_solution_is_exact():
    # More elements along x than along y, so that the order of the degrees
    # of freedom matters, and sides that set positions apart from parameters
    width, height = 2., 3.
    patch = iguana.create_rectangle(lengths=(width, height), elements=(4, 3))
    space = FunctionSpace(patch)
    element = PoissonElement()

    def source(points):
        x, y = points.T
        return 2. * (x * (width - x) + y * (height - y))

    # The quadratic splines hold u = x (width - x) y (height - y), which
    # vanishes on the boundary, so Nitsche's method, consistent, finds it
    # exactly
    condition = NitscheCondition(element.u, edges(patch),
                                 lambda x: np.zeros(len(x)), 40.)
    problem = LinearProblem(element, space, gauss_quadrature(patch), source,
                            conditions=[condition],
                            petsc_options_prefix='quadratic_')
    u = problem.solve()

    # In parameters, x (width - x) is width^2 t (1 - t). With every cell
    # inside, each function keeps its own index, the first direction
    # running fastest
    along_x = width ** 2 * coefficients(patch.knots[0])
    along_y = height ** 2 * coefficients(patch.knots[1])

    assert np.allclose(u, np.kron(along_y, along_x), rtol=0., atol=1e-12)


def test_backward_euler_steps_a_field_linear_in_time():
    """Backward Euler is exact for a field linear in time, and the space
    holds a field linear in space, so steps of u = t (b . x) from zero, its
    boundary values imposed by Nitsche's method, reproduce it at every
    step."""
    patch = rectangle()
    space = FunctionSpace(patch)
    element = PoissonElement()
    b = np.array([1., -2.])
    field = space.control_points @ b

    # du/dt = b . x, and the Laplacian vanishes
    condition = NitscheCondition(element.u, edges(patch),
                                 lambda x: np.zeros(len(x)), 40.)
    problem = LinearProblem(element, space, gauss_quadrature(patch),
                            lambda x: x @ b, conditions=[condition],
                            step=.1,
                            petsc_options_prefix='linear_in_time_')

    for time in np.arange(1, 11) / 10.:
        condition.values = lambda x, time=time: time * (x @ b)
        u = problem.solve()
        problem.previous = u

        assert np.allclose(u, time * field, rtol=0., atol=1e-12)


def test_backward_euler_is_first_order_in_time():
    """For u = t^2 (b . x), imposed on the boundary, the space adds no
    error, so the error at time one is that of backward Euler, which halves
    with the time step."""
    patch = rectangle()
    space = FunctionSpace(patch)
    element = PoissonElement()
    b = np.array([1., -2.])
    field = space.control_points @ b
    errors = []

    for step in (.1, .05, .025):
        condition = NitscheCondition(element.u, edges(patch),
                                     lambda x: np.zeros(len(x)), 40.)
        problem = LinearProblem(element, space, gauss_quadrature(patch),
                                lambda x: np.zeros(len(x)),
                                conditions=[condition], step=step,
                                petsc_options_prefix='first_order_')

        # du/dt = 2 t (b . x), and the Laplacian vanishes
        for time in step * np.arange(1, round(1. / step) + 1):
            problem.source = lambda x, time=time: 2. * time * (x @ b)
            condition.values = lambda x, time=time: time**2 * (x @ b)
            problem.previous = problem.solve()

        errors.append(np.abs(problem.previous - field).max())

    assert errors[0] / errors[1] == pytest.approx(2., rel=.01)
    assert errors[1] / errors[2] == pytest.approx(2., rel=.01)


def test_petsc_options_set_up_the_solver():
    patch = rectangle()
    space = FunctionSpace(patch)
    quadrature = gauss_quadrature(patch)
    element = PoissonElement()
    b = np.array([1., -2.])
    condition = NitscheCondition(element.u, edges(patch), lambda x: x @ b,
                                 40.)

    def problem(prefix, options=None):
        return LinearProblem(element, space, quadrature,
                             np.ones(quadrature.num_points),
                             conditions=[condition],
                             petsc_options_prefix=prefix,
                             petsc_options=options)

    # An iterative solver, chosen by its options, converges to the solution
    # of the default factorization, within its tolerance
    direct = problem('direct_')
    iterative = problem('iterative_', {'ksp_type': 'gmres',
                                       'pc_type': 'jacobi',
                                       'ksp_rtol': 1e-12})

    assert direct.solver.getType() == 'preonly'
    assert iterative.solver.getType() == 'gmres'
    assert np.allclose(iterative.solve(), direct.solve(), rtol=0.,
                       atol=1e-9)

    # The options leave the database once applied, so another problem of
    # the same prefix takes the defaults
    assert problem('iterative_').solver.getType() == 'preonly'

    # A solver stopped before it converges fails the solve
    stopped = problem('stopped_', {'ksp_type': 'gmres', 'pc_type': 'none',
                                   'ksp_max_it': 1})

    with pytest.raises(RuntimeError):
        stopped.solve()


def test_invalid_arguments():
    patch = rectangle()
    space = FunctionSpace(patch)
    quadrature = gauss_quadrature(patch)
    ones = np.ones(quadrature.num_points)
    element = PoissonElement()
    prefix = 'invalid_'

    with pytest.raises(TypeError):
        LinearProblem('element', space, quadrature, ones,
                      petsc_options_prefix=prefix)

    with pytest.raises(TypeError):
        LinearProblem(element, 'space', quadrature, ones,
                      petsc_options_prefix=prefix)

    # A surface has no physical gradients for the Poisson element, and a
    # volume no conditions so far
    surface = iguana.create_surface(degrees=(1, 1),
                                    knots=([0., 0., 1., 1.],
                                           [0., 0., 1., 1.]),
                                    control_points=np.eye(4, 3))
    box = iguana.create_box(lengths=(1., 1., 1.), elements=(1, 1, 1))

    for other in (surface, box):
        with pytest.raises(TypeError):
            LinearProblem(element, FunctionSpace(other), quadrature, ones,
                          petsc_options_prefix=prefix)

    with pytest.raises(TypeError):
        LinearProblem(element, space, 'quadrature', ones,
                      petsc_options_prefix=prefix)

    # Neither another patch nor a refined grid numbers the cells as the
    # elements of the patch of the space
    with pytest.raises(ValueError):
        LinearProblem(element, space, gauss_quadrature(rectangle()), ones,
                      petsc_options_prefix=prefix)

    refined = QuadratureDomain(
        patch, grid=HierarchicalGrid(patch.degrees, patch.knots).refine([0]))
    refined.fill_gauss_legendre(CellType.inside, 3)

    with pytest.raises(ValueError):
        LinearProblem(element, space, refined, np.ones(refined.num_points),
                      petsc_options_prefix=prefix)

    # A column is not one value per point, however many rows it has
    with pytest.raises(ValueError):
        LinearProblem(element, space, quadrature, ones[:, np.newaxis],
                      petsc_options_prefix=prefix)

    # One weight short of the six cells
    with pytest.raises(ValueError):
        LinearProblem(element, space, quadrature, ones,
                      cell_weights=np.ones(5), petsc_options_prefix=prefix)

    # A previous solution belongs to a transient problem, whose step is
    # positive, and gives one coefficient per degree of freedom
    steady = LinearProblem(element, space, quadrature, ones,
                           petsc_options_prefix=prefix)

    assert steady.previous is None

    with pytest.raises(ValueError):
        steady.previous = np.zeros(space.num_dofs)

    with pytest.raises(ValueError):
        LinearProblem(element, space, quadrature, ones, step=0.,
                      petsc_options_prefix=prefix)

    transient = LinearProblem(element, space, quadrature, ones, step=.1,
                              petsc_options_prefix=prefix)

    with pytest.raises(ValueError):
        transient.previous = np.zeros(space.num_dofs - 1)
