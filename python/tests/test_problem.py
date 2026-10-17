# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the linear problems on a function space, solved with PETSc"""

import numpy as np
import pytest

import iguana
from iguana import (Boundary, CellType, FunctionSpace, GhostPenalty,
                    HierarchicalGrid, NitscheCondition, PoissonElement,
                    QuadratureBoundary, QuadratureDomain,
                    SkeletonNitscheCondition, StokesElement,
                    SurrogateBoundary, classify_cells)

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


def quadratic_flow(patch):
    """Coefficients of the flow u = (x^2, −2 x y), which has no divergence,
    and of the pressure p = x + y on a rectangle of quadratic elements from
    the origin, of shape `(num_dofs, 3)`. In parameters, x is the width
    times t, whose coefficients are the means of the inner knots of each
    function, and those of t^2 their products; those of x + y are its values
    at the control points. With every cell inside, each function keeps its
    own index, the first direction running fastest."""
    width, height = patch.control_points.max(axis=0)
    t, s = ((knots[1:-2] + knots[2:-1]) / 2. for knots in patch.knots)
    t_squared = patch.knots[0][1:-2] * patch.knots[0][2:-1]

    return np.column_stack([np.kron(np.ones(len(s)), width ** 2 * t_squared),
                            -2. * np.kron(height * s, width * t),
                            patch.control_points.sum(axis=1)])


def stokes():
    """Element of a fluid, its velocity the quadratic flow, and the body
    force that drives it, f = −div(2μ ε(u)) + ∇p = (1 − 2μ, 1)."""
    element = StokesElement(1.5, 3.)

    def velocity(points):
        x, y = points.T
        return np.column_stack([x * x, -2. * x * y])

    def body_force(points):
        return np.tile([1. - 2. * element.viscosity, 1.], (len(points), 1))

    return element, velocity, body_force


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


def test_stokes_flow_is_exact():
    """The quadratic splines hold the flow and the pressure, so with the
    velocity imposed all around by Nitsche's method the Stokes element
    finds them exactly, the pressure up to the constant its reference
    fixes, zero at the function nearest the reference."""
    patch = rectangle()
    space = FunctionSpace(patch, block_size=3)
    element, velocity, body_force = stokes()

    condition = NitscheCondition(element.u, edges(patch), velocity,
                                 40. * element.viscosity)
    problem = LinearProblem(element, space, gauss_quadrature(patch),
                            body_force, conditions=[condition],
                            pressure_reference=(1., 1.6),
                            petsc_options_prefix='stokes_')
    u = problem.solve()
    exact = quadratic_flow(patch)
    nearest = np.argmin(np.linalg.norm(space.control_points - [1., 1.6],
                                       axis=1))

    assert u.shape == (space.num_dofs, 3)
    assert np.allclose(u[:, :2], exact[:, :2], rtol=0., atol=1e-12)
    assert np.allclose(u[:, 2] - u[nearest, 2], exact[:, 2]
                       - exact[nearest, 2], rtol=0., atol=1e-12)
    assert u[nearest, 2] == 0.


def test_stokes_flow_by_the_generalized_shifted_boundary_method():
    """Inside a square, with the volume fractions weighting the cells of a
    space that keeps every function, Nitsche's condition on the faces where
    they jump and ghost penalties on the velocity and the pressure find the
    flow and the pressure at every degree of freedom, those beyond the
    square included."""
    patch = iguana.create_rectangle(lengths=(2., 3.), elements=(6, 6))
    lower, upper = np.array([.5, 1.25]), np.array([1.5, 2.25])
    corners = np.array([lower, [upper[0], lower[1]], upper,
                        [lower[0], upper[1]]])
    square = Boundary([iguana.create_curve(
        degree=1, knots=[0., 0., 1., 1.],
        control_points=[corners[side], corners[(side + 1) % 4]])
        for side in range(4)])
    cell_types, fractions = classify_cells(patch, square)
    element, velocity, body_force = stokes()

    quadrature = QuadratureDomain(patch, cell_types)

    for cell_type in (CellType.inside, CellType.cut, CellType.outside):
        quadrature.fill_gauss_legendre(cell_type, 3)

    condition = SkeletonNitscheCondition(element.u, patch, cell_types, square,
                                         velocity, 40. * element.viscosity)
    ghosts = [GhostPenalty(trace, patch, cell_types, .1)
              for trace in (element.u, element.p)]
    u = LinearProblem(element, FunctionSpace(patch, block_size=3),
                      quadrature, body_force, cell_weights=fractions,
                      conditions=[condition], stabilizations=ghosts,
                      pressure_reference=(1., 1.75),
                      petsc_options_prefix='stokes_gsbm_').solve()
    exact = quadratic_flow(patch)

    # Exact to round-off, which the conditioning of the cells the penalties
    # alone determine amplifies
    assert np.abs(u[:, :2] - exact[:, :2]).max() < 1e-9
    assert np.abs((u[:, 2] - u[:, 2].mean())
                  - (exact[:, 2] - exact[:, 2].mean())).max() < 1e-9


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

    # The space of a fluid holds the velocity and the pressure, its source
    # is the body force, its conditions act on its velocity, and its
    # pressure reference is a point
    fluid, velocity, body_force = stokes()
    vector = FunctionSpace(patch, block_size=3)
    force = body_force(quadrature.positions)

    with pytest.raises(ValueError):
        LinearProblem(element, vector, quadrature, ones,
                      petsc_options_prefix=prefix)

    with pytest.raises(ValueError):
        LinearProblem(fluid, space, quadrature, force,
                      petsc_options_prefix=prefix)

    with pytest.raises(ValueError):
        LinearProblem(fluid, vector, quadrature, ones,
                      petsc_options_prefix=prefix)

    field = NitscheCondition(element.u, edges(patch),
                             lambda x: np.zeros(len(x)), 40.)

    with pytest.raises(ValueError):
        LinearProblem(fluid, vector, quadrature, force, conditions=[field],
                      petsc_options_prefix=prefix)

    with pytest.raises(ValueError):
        LinearProblem(element, space, quadrature, ones,
                      pressure_reference=(1., 1.),
                      petsc_options_prefix=prefix)

    with pytest.raises(ValueError):
        LinearProblem(fluid, vector, quadrature, force,
                      pressure_reference=(1., 1., 1.),
                      petsc_options_prefix=prefix)

    for viscosity, stabilization in ((0., 3.), (1., 0.)):
        with pytest.raises(ValueError):
            StokesElement(viscosity, stabilization)

    # A field of several components has a coefficient per component
    stepped = LinearProblem(fluid, vector, quadrature, force, step=.1,
                            petsc_options_prefix=prefix)

    assert stepped.previous.shape == (vector.num_dofs, 3)

    with pytest.raises(ValueError):
        stepped.previous = np.zeros(3 * vector.num_dofs)
