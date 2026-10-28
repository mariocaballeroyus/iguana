# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the conditions imposed along a boundary"""

import numpy as np
import pytest

import iguana
from iguana import (Boundary, CellType, FunctionSpace, GhostPenalty,
                    HierarchicalGrid, NeumannCondition, NitscheCondition,
                    PenaltyCondition, PoissonElement, QuadratureBoundary,
                    QuadratureDomain, SkeletonNitscheCondition,
                    StokesElement, SurrogateBoundary, classify_cells, cpp)

try:
    from iguana.petsc import LinearProblem
except ImportError:
    # A build without PETSc skips the tests that solve
    LinearProblem = None

# Sides of the rectangle, and of the square inside it, whose edges halve
# the cells they cut, so that no cut is a sliver
SIDES = np.array([2., 3.])
LOWER = np.array([.5, 1.25])
UPPER = np.array([1.5, 2.25])

# Gradient of the linear field the conditions impose
GRADIENT = np.array([1., -2.])


def rectangle():
    """Rectangle of quadratic elements, six along each direction."""
    return iguana.create_rectangle(lengths=tuple(SIDES), elements=(6, 6))


def square_cell_types(patch):
    """Type of each cell of the patch against the square."""
    bounds = HierarchicalGrid(patch.degrees, patch.knots).bounds * SIDES
    cell_types = []

    for start, end in bounds:
        if np.all(start >= LOWER) and np.all(end <= UPPER):
            cell_types.append(CellType.inside)
        elif np.any(end <= LOWER) or np.any(start >= UPPER):
            cell_types.append(CellType.outside)
        else:
            cell_types.append(CellType.cut)

    return cell_types


def square_sides(sides=range(4)):
    """Sides of the square, numbered counterclockwise from the bottom one,
    one face per side."""
    corners = np.array([LOWER, [UPPER[0], LOWER[1]], UPPER,
                        [LOWER[0], UPPER[1]]])

    return Boundary([iguana.create_curve(degree=1, knots=[0., 0., 1., 1.],
                                         control_points=[corners[side],
                                                         corners[(side + 1)
                                                                 % 4]])
                     for side in sides])


def square_boundary(patch, sides=range(4)):
    """Quadrature of sides of the square."""
    return QuadratureBoundary(patch, square_sides(sides), 5)


def square_surrogate(patch):
    """Quadrature of the surrogate boundary of the cells inside the square,
    shifted towards its sides."""
    surrogate = SurrogateBoundary(patch, square_cell_types(patch))

    return QuadratureBoundary(patch, surrogate, 3, shift=square_sides())


def square_problem():
    """Patch, space and domain quadrature of the square, the inside cells
    with Gauss and the cut ones fitted to the square."""
    patch = rectangle()
    cell_types = square_cell_types(patch)
    vertices = np.array([LOWER, [UPPER[0], LOWER[1]], UPPER,
                         [LOWER[0], UPPER[1]]])

    quadrature = QuadratureDomain(patch, cell_types)
    quadrature.fill_gauss_legendre(CellType.inside, 3)
    quadrature.fill_moment_fitting(CellType.cut, vertices,
                                   [[0, 1], [1, 2], [2, 3], [3, 0]], order=3)

    return patch, FunctionSpace(patch, cell_types), quadrature


def surrogate_problem():
    """Patch, space and domain quadrature of the surrogate domain of the
    square, its inside cells, with every cut cell left out."""
    patch = rectangle()
    cell_types = [CellType.inside if cell == CellType.inside
                  else CellType.outside for cell in square_cell_types(patch)]

    quadrature = QuadratureDomain(patch, cell_types)
    quadrature.fill_gauss_legendre(CellType.inside, 3)

    return patch, FunctionSpace(patch, cell_types), quadrature


@pytest.mark.petsc
def test_penalty_imposes_a_linear_field():
    patch, space, quadrature = square_problem()
    boundary = square_boundary(patch)
    element = PoissonElement()
    source = np.zeros(quadrature.num_points)

    # The space holds the linear field, which has no Laplacian, so the
    # error of the penalty alone remains, decaying like one over it
    exact = space.control_points @ GRADIENT
    errors = []

    for penalty in (1e3, 1e5):
        condition = PenaltyCondition(element.u, boundary,
                                     lambda x: x @ GRADIENT, penalty)
        u = LinearProblem(element, space, quadrature, source,
                          conditions=[condition],
                          petsc_options_prefix='penalty_').solve()
        errors.append(np.abs(u - exact).max())

    assert errors[1] < 1e-3
    assert errors[1] / errors[0] == pytest.approx(1e-2, rel=.05)

    # Values given at the points impose the same as their function
    at_points = PenaltyCondition(element.u, boundary,
                                 boundary.positions @ GRADIENT, 1e5)
    u = LinearProblem(element, space, quadrature, source,
                      conditions=[at_points],
                      petsc_options_prefix='penalty_').solve()

    assert np.abs(u - exact).max() == pytest.approx(errors[1])


@pytest.mark.petsc
def test_neumann_condition_imposes_a_flux():
    patch, space, quadrature = square_problem()
    element = PoissonElement()
    source = np.zeros(quadrature.num_points)

    # The penalty imposes the linear field on the bottom and left sides, and
    # the Neumann condition its outward flux on the right and top ones. The
    # flux is exact, so only the error of the penalty remains
    dirichlet = square_boundary(patch, sides=(0, 3))
    neumann = square_boundary(patch, sides=(1, 2))
    flux = NeumannCondition(element.u, neumann, neumann.normals @ GRADIENT)

    exact = space.control_points @ GRADIENT
    errors = []

    for penalty in (1e3, 1e5):
        condition = PenaltyCondition(element.u, dirichlet,
                                     lambda x: x @ GRADIENT, penalty)
        u = LinearProblem(element, space, quadrature, source,
                          conditions=[condition, flux],
                          petsc_options_prefix='neumann_').solve()
        errors.append(np.abs(u - exact).max())

    assert errors[1] < 1e-3
    assert errors[1] / errors[0] == pytest.approx(1e-2, rel=.05)


@pytest.mark.petsc
def test_nitsche_condition_imposes_a_linear_field():
    """On the surrogate domain of the square, Nitsche's method imposes the
    linear field from the sides exactly, symmetric whatever the penalty and
    skew-symmetric without one, where a penalty alone leaves an error of
    the order of the cells."""
    patch, space, quadrature = surrogate_problem()
    shifted = square_surrogate(patch)
    element = PoissonElement()
    source = np.zeros(quadrature.num_points)
    exact = space.control_points @ GRADIENT

    for penalty, symmetric in ((40., True), (4e3, True), (0., False)):
        condition = NitscheCondition(element.u, shifted,
                                     lambda x: x @ GRADIENT, penalty,
                                     symmetric=symmetric)
        u = LinearProblem(element, space, quadrature, source,
                          conditions=[condition],
                          petsc_options_prefix='nitsche_').solve()

        assert np.abs(u - exact).max() < 1e-10

    condition = PenaltyCondition(element.u, shifted, lambda x: x @ GRADIENT,
                                 4e3)
    u = LinearProblem(element, space, quadrature, source,
                      conditions=[condition],
                      petsc_options_prefix='nitsche_').solve()

    assert np.abs(u - exact).max() > 1e-2


@pytest.mark.petsc
def test_skeleton_nitsche_condition_imposes_a_linear_field():
    """With the volume fractions of the square weighting the cells, of a
    space that keeps every function and a quadrature over every cell,
    Nitsche's condition on the faces where they jump and a ghost penalty
    impose the linear field at every degree of freedom, as the generalized
    shifted boundary method does, symmetric with a penalty or
    skew-symmetric without one."""
    patch = rectangle()
    cell_types, fractions = classify_cells(patch, square_sides())
    space = FunctionSpace(patch, [CellType.inside] * len(cell_types))

    quadrature = QuadratureDomain(patch, cell_types)

    for cell_type in (CellType.inside, CellType.cut, CellType.outside):
        quadrature.fill_gauss_legendre(cell_type, 3)

    element = PoissonElement()
    ghost = GhostPenalty(element.u, patch, cell_types, .1)

    for penalty, symmetric in ((40., True), (0., False)):
        condition = SkeletonNitscheCondition(element.u, patch, cell_types,
                                             square_sides(),
                                             lambda x: x @ GRADIENT, penalty,
                                             symmetric=symmetric)
        u = LinearProblem(element, space, quadrature,
                          np.zeros(quadrature.num_points),
                          cell_weights=fractions, conditions=[condition],
                          stabilizations=[ghost],
                          petsc_options_prefix='skeleton_').solve()

        # Exact to round-off, which the conditioning of the cells the
        # ghost penalty alone determines amplifies
        assert np.abs(u - space.control_points @ GRADIENT).max() < 1e-9


@pytest.mark.petsc
def test_skeleton_nitsche_condition_steps_a_field_linear_in_time():
    """With the volume fractions of the square weighting the mass like the
    other volume terms, steps of backward Euler of u = t (b . x) from zero,
    imposed on the faces where the fractions jump, reproduce it at every
    degree of freedom."""
    patch = rectangle()
    cell_types, fractions = classify_cells(patch, square_sides())
    space = FunctionSpace(patch, [CellType.inside] * len(cell_types))

    quadrature = QuadratureDomain(patch, cell_types)

    for cell_type in (CellType.inside, CellType.cut, CellType.outside):
        quadrature.fill_gauss_legendre(cell_type, 3)

    element = PoissonElement()
    condition = SkeletonNitscheCondition(element.u, patch, cell_types,
                                         square_sides(),
                                         lambda x: np.zeros(len(x)), 40.)
    ghost = GhostPenalty(element.u, patch, cell_types, .1)
    problem = LinearProblem(element, space, quadrature,
                            lambda x: x @ GRADIENT, cell_weights=fractions,
                            conditions=[condition], stabilizations=[ghost],
                            step=.2,
                            petsc_options_prefix='skeleton_in_time_')
    field = space.control_points @ GRADIENT

    # du/dt = b . x, and the Laplacian vanishes
    for time in np.arange(1, 6) / 5.:
        condition.values = lambda x, time=time: time * (x @ GRADIENT)
        u = problem.solve()
        problem.previous = u

        assert np.abs(u - time * field).max() < 1e-9


def test_data_at_the_closest_points():
    """On a shifted quadrature, a condition takes its data where the points
    are shifted to, the sides of the square, half a cell away from the
    surrogate boundary."""
    shifted = square_surrogate(rectangle())
    condition = PenaltyCondition(PoissonElement().u, shifted,
                                 lambda x: x @ GRADIENT, 1.)

    np.testing.assert_allclose(condition.values,
                               shifted.projections @ GRADIENT)
    assert not np.allclose(condition.values, shifted.positions @ GRADIENT)


@pytest.mark.petsc
def test_invalid_arguments():
    patch, space, quadrature = square_problem()
    boundary = square_boundary(patch)
    element = PoissonElement()
    values = np.zeros(boundary.num_points)
    source = np.zeros(quadrature.num_points)

    with pytest.raises(TypeError):
        PenaltyCondition('u', boundary, values, 1.)

    with pytest.raises(TypeError):
        PenaltyCondition(element.u, quadrature, values, 1.)

    with pytest.raises(ValueError):
        PenaltyCondition(element.u, boundary, values[:-1], 1.)

    with pytest.raises(ValueError):
        PenaltyCondition(element.u, boundary, values, 0.)

    with pytest.raises(TypeError):
        NitscheCondition('u', boundary, values, 1.)

    with pytest.raises(TypeError):
        NitscheCondition(element.u, quadrature, values, 1.)

    with pytest.raises(ValueError):
        NitscheCondition(element.u, boundary, values[:-1], 1.)

    with pytest.raises(ValueError):
        NitscheCondition(element.u, boundary, values, 0.)

    # The skew-symmetric method needs no penalty, but not a negative one
    with pytest.raises(ValueError):
        NitscheCondition(element.u, boundary, values, -1., symmetric=False)

    with pytest.raises(TypeError):
        NeumannCondition('u', boundary, values)

    with pytest.raises(TypeError):
        NeumannCondition(element.u, quadrature, values)

    with pytest.raises(ValueError):
        NeumannCondition(element.u, boundary, values[:-1])

    # A shifted flux would need the gradients at the closest points
    shifted = square_surrogate(patch)

    with pytest.raises(ValueError):
        NeumannCondition(element.u, shifted, np.zeros(shifted.num_points))

    with pytest.raises(TypeError):
        LinearProblem(element, space, quadrature, source,
                      conditions=['condition'],
                      petsc_options_prefix='invalid_')

    # The boundary must lie on the patch of the space
    other = PenaltyCondition(element.u, square_boundary(rectangle()), values,
                             1.)

    with pytest.raises(ValueError):
        LinearProblem(element, space, quadrature, source, conditions=[other],
                      petsc_options_prefix='invalid_')

    # Values set later are checked as at initialization
    condition = PenaltyCondition(element.u, boundary, values, 1.)

    with pytest.raises(ValueError):
        condition.values = values[:-1]

    # The velocity of a fluid takes two values per point, given back as they
    # were given
    fluid = StokesElement(1., 3.)
    velocity = np.column_stack([values, values + 1.])

    with pytest.raises(ValueError):
        NitscheCondition(fluid.u, boundary, values, 1.)

    assert np.array_equal(
        NitscheCondition(fluid.u, boundary, velocity, 1.).values, velocity)


@pytest.mark.petsc
def test_invalid_skeleton_nitsche_conditions():
    patch = rectangle()
    cell_types, fractions = classify_cells(patch, square_sides())
    element = PoissonElement()
    sides = square_sides()

    with pytest.raises(TypeError):
        SkeletonNitscheCondition('u', patch, cell_types, sides, 0., 1.)

    with pytest.raises(TypeError):
        SkeletonNitscheCondition(element.u, patch.isocurves()[0], cell_types,
                                 sides, 0., 1.)

    with pytest.raises(TypeError):
        SkeletonNitscheCondition(element.u, patch, cell_types, patch, 0., 1.)

    with pytest.raises(ValueError):
        SkeletonNitscheCondition(element.u, patch, cell_types[:-1], sides,
                                 0., 1.)

    with pytest.raises(ValueError):
        SkeletonNitscheCondition(element.u, patch, cell_types, sides, [0.], 1.)

    with pytest.raises(ValueError):
        SkeletonNitscheCondition(element.u, patch, cell_types, sides,
                                 lambda x: x[:, 0], 0.)

    with pytest.raises(ValueError):
        SkeletonNitscheCondition(element.u, patch, cell_types, sides,
                                 lambda x: x[:, 0], 1., order=-1)

    with pytest.raises(ValueError):
        SkeletonNitscheCondition(element.u, patch, cell_types, sides,
                                 lambda x: x[:, 0], 1., center='inside')

    # The skew-symmetric method needs no penalty, but not a negative one,
    # and the faces need Gauss points
    with pytest.raises(ValueError):
        SkeletonNitscheCondition(element.u, patch, cell_types, sides,
                                 lambda x: x[:, 0], -1., symmetric=False)

    with pytest.raises(ValueError):
        SkeletonNitscheCondition(element.u, patch, cell_types, sides,
                                 lambda x: x[:, 0], 1., num_points=0)

    # A source gives the values of the trace at each point, two for the
    # body force of the velocity, and is derivatives or a function of the
    # points
    def zero(x):
        return np.zeros(len(x))

    with pytest.raises(ValueError):
        SkeletonNitscheCondition(StokesElement(1., 1.).u, patch, cell_types,
                                 sides, lambda x: x, 1., source=zero)

    with pytest.raises(TypeError):
        SkeletonNitscheCondition(element.u, patch, cell_types, sides, zero,
                                 1., source=0.)

    # The map of the patch must be affine
    bent = patch.control_points.copy()
    bent[20, 1] += .1
    curved = iguana.PlanarPatch(cpp.PlanarPatch(
        basis=patch._cpp_object.basis, coefficients=bent))

    with pytest.raises(ValueError):
        SkeletonNitscheCondition(element.u, curved, cell_types, sides,
                                 lambda x: x[:, 0], 1.)

    # A problem gives the condition the weights of the cells, which it
    # needs, and its faces must lie on the patch of the space
    space = FunctionSpace(patch, [CellType.inside] * len(cell_types))
    quadrature = QuadratureDomain(patch)
    quadrature.fill_gauss_legendre(CellType.inside, 3)
    source = np.zeros(quadrature.num_points)
    condition = SkeletonNitscheCondition(element.u, patch, cell_types, sides,
                                         lambda x: x[:, 0], 1.)
    other = SkeletonNitscheCondition(element.u, rectangle(), cell_types,
                                     sides, lambda x: x[:, 0], 1.)

    with pytest.raises(ValueError):
        LinearProblem(element, space, quadrature, source,
                      conditions=[condition], petsc_options_prefix='invalid_')

    with pytest.raises(ValueError):
        LinearProblem(element, space, quadrature, source,
                      cell_weights=fractions, conditions=[other],
                      petsc_options_prefix='invalid_')

    # Values set later are checked as at initialization
    with pytest.raises(ValueError):
        condition.values = [0.]
