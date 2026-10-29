# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the stabilizations of a discrete problem"""

import numpy as np
import pytest

import iguana
from iguana import (Boundary, CellType, FunctionSpace, GhostPenalty,
                    NitscheCondition, PoissonElement, QuadratureBoundary,
                    QuadratureDomain, SkeletonNitscheCondition,
                    StokesElement, SurrogateBoundary, VolumeGhostPenalty,
                    classify_cells, cpp)

try:
    from iguana.petsc import LinearProblem
except ImportError:
    # A build without PETSc skips the tests that solve
    LinearProblem = None

# Centre and radius of a circle off the knot lines of the rectangle three by
# two of fourteen elements along each direction
CENTRE = np.array([1.43, 1.07])
RADIUS = .71

# Gradient of the linear field the shifted problem imposes
GRADIENT = np.array([1., -2.])


def rectangle():
    """Rectangle three by two of quadratic elements, fourteen along each
    direction."""
    return iguana.create_rectangle(lengths=(3., 2.), elements=(14, 14))


def circle():
    """The circle as four quadratic arcs joined at double knots."""
    corner = np.sqrt(.5)
    points = np.array([[1., 0.], [1., 1.], [0., 1.], [-1., 1.], [-1., 0.],
                       [-1., -1.], [0., -1.], [1., -1.], [1., 0.]])

    return Boundary([iguana.create_curve(
        degree=2, knots=[0., 0., 0., .25, .25, .5, .5, .75, .75, 1., 1., 1.],
        control_points=CENTRE + RADIUS * points,
        weights=[1., corner, 1., corner, 1., corner, 1., corner, 1.])])


@pytest.mark.petsc
def test_ghost_penalty_extends_a_shifted_problem():
    """With every function of the patch kept, the inside cells carry the
    shifted boundary method and the ghost penalty extends its solution over
    the rest, so that a linear field is recovered at every degree of
    freedom, those of the cells outside the circle included. So it is with
    two penalties sharing the faces, a weaker one between outside cells,
    and with the volume penalty."""
    patch = rectangle()
    cell_types = classify_cells(patch, circle()).cell_types
    inside = [CellType.inside if cell == CellType.inside
              else CellType.outside for cell in cell_types]

    quadrature = QuadratureDomain(patch, inside)
    quadrature.fill_gauss_legendre(CellType.inside, 3)
    boundary = QuadratureBoundary(patch, SurrogateBoundary(patch, inside),
                                  3, shift=circle())

    space = FunctionSpace(patch, [CellType.inside] * len(cell_types))
    element = PoissonElement()
    condition = NitscheCondition(element.u, boundary,
                                 lambda x: x @ GRADIENT, 40.)
    ghost = GhostPenalty(element.u, patch, cell_types, .1)
    cut = GhostPenalty(element.u, patch, cell_types, .1, faces='cut')
    void = GhostPenalty(element.u, patch, cell_types, .01, faces='outside')

    volume = VolumeGhostPenalty(element.u, patch, cell_types, .1)

    assert cut.num_faces + void.num_faces == ghost.num_faces

    for stabilizations in ([ghost], [cut, void], [volume]):
        u = LinearProblem(element, space, quadrature,
                          np.zeros(quadrature.num_points),
                          conditions=[condition],
                          stabilizations=stabilizations,
                          petsc_options_prefix='ghost_').solve()

        # Exact to round-off, which the conditioning of the cells the
        # penalty alone determines amplifies
        assert np.abs(u - space.control_points @ GRADIENT).max() < 1e-9


@pytest.mark.petsc
def test_ghost_penalties_of_a_fluid_take_the_signs_of_its_blocks():
    """For one positive penalty, the ghost penalty of the velocity adds to
    its positive block and that of the pressure to the negative pressure
    block of the symmetric form, so doubling the penalty adds a positive
    block on the velocity and a negative one on the pressure."""
    patch = rectangle()
    cell_types, fractions = classify_cells(patch, circle())
    element = StokesElement(1.5, 3.)

    quadrature = QuadratureDomain(patch, cell_types)

    for cell_type in (CellType.inside, CellType.cut, CellType.outside):
        quadrature.fill_gauss_legendre(cell_type, 3)

    condition = SkeletonNitscheCondition(element.u, patch, cell_types,
                                         circle(),
                                         lambda x: np.zeros((len(x), 2)),
                                         40. * element.viscosity)

    def operator(penalty):
        ghosts = [GhostPenalty(trace, patch, cell_types, penalty)
                  for trace in (element.u, element.p)]
        problem = LinearProblem(element, FunctionSpace(patch, block_size=3),
                                quadrature,
                                np.zeros((quadrature.num_points, 2)),
                                cell_weights=fractions, conditions=[condition],
                                stabilizations=ghosts,
                                pressure_reference=CENTRE,
                                petsc_options_prefix='signs_')
        problem.solve()

        return problem.A

    # Everything but the penalties cancels, the pinned pressure included
    difference = operator(.2)
    difference.axpy(-1., operator(.1))

    rng = np.random.default_rng(0)
    num_dofs = difference.getSize()[0] // 3

    def energy(component):
        coefficients = np.zeros((num_dofs, 3))
        coefficients[:, component] = rng.standard_normal(num_dofs)
        field = difference.createVecRight()
        field.setArray(coefficients.ravel())
        product = difference.createVecLeft()
        difference.mult(field, product)

        return field.dot(product)

    assert energy(0) > 0.
    assert energy(1) > 0.
    assert energy(2) < 0.


@pytest.mark.petsc
def test_invalid_ghost_penalties():
    patch = rectangle()
    cell_types = classify_cells(patch, circle()).cell_types
    element = PoissonElement()

    with pytest.raises(TypeError):
        GhostPenalty('u', patch, cell_types, 1.)

    with pytest.raises(TypeError):
        GhostPenalty(element.u, patch.isocurves()[0], cell_types, 1.)

    with pytest.raises(ValueError):
        GhostPenalty(element.u, patch, cell_types[:-1], 1.)

    with pytest.raises(ValueError):
        GhostPenalty(element.u, patch, cell_types, 0.)

    with pytest.raises(ValueError):
        GhostPenalty(element.u, patch, cell_types, 1., faces='void')

    # The volume penalty checks its arguments as the face one does
    with pytest.raises(TypeError):
        VolumeGhostPenalty('u', patch, cell_types, 1.)

    with pytest.raises(ValueError):
        VolumeGhostPenalty(element.u, patch, cell_types, 0.)

    quadrature = QuadratureDomain(patch)
    quadrature.fill_gauss_legendre(CellType.inside, 3)
    source = np.zeros(quadrature.num_points)

    with pytest.raises(TypeError):
        LinearProblem(element, FunctionSpace(patch), quadrature, source,
                      stabilizations=['ghost'],
                      petsc_options_prefix='invalid_')

    # The pressure of a fluid, foreign to the Poisson element
    pressure = GhostPenalty(StokesElement(1., 3.).p, patch, cell_types, 1.)

    with pytest.raises(ValueError):
        LinearProblem(element, FunctionSpace(patch), quadrature, source,
                      stabilizations=[pressure],
                      petsc_options_prefix='invalid_')

    # The cells must lie on the patch of the space
    other = GhostPenalty(element.u, rectangle(), cell_types, 1.)

    with pytest.raises(ValueError):
        LinearProblem(element, FunctionSpace(patch), quadrature, source,
                      stabilizations=[other], petsc_options_prefix='invalid_')

    # The map of the patch must be affine, which the assembly checks
    bent = patch.control_points.copy()
    bent[20, 1] += .1
    curved = iguana.PlanarPatch(cpp.PlanarPatch(
        basis=patch._cpp_object.basis, coefficients=bent))
    curved_quadrature = QuadratureDomain(curved)
    curved_quadrature.fill_gauss_legendre(CellType.inside, 3)
    ghost = GhostPenalty(element.u, curved, cell_types, 1.)

    problem = LinearProblem(element, FunctionSpace(curved), curved_quadrature,
                            np.zeros(curved_quadrature.num_points),
                            stabilizations=[ghost],
                            petsc_options_prefix='invalid_')

    with pytest.raises(ValueError):
        problem.solve()
