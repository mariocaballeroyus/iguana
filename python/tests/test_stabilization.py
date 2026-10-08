# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the stabilizations of a discrete problem"""

import numpy as np
import pytest

import iguana
from iguana import (Boundary, BoundaryQuadrature, CellType, DomainQuadrature,
                    FunctionSpace, GhostPenalty, NitscheCondition,
                    PoissonElement, SurrogateBoundary, classify_cells, cpp,
                    solve)

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


def test_ghost_penalty_extends_a_shifted_problem():
    """With every function of the patch kept, the inside cells carry the
    shifted boundary method and the ghost penalty extends its solution over
    the rest, so that a linear field is recovered at every degree of
    freedom, those of the cells outside the circle included."""
    patch = rectangle()
    cell_types = classify_cells(patch, circle()).cell_types
    inside = [CellType.inside if cell == CellType.inside
              else CellType.outside for cell in cell_types]

    quadrature = DomainQuadrature(patch, inside)
    quadrature.fill_gauss_legendre(CellType.inside, 3)
    boundary = BoundaryQuadrature(patch, SurrogateBoundary(patch, inside),
                                  3, shift=circle())

    space = FunctionSpace(patch, [CellType.inside] * len(cell_types))
    element = PoissonElement()
    condition = NitscheCondition(element.u, boundary,
                                 lambda x: x @ GRADIENT, 40.)
    ghost = GhostPenalty(element.u, patch, cell_types, .1)

    u = solve(element, space, quadrature, np.zeros(quadrature.num_points),
              conditions=[condition], stabilizations=[ghost])

    # Exact to round-off, which the conditioning of the cells the penalty
    # alone determines amplifies
    assert np.abs(u - space.control_points @ GRADIENT).max() < 1e-9


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

    quadrature = DomainQuadrature(patch)
    quadrature.fill_gauss_legendre(CellType.inside, 3)
    source = np.zeros(quadrature.num_points)

    with pytest.raises(TypeError):
        solve(element, FunctionSpace(patch), quadrature, source,
              stabilizations=['ghost'])

    # The cells must lie on the patch of the space
    other = GhostPenalty(element.u, rectangle(), cell_types, 1.)

    with pytest.raises(ValueError):
        solve(element, FunctionSpace(patch), quadrature, source,
              stabilizations=[other])

    # The map of the patch must be affine
    bent = patch.control_points.copy()
    bent[20, 1] += .1
    curved = iguana.PlanarPatch(cpp.PlanarPatch(
        basis=patch._cpp_object.basis, coefficients=bent))
    curved_quadrature = DomainQuadrature(curved)
    curved_quadrature.fill_gauss_legendre(CellType.inside, 3)
    ghost = GhostPenalty(element.u, curved, cell_types, 1.)

    with pytest.raises(ValueError):
        solve(element, FunctionSpace(curved), curved_quadrature,
              np.zeros(curved_quadrature.num_points), stabilizations=[ghost])

    # Faces between cells lie in the plane, so a volume takes none
    box = iguana.create_box(lengths=(1., 1., 1.), elements=(1, 1, 1))
    box_quadrature = DomainQuadrature(box)
    box_quadrature.fill_gauss_legendre(CellType.inside, 3)

    with pytest.raises(TypeError):
        solve(element, FunctionSpace(box), box_quadrature,
              np.zeros(box_quadrature.num_points), stabilizations=[ghost])
