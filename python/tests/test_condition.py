# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the conditions imposed along a boundary"""

import numpy as np
import pytest

import iguana
from iguana import (Boundary, BoundaryQuadrature, CellType, DomainQuadrature,
                    FunctionSpace, HierarchicalGrid, PenaltyCondition,
                    PoissonElement, solve)

# Sides of the rectangle, and of the square inside it, whose edges halve
# the cells they cut, so that no cut is a sliver
SIDES = np.array([2., 3.])
LOWER = np.array([.5, 1.25])
UPPER = np.array([1.5, 2.25])

# Gradient of the linear field the penalty imposes
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


def square_boundary(patch):
    """Quadrature of the square, counterclockwise, one face per side."""
    corners = np.array([LOWER, [UPPER[0], LOWER[1]], UPPER,
                        [LOWER[0], UPPER[1]]])
    faces = [iguana.create_curve(degree=1, knots=[0., 0., 1., 1.],
                                 control_points=[corners[k],
                                                 corners[(k + 1) % 4]])
             for k in range(4)]

    return BoundaryQuadrature(patch, Boundary(faces), 5)


def square_problem():
    """Patch, space and domain quadrature of the square, the inside cells
    with Gauss and the cut ones fitted to the square."""
    patch = rectangle()
    cell_types = square_cell_types(patch)
    vertices = np.array([LOWER, [UPPER[0], LOWER[1]], UPPER,
                         [LOWER[0], UPPER[1]]])

    quadrature = DomainQuadrature(patch, cell_types)
    quadrature.fill_gauss_legendre(CellType.inside, 3)
    quadrature.fill_moment_fitting(CellType.cut, vertices,
                                   [[0, 1], [1, 2], [2, 3], [3, 0]], order=3)

    return patch, FunctionSpace(patch, cell_types), quadrature


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
        u = solve(element, space, quadrature, source,
                  conditions=[condition])
        errors.append(np.abs(u - exact).max())

    assert errors[1] < 1e-3
    assert errors[1] / errors[0] == pytest.approx(1e-2, rel=.05)

    # Values given at the points impose the same as their function
    at_points = PenaltyCondition(element.u, boundary,
                                 boundary.positions @ GRADIENT, 1e5)
    u = solve(element, space, quadrature, source, conditions=[at_points])

    assert np.abs(u - exact).max() == pytest.approx(errors[1])


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
        solve(element, space, quadrature, source, conditions=['condition'])

    # The boundary must lie on the patch of the space
    other = PenaltyCondition(element.u, square_boundary(rectangle()), values,
                             1.)

    with pytest.raises(ValueError):
        solve(element, space, quadrature, source, conditions=[other])

    # Boundaries lie in the plane, so a volume takes no conditions
    box = iguana.create_box(lengths=(1., 1., 1.), elements=(1, 1, 1))
    box_quadrature = DomainQuadrature(box)
    box_quadrature.fill_gauss_legendre(CellType.inside, 3)
    condition = PenaltyCondition(element.u, boundary, values, 1.)

    with pytest.raises(TypeError):
        solve(element, FunctionSpace(box), box_quadrature,
              np.zeros(box_quadrature.num_points), conditions=[condition])
