# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the solve of a problem on a function space"""

import numpy as np
import pytest

import iguana
from iguana import (CellType, DomainQuadrature, FunctionSpace,
                    HierarchicalGrid, PoissonElement, solve)


def rectangle():
    """Rectangle of sides 2 and 3 with quadratic elements, three along x and
    two along y."""
    return iguana.create_rectangle(lengths=(2., 3.), elements=(3, 2))


def gauss_quadrature(patch):
    """Gauss quadrature of a patch whose cells all lie inside, exact for the
    products of quadratic functions through an affine map."""
    quadrature = DomainQuadrature(patch)
    quadrature.fill_gauss_legendre(CellType.inside, 3)

    return quadrature


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

    def source(points):
        x, y = points.T
        return 2. * (x * (width - x) + y * (height - y))

    # The quadratic splines hold u = x (width - x) y (height - y), which
    # vanishes on the boundary, so the solve finds it exactly
    u = solve(PoissonElement(), space, gauss_quadrature(patch), source,
              fixed=space.boundary_dofs)

    # In parameters, x (width - x) is width^2 t (1 - t). With every cell
    # inside, each function keeps its own index, the first direction
    # running fastest
    along_x = width ** 2 * coefficients(patch.knots[0])
    along_y = height ** 2 * coefficients(patch.knots[1])

    assert np.allclose(u, np.kron(along_y, along_x), rtol=0., atol=1e-13)


def test_fixed_values_reproduce_a_linear_field():
    patch = iguana.create_box(lengths=(2., 3., 1.), elements=(2, 2, 1))
    space = FunctionSpace(patch)
    quadrature = gauss_quadrature(patch)

    # A linear field has no Laplacian, so fixing its values on the boundary
    # without a source gives it back everywhere
    b = np.array([1., -2., .5])
    fixed = space.boundary_dofs

    u = solve(PoissonElement(), space, quadrature,
              np.zeros(quadrature.num_points), fixed=fixed,
              values=space.control_points[fixed] @ b)

    assert np.allclose(u, space.control_points @ b, rtol=0., atol=1e-12)


def test_invalid_arguments():
    patch = rectangle()
    space = FunctionSpace(patch)
    quadrature = gauss_quadrature(patch)
    ones = np.ones(quadrature.num_points)
    element = PoissonElement()

    with pytest.raises(TypeError):
        solve('element', space, quadrature, ones)

    with pytest.raises(TypeError):
        solve(element, 'space', quadrature, ones)

    # A surface has no physical gradients for the Poisson element
    surface = iguana.create_surface(degrees=(1, 1),
                                    knots=([0., 0., 1., 1.],
                                           [0., 0., 1., 1.]),
                                    control_points=np.eye(4, 3))

    with pytest.raises(TypeError):
        solve(element, FunctionSpace(surface), quadrature, ones)

    with pytest.raises(TypeError):
        solve(element, space, 'quadrature', ones)

    # Neither another patch nor a refined grid numbers the cells as the
    # elements of the patch of the space
    with pytest.raises(ValueError):
        solve(element, space, gauss_quadrature(rectangle()), ones)

    refined = DomainQuadrature(
        patch, grid=HierarchicalGrid(patch.degrees, patch.knots).refine([0]))
    refined.fill_gauss_legendre(CellType.inside, 3)

    with pytest.raises(ValueError):
        solve(element, space, refined, np.ones(refined.num_points))

    # A column is not one value per point, however many rows it has
    with pytest.raises(ValueError):
        solve(element, space, quadrature, ones[:, np.newaxis])

    with pytest.raises(ValueError):
        solve(element, space, quadrature, ones, fixed=[0, space.num_dofs])

    with pytest.raises(ValueError):
        solve(element, space, quadrature, ones, fixed=[0, 0])

    with pytest.raises(ValueError):
        solve(element, space, quadrature, ones, fixed=[0, 1], values=[1.])

    # One weight short of the six cells
    with pytest.raises(ValueError):
        solve(element, space, quadrature, ones, cell_weights=np.ones(5))
