# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the quadrature over the cells of a patch"""

import numpy as np
import pytest

import iguana
from iguana import CellType, DomainQuadrature, HierarchicalDomain

ORIGIN = np.array([1., 0., -1.])


def box():
    """Block of unit elements, four by two by one, away from the origin."""
    return iguana.create_box(lengths=(4., 2., 1.), elements=(4, 2, 1),
                             degrees=(2, 1, 1), origin=tuple(ORIGIN))


def mixed():
    """Cell types of the block with cut cells 1 and 6 and outside cell 3."""
    cell_types = [CellType.inside] * 8
    cell_types[1] = cell_types[6] = CellType.cut
    cell_types[3] = CellType.outside

    return cell_types


def centre(element):
    """Centre of an element of the block in space."""
    return ORIGIN + [element % 4 + .5, element // 4 + .5, .5]


def solid():
    """Triangle mesh of a box ending halfway along the second element in x,
    counterclockwise seen from outside."""
    lower, upper = ORIGIN - 1, ORIGIN + [1.5, 3., 2.]

    # Corner c takes the upper bound in the directions of the bits of c
    bits = np.arange(8)[:, None] >> np.arange(3) & 1
    triangles = [[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5],
                 [0, 5, 4], [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2],
                 [1, 3, 7], [1, 7, 5]]

    return np.where(bits, upper, lower), triangles


def square():
    """The unit square as a surface of two by two quadratic elements,
    mapped onto itself so that its parameters are its coordinates."""
    knots = [0., 0., 0., .5, 1., 1., 1.]
    greville = [0., .25, .75, 1.]

    # The first direction runs fastest
    first, second = np.meshgrid(greville, greville, indexing='xy')

    return iguana.create_surface(
        degrees=(2, 2), knots=(knots, knots),
        control_points=np.column_stack([first.ravel(), second.ravel(),
                                        np.zeros(first.size)]))


def triangle():
    """The triangle trimming the square below its diagonal u + v = 1, as
    segments with it on their left."""
    return [[0., 0.], [1., 0.], [0., 1.]], [[0, 1], [1, 2], [2, 0]]


def trimmed():
    """Cell types of the square under the triangle: the corner element
    inside, the two beside it cut, and the far one, which only touches
    it, outside."""
    return [CellType.inside, CellType.cut, CellType.cut, CellType.outside]


def test_gauss_legendre_positions():
    """The points of each inside cell are its Gauss points in space."""
    quadrature = DomainQuadrature(box(), mixed())
    quadrature.fill_gauss_legendre(CellType.inside, (2, 3, 1))

    # Gauss nodes about the centre of a unit cell, first direction fastest
    nodes = [np.polynomial.legendre.leggauss(n)[0] / 2 for n in (2, 3, 1)]
    offsets = np.column_stack([node.ravel(order='F') for node in
                               np.meshgrid(*nodes, indexing='ij')])

    np.testing.assert_allclose(quadrature.positions,
                               np.vstack([centre(element) + offsets
                                          for element in (0, 2, 4, 5, 7)]))


def test_fill_order():
    """Each filled cell type follows the ones filled before."""
    quadrature = DomainQuadrature(box(), mixed())
    assert quadrature.positions.shape == (0, 3)

    quadrature.fill_gauss_legendre(CellType.cut, 1)
    quadrature.fill_gauss_legendre(CellType.inside, 1)

    np.testing.assert_allclose(quadrature.positions,
                               [centre(element)
                                for element in (1, 6, 0, 2, 4, 5, 7)])


def test_hierarchical_domain():
    """The cells of a refined domain integrate the patch exactly, each
    through the patch element holding it."""
    block = box()
    net = block.control_points.copy()

    # A bend along x, so that the map differs from element to element
    net[:, 2] += .1 * np.sin(net[:, 0])
    patch = iguana.VolumePatch(
        iguana.cpp.VolumePatch(block._cpp_object.basis, net))

    domain = HierarchicalDomain(patch.degrees, patch.knots)
    domain = domain.refine([5, 1]).refine([7])

    # Exact for the map, of degree 2 along x and 1 along y and z
    quadrature = DomainQuadrature(patch, domain=domain)
    quadrature.fill_gauss_legendre(CellType.inside, (2, 1, 1))

    # A B-spline integrates to (t[i + p + 1] - t[i]) / (p + 1), and the
    # control points run with the first direction fastest
    integrals = [(knots[degree + 1:] - knots[:-degree - 1]) / (degree + 1)
                 for degree, knots in zip(patch.degrees, patch.knots)]
    exact = np.kron(integrals[2], np.kron(integrals[1], integrals[0])) @ net

    assert quadrature.num_elements == 29
    np.testing.assert_allclose(quadrature.weights @ quadrature.positions,
                               exact, rtol=1e-12)


def test_moment_fitting():
    """Gauss-Legendre on the inside cells and moment fitting on the cut ones
    integrate the part of the block inside the solid."""
    cell_types = [CellType.outside] * 8
    cell_types[0] = cell_types[4] = CellType.inside
    cell_types[1] = cell_types[5] = CellType.cut

    quadrature = DomainQuadrature(box(), cell_types)
    quadrature.fill_gauss_legendre(CellType.inside, 2)
    quadrature.fill_moment_fitting(CellType.cut, *solid())

    # The part inside, 1.5 by 2 by 1, out of the block, 4 by 2 by 1, whose
    # parameters span the unit cube
    np.testing.assert_allclose(quadrature.weights.sum(), 1.5 * 2 / 8)
    assert (quadrature.positions[:, 0] <= ORIGIN[0] + 1.5).all()


def test_surface():
    """A surface patch takes one number of points per surface direction."""
    # The first isosurface is the face of the block at x = 1, whose cells
    # all lie inside without cell types
    quadrature = DomainQuadrature(box().isosurfaces()[0])
    quadrature.fill_gauss_legendre(CellType.inside, (3, 2))

    assert quadrature.num_points == 2 * 3 * 2
    np.testing.assert_allclose(quadrature.positions[:, 0], ORIGIN[0])


def test_planar():
    """A planar patch places the points in the plane, at the Gauss points
    of its elements."""
    rectangle = iguana.create_rectangle(lengths=(4., 2.), elements=(4, 2),
                                        degrees=(2, 1), origin=ORIGIN[:2])
    quadrature = DomainQuadrature(rectangle)
    quadrature.fill_gauss_legendre(CellType.inside, 1)

    # One point per unit element, at its centre
    element = np.arange(8)
    centres = ORIGIN[:2] + np.column_stack([element % 4 + .5,
                                            element // 4 + .5])

    np.testing.assert_allclose(quadrature.positions, centres)


def test_trimmed_surface():
    """Moment fitting on the cut cells of a trimmed surface integrates the
    area and first moments of the trimmed region."""
    quadrature = DomainQuadrature(square(), trimmed())
    quadrature.fill_gauss_legendre(CellType.inside, 3)
    quadrature.fill_moment_fitting(CellType.cut, *triangle())

    weights = quadrature.weights
    positions = quadrature.positions

    # Area 1/2, and 1/6 for the integral of each coordinate
    np.testing.assert_allclose(weights.sum(), .5)
    np.testing.assert_allclose(weights @ positions[:, :2], [1 / 6, 1 / 6])

    assert (weights > 0.).all()
    assert (positions[:, :2].sum(axis=1) <= 1. + 1e-12).all()


def test_planar_region():
    """Moment fitting on the cut cells of a planar patch integrates the
    area and first moments of a polygon given in the plane."""
    rectangle = iguana.create_rectangle(lengths=(4., 2.), elements=(4, 2),
                                        degrees=(2, 1), origin=ORIGIN[:2])

    # The triangle below the diagonal of the rectangle, with it on the
    # left of its segments. Of the unit elements, the two at its right
    # angle lie inside it and the two in the far corner outside
    corners = ORIGIN[:2] + np.array([[0., 0.], [4., 0.], [0., 2.]])
    segments = [[0, 1], [1, 2], [2, 0]]
    cell_types = [CellType.inside] * 2 + [CellType.cut] * 4 \
        + [CellType.outside] * 2

    quadrature = DomainQuadrature(rectangle, cell_types)
    quadrature.fill_gauss_legendre(CellType.inside, 3)
    quadrature.fill_moment_fitting(CellType.cut, corners, segments)

    # Weights in parameter space, times the area of the rectangle
    weights = quadrature.weights * 8.
    positions = quadrature.positions

    np.testing.assert_allclose(weights.sum(), 4.)
    np.testing.assert_allclose(weights @ positions, 4. * corners.mean(axis=0))

    assert (weights > 0.).all()
    assert ((positions - ORIGIN[:2]) @ [1 / 4, 1 / 2] <= 1. + 1e-12).all()


def test_invalid_arguments():
    """Invalid arguments raise and leave the quadrature unchanged."""
    # A curve patch, and one cell type too few
    with pytest.raises(TypeError):
        DomainQuadrature(box().isosurfaces()[0].isocurves()[0])

    with pytest.raises(ValueError):
        DomainQuadrature(box(), mixed()[:-1])

    # A domain must lie on the knots of the patch
    with pytest.raises(TypeError):
        DomainQuadrature(box(), domain=box())

    with pytest.raises(ValueError):
        DomainQuadrature(box(), domain=HierarchicalDomain(
            (2, 1, 1), [[0., 0., 0., 1., 1., 1.], [0., 0., 1., 1.],
                        [0., 0., 1., 1.]]))

    quadrature = DomainQuadrature(box(), mixed())

    with pytest.raises(ValueError):
        quadrature.fill_gauss_legendre(CellType.inside, (2, 2))

    quadrature.fill_gauss_legendre(CellType.inside, 1)

    with pytest.raises(ValueError):
        quadrature.fill_gauss_legendre(CellType.inside, 2)

    assert quadrature.num_points == 5

    # A surface polygon lies in parameter space and closes by segments
    surface = DomainQuadrature(square(), trimmed())
    vertices, segments = triangle()

    with pytest.raises(ValueError):
        surface.fill_moment_fitting(CellType.cut, *solid())

    with pytest.raises(ValueError):
        surface.fill_moment_fitting(CellType.cut, vertices, [[0, 1], [1, 3]])

    with pytest.raises(ValueError):
        surface.fill_moment_fitting(CellType.cut, vertices, segments, order=8)

    assert surface.num_points == 0
