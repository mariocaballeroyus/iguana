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

    # Moment fitting needs a volume patch
    surface = DomainQuadrature(box().isosurfaces()[0])

    with pytest.raises(TypeError):
        surface.fill_moment_fitting(CellType.inside, *solid())
