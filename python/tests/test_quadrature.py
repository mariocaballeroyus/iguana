# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the quadrature over the cells of a domain"""

import numpy as np
import pytest

import iguana
from iguana import CellType, DomainQuadrature, TensorDomain

ORIGIN = np.array([1., 0., -1.])


def box():
    """Block of unit elements, four by two by one, away from the origin."""
    return iguana.create_box(lengths=(4., 2., 1.), elements=(4, 2, 1),
                             degrees=(2, 1, 1), origin=tuple(ORIGIN))


def mixed():
    """Domain over the block with cut cells 1 and 6 and outside cell 3."""
    cell_types = [CellType.inside] * 8
    cell_types[1] = cell_types[6] = CellType.cut
    cell_types[3] = CellType.outside

    return TensorDomain(box(), cell_types)


def centre(element):
    """Centre of an element of the block in space."""
    return ORIGIN + [element % 4 + .5, element // 4 + .5, .5]


def test_gauss_legendre_positions():
    """The points of each inside cell are its Gauss points in space."""
    quadrature = DomainQuadrature(mixed())
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
    quadrature = DomainQuadrature(mixed())
    assert quadrature.positions.shape == (0, 3)

    quadrature.fill_gauss_legendre(CellType.cut, 1)
    quadrature.fill_gauss_legendre(CellType.inside, 1)

    np.testing.assert_allclose(quadrature.positions,
                               [centre(element)
                                for element in (1, 6, 0, 2, 4, 5, 7)])


def test_surface():
    """A surface domain takes one number of points per surface direction."""
    # The first isosurface is the face of the block at x = 1
    quadrature = DomainQuadrature(TensorDomain(box().isosurfaces()[0]))
    quadrature.fill_gauss_legendre(CellType.inside, (3, 2))

    assert quadrature.num_points == 2 * 3 * 2
    np.testing.assert_allclose(quadrature.positions[:, 0], ORIGIN[0])


def test_invalid_arguments():
    """Invalid arguments raise and leave the quadrature unchanged."""
    with pytest.raises(TypeError):
        DomainQuadrature(box())

    quadrature = DomainQuadrature(mixed())

    with pytest.raises(ValueError):
        quadrature.fill_gauss_legendre(CellType.inside, (2, 2))

    quadrature.fill_gauss_legendre(CellType.inside, 1)

    with pytest.raises(ValueError):
        quadrature.fill_gauss_legendre(CellType.inside, 2)

    assert quadrature.num_points == 5
