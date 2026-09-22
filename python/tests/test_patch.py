# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the volume patch and the isocurves read from it"""

import numpy as np
import pytest

import iguana

LENGTHS = (1., 2., 3.)


def test_invalid_arguments():
    """The factory rejects an argument that is not one per direction."""
    with pytest.raises(ValueError):
        iguana.create_box(lengths=(1., 1.), elements=(1, 1, 1))

    with pytest.raises(ValueError):
        iguana.create_box(lengths=(1., 1., 1.), elements=(1, 1))


def test_control_net():
    """The net of a block holds one point per function and spans it."""
    patch = iguana.create_box(lengths=LENGTHS, elements=(3, 2, 2),
                              degrees=(2, 3, 1))
    points = patch.control_points

    # A direction of degree p and n elements carries p + n functions
    assert points.shape == (5 * 5 * 3, 3)
    assert np.allclose(points.min(axis=0), 0.)
    assert np.allclose(points.max(axis=0), LENGTHS)


def test_box_edges():
    """The isocurves of a trilinear box are its twelve edges."""
    patch = iguana.create_box(lengths=LENGTHS, elements=(1, 1, 1),
                              degrees=(1, 1, 1))
    curves = patch.isocurves()

    corners = [(x, y, z)
               for x in (0., LENGTHS[0])
               for y in (0., LENGTHS[1])
               for z in (0., LENGTHS[2])]

    # An edge joins the two corners that differ along one direction
    edges = {tuple(sorted((first, second)))
             for first in corners
             for second in corners
             if sum(a != b for a, b in zip(first, second)) == 1}

    assert len(edges) == 12
    assert len(curves) == 12
    assert {tuple(sorted(map(tuple, curve.control_points.tolist())))
            for curve in curves} == edges


def test_grid_lines():
    """The isocurves of a linear box are the lines of its own grid."""
    # A different number of elements per direction, so that pairing a
    # direction with the wrong one cannot pass unnoticed
    elements = (1, 2, 3)

    patch = iguana.create_box(lengths=LENGTHS, elements=elements,
                              degrees=(1, 1, 1))

    lines = [np.linspace(0., LENGTHS[axis], elements[axis] + 1).tolist()
             for axis in range(3)]

    # A linear box is its own grid: one curve per direction and knot
    # line of the other two, running the whole side
    expected = set()

    for direction in range(3):
        first, second = (axis for axis in range(3) if axis != direction)

        for one in lines[first]:
            for other in lines[second]:
                point = [0., 0., 0.]
                point[first] = one
                point[second] = other

                curve = []

                for station in lines[direction]:
                    point[direction] = station
                    curve.append(tuple(point))

                expected.add(tuple(curve))

    found = {tuple(map(tuple, curve.control_points.tolist()))
             for curve in patch.isocurves()}

    assert len(expected) == 26
    assert found == expected


def test_curve_basis():
    """Every curve carries the degree and knots of its direction."""
    patch = iguana.create_box(lengths=LENGTHS, elements=(1, 1, 1),
                              degrees=(1, 2, 3))
    curves = patch.isocurves()

    # Four curves run along each direction, carrying its degree
    degrees = sorted(curve.degree for curve in curves)

    assert degrees == [1] * 4 + [2] * 4 + [3] * 4

    for curve in curves:
        direction = patch.degrees.index(curve.degree)

        assert np.allclose(curve.knots, patch.knots[direction])
        assert len(curve.control_points) == curve.degree + 1
