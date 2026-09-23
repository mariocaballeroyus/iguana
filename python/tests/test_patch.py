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
    """The patch of a block carries its basis and a net spanning it."""
    patch = iguana.create_box(lengths=LENGTHS, elements=(3, 2, 2),
                              degrees=(2, 3, 1))
    points = patch.control_points

    assert patch.degrees == (2, 3, 1)

    # A direction of degree p and n elements carries p + n functions,
    # and p + 1 more knots than functions
    assert [len(knot) for knot in patch.knots] == [8, 9, 5]
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


def test_boundary_grid_lines():
    """The isocurves of a linear box are the grid lines on its faces."""
    # A different number of elements per direction, so that pairing a
    # direction with the wrong one cannot pass unnoticed, and at least
    # two per direction, so that each has interior lines to leave out
    elements = (2, 3, 4)

    patch = iguana.create_box(lengths=LENGTHS, elements=elements,
                              degrees=(1, 1, 1))

    lines = [np.linspace(0., LENGTHS[axis], elements[axis] + 1).tolist()
             for axis in range(3)]

    def on_face(axis, station):
        return station in (lines[axis][0], lines[axis][-1])

    # A linear box is its own grid: one curve per direction and knot
    # line of the other two on a face, running the whole side
    expected = set()

    for direction in range(3):
        first, second = (axis for axis in range(3) if axis != direction)

        for one in lines[first]:
            for other in lines[second]:
                if not (on_face(first, one) or on_face(second, other)):
                    continue

                point = [0., 0., 0.]
                point[first] = one
                point[second] = other

                curve = []

                for station in lines[direction]:
                    point[direction] = station
                    curve.append(tuple(point))

                expected.add(tuple(curve))

    curves = patch.isocurves()
    found = {tuple(map(tuple, curve.control_points.tolist()))
             for curve in curves}

    # 14, 12 and 10 of the 20, 15 and 12 lines along each direction
    assert len(expected) == 36
    assert len(curves) == len(expected)
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
