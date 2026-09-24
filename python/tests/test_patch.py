# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the patches and the isoparametric patches read from them"""

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


# Quadratic and cubic, so that a knot plane inside the block weighs
# several layers of the control net rather than a single one
ELEMENTS = (2, 3, 4)
DEGREES = (2, 3, 2)


def knot_planes():
    """Pinned direction and knot line of each isosurface, in order."""
    return [(direction, line)
            for direction in range(3)
            for line in range(ELEMENTS[direction] + 1)]


def station(direction, line):
    """Position of a knot line of the block along its direction."""
    return LENGTHS[direction] * line / ELEMENTS[direction]


def test_boundary_isosurfaces():
    """The first and last isosurface of each direction are the outer
    layers of the control net, in the same numbering."""
    patch = iguana.create_box(lengths=LENGTHS, elements=ELEMENTS,
                              degrees=DEGREES)
    surfaces = patch.isosurfaces()

    # The net numbers the first direction fastest, so it reshapes to
    # (third, second, first) counts
    counts = [degree + span for degree, span in zip(DEGREES, ELEMENTS)]
    net = patch.control_points.reshape(counts[2], counts[1], counts[0], 3)

    layers = {(0, 0): net[:, :, 0], (0, ELEMENTS[0]): net[:, :, -1],
              (1, 0): net[:, 0, :], (1, ELEMENTS[1]): net[:, -1, :],
              (2, 0): net[0, :, :], (2, ELEMENTS[2]): net[-1, :, :]}

    for plane, surface in zip(knot_planes(), surfaces):
        if plane in layers:
            assert np.array_equal(surface.control_points,
                                  layers[plane].reshape(-1, 3))


def test_isosurfaces():
    """Each isosurface lies on its knot plane and carries the basis of
    the two directions it keeps."""
    patch = iguana.create_box(lengths=LENGTHS, elements=ELEMENTS,
                              degrees=DEGREES)
    surfaces = patch.isosurfaces()

    assert len(surfaces) == len(knot_planes())

    for (direction, line), surface in zip(knot_planes(), surfaces):
        kept = [axis for axis in range(3) if axis != direction]
        points = surface.control_points

        # The map is affine, so even a plane inside the block is exact
        assert np.allclose(points[:, direction], station(direction, line))

        assert surface.degrees == tuple(DEGREES[axis] for axis in kept)

        for own, axis in enumerate(kept):
            assert np.allclose(surface.knots[own], patch.knots[axis])

        # It spans the block along the directions it keeps
        assert np.allclose(points[:, kept].min(axis=0), 0.)
        assert np.allclose(points[:, kept].max(axis=0),
                           [LENGTHS[axis] for axis in kept])


def test_surface_isocurves():
    """The isocurves of a surface run along all of its knot lines."""
    patch = iguana.create_box(lengths=LENGTHS, elements=ELEMENTS,
                              degrees=DEGREES)

    for (direction, line), surface in zip(knot_planes(),
                                          patch.isosurfaces()):
        kept = [axis for axis in range(3) if axis != direction]
        curves = surface.isocurves()

        # Pinned in each direction the surface keeps, at every line
        pins = [(axis, curve_line)
                for axis in kept
                for curve_line in range(ELEMENTS[axis] + 1)]

        assert len(curves) == len(pins)

        for (axis, curve_line), curve in zip(pins, curves):
            running = kept[1] if axis == kept[0] else kept[0]
            points = curve.control_points

            assert np.allclose(points[:, direction],
                               station(direction, line))
            assert np.allclose(points[:, axis], station(axis, curve_line))

            assert curve.degree == DEGREES[running]
            assert np.allclose(curve.knots, patch.knots[running])
            assert np.isclose(points[:, running].min(), 0.)
            assert np.isclose(points[:, running].max(), LENGTHS[running])
