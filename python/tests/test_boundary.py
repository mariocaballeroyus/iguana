# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the boundaries of a domain"""

import numpy as np
import pytest

import iguana
from iguana import Boundary


def quarter_circle():
    """Quarter of the unit circle, counterclockwise from (1, 0), as a NURBS
    curve."""
    return iguana.create_curve(degree=2, knots=[0., 0., 0., 1., 1., 1.],
                               control_points=[[1., 0.], [1., 1.], [0., 1.]],
                               weights=[1., np.sqrt(.5), 1.])


def square():
    """Unit square in the plane z = 0, as a bilinear B-spline surface."""
    return iguana.create_surface(degrees=(1, 1),
                                 knots=([0., 0., 1., 1.], [0., 0., 1., 1.]),
                                 control_points=[[0., 0., 0.], [1., 0., 0.],
                                                 [0., 1., 0.], [1., 1., 0.]])


def test_curve_faces():
    """A boundary of curves keeps their control points, weights and
    signs."""
    arc = quarter_circle()
    boundary = Boundary([arc, arc], signs=[1, -1])

    assert np.array_equal(boundary.signs, [1, -1])

    for face in boundary.faces:
        assert np.array_equal(face.control_points, arc.control_points)
        assert np.array_equal(face.weights, arc.weights)


def test_bspline_face():
    """A B-spline surface becomes a face of unit weights, whose normal points
    out of the domain unless told otherwise."""
    boundary = Boundary([square()])
    face = boundary.faces[0]

    assert np.array_equal(boundary.signs, [1])
    assert face.degrees == (1, 1)
    assert np.array_equal(face.control_points, square().control_points)
    assert np.array_equal(face.weights, np.ones(4))


def test_invalid_arguments():
    """Faces of mixed kinds or curves in space are rejected, as are signs that
    do not match the faces."""
    with pytest.raises(TypeError, match='curves in the plane'):
        Boundary([quarter_circle(), square()])

    with pytest.raises(TypeError, match='curves in the plane'):
        Boundary([iguana.create_curve(degree=1, knots=[0., 0., 1., 1.],
                                      control_points=[[0., 0., 0.],
                                                      [1., 1., 1.]])])

    with pytest.raises(ValueError):
        Boundary([quarter_circle()], signs=[1, 1])
