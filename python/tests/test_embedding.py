# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of boundaries divided over the elements of a patch"""

import numpy as np
import pytest

import iguana
from iguana import (Boundary, BoundaryQuadrature, CellType, SurrogateBoundary,
                    cpp)


def embed(patch, boundary):
    """Boundary in the plane of a patch, divided over its elements."""
    return cpp.EmbeddedBoundary2d(patch=patch._cpp_object,
                                  boundary=boundary._cpp_object)


def test_circle_in_the_plane():
    """A circle given in the plane, on a rectangle away from the origin, is
    divided as in parameter space: one piece in each corner element, two
    in each edge element, none in the middle one, together covering the
    whole circle."""
    corner = np.sqrt(.5)
    points = [[1., 0.], [1., 1.], [0., 1.], [-1., 1.], [-1., 0.],
              [-1., -1.], [0., -1.], [1., -1.], [1., 0.]]
    circle = iguana.create_curve(
        degree=2, knots=[0., 0., 0., .25, .25, .5, .5, .75, .75, 1., 1., 1.],
        control_points=[5., 3.] + 2. * np.array(points),
        weights=[1., corner, 1., corner, 1., corner, 1., corner, 1.])

    patch = iguana.create_rectangle(lengths=(6., 6.), elements=(3, 3),
                                    degrees=(1, 1), origin=(2., 0.))
    embedded = embed(patch, Boundary([circle]))

    counts = np.bincount(embedded.elements, minlength=9)
    lengths = embedded.intervals[:, 1] - embedded.intervals[:, 0]

    assert np.array_equal(counts, [1, 2, 1, 2, 0, 2, 1, 2, 1])
    assert np.isclose(lengths.sum(), 1., rtol=0., atol=1e-15)


def test_face_on_a_knot_line():
    """A face drawn along x = 0.3 lies on the knot line 3/7 of a rectangle
    0.7 wide, although inverting the map misses it by rounding, and goes to
    the element opposite its normal."""
    line = iguana.create_curve(degree=2, knots=[0., 0., 0., 1., 1., 1.],
                               control_points=[[.3, .05], [.3, .065],
                                               [.3, .08]])

    patch = iguana.create_rectangle(lengths=(.7, .7), elements=(7, 7),
                                    degrees=(1, 1))
    embedded = embed(patch, Boundary([line, line], signs=[1, -1]))

    # Up along the line, the normal points to +x with sign 1 and to -x
    # with sign -1
    assert np.array_equal(embedded.faces, [0, 1])
    assert np.array_equal(embedded.elements, [2, 3])


def test_patch_not_affine():
    """A patch whose map is not affine cannot receive a boundary."""
    patch = iguana.create_rectangle(lengths=(1., 1.), elements=(2, 1),
                                    degrees=(1, 1))
    bent = patch.control_points.copy()
    bent[1, 1] += .1

    line = iguana.create_curve(degree=1, knots=[0., 0., 1., 1.],
                               control_points=[[.25, .2], [.25, .8]])

    with pytest.raises(ValueError):
        cpp.EmbeddedBoundary2d(
            patch=cpp.PlanarPatch(basis=patch._cpp_object.basis,
                                  coefficients=bent),
            boundary=Boundary([line])._cpp_object)


def test_surrogate_boundary():
    """The surrogate boundary of two inside cells runs along the six sides
    of their block, a cut cell beside them included, and the quadrature on
    it measures the block in the plane."""
    patch = iguana.create_rectangle(lengths=(2., 3.), elements=(6, 6))

    # Cells 20 and 21 cover [2/3, 4/3] x [1.5, 2], and cell 22 is cut
    cell_types = [CellType.outside] * 36
    cell_types[20] = cell_types[21] = CellType.inside
    cell_types[22] = CellType.cut

    surrogate = SurrogateBoundary(patch, cell_types)

    assert surrogate.num_faces == 6
    assert np.array_equal(surrogate.elements, [20, 20, 20, 21, 21, 21])

    # The weights give the perimeter, and the flux of the position, whose
    # divergence is two, twice the area
    quadrature = BoundaryQuadrature(patch, surrogate, 2)
    flux = quadrature.weights @ np.sum(quadrature.positions
                                       * quadrature.normals, axis=1)

    assert quadrature.boundary is surrogate
    assert quadrature.weights.sum() == pytest.approx(2. * (2. / 3. + .5))
    assert flux == pytest.approx(2. * (2. / 3.) * .5)


def test_invalid_surrogate_boundaries():
    patch = iguana.create_rectangle(lengths=(2., 3.), elements=(6, 6))
    cell_types = [CellType.inside] * 36

    with pytest.raises(ValueError):
        SurrogateBoundary(patch, cell_types[:-1])

    surface = iguana.create_surface(degrees=(1, 1),
                                    knots=([0., 0., 1., 1.],
                                           [0., 0., 1., 1.]),
                                    control_points=np.eye(4, 3))

    with pytest.raises(TypeError):
        SurrogateBoundary(surface, [CellType.inside])

    # A quadrature takes the surrogate boundary of its own patch only
    other = iguana.create_rectangle(lengths=(2., 3.), elements=(6, 6))

    with pytest.raises(ValueError):
        BoundaryQuadrature(other, SurrogateBoundary(patch, cell_types), 2)
