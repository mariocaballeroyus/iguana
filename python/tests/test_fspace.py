# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the function spaces of a patch"""

import numpy as np
import pytest

import iguana
from iguana import CellType, FunctionSpace, MixedFunctionSpace


def rectangle():
    """Rectangle of three linear elements along x and one along y."""
    return iguana.create_rectangle(lengths=(3., 1.), elements=(3, 1),
                                   degrees=(1, 1))


def test_outside_functions():
    patch = rectangle()
    space = FunctionSpace(patch, [CellType.inside, CellType.cut,
                                  CellType.outside])

    # The functions at x = 3 live on the outside cell alone, while those at
    # x = 2 reach the cut cell and keep their unknowns
    kept = patch.control_points[:, 0] < 3.

    assert space.num_dofs == 6
    assert np.array_equal(space.control_points, patch.control_points[kept])


def test_every_cell_inside():
    patch = iguana.create_box(lengths=(1., 2., 3.), elements=(2, 1, 2),
                              degrees=(2, 1, 1))
    space = FunctionSpace(patch)

    assert space.patch is patch
    assert space.num_dofs == len(patch.control_points)
    assert np.array_equal(space.control_points, patch.control_points)


def test_nurbs_surface():
    """A NURBS surface spans its space with its rational functions."""
    net = [[0., 0., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 0.]]
    surface = iguana.create_surface(degrees=(1, 1),
                                    knots=([0., 0., 1., 1.],
                                           [0., 0., 1., 1.]),
                                    control_points=net,
                                    weights=[1., 2., 3., 4.])
    space = FunctionSpace(surface)

    assert space.num_dofs == 4
    assert np.array_equal(space.control_points, surface.control_points)


def test_block_size():
    """A field of several components gives each degree of freedom a
    coefficient per component, the degrees of freedom staying those of the
    functions."""
    patch = rectangle()
    space = FunctionSpace(patch, block_size=3)

    assert space.block_size == 3
    assert space.num_dofs == FunctionSpace(patch).num_dofs


def test_other_degrees():
    """A space of other degrees spans the B-splines of those degrees on the
    knots of its patch, its interior knots kept and its ends repeated once
    per degree plus one, and drops those supported on outside cells alone,
    as the space of the patch does."""
    patch = rectangle()
    space = FunctionSpace(patch, [CellType.inside, CellType.cut,
                                  CellType.outside], degrees=(2, 2))

    assert space.degrees == (2, 2)

    for knots, linear in zip(space.knots, patch.knots):
        assert np.array_equal(knots, np.concatenate([[0.] * 3, linear[2:-2],
                                                     [1.] * 3]))

    # Five quadratic functions along x, the last on the outside cell alone,
    # by three along y, the first direction running fastest
    kept = [along_x + 5 * along_y for along_y in range(3)
            for along_x in range(4)]

    assert space.num_dofs == 12
    assert np.array_equal(space.functions, kept)

    # The first function along x and those first or last along y reach the
    # boundary, the last along x being dropped
    assert np.array_equal(space.boundary_dofs, [0, 1, 2, 3, 4, 8, 9, 10, 11])

    with pytest.raises(ValueError):
        space.control_points

    # The degrees of the patch give its own space
    same = FunctionSpace(patch, degrees=(1, 1))

    assert np.array_equal(same.control_points, patch.control_points)


def test_divergence_conforming_space():
    """Each component of the velocity takes one degree more along its own
    direction than the pressure, which takes the basis of the patch, the
    fields numbered one after another."""
    patch = iguana.create_rectangle(lengths=(2., 3.), elements=(3, 2))
    space = MixedFunctionSpace.divergence_conforming(patch)

    assert space.patch is patch
    assert all(field.patch is patch for field in space.fields)
    assert [field.degrees for field in space.fields] == [(3, 2), (2, 3),
                                                        (2, 2)]

    # Six by four, five by five and five by four functions
    assert space.offsets == (0, 24, 49)
    assert space.num_dofs == 69


def test_invalid_arguments():
    with pytest.raises(TypeError):
        FunctionSpace('patch')

    with pytest.raises(ValueError):
        FunctionSpace(rectangle(), [CellType.inside])

    with pytest.raises(ValueError):
        FunctionSpace(rectangle(), block_size=0)

    # A degree per direction, non-negative, on open knots of B-splines
    for degrees in ((2,), (-1, 1)):
        with pytest.raises(ValueError):
            FunctionSpace(rectangle(), degrees=degrees)

    unclamped = iguana.create_surface(degrees=(1, 1),
                                      knots=([0., .5, 1., 1.5],
                                             [0., 0., 1., 1.]),
                                      control_points=np.eye(4, 3))

    with pytest.raises(ValueError):
        FunctionSpace(unclamped, degrees=(2, 1))

    rational = iguana.create_surface(degrees=(1, 1),
                                     knots=([0., 0., 1., 1.],
                                            [0., 0., 1., 1.]),
                                     control_points=np.eye(4, 3),
                                     weights=[1., 2., 3., 4.])

    with pytest.raises(TypeError):
        FunctionSpace(rational, degrees=(2, 2))

    # The fields of a mixed space, function spaces on one planar patch
    with pytest.raises(ValueError):
        MixedFunctionSpace([])

    with pytest.raises(TypeError):
        MixedFunctionSpace(['field'])

    with pytest.raises(ValueError):
        MixedFunctionSpace([FunctionSpace(rectangle()),
                            FunctionSpace(rectangle())])

    box = iguana.create_box(lengths=(1., 1., 1.), elements=(1, 1, 1))

    with pytest.raises(TypeError):
        MixedFunctionSpace([FunctionSpace(box)])

    with pytest.raises(TypeError):
        MixedFunctionSpace.divergence_conforming(box)
