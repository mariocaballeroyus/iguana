# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the function spaces of a patch"""

import numpy as np
import pytest

import iguana
from iguana import CellType, FunctionSpace


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


def test_invalid_arguments():
    with pytest.raises(TypeError):
        FunctionSpace('patch')

    with pytest.raises(ValueError):
        FunctionSpace(rectangle(), [CellType.inside])
