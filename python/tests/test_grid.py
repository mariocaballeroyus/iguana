# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the hierarchical grid"""

import numpy as np
import pytest

import iguana
from iguana import HierarchicalGrid


def box():
    """Block of four by two by one elements, with the unit parameter box."""
    return iguana.create_box(lengths=(4., 2., 1.), elements=(4, 2, 1),
                             degrees=(2, 1, 1))


def grid():
    """Grid on the knots of the block."""
    patch = box()

    return HierarchicalGrid(patch.degrees, patch.knots)


def test_unrefined_grid_holds_the_patch_elements():
    """Before any refinement, the grid on the knots of a patch holds its
    elements in their numbering, the first direction running fastest."""
    coarse = grid()

    assert coarse.num_levels == 1
    assert coarse.num_elements == 8
    np.testing.assert_array_equal(coarse.levels, np.zeros(8))

    element = np.arange(8)
    starts = np.column_stack([element % 4 / 4, element // 4 / 2,
                              np.zeros(8)])
    ends = starts + [1 / 4, 1 / 2, 1]

    np.testing.assert_allclose(coarse.bounds,
                               np.stack([starts, ends], axis=1))


def test_refine_replaces_elements_by_their_children():
    """Refining returns a new grid, in which the marked element gives way
    to its children on the next level."""
    coarse = grid()
    refined = coarse.refine(np.array([5, 5]))

    assert coarse.num_elements == 8
    assert refined.num_levels == 2
    np.testing.assert_array_equal(np.bincount(refined.levels), [7, 8])

    # The children, numbered after the coarse elements, fill the parent
    children = refined.bounds[refined.levels == 1]

    np.testing.assert_allclose(children[:, 0].min(axis=0), [.25, .5, 0.])
    np.testing.assert_allclose(children[:, 1].max(axis=0), [.5, 1., 1.])
    np.testing.assert_allclose(
        np.prod(children[:, 1] - children[:, 0], axis=1).sum(), 1 / 8)


def test_rejects_what_it_cannot_build_or_refine():
    """A grid has two or three directions with one degree each, and only
    its active elements can be refined."""
    knots = box().knots

    with pytest.raises(ValueError):
        HierarchicalGrid((2, 1), knots)

    with pytest.raises(ValueError):
        HierarchicalGrid((2,), knots[:1])

    with pytest.raises(ValueError):
        grid().refine([8])
