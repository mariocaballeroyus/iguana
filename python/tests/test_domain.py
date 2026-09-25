# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Tests of the domain over the elements of a patch"""

import pytest

import iguana
from iguana import CellType, TensorDomain


def box():
    """Patch of a block with a different number of elements per direction."""
    return iguana.create_box(lengths=(1., 2., 3.), elements=(2, 3, 4),
                             degrees=(2, 3, 1))


def test_whole_patch():
    """Without cell types, every element of the patch lies inside."""
    patch = box()
    volume = TensorDomain(patch)
    surface = TensorDomain(patch.isosurfaces()[0])

    assert volume.patch is patch
    assert volume.num_elements == 2 * 3 * 4
    assert volume.cell_types == [CellType.inside] * 24

    # The first isosurface keeps the second and third directions
    assert surface.num_elements == 3 * 4
    assert surface.cell_types == [CellType.inside] * 12


def test_given_cell_types():
    """The cell types are kept in element order, from any sequence."""
    cell_types = [(CellType.inside, CellType.cut, CellType.outside)[e % 3]
                  for e in range(24)]

    assert TensorDomain(box(), cell_types).cell_types == cell_types
    assert TensorDomain(box(), tuple(cell_types)).cell_types == cell_types


def test_invalid_arguments():
    """The domain needs a surface or volume patch and one type per cell."""
    curve = box().isosurfaces()[0].isocurves()[0]

    with pytest.raises(TypeError):
        TensorDomain(curve)

    with pytest.raises(ValueError):
        TensorDomain(box(), [CellType.inside] * 23)
