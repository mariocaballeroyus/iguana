# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Solver, the solution of a problem on a function space

The solve assembles the stiffness and load of an element over a quadrature,
in compiled code, holds some degrees of freedom at given values, and solves
the sparse system for the others.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from iguana import cpp as _cpp
from iguana.element import PoissonElement
from iguana.fspace import FunctionSpace
from iguana.patch import PlanarPatch, VolumePatch
from iguana.quadrature import DomainQuadrature


def solve(
    element: PoissonElement,
    space: FunctionSpace,
    quadrature: DomainQuadrature,
    source: (npt.ArrayLike
             | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
    fixed: npt.ArrayLike | None = None,
    values: npt.ArrayLike | None = None,
) -> npt.NDArray[np.float64]:
    """Solve the problem of an element on a space.

    The free degrees of freedom solve K_ff u_f = F_f - K_fc u_c, with the
    stiffness K and load F assembled over the quadrature and the fixed
    degrees of freedom held at their values u_c.

    Args:
        element: The element stating the problem.
        space: The space of the solution, on a planar or volume patch.
        quadrature: A quadrature over the elements of the patch of the
            space, not over a refined grid.
        source: The value of the source at each point of the quadrature, in
            its order, or a function that takes their positions, of shape
            `(num_points, n)`, and returns those values.
        fixed: Degrees of freedom held fixed, each listed once, such as
            `space.boundary_dofs`. Without them none is fixed, and the
            problem must determine the solution by itself.
        values: Value of each fixed degree of freedom. Without them, the
            fixed ones are held at zero.

    Returns:
        The coefficient of each degree of freedom, of shape `(num_dofs,)`.

    Raises:
        TypeError: If the element is not a Poisson element, if the space
            is not a function space on a planar or volume patch, or if the
            quadrature is not a domain quadrature.
        ValueError: If the quadrature does not lie on the elements of the
            patch of the space, if the source does not give one value per
            point, if a fixed degree of freedom lies outside the space or
            repeats, or if there is not one value per fixed one.
    """
    stiffness, load = _assemble(element, space, quadrature, source)
    fixed, values = _fixed(space.num_dofs, fixed, values)

    # The fixed degrees of freedom carry their part of the stiffness over to
    # the right-hand side of the free ones
    free = np.setdiff1d(np.arange(space.num_dofs), fixed)
    solution = np.zeros(space.num_dofs)
    solution[fixed] = values

    rhs = load[free] - stiffness[free][:, fixed] @ values
    solution[free] = spsolve(stiffness[free][:, free].tocsc(), rhs)

    return solution


def _assemble(
    element: PoissonElement,
    space: FunctionSpace,
    quadrature: DomainQuadrature,
    source: (npt.ArrayLike
             | Callable[[npt.NDArray[np.float64]], npt.ArrayLike]),
) -> tuple[csr_matrix, npt.NDArray[np.float64]]:
    """Stiffness and load of an element over a quadrature, in the numbering
    of the degrees of freedom of a space."""
    if not isinstance(element, PoissonElement):
        raise TypeError('the element must be a Poisson element')

    if not isinstance(space, FunctionSpace):
        raise TypeError('the space must be a function space')

    patch = space.patch

    if isinstance(patch, PlanarPatch):
        assembler = _cpp.Assembler2d(space._cpp_object, patch._cpp_object)
        cpp_element = element._cpp_element(2)
    elif isinstance(patch, VolumePatch):
        assembler = _cpp.Assembler3d(space._cpp_object, patch._cpp_object)
        cpp_element = element._cpp_element(3)
    else:
        raise TypeError('the space must lie on a planar or volume patch')

    if not isinstance(quadrature, DomainQuadrature):
        raise TypeError('the quadrature must be a domain quadrature')

    # The system numbers the cells as the elements of the patch, which
    # another patch or a refined grid would not
    refined = np.any(quadrature.grid.levels != 0)

    if quadrature.patch is not patch or refined:
        raise ValueError('the quadrature must lie on the elements of the '
                         'patch of the space')

    if callable(source):
        source = source(quadrature.positions)

    source = np.asarray(source, dtype=np.float64)

    if source.shape != (quadrature.num_points,):
        raise ValueError('the source must give one value per point')

    assembler.assemble_stiffness(cpp_element, quadrature._cpp_object)
    assembler.assemble_load(cpp_element, quadrature._cpp_object, source)

    return assembler.stiffness, assembler.load


def _fixed(
    num_dofs: int,
    fixed: npt.ArrayLike | None,
    values: npt.ArrayLike | None,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    """Fixed degrees of freedom and their values, checked against the
    number of degrees of freedom of a space."""
    fixed = np.asarray([] if fixed is None else fixed, dtype=np.int64)

    if values is None:
        values = np.zeros(fixed.shape)

    values = np.asarray(values, dtype=np.float64)

    if (fixed.ndim != 1 or np.any((fixed < 0) | (fixed >= num_dofs))
            or len(np.unique(fixed)) != len(fixed)):
        raise ValueError('the fixed degrees of freedom must lie in the '
                         'space, each listed once')

    if values.shape != fixed.shape:
        raise ValueError('there must be one value per fixed degree of '
                         'freedom')

    return fixed, values
