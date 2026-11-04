# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Field evaluation, a solved field read back at points

A problem returns the coefficients of its field. These functions read the
field at points of its patch through the traces of its element, such as the
velocity and the pressure of a fluid, and through their fluxes, such as its
traction. The compiled traces need a build of iguana with PETSc, as the
problems do.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp
from iguana.element import ElasticityElement, PoissonElement, StokesElement
from iguana.fspace import FunctionSpace, MixedFunctionSpace
from iguana.patch import PlanarPatch

# The element of each trace, whose unknowns the space must hold
_ELEMENTS = {PoissonElement.U: PoissonElement,
             StokesElement.U: StokesElement,
             StokesElement.P: StokesElement,
             ElasticityElement.U: ElasticityElement}


def trace_at_points(
    trace: (PoissonElement.U | StokesElement.U | StokesElement.P
            | ElasticityElement.U),
    space: FunctionSpace | MixedFunctionSpace,
    coefficients: npt.ArrayLike | Sequence[npt.ArrayLike],
    points: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Values of a trace of a field at points of its patch.

    Each point takes the cell holding it, which on a knot line is the one
    after the line.

    Args:
        trace: The trace to evaluate, such as `fluid.u` or `fluid.p`.
        space: The space of the field, on a planar patch whose map is
            affine.
        coefficients: The coefficients of the field, as a problem returns
            them: of shape `(num_dofs,)`, or `(num_dofs, block_size)` for a
            space of several components, or those of each field of a mixed
            space.
        points: The points in the patch, of shape `(num_points, 2)`.

    Returns:
        The values at the points, of shape `(num_points,)` for a trace of
        one value per point, such as a pressure, and `(num_points, 2)` for
        one of two, such as a velocity.

    Raises:
        TypeError: If the trace is not one of an element, or if the space is
            neither a function space nor a mixed space on a planar patch.
        ValueError: If the space does not hold the unknowns of the element
            of the trace, if the coefficients do not give every degree of
            freedom, if the points are not of the plane, if the map of the
            patch is not affine, or if a point lies outside the patch or on
            a cell outside the physical domain.
    """
    compiled_space, patch, entries, points = _arguments(
        trace, space, coefficients, points)

    return _components(trace, [
        _cpp.trace_at_points(compiled_space, patch, compiled, entries, points)
        for compiled in trace._cpp_traces()])


def flux_at_points(
    trace: PoissonElement.U | StokesElement.U | ElasticityElement.U,
    space: FunctionSpace | MixedFunctionSpace,
    coefficients: npt.ArrayLike | Sequence[npt.ArrayLike],
    points: npt.ArrayLike,
    normals: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Values of the flux conjugate to a trace of a field at points of its
    patch, such as the traction of a fluid, with its pressure.

    Each point takes the cell holding it, so that a flux jumping across a
    knot line, as at degree one, gives the value of the cell after the line.
    The normals need not be those of a boundary of the field: the traction of
    a fluid along the normal of a solid, pointing into the fluid, is the load
    of the fluid on the solid. The traction of a solid itself is that of small
    displacements, as a flux is linear in the coefficients, which holds for
    the solution of a `LinearProblem` and not for one of large displacements.

    Args:
        trace: The trace whose flux to evaluate, such as `fluid.u` for the
            traction.
        space: The space of the field, on a planar patch whose map is
            affine.
        coefficients: The coefficients of the field, as a problem returns
            them: of shape `(num_dofs,)`, or `(num_dofs, block_size)` for a
            space of several components, or those of each field of a mixed
            space.
        points: The points in the patch, of shape `(num_points, 2)`.
        normals: The unit normals at the points, of the same shape.

    Returns:
        The values at the points, of shape `(num_points,)` for a flux of one
        value per point, such as a normal derivative, and `(num_points, 2)`
        for one of two, such as a traction.

    Raises:
        TypeError: If the trace is not one of an element or has no flux, or
            if the space is neither a function space nor a mixed space on a
            planar patch.
        ValueError: If the space does not hold the unknowns of the element
            of the trace, if the coefficients do not give every degree of
            freedom, if the points are not of the plane, if the normals are
            not one per point, if the map of the patch is not affine, or if
            a point lies outside the patch or on a cell outside the physical
            domain.
    """
    compiled_space, patch, entries, points = _arguments(
        trace, space, coefficients, points)

    if not hasattr(trace, '_cpp_fluxes'):
        raise TypeError('the trace has no flux')

    normals = np.asarray(normals, dtype=np.float64)

    if normals.shape != points.shape:
        raise ValueError('the normals must be one per point')

    return _components(trace, [
        _cpp.flux_at_points(compiled_space, patch, compiled, entries, points,
                            normals)
        for compiled in trace._cpp_fluxes()])


def _arguments(
    trace: (PoissonElement.U | StokesElement.U | StokesElement.P
            | ElasticityElement.U),
    space: FunctionSpace | MixedFunctionSpace,
    coefficients: npt.ArrayLike | Sequence[npt.ArrayLike],
    points: npt.ArrayLike,
) -> tuple[_cpp.MixedFunctionSpace2d, _cpp.PlanarPatch,
           npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compiled space and patch, entries of the coefficients and points of an
    evaluation, checked as the functions of this module document."""
    element = _ELEMENTS.get(type(trace))

    if element is None:
        raise TypeError('the trace must be one of an element')

    if not isinstance(space, (FunctionSpace, MixedFunctionSpace)):
        raise TypeError('the space must be a function space or a mixed '
                        'space')

    if not isinstance(space.patch, PlanarPatch):
        raise TypeError('the space must lie on a planar patch')

    if not space._holds_unknowns(element._num_unknowns):
        raise ValueError('the space must give each unknown of the element a '
                         'component of one field or a field of its own')

    # Its local matrices read the components of one field
    if (element is ElasticityElement
            and isinstance(space, MixedFunctionSpace)):
        raise ValueError('a solid takes the components of one field')

    entries = space._entries(coefficients)
    points = np.asarray(points, dtype=np.float64)

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('the points must be points of the plane, of shape '
                         '(num_points, 2)')

    # A plain space is a mixed one of one field
    compiled_space = (space._cpp_object
                      if isinstance(space, MixedFunctionSpace)
                      else _cpp.MixedFunctionSpace2d([space._cpp_object]))

    return compiled_space, space.patch._cpp_object, entries, points


def _components(
    trace: (PoissonElement.U | StokesElement.U | StokesElement.P
            | ElasticityElement.U),
    values: list[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    """Values at the points of each component of a trace, shaped as its
    values per point."""
    if trace._shape == ():
        return values[0]

    return np.column_stack(values)
