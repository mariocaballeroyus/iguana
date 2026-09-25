# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Quadratures, the points integrating over the cells of a domain

A quadrature is filled one cell type at a time, each with a rule of its
own, such as Gauss-Legendre on the inside cells.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp
from iguana.domain import CellType, TensorDomain


class DomainQuadrature:
    """The quadrature points over the cells of a domain."""

    _cpp_object: _cpp.SurfaceQuadrature | _cpp.VolumeQuadrature

    def __init__(self, domain: TensorDomain) -> None:
        """Initialize an empty quadrature over a domain.

        Args:
            domain: The domain whose cells the quadrature integrates.

        Raises:
            TypeError: If the domain is not a tensor domain.
        """
        if not isinstance(domain, TensorDomain):
            raise TypeError('the domain must be a tensor domain')

        if isinstance(domain._cpp_object, _cpp.VolumeDomain):
            self._cpp_object = _cpp.VolumeQuadrature()
            self._dimension = 3
        else:
            self._cpp_object = _cpp.SurfaceQuadrature()
            self._dimension = 2

        self._domain = domain

    @property
    def domain(self) -> TensorDomain:
        """The domain whose cells the quadrature integrates."""
        return self._domain

    @property
    def num_elements(self) -> int:
        """Number of filled elements."""
        return self._cpp_object.num_elements

    @property
    def num_points(self) -> int:
        """Number of points over all filled elements."""
        return self._cpp_object.num_points

    @property
    def positions(self) -> npt.NDArray[np.float64]:
        """Positions of the points in space, of shape ``(num_points, 3)``."""
        return self._cpp_object.positions(self._domain._cpp_object)

    def fill_gauss_legendre(self, cell_type: CellType,
                            num_points: int | Sequence[int]) -> None:
        """Fill the cells of one type with a Gauss-Legendre rule.

        Args:
            cell_type: The type of the cells to fill.
            num_points: Number of points of each direction, or a single
                number for all of them.

        Raises:
            ValueError: If there is not one number per direction, if a
                number lies outside [1, 8], or if the cells of this type
                are already filled.
        """
        if isinstance(num_points, int):
            num_points = [num_points] * self._dimension

        if len(num_points) != self._dimension:
            raise ValueError('there must be one number of points per '
                             'direction')

        self._cpp_object.fill_gauss_legendre(self._domain._cpp_object,
                                             cell_type, list(num_points))

    def __repr__(self) -> str:
        return (f'DomainQuadrature(num_elements={self.num_elements}, '
                f'num_points={self.num_points})')
