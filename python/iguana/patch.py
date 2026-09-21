# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Spline patches, the maps from a parameter box into physical space"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from iguana import cpp as _cpp


class VolumePatch:
    """A spline map from a three-dimensional parameter box into space."""

    _cpp_object: _cpp.VolumePatch

    def __init__(
        self,
        patch: _cpp.VolumePatch,
        degrees: Sequence[int],
        knots: Sequence[Sequence[float]],
    ) -> None:
        """Initialize the patch from a C++ patch.

        Args:
            patch: A C++ patch object.
            degrees: Polynomial degree of each parametric direction.
            knots: Knot vector of each parametric direction.
        """
        self._cpp_object = patch
        self._degrees = tuple(degrees)
        self._knots = tuple(np.asarray(k, float) for k in knots)

    @property
    def degrees(self) -> tuple[int, ...]:
        """Polynomial degree of each parametric direction."""
        return self._degrees

    @property
    def knots(self) -> tuple[npt.NDArray[np.float64], ...]:
        """Knot vector of each parametric direction."""
        return self._knots

    @property
    def control_points(self) -> npt.NDArray[np.float64]:
        """Control points, of shape ``(num_control_points, 3)``.

        A read-only view of the patch, not a copy. It stays valid for as
        long as the patch does.
        """
        return self._cpp_object.coefficients

    def __repr__(self) -> str:
        return (f'VolumePatch(degrees={self._degrees}, '
                f'num_control_points={len(self.control_points)})')
