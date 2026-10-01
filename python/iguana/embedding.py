# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""Embedded domains, how the physical domain lies on the elements of a patch

CellType tells where an element lies with respect to the physical domain:
CellType.inside, CellType.outside or CellType.cut. The type is geometric
only; how each type of cell is integrated is up to the method built on
the domain.
"""

from iguana import cpp as _cpp

CellType = _cpp.CellType
