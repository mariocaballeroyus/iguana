# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""IGUANA: Immersogeometric framework for boundary-unfitted analysis

The compiled classes live in iguana.cpp. This package wraps them in Python
types that hold their C++ instance as _cpp_object
"""

from iguana import cpp
from iguana.domain import HierarchicalDomain
from iguana.embedding import CellType
from iguana.patch import (CurvePatch, PlanarPatch, SurfacePatch, VolumePatch,
                          create_box, create_rectangle)
from iguana.quadrature import DomainQuadrature

__version__ = cpp.__version__

__all__ = ['CellType', 'CurvePatch', 'DomainQuadrature', 'HierarchicalDomain',
           'PlanarPatch', 'SurfacePatch', 'VolumePatch', 'create_box',
           'create_rectangle']
