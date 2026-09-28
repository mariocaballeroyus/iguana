# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""IGUANA: Immersogeometric framework for boundary-unfitted analysis

The compiled classes live in iguana.cpp. This package wraps them in Python
types that hold their C++ instance as _cpp_object
"""

from iguana import cpp
from iguana.embedding import CellType
from iguana.patch import (CurvePatch, SurfacePatch, VolumePatch,
                          create_box)
from iguana.quadrature import DomainQuadrature

__version__ = cpp.__version__

__all__ = ['CellType', 'CurvePatch', 'DomainQuadrature', 'SurfacePatch',
           'VolumePatch', 'create_box']
