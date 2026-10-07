# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""IGUANA: Immersogeometric framework for boundary-unfitted analysis

The compiled classes live in iguana.cpp. This package wraps them in Python
types that hold their C++ instance as _cpp_object
"""

from iguana import cpp
from iguana.boundary import Boundary
from iguana.condition import NeumannCondition, PenaltyCondition
from iguana.element import PoissonElement
from iguana.embedding import CellType
from iguana.fspace import FunctionSpace
from iguana.grid import HierarchicalGrid
from iguana.patch import (CurvePatch, PlanarPatch, SurfacePatch, VolumePatch,
                          create_box, create_curve, create_rectangle,
                          create_surface)
from iguana.quadrature import BoundaryQuadrature, DomainQuadrature
from iguana.solver import solve

__version__ = cpp.__version__

__all__ = ['Boundary', 'BoundaryQuadrature', 'CellType', 'CurvePatch',
           'DomainQuadrature', 'FunctionSpace', 'HierarchicalGrid',
           'NeumannCondition', 'PenaltyCondition', 'PlanarPatch',
           'PoissonElement', 'SurfacePatch', 'VolumePatch', 'create_box',
           'create_curve', 'create_rectangle', 'create_surface', 'solve']
