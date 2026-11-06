# Copyright (c) 2026 Mario Caballero
# SPDX-License-Identifier: MIT

"""IGUANA: Immersogeometric framework for boundary-unfitted analysis

The compiled classes live in iguana.cpp. This package wraps them in Python
types that hold their C++ instance as _cpp_object. The problems that solve
with PETSc live in iguana.petsc, which needs a build with PETSc
"""

from iguana import cpp
from iguana.boundary import Boundary
from iguana.condition import (Derivatives, LagrangeMultiplierCondition,
                              NeumannCondition, NitscheCondition,
                              PenaltyCondition, SkeletonMultiplierCondition,
                              SkeletonNitscheCondition)
from iguana.element import ElasticityElement, PoissonElement, StokesElement
from iguana.embedding import CellType, SurrogateBoundary, classify_cells
from iguana.evaluation import flux_at_points, trace_at_points
from iguana.fspace import FunctionSpace, MixedFunctionSpace
from iguana.grid import HierarchicalGrid
from iguana.patch import (CurvePatch, PlanarPatch, SurfacePatch, VolumePatch,
                          create_box, create_curve, create_rectangle,
                          create_surface)
from iguana.quadrature import QuadratureBoundary, QuadratureDomain
from iguana.stabilization import GhostPenalty, VolumeGhostPenalty

__version__ = cpp.__version__

__all__ = ['Boundary', 'CellType', 'CurvePatch', 'Derivatives',
           'ElasticityElement', 'FunctionSpace', 'GhostPenalty',
           'HierarchicalGrid', 'LagrangeMultiplierCondition',
           'MixedFunctionSpace', 'NeumannCondition', 'NitscheCondition',
           'PenaltyCondition', 'PlanarPatch', 'PoissonElement',
           'QuadratureBoundary', 'QuadratureDomain',
           'SkeletonMultiplierCondition', 'SkeletonNitscheCondition',
           'StokesElement', 'SurfacePatch', 'SurrogateBoundary',
           'VolumeGhostPenalty', 'VolumePatch', 'classify_cells',
           'create_box', 'create_curve', 'create_rectangle', 'create_surface',
           'flux_at_points', 'trace_at_points']
