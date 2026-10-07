/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_IGUANA_HPP
#define IGUANA_IGUANA_HPP

#include "basis/bspline.hpp"
#include "basis/tensor_bspline.hpp"
#include "basis/tensor_nurbs.hpp"
#include "grid/hierarchical_grid.hpp"
#include "grid/hierarchical_grid_iterator.hpp"
#include "grid/knot_vector.hpp"
#include "grid/tensor_grid.hpp"
#include "grid/tensor_grid_iterator.hpp"
#include "embedding/embedded_boundary.hpp"
#include "embedding/embedded_domain.hpp"
#include "embedding/surrogate_boundary.hpp"
#include "fspace/dof_map.hpp"
#include "fspace/function_space.hpp"
#include "geometry/boundary.hpp"
#include "geometry/nurbs/bezier_points.hpp"
#include "geometry/nurbs/curve_projection.hpp"
#include "geometry/patch.hpp"
#include "quadrature/boundary_quadrature.hpp"
#include "quadrature/box_rule.hpp"
#include "quadrature/domain_quadrature.hpp"
#include "quadrature/gauss_legendre/gauss_legendre.hpp"
#include "quadrature/moment_fitting/moment_fitting.hpp"
#include "quadrature/moment_fitting/moments.hpp"
#include "quadrature/xiao_gimbutas/xiao_gimbutas.hpp"
#include "assembly/assembler.hpp"
#include "assembly/element_values.hpp"
#include "element/element.hpp"
#include "element/poisson/poisson_element.hpp"
#include "element/poisson/poisson_traces.hpp"
#include "element/trace.hpp"
#include "condition/condition.hpp"
#include "condition/penalty_condition.hpp"
#include "condition/neumann_condition.hpp"

#endif // IGUANA_IGUANA_HPP
