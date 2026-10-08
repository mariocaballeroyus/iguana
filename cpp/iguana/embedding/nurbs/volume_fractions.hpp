/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_NURBS_VOLUME_FRACTIONS_HPP
#define IGUANA_EMBEDDING_NURBS_VOLUME_FRACTIONS_HPP

#include <Eigen/Core>

#include "iguana/geometry/boundary.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Volume fraction of each element of the grid of a patch, the part
 *        of its measure inside the domain a boundary encloses
 *
 * The fractions are computed in parameter space, where the boundary is
 * pulled back, as the map is affine and keeps the ratios of areas. For a
 * cell C = [x0, x1] x [y0, y1], the field (clamp(x, x0, x1) - x0, 0) on the
 * row of C has the indicator of C as its divergence, so by the divergence
 * theorem the area of C inside the domain is
 *
 *     a_C + (x1 - x0) (sum of b over the cells right of C in its row + s h)
 *
 * with a_C the integral of (x - x0) n_x ds and b the integral of n_x ds
 * over the pieces of the boundary in each cell, h the height of the row,
 * and s = 1 for an unbounded domain, the fluid around an obstacle whose
 * faces point into it, as one enclosing a negative area is. Cells holding no
 * piece have fractions of exactly 0 or 1, and the others are set to 0 or 1
 * within rounding, as a cell whose boundary runs along its side
 *
 * @param patch Patch whose grid holds the elements
 * @param boundary Boundary of the domain in the physical space of the patch
 * @return Volume fraction of each element, in the numbering of the grid
 *
 * @throws std::invalid_argument If the map of the patch is not affine, or
 *         if the knot vector of a face is not clamped
 *
 * @pre The boundary is closed and lies inside the grid, and no face runs
 *      back and forth along a knot line
 */
template<typename Basis>
    requires (Basis::dimension == 2)
Eigen::VectorX<typename Basis::Scalar> volume_fractions(
    const Patch<Basis, 2>& patch,
    const Boundary<typename Basis::Scalar, 2>& boundary);

} // namespace iguana

#endif // IGUANA_EMBEDDING_NURBS_VOLUME_FRACTIONS_HPP
