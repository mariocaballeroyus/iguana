/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_NURBS_GRID_CROSSINGS_HPP
#define IGUANA_EMBEDDING_NURBS_GRID_CROSSINGS_HPP

#include <array>
#include <concepts>
#include <vector>

#include <Eigen/Core>

#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/// @brief Part of an element of a curve inside one cell of a grid
template<std::floating_point T>
struct CurvePiece
{
    /// @brief Parameter of [0, 1] of the element at which the piece starts
    T start;

    /// @brief Parameter of [0, 1] of the element at which the piece ends
    T end;

    /// @brief Cell holding the piece, with the first direction running
    ///        fastest
    int cell;
};

/**
 * @brief Pieces into which the lines of a grid divide an element of a
 *        planar NURBS curve, in increasing parameter
 *
 * The element is split where it crosses the lines, and crossings closer
 * than the tolerance to each other, as those of both axes at a vertex, or
 * to an end of the element merge. Each piece goes to the cell holding
 * points inside it, and an element lying on a line to the cell opposite
 * its normal sign (y', -x'). Parts outside the grid are left out, and so
 * is an element whose control points coincide
 *
 * @param curve NURBS curve in the plane of the grid
 * @param element Element of the curve
 * @param extraction Bezier extraction operator of the element, as given
 *        by extraction_operators()
 * @param lines Coordinates of the lines of the grid along each axis, in
 *        increasing order, cell i spanning [lines[i], lines[i + 1]]
 * @param sign Sign of the curve, which orients its normal
 * @param tolerance Width, relative to the element, below which crossings
 *        merge
 * @return Pieces in increasing parameter
 *
 * @pre The weights are positive, each axis has at least two lines, an
 *      element lying on a line has exactly the coordinate of the line at
 *      its control points, and no element runs back and forth along a line
 */
template<std::floating_point T>
std::vector<CurvePiece<T>> divide_curve(
    const Patch<TensorNURBS<T, 1>, 2>& curve, int element,
    const Eigen::MatrixX<T>& extraction,
    const std::array<std::vector<T>, 2>& lines, int sign, T tolerance);

} // namespace iguana

#endif // IGUANA_EMBEDDING_NURBS_GRID_CROSSINGS_HPP
