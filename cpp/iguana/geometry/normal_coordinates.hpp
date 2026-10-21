/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GEOMETRY_NORMAL_COORDINATES_HPP
#define IGUANA_GEOMETRY_NORMAL_COORDINATES_HPP

#include <concepts>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Derivatives in the normal coordinates of a plane curve at a point
 *        of it, as combinations of the partial derivatives there
 *
 *     x = γ(s) + t ν(s),    τ' = −κ ν,    ν' = κ τ
 *
 * @details Row (j, i) holds the weights of the partial derivatives that
 *          make up ∂_s^j ∂_t^i at t = 0, the j-th derivative along the curve
 *          of the i-th derivative along its normal. Rows and columns follow
 *          simplex_index(), with j and a running fastest. The derivatives
 *          are those of the coordinates the frame is given in: physical ones
 *          with the frame itself, parametric ones of an affine map
 *          x = a + A ξ with the frame mapped by A^−1
 *
 * @tparam T Floating-point type
 *
 * @param frame Tangent τ and normal ν at the point, the columns of the
 *        matrix
 * @param curvatures Curvature and its arc-length derivatives at the point,
 *        from order 0
 * @param order Highest total order j + i, and of the partial derivatives
 * @return Weights of size (simplex_size(order), simplex_size(order))
 *
 * @throws std::invalid_argument If @p order is negative, or if
 *         @p curvatures has fewer than order − 1 entries
 */
template<std::floating_point T>
Eigen::MatrixX<T> normal_coordinate_derivs(const Eigen::Matrix2<T>& frame,
                                           const Eigen::VectorX<T>& curvatures,
                                           int order);

} // namespace iguana

#endif // IGUANA_GEOMETRY_NORMAL_COORDINATES_HPP
