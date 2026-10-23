/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GEOMETRY_DIRECTIONAL_DERIVS_HPP
#define IGUANA_GEOMETRY_DIRECTIONAL_DERIVS_HPP

#include <concepts>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Derivatives along the two directions of a frame, as combinations
 *        of the partial derivatives
 *
 *     (f₀ · ∇)^j (f₁ · ∇)^i = Σ_{a + b = j + i} c_ab ∂_1^a ∂_2^b
 *
 * @details Row (j, i) holds the weights c of the partial derivatives that
 *          make up the j-th derivative along f₀ of the i-th along f₁. Rows
 *          and columns follow simplex_index(), with j and a running
 *          fastest. The derivatives are those of the coordinates the frame
 *          is given in: physical ones with the frame itself, parametric
 *          ones of an affine map x = a + A ξ with the frame mapped by A^−1
 *
 * @tparam T Floating-point type
 *
 * @param frame Directions f₀ and f₁, the columns of the matrix
 * @param order Highest total order j + i, and of the partial derivatives
 * @return Weights of size (simplex_size(order), simplex_size(order))
 *
 * @throws std::invalid_argument If @p order is negative
 */
template<std::floating_point T>
Eigen::MatrixX<T> directional_derivs(const Eigen::Matrix2<T>& frame,
                                     int order);

} // namespace iguana

#endif // IGUANA_GEOMETRY_DIRECTIONAL_DERIVS_HPP
