/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_ELEMENT_POISSON_POISSON_NORMAL_SERIES_HPP
#define IGUANA_ELEMENT_POISSON_POISSON_NORMAL_SERIES_HPP

#include <concepts>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Derivatives of the solution of the Poisson equation along the
 *        normal of a boundary, at a point of it, as forms over its trace,
 *        its normal derivative and the source
 *
 * @details Row k of each matrix gives U_k = ∂_t^k u in the normal
 *          coordinates x = γ(s) + t ν(s) of the boundary. Its columns
 *          follow simplex_index() over the derivatives ∂_s^j ∂_t^i at the
 *          point, of u with i at most 1, the trace g and the normal
 *          derivative q along the boundary, and of f
 *
 * @tparam T Floating-point type
 */
template<std::floating_point T>
struct PoissonNormalSeries
{
    /// @brief Weights of the derivatives of u, with size
    ///        (order + 1, simplex_size(order))
    Eigen::MatrixX<T> field;

    /// @brief Weights of the derivatives of f, with size
    ///        (order + 1, simplex_size(order − 2)), and no columns below
    ///        order 2
    Eigen::MatrixX<T> source;
};

/**
 * @brief Normal series of the Poisson equation at a point of a boundary,
 *        up to an order
 *
 *     u(γ + t ν) = Σ_{k ≤ m} t^k / k! U_k,    U_0 = g,    U_1 = q
 *     J³ u_tt + κ J² u_t + J u_ss − t κ' u_s = −J³ f,    J = 1 + t κ
 *
 * @details The second line is −Δu = f in the normal coordinates of the
 *          boundary, times J², whose derivatives along t give each U_k from
 *          the lower ones
 *
 * @tparam T Floating-point type
 *
 * @param curvatures Curvature and its arc-length derivatives at the point,
 *        from order 0
 * @param order Highest order m of the series
 * @return The forms of U_0 to U_m
 *
 * @throws std::invalid_argument If @p order is negative, or if
 *         @p curvatures has fewer than order − 1 entries
 */
template<std::floating_point T>
PoissonNormalSeries<T> poisson_normal_series(
    const Eigen::VectorX<T>& curvatures, int order);

} // namespace iguana

#endif // IGUANA_ELEMENT_POISSON_POISSON_NORMAL_SERIES_HPP
