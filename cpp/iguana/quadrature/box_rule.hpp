/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_BOX_RULE_HPP
#define IGUANA_QUADRATURE_BOX_RULE_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Quadrature rule defined on a box cell
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class BoxRule
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    virtual ~BoxRule() = default;

    /**
     * @brief Fills the quadrature of a box on the reference cell [-1, 1]^d
     *
     * @param start Parameters at which the box starts
     * @param end Parameters at which the box ends
     * @param points Output matrix of the points in [-1, 1]^d, with size
     *        (num_points, d). It is resized when necessary
     * @param weights Output vector of their weights, with size num_points.
     *        It is resized when necessary
     *
     * @pre @p start lies below @p end in every direction
     */
    virtual void fill_to_reference_space(const std::array<T, d>& start,
                                         const std::array<T, d>& end,
                                         Eigen::MatrixX<T>& points,
                                         Eigen::VectorX<T>& weights) const = 0;

    /**
     * @brief Places a quadrature on the reference cell [-1, 1]^d on a box
     *        in parameter space
     *
     * Each direction k maps through the affine map
     *
     *     x_k(xi_k) = (end_k - start_k) / 2 xi_k + (start_k + end_k) / 2
     *
     * and each weight scales by its Jacobian, the product of
     * (end_k - start_k) / 2 over the directions
     *
     * @param start Parameters at which the box starts
     * @param end Parameters at which the box ends
     * @param reference_points Points in [-1, 1]^d, with size
     *        (num_points, d)
     * @param reference_weights Their weights, with size num_points
     * @param points Output matrix of the points in the box, with size
     *        (num_points, d), as TensorBSpline::eval_on_element() takes it.
     *        It is resized when necessary
     * @param weights Output vector of their weights, with size num_points.
     *        It is resized when necessary
     *
     * @pre @p start lies below @p end in every direction
     */
    static void
    map_to_parameter_space(const std::array<T, d>& start,
                           const std::array<T, d>& end,
                           const Eigen::MatrixX<T>& reference_points,
                           const Eigen::VectorX<T>& reference_weights,
                           Eigen::MatrixX<T>& points,
                           Eigen::VectorX<T>& weights);
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_BOX_RULE_HPP
