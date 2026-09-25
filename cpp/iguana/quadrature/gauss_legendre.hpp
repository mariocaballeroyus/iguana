/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_GAUSS_LEGENDRE_HPP
#define IGUANA_QUADRATURE_GAUSS_LEGENDRE_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/quadrature/quadrature_rule.hpp"

namespace iguana
{

/**
 * @brief Gauss-Legendre rule
 *
 * The rule is the product of one univariate rule per direction, with the
 * first direction running fastest
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class GaussLegendre final : public QuadratureRule<T, d>
{
    static_assert(d > 0, "GaussLegendre: "
                         "the parametric dimension must be positive");

public:
    /// @brief Largest tabulated number of points per direction
    static constexpr int max_points = 8;

    /**
     * @brief Constructs the rule with the same count in every direction
     *
     * @param num_points Number of points per direction
     *
     * @throws std::invalid_argument If @p num_points lies outside
     *         [1, max_points]
     */
    explicit GaussLegendre(int num_points);

    /**
     * @brief Constructs the rule with one count per direction
     *
     * @param num_points Number of points of each direction
     *
     * @throws std::invalid_argument If a count lies outside
     *         [1, max_points]
     */
    explicit GaussLegendre(const std::array<int, d>& num_points);

    /// @brief Number of points over all directions
    constexpr int num_points() const noexcept
    { return static_cast<int>(weights_.size()); }

    /// @brief Reference points on [-1, 1]^d, with size (num_points, d)
    constexpr const Eigen::MatrixX<T>& points() const noexcept
    { return points_; }

    /// @brief Reference weights, one per point, summing to 2^d
    constexpr const Eigen::VectorX<T>& weights() const noexcept
    { return weights_; }

    /**
     * @brief Maps the rule onto an element
     *
     * Each direction k maps through the affine map
     *
     *     x_k(xi_k) = (end_k - start_k) / 2 xi_k + (start_k + end_k) / 2
     *
     * and each weight scales by its Jacobian, the product of
     * (end_k - start_k) / 2 over the directions
     *
     * @param start Parameters at which the element starts
     * @param end Parameters at which the element ends
     * @param points Output matrix with size (num_points, d), as
     *        TensorBSpline::eval_on_element() takes it. It is resized when
     *        necessary
     * @param weights Output vector with size num_points. It is resized
     *        when necessary
     *
     * @pre @p start lies below @p end in every direction
     */
    void map_to(const std::array<T, d>& start, const std::array<T, d>& end,
                Eigen::MatrixX<T>& points,
                Eigen::VectorX<T>& weights) const override;

private:
    /// @brief Reference points on [-1, 1]^d, with size (num_points, d)
    Eigen::MatrixX<T> points_;

    /// @brief Reference weights, one per point, summing to 2^d
    Eigen::VectorX<T> weights_;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_GAUSS_LEGENDRE_HPP
