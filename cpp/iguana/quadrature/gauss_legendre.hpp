/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_GAUSS_LEGENDRE_HPP
#define IGUANA_QUADRATURE_GAUSS_LEGENDRE_HPP

#include <concepts>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Univariate Gauss-Legendre rule
 *
 * @tparam T Floating-point type
 */
template<std::floating_point T>
class GaussLegendre
{
public:
    /// @brief Largest tabulated number of points
    static constexpr int max_points = 8;

    /**
     * @brief Constructs the rule with a given number of points
     *
     * @param num_points Number of points
     *
     * @throws std::invalid_argument If @p num_points lies outside
     *         [1, max_points]
     */
    explicit GaussLegendre(int num_points);

    /// @brief Number of points
    constexpr int num_points() const noexcept
    { return static_cast<int>(weights_.size()); }

    /// @brief Reference points on [-1, 1], in increasing order
    constexpr const Eigen::VectorX<T>& points() const noexcept
    { return points_; }

    /// @brief Reference weights, one per point, summing to 2
    constexpr const Eigen::VectorX<T>& weights() const noexcept
    { return weights_; }

    /**
     * @brief Maps the rule onto an element
     *
     * Each reference point xi maps through the affine map
     *
     *     x(xi) = (end - start) / 2 xi + (start + end) / 2
     *
     * and each weight scales by its Jacobian (end - start) / 2
     *
     * @param start Parameter at which the element starts
     * @param end Parameter at which the element ends
     * @param points Output vector with size num_points. It is resized
     *        when necessary
     * @param weights Output vector with size num_points. It is resized
     *        when necessary
     *
     * @pre @p start lies below @p end
     */
    void map_to(T start, T end, Eigen::VectorX<T>& points,
                Eigen::VectorX<T>& weights) const;

private:
    /// @brief Reference points on [-1, 1], in increasing order
    Eigen::VectorX<T> points_;

    /// @brief Reference weights, one per point, summing to 2
    Eigen::VectorX<T> weights_;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_GAUSS_LEGENDRE_HPP
