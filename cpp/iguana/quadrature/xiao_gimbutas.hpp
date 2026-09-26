/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_XIAO_GIMBUTAS_HPP
#define IGUANA_QUADRATURE_XIAO_GIMBUTAS_HPP

#include <concepts>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Xiao-Gimbutas rule on the triangle s, t >= 0, s + t <= 1
 *
 * The fully symmetric rules of Xiao and Gimbutas (Comput. Math. Appl. 59,
 * 2010), as Basix tabulates them, integrate polynomials up to their total
 * degree exactly, with positive weights at points inside the triangle
 *
 * @tparam T Floating-point type
 */
template<std::floating_point T>
class XiaoGimbutas
{
public:
    /// @brief Highest tabulated degree
    static constexpr int max_degree = 13;

    /**
     * @brief Constructs the rule of a degree
     *
     * @param degree Highest total degree the rule integrates exactly
     *
     * @throws std::invalid_argument If @p degree lies outside
     *         [1, max_degree]
     */
    explicit XiaoGimbutas(int degree);

    /// @brief Number of points
    constexpr int num_points() const noexcept
    { return static_cast<int>(weights_.size()); }

    /// @brief Points (s, t) of the triangle, with size (num_points, 2)
    constexpr const Eigen::MatrixX<T>& points() const noexcept
    { return points_; }

    /// @brief Weights, one per point, summing to 1 / 2
    constexpr const Eigen::VectorX<T>& weights() const noexcept
    { return weights_; }

private:
    /// @brief Points (s, t) of the triangle, with size (num_points, 2)
    Eigen::MatrixX<T> points_;

    /// @brief Weights, one per point, summing to 1 / 2
    Eigen::VectorX<T> weights_;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_XIAO_GIMBUTAS_HPP
