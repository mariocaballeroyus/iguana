/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_QUADRATURE_HPP
#define IGUANA_QUADRATURE_QUADRATURE_HPP

#include <concepts>
#include <cstddef>
#include <utility>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Points and weights of a quadrature
 *
 * @details The points are stored group after group, and an offset marks
 *          where the points of each group start. The points lie in the 
 *          parameter space, and are grouped to share their active functions.
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class Quadrature
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /// @brief Number of points over all groups
    constexpr int num_points() const noexcept
    { return static_cast<int>(weights_.size()); }

    /// @brief First point of each group, followed by the number of points
    constexpr const Eigen::VectorXi& offsets() const noexcept
    { return offsets_; }

    /// @brief Points in parameter space, with size (num_points, d)
    constexpr const Eigen::MatrixX<T>& points() const noexcept
    { return points_; }

    /// @brief Weights, one per point
    constexpr const Eigen::VectorX<T>& weights() const noexcept
    { return weights_; }

protected:
    /// @brief Constructs an empty quadrature, with no group and no point
    Quadrature()
        : offsets_(Eigen::VectorXi::Zero(1)),
          points_(0, d)
    {
    }

    /**
     * @brief Constructs the quadrature from its flat arrays, which the
     *        derived quadrature checks
     *
     * @param offsets First point of each group, followed by the number of
     *        points
     * @param points Points in parameter space, with size (num_points, d)
     * @param weights Weights, one per point
     */
    Quadrature(Eigen::VectorXi offsets, Eigen::MatrixX<T> points,
               Eigen::VectorX<T> weights)
        : offsets_(std::move(offsets)),
          points_(std::move(points)),
          weights_(std::move(weights))
    {
    }

    /// @brief First point of each group, followed by the number of points
    Eigen::VectorXi offsets_;

    /// @brief Points in parameter space, with size (num_points, d)
    Eigen::MatrixX<T> points_;

    /// @brief Weights, one per point
    Eigen::VectorX<T> weights_;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_QUADRATURE_HPP
