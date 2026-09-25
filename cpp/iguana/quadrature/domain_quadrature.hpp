/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_DOMAIN_QUADRATURE_HPP
#define IGUANA_QUADRATURE_DOMAIN_QUADRATURE_HPP

#include <concepts>
#include <cstddef>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Quadrature over the elements of a domain
 *
 * The points and weights are stored element after element, and an offset
 * marks where the points of each element start. Elements may hold different
 * numbers of points, so that cut elements can carry rules of their own, and
 * elements that are not integrated are left out
 *
 * The points lie in the parameter space of the domain. The weights include
 * the measure of their element in parameter space but not the Jacobian of
 * the geometry map, which the assembly applies
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class DomainQuadrature
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    /**
     * @brief Constructs the quadrature from its flat arrays
     *
     * @param elements Index of each integrated element, in the numbering of
     *        the domain the quadrature is built on
     * @param offsets First point of each integrated element, followed by
     *        the number of points, with size num_elements + 1
     * @param points Points in parameter space, with size (num_points, d)
     * @param weights Weights, one per point
     *
     * @throws std::invalid_argument If the points do not have d columns, if
     *         there is not one weight per point, if there is not one offset
     *         more than elements, or if the offsets do not run from zero to
     *         the number of points without decreasing
     */
    DomainQuadrature(Eigen::VectorXi elements, Eigen::VectorXi offsets,
                     Eigen::MatrixX<T> points, Eigen::VectorX<T> weights);

    /// @brief Number of integrated elements
    constexpr int num_elements() const noexcept
    { return static_cast<int>(elements_.size()); }

    /// @brief Number of points over all elements
    constexpr int num_points() const noexcept
    { return static_cast<int>(weights_.size()); }

    /// @brief Index of each integrated element, in the domain numbering
    constexpr const Eigen::VectorXi& elements() const noexcept
    { return elements_; }

    /// @brief First point of each element, followed by the number of points
    constexpr const Eigen::VectorXi& offsets() const noexcept
    { return offsets_; }

    /// @brief Points in parameter space, with size (num_points, d)
    constexpr const Eigen::MatrixX<T>& points() const noexcept
    { return points_; }

    /// @brief Weights, one per point
    constexpr const Eigen::VectorX<T>& weights() const noexcept
    { return weights_; }

private:
    /// @brief Index of each integrated element, in the domain numbering
    Eigen::VectorXi elements_;

    /// @brief First point of each element, followed by the number of points
    Eigen::VectorXi offsets_;

    /// @brief Points in parameter space, with size (num_points, d)
    Eigen::MatrixX<T> points_;

    /// @brief Weights, one per point
    Eigen::VectorX<T> weights_;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_DOMAIN_QUADRATURE_HPP
