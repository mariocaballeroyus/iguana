/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_QUADRATURE_RULE_HPP
#define IGUANA_QUADRATURE_QUADRATURE_RULE_HPP

#include <array>
#include <concepts>
#include <cstddef>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Quadrature rule on the cells of a domain
 *
 * A rule gives the points of any cell and their weights. Some rules map one
 * reference rule onto every cell, as GaussLegendre for uncut cells, and
 * others build a rule of their own for each, as MomentFitting for cut cells
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions
 */
template<std::floating_point T, std::size_t d>
class QuadratureRule
{
public:
    /// @brief Number of parametric directions
    static constexpr std::size_t dimension = d;

    virtual ~QuadratureRule() = default;

    /**
     * @brief Rule of a cell
     *
     * @param start Parameters at which the cell starts
     * @param end Parameters at which the cell ends
     * @param points Output matrix of the points, each in the cell, with
     *        size (num_points, d). It is resized when necessary
     * @param weights Output vector of their weights, with size num_points.
     *        It is resized when necessary
     *
     * @pre @p start lies below @p end in every direction
     */
    virtual void map_to(const std::array<T, d>& start,
                        const std::array<T, d>& end,
                        Eigen::MatrixX<T>& points,
                        Eigen::VectorX<T>& weights) const = 0;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_QUADRATURE_RULE_HPP
