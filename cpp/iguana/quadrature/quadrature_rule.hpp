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
 * A rule gives the points and weights of any cell on the reference cell
 * [-1, 1]^d, which map_to_cell() places on the cell itself. Some rules give
 * the same reference rule for every cell, as GaussLegendre for uncut cells,
 * and others build a rule of their own for each, as MomentFitting for cut
 * cells
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
     * @brief Rule of a cell on the reference cell [-1, 1]^d
     *
     * @param start Parameters at which the cell starts
     * @param end Parameters at which the cell ends
     * @param points Output matrix of the points in [-1, 1]^d, with size
     *        (num_points, d). It is resized when necessary
     * @param weights Output vector of their weights, with size num_points.
     *        It is resized when necessary
     *
     * @pre @p start lies below @p end in every direction
     */
    virtual void reference_rule(const std::array<T, d>& start,
                                const std::array<T, d>& end,
                                Eigen::MatrixX<T>& points,
                                Eigen::VectorX<T>& weights) const = 0;
};

/**
 * @brief Places a rule on the reference cell [-1, 1]^d on a cell
 *
 * Each direction k maps through the affine map
 *
 *     x_k(xi_k) = (end_k - start_k) / 2 xi_k + (start_k + end_k) / 2
 *
 * and each weight scales by its Jacobian, the product of
 * (end_k - start_k) / 2 over the directions
 *
 * @param start Parameters at which the cell starts
 * @param end Parameters at which the cell ends
 * @param reference_points Points in [-1, 1]^d, with size (num_points, d)
 * @param reference_weights Their weights, with size num_points
 * @param points Output matrix of the points in the cell, with size
 *        (num_points, d), as TensorBSpline::eval_on_element() takes it. It
 *        is resized when necessary
 * @param weights Output vector of their weights, with size num_points. It
 *        is resized when necessary
 *
 * @pre @p start lies below @p end in every direction
 */
template<std::floating_point T, std::size_t d>
void map_to_cell(const std::array<T, d>& start, const std::array<T, d>& end,
                 const Eigen::MatrixX<T>& reference_points,
                 const Eigen::VectorX<T>& reference_weights,
                 Eigen::MatrixX<T>& points, Eigen::VectorX<T>& weights)
{
    points.resize(reference_points.rows(), d);

    T jacobian{1};

    for (std::size_t direction = 0; direction < d; ++direction) {
        const T half = (end[direction] - start[direction]) / T{2};
        const T middle = (start[direction] + end[direction]) / T{2};

        points.col(direction).array() =
            half * reference_points.col(direction).array() + middle;
        jacobian *= half;
    }

    weights = jacobian * reference_weights;
}

} // namespace iguana

#endif // IGUANA_QUADRATURE_QUADRATURE_RULE_HPP
