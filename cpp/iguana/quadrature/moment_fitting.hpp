/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_MOMENT_FITTING_HPP
#define IGUANA_QUADRATURE_MOMENT_FITTING_HPP

#include <array>
#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

#include "iguana/quadrature/gauss_legendre.hpp"
#include "iguana/quadrature/quadrature_rule.hpp"

namespace iguana
{

/**
 * @brief Moment-fitted rules on the cut cells of a domain bounded by closed
 *        facets
 *
 * The rule of a cell integrates the Legendre products up to an order in
 * each direction over its part inside the domain, fitted by non-negative
 * least squares after Messmer et al. (CMAME 400, 2022)
 *
 * @tparam T Floating-point type
 * @tparam d Number of parametric directions, two or three
 */
template<std::floating_point T, std::size_t d>
class MomentFitting final : public QuadratureRule<T, d>
{
    static_assert(d == 2 || d == 3, "MomentFitting: "
                                    "the domain must have two or three "
                                    "parametric directions");

public:
    /// @brief Highest order, the one the rules on the facets reach
    static constexpr int max_order = d == 3 ? 4 : 7;

    /**
     * @brief Constructs the rules of a domain
     *
     * @param vertices Vertices of the facets in parameter space, with size
     *        (num_vertices, d)
     * @param facets Vertices of each facet, with size (num_facets, d):
     *        segments with the domain on their left, or triangles
     *        counterclockwise seen from outside
     * @param order Highest Legendre degree of each direction
     *
     * @throws std::invalid_argument If the vertices or facets do not have
     *         d columns, if a facet refers to a missing vertex, or if
     *         @p order lies outside [0, max_order]
     *
     * @pre The facets close the domain
     */
    MomentFitting(const Eigen::MatrixX<T>& vertices,
                  const Eigen::MatrixXi& facets, int order);

    /**
     * @brief Rule of a cell on the reference cell, fitted to the moments of
     *        its part inside the domain
     *
     * Non-negative least squares weights the Gauss points inside the domain
     * of 2, then 4, boxes per direction to reproduce reference_moments(),
     * keeping at most (order + 1)^d of positive weight. A cell whose part
     * inside is below a thousandth of it, or whose fit misses by more than
     * a hundredth, gets no points
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
    void reference_rule(const std::array<T, d>& start,
                        const std::array<T, d>& end,
                        Eigen::MatrixX<T>& points,
                        Eigen::VectorX<T>& weights) const override;

private:
    /// @brief Vertices of each facet in parameter space, one per column
    std::vector<Eigen::Matrix<T, d, d>> facets_;

    /// @brief Highest Legendre degree of each direction
    int order_;

    /// @brief Rule on each interval whose nodes, per direction, grid the
    ///        candidates
    GaussLegendre<T, 1> candidate_rule_;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_MOMENT_FITTING_HPP
