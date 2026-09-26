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

#include "iguana/quadrature/quadrature_rule.hpp"

namespace iguana
{

/**
 * @brief Moments of the part of [-1, 1]^d inside a domain bounded by facets
 *
 * The integrals of P_{j_1} ... P_{j_d} over that part, with j_1 running
 * fastest, by the divergence theorem with the field
 * (Q_{j_1} P_{j_2} ... P_{j_d}, 0, ...), Q_{j_1} vanishing at -1: the facets
 * clipped to the cell, plus twice the section's moments at x_1 = 1 where
 * j_1 = 0
 *
 * @tparam d Number of directions, two or three
 * @param facets Facets bounding the domain, one vertex per column: segments
 *        with the domain on their left, or triangles counterclockwise seen
 *        from outside
 * @param order Highest Legendre degree of each direction
 * @return Moments, with size (order + 1)^d
 *
 * @pre @p order lies in [0, 7] in two directions and in [0, 4] in three
 */
template<std::floating_point T, std::size_t d>
Eigen::VectorX<T>
reference_moments(const std::vector<Eigen::Matrix<T, d, d>>& facets,
                  int order);

/**
 * @brief Moment-fitted rules on the cut cells of a domain bounded by
 *        closed facets
 *
 * The rule of a cell integrates exactly the tensor-product Legendre
 * polynomials up to an order in each direction, over the part of the cell
 * inside the trimmed domain
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
    /**
     * @brief Constructs the rules of a domain
     *
     * @param vertices Vertices of the facets in parameter space, with size
     *        (num_vertices, d)
     * @param facets Vertices of each facet, triangles in three directions
     *        and segments in two, ordered so that its normal points
     *        outwards, with size (num_facets, d)
     * @param order Highest Legendre degree of each direction
     * @param subdivisions Number of intervals per direction over a cell
     *        whose Gauss points the rays pass through
     *
     * @throws std::invalid_argument If the vertices or facets do not have
     *         d columns, if a facet refers to a missing vertex, if
     *         @p order is negative, or if @p subdivisions is not positive
     */
    MomentFitting(Eigen::MatrixX<T> vertices, Eigen::MatrixXi facets,
                  int order, int subdivisions);

    /**
     * @brief Rule of a cell on the reference cell, fitted to the moments of
     *        its part inside the domain
     *
     * The moments, the integrals of the Legendre polynomials over that
     * part, are taken along rays through the cell. Non-negative least
     * squares then keeps some points of the rays inside the domain, at most
     * (order + 1)^d, with positive weights that reproduce the moments. A
     * cell outside the domain gets no points
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
    /// @brief Vertices of the facets, with size (num_vertices, d)
    Eigen::MatrixX<T> vertices_;

    /// @brief Vertices of each facet, with size (num_facets, d)
    Eigen::MatrixXi facets_;

    /// @brief Highest Legendre degree of each direction
    int order_;

    /// @brief Number of intervals per direction the rays of a cell cover
    int subdivisions_;
};

} // namespace iguana

#endif // IGUANA_QUADRATURE_MOMENT_FITTING_HPP
