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
 * @brief Moments of the part of the reference square [-1, 1]^2 inside a
 *        domain bounded by segments
 *
 * The moments, the integrals of P_{j_1}(u) P_{j_2}(v) over that part with
 * j_1 running fastest, follow from the divergence theorem with the field
 * (Q_{j_1}(u) P_{j_2}(v), 0), where Q_{j_1} is the antiderivative of
 * P_{j_1} vanishing at -1. Its flux crosses the segments clipped to the
 * square and, only when j_1 = 0, the upper edge u = 1, where it adds twice
 * the moments of the section of the domain
 *
 * @param segments Segments bounding the domain, with the domain on their
 *        left, one end per column
 * @param order Highest Legendre degree of each direction
 * @return Moments, with size (order + 1)^2
 *
 * @pre @p order lies in [0, GaussLegendre::max_points - 1]
 */
template<std::floating_point T>
Eigen::VectorX<T>
reference_moments(const std::vector<Eigen::Matrix2<T>>& segments, int order);

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
