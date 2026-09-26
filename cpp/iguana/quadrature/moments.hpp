/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_QUADRATURE_MOMENTS_HPP
#define IGUANA_QUADRATURE_MOMENTS_HPP

#include <concepts>
#include <cstddef>
#include <vector>

#include <Eigen/Core>

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

} // namespace iguana

#endif // IGUANA_QUADRATURE_MOMENTS_HPP
