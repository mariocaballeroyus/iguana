/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_INSIDE_HPP
#define IGUANA_EMBEDDING_INSIDE_HPP

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "iguana/embedding/slicer.hpp"

namespace iguana
{

/**
 * @brief Signed crossings of facets with the line along x_1 through a point
 *
 * The facets are sliced down to the line, through the plane at x_3 in three
 * directions, so each crossing is where the line meets a facet along x_1,
 * with its sign as slice_segment() gives it
 *
 * @param facets Facets bounding a domain, one vertex per column: segments
 *        with the domain on their left, or triangles counterclockwise seen
 *        from outside
 * @param point Point the line passes through
 * @return Crossings, in the order of the facets
 */
template<std::floating_point T, std::size_t d>
std::vector<std::pair<T, int>>
line_crossings(const std::vector<Eigen::Matrix<T, d, d>>& facets,
               const Eigen::Vector<T, d>& point)
{
    static_assert(d == 2 || d == 3, "line_crossings: "
                                    "the facets must have two or three "
                                    "directions");

    std::vector<std::pair<T, int>> result;

    for (const Eigen::Matrix<T, d, d>& facet : facets) {
        if constexpr (d == 2) {
            if (const auto crossing = slice_segment<T>(facet, 1, point(1)))
                result.push_back(*crossing);
        }
        else if constexpr (d == 3) {
            const auto segment = slice_triangle<T>(facet, 2, point(2));

            if (!segment)
                continue;

            if (const auto crossing = slice_segment<T>(*segment, 1, point(1)))
                result.push_back(*crossing);
        }
    }

    return result;
}

/**
 * @brief Whether a line lies inside a domain at a coordinate, having entered
 *        it more often than left it before
 *
 * @param crossings Signed crossings of the line with the boundary, +1 where
 *        it leaves the domain and -1 where it enters, in any order
 * @param coordinate Coordinate along the line, where a crossing counts as
 *        before it
 */
template<std::floating_point T>
bool is_inside(const std::vector<std::pair<T, int>>& crossings, T coordinate)
{
    int depth = 0;

    for (const auto& [position, sign] : crossings)
        if (position <= coordinate)
            depth -= sign;

    return depth > 0;
}

/**
 * @brief Whether points on one line along x_1 lie inside a domain
 *
 * The crossings of the line are taken once, sorted, and swept together with
 * the points in their order along it
 *
 * @param facets Facets bounding the domain, as line_crossings() takes them
 * @param points Points on the line, one per row
 * @return Whether each point is inside, a crossing on it counting as before
 *         it
 *
 * @pre The points share their coordinates across x_1
 */
template<std::floating_point T, std::size_t d>
Eigen::Array<bool, Eigen::Dynamic, 1>
is_inside(const std::vector<Eigen::Matrix<T, d, d>>& facets,
          const Eigen::MatrixX<T>& points)
{
    Eigen::Array<bool, Eigen::Dynamic, 1> result(points.rows());

    if (points.rows() == 0)
        return result;

    std::vector<std::pair<T, int>> crossings =
        line_crossings<T, d>(facets, points.row(0).transpose());
    std::sort(crossings.begin(), crossings.end());

    std::vector<Eigen::Index> order(points.rows());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&points](Eigen::Index first, Eigen::Index second) {
                  return points(first, 0) < points(second, 0);
              });

    // Entries minus exits up to each point, swept along the line
    int depth = 0;
    std::size_t crossing = 0;

    for (const Eigen::Index row : order) {
        for (; crossing < crossings.size()
               && crossings[crossing].first <= points(row, 0);
             ++crossing)
            depth -= crossings[crossing].second;

        result(row) = depth > 0;
    }

    return result;
}

} // namespace iguana

#endif // IGUANA_EMBEDDING_INSIDE_HPP
