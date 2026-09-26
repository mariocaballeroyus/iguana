/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_SLICER_HPP
#define IGUANA_EMBEDDING_SLICER_HPP

#include <concepts>
#include <cstddef>
#include <optional>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace iguana
{

/**
 * @brief Point where a segment of a two-dimensional parameter space
 *        crosses the line x_axis = coordinate
 *
 * The segment has the domain on its left, so its outward normal is
 * (dy, -dx). An end on the line counts as below it, which keeps the
 * crossings of a closed boundary paired
 *
 * @param segment Ends of the segment, one per column
 * @param axis Axis the line fixes
 * @param coordinate Coordinate at which it fixes it
 * @return The other coordinate of the point, with +1 where the line leaves
 *         the domain as that coordinate grows and -1 where it enters, or
 *         nothing if the segment misses the line
 *
 * @pre @p axis is 0 or 1
 */
template<std::floating_point T>
std::optional<std::pair<T, int>>
slice_segment(const Eigen::Matrix2<T>& segment,
              std::size_t axis, T coordinate)
{
    const T first = segment(axis, 0);
    const T second = segment(axis, 1);

    if ((first > coordinate) == (second > coordinate))
        // Return early if there is no slicing
        return std::nullopt;

    const T ratio = (coordinate - first) / (second - first);
    const Eigen::Vector2<T> edge = segment.col(1) - segment.col(0);
    const Eigen::Vector2<T> point = segment.col(0) + ratio * edge;

    // Outward normal, with the domain on the left of the segment
    const Eigen::Vector2<T> normal(edge(1), -edge(0));

    const std::size_t other = 1 - axis;

    return std::pair{point(other), normal(other) > T{0} ? 1 : -1};
}

/**
 * @brief Segment where a triangle of a three-dimensional parameter space
 *        crosses the plane x_axis = coordinate
 *
 * The triangle runs counterclockwise seen from outside the domain, and the
 * segment has the slice of the domain on its left, as slice_segment()
 * expects. A vertex on the plane counts as below it, which keeps the
 * slices of a closed boundary closed
 *
 * @param triangle Vertices of the triangle, one per column
 * @param axis Axis the plane fixes
 * @param coordinate Coordinate at which it fixes it
 * @return Ends of the segment in the other two axes, in increasing order,
 *         one per column, or nothing if the triangle misses the plane
 *
 * @pre @p axis is 0, 1 or 2
 */
template<std::floating_point T>
std::optional<Eigen::Matrix2<T>>
slice_triangle(const Eigen::Matrix3<T>& triangle,
               std::size_t axis, T coordinate)
{
    const std::size_t first = axis == 0 ? 1 : 0;
    const std::size_t second = axis == 2 ? 1 : 2;

    Eigen::Matrix2<T> result;
    int num_ends = 0;

    for (int corner = 0; corner < 3; ++corner) {
        // Two of the three edges of a triangle cross a plane
        const Eigen::Vector3<T> from = triangle.col(corner);
        const Eigen::Vector3<T> to = triangle.col((corner + 1) % 3);

        if ((from(axis) > coordinate) == (to(axis) > coordinate))
            continue;

        const T ratio = (coordinate - from(axis)) / (to(axis) - from(axis));
        const Eigen::Vector3<T> end = from + ratio * (to - from);

        result.col(num_ends) << end(first), end(second);
        ++num_ends;
    }

    if (num_ends == 0)
        // Return early if there is no slicing
        return std::nullopt;

    // Outward normal of the triangle, by the right-hand rule
    const Eigen::Vector3<T> edge_1 = triangle.col(1) - triangle.col(0);
    const Eigen::Vector3<T> edge_2 = triangle.col(2) - triangle.col(0);
    const Eigen::Vector3<T> triangle_normal = edge_1.cross(edge_2);

    // Outward normal of the segment, (dy, -dx)
    const Eigen::Vector2<T> edge = result.col(1) - result.col(0);
    const Eigen::Vector2<T> segment_normal(edge(1), -edge(0));

    // Both must point the same way within the plane
    const Eigen::Vector2<T> projected(triangle_normal(first),
                                      triangle_normal(second));

    if (segment_normal.dot(projected) < T{0})
        result.col(0).swap(result.col(1));

    return result;
}

} // namespace iguana

#endif // IGUANA_EMBEDDING_SLICER_HPP
