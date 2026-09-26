/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_CLIPPER_HPP
#define IGUANA_EMBEDDING_CLIPPER_HPP

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <optional>

#include <Eigen/Core>

namespace iguana
{

/**
 * @brief Part of a segment of a two-dimensional parameter space inside the
 *        closed box [lower, upper]
 *
 * @param segment Ends of the segment, one per column
 * @param lower Lower corner of the box
 * @param upper Upper corner of the box
 * @return Ends of the part, one per column, in the order of the segment,
 *         or nothing if the segment misses the box
 */
template<std::floating_point T>
std::optional<Eigen::Matrix2<T>>
clip_segment(const Eigen::Matrix2<T>& segment,
             const std::array<T, 2>& lower, const std::array<T, 2>& upper)
{
    const Eigen::Vector2<T> edge = segment.col(1) - segment.col(0);

    // Range of t in [0, 1] whose points segment.col(0) + t * edge lie in
    // the box, narrowed axis by axis
    T enter{0};
    T leave{1};

    for (std::size_t axis = 0; axis < 2; ++axis) {
        const T origin = segment(axis, 0);
        const T step = edge(axis);

        if (step == T{0}) {
            // Parallel to the sides of this axis, inside or not at all
            if (origin < lower[axis] || origin > upper[axis])
                return std::nullopt;

            continue;
        }

        const T to_lower = (lower[axis] - origin) / step;
        const T to_upper = (upper[axis] - origin) / step;

        enter = std::max(enter, std::min(to_lower, to_upper));
        leave = std::min(leave, std::max(to_lower, to_upper));
    }

    if (enter >= leave)
        // Return early if the segment misses the box or only touches it
        return std::nullopt;

    Eigen::Matrix2<T> result;
    result.col(0) = segment.col(0) + enter * edge;
    result.col(1) = segment.col(0) + leave * edge;

    return result;
}

/**
 * @brief Part of a triangle of a three-dimensional parameter space inside
 *        the closed box [lower, upper]
 *
 * The part is a convex polygon, cut from the triangle by the six planes of
 * the box in turn. Each cut adds at most one vertex, so it has at most nine
 *
 * @param triangle Vertices of the triangle, one per column
 * @param lower Lower corner of the box
 * @param upper Upper corner of the box
 * @return Vertices of the polygon, one per column, in the order of the
 *         triangle, or none if the triangle misses the box
 */
template<std::floating_point T>
Eigen::Matrix<T, 3, Eigen::Dynamic, 0, 3, 9>
clip_triangle(const Eigen::Matrix3<T>& triangle,
              const std::array<T, 3>& lower, const std::array<T, 3>& upper)
{
    // At most nine columns, which keeps the polygon off the heap
    using Polygon = Eigen::Matrix<T, 3, Eigen::Dynamic, 0, 3, 9>;

    Polygon polygon = triangle;

    for (std::size_t axis = 0; axis < 3; ++axis) {
        // The lower plane keeps x >= lower and the upper one x <= upper
        for (const T sense : {T{1}, T{-1}}) {
            if (polygon.cols() == 0)
                // Return early if the triangle misses the box
                return polygon;

            const T bound = sense > T{0} ? lower[axis] : upper[axis];
            const Polygon previous = polygon;
            polygon.resize(3, 0);

            const auto keep = [&polygon](const Eigen::Vector3<T>& point) {
                polygon.conservativeResize(Eigen::NoChange, polygon.cols() + 1);
                polygon.col(polygon.cols() - 1) = point;
            };

            for (Eigen::Index vertex = 0; vertex < previous.cols(); ++vertex) {
                const Eigen::Vector3<T> from = previous.col(vertex);
                const Eigen::Vector3<T> to =
                    previous.col((vertex + 1) % previous.cols());

                // Signed distances to the plane, positive on the box side
                const T from_distance = sense * (from(axis) - bound);
                const T to_distance = sense * (to(axis) - bound);

                if (from_distance >= T{0})
                    keep(from);

                if ((from_distance >= T{0}) != (to_distance >= T{0}))
                    keep(from + from_distance / (from_distance - to_distance)
                                    * (to - from));
            }
        }
    }

    return polygon;
}

} // namespace iguana

#endif // IGUANA_EMBEDDING_CLIPPER_HPP
