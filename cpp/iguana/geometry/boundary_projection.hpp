/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GEOMETRY_BOUNDARY_PROJECTION_HPP
#define IGUANA_GEOMETRY_BOUNDARY_PROJECTION_HPP

#include <concepts>

#include <Eigen/Core>

#include "iguana/geometry/boundary.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Closest points of a boundary to some points, one row or entry per
 *        point
 *
 * @tparam T Floating-point type
 */
template<std::floating_point T>
struct BoundaryProjection
{
    /// @brief Face holding each closest point
    Eigen::VectorXi faces;

    /// @brief Parameter of the face at each closest point
    Eigen::VectorX<T> parameters;

    /// @brief Closest points, with size (num_points, 2)
    PointMatrix<T, 2> positions;

    /// @brief Unit normals of the faces at the closest points, pointing out
    ///        of the domain, with size (num_points, 2)
    PointMatrix<T, 2> normals;
};

/**
 * @brief Closest points of a boundary of curves to points of the plane
 *
 * Every element of every face is searched by the sign changes of the
 * derivative of the distance in Bernstein form, as closest_on_element()
 * does, so that the closest point is the global one, found with no initial
 * guess. Of points at one distance, the first face and element win, so that
 * at a vertex shared by two faces the normal is that of the first. A point
 * farther from a face than its radius of curvature can have several closest
 * points, of which one is given
 *
 * @param boundary Boundary of curves in the plane
 * @param points Points in physical space, one per row
 * @return The closest point to each, with its face, parameter and normal
 *
 * @throws std::invalid_argument If the boundary has no face, or if the knot
 *         vector of a face is not clamped
 */
template<std::floating_point T>
BoundaryProjection<T> project_points(const Boundary<T, 2>& boundary,
                                     const PointMatrix<T, 2>& points);

} // namespace iguana

#endif // IGUANA_GEOMETRY_BOUNDARY_PROJECTION_HPP
