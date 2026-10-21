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
 * @brief Projection of points onto their closest points on a boundary
 *
 * @details Used in the Shifted Boundary Method to shift the conditions
 *          from the true boundary to the surrogate one, along the distance
 *          vector from each point to its closest point
 *
 * @tparam T Floating-point type
 */
template<std::floating_point T>
struct BoundaryProjection
{
    /// @brief Face holding each closest point
    Eigen::VectorXi faces;

    /// @brief Element of the face holding each closest point, on which its
    ///        position and normal are evaluated, one-sided at a knot
    Eigen::VectorXi elements;

    /// @brief Parameter of the face at each closest point
    Eigen::VectorX<T> parameters;

    /// @brief Closest points in physical space, with size (num_points, 2)
    PointMatrix<T, 2> positions;

    /// @brief Unit normals of the faces at the closest points, in physical
    ///        space, pointing out of the domain, with size (num_points, 2)
    PointMatrix<T, 2> normals;

    /// @brief Distance vector from each point to its closest point, in the
    ///        parameter space of the patch holding the points, with size
    ///        (num_points, 2)
    Eigen::MatrixX<T> distances;
};

/**
 * @brief Closest points of a boundary of curves to points of the parameter
 *        space of a patch, with the distance vectors to them
 *
 * @details Each closest point is found on the exact NURBS curve of the 
 *          boundary, as a root of the derivative of the distance written
 *          in Bernstein form on each element
 *
 * @tparam Basis Basis type of the patch
 *
 * @param patch Patch whose parameter space holds the points
 * @param boundary Boundary of curves in the physical space of the patch
 * @param points Points in the parameter space of the patch, one per row
 * @return The closest point to each, with its face, element, parameter,
 *         normal and distance vector
 *
 * @throws std::invalid_argument If the map of the patch is not affine, if
 *         the boundary has no face, or if the knot vector of a face is not
 *         clamped
 */
template<typename Basis> requires (Basis::dimension == 2)
BoundaryProjection<typename Basis::Scalar> project_closest_points(
    const Patch<Basis, 2>& patch,
    const Boundary<typename Basis::Scalar, 2>& boundary,
    const PointMatrix<typename Basis::Scalar, 2>& points);

} // namespace iguana

#endif // IGUANA_GEOMETRY_BOUNDARY_PROJECTION_HPP
