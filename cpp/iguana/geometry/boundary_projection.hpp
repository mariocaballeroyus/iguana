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

    /// @brief Weights of the parametric derivatives at the closest points
    ///        that shift the values at the points in place of the Taylor
    ///        series, empty for it, with one row per point and one column
    ///        per multi-index, the first direction running fastest
    Eigen::MatrixX<T> shift_weights;
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

/**
 * @brief Curvature of a boundary at closest points, and its derivatives
 *        along the boundary up to an order
 *
 *     τ' = −κ ν,    ν' = κ τ
 *
 * @details Derivatives are in the arc length along the tangent τ, the
 *          outward normal ν turned a quarter turn counterclockwise, so that
 *          κ is positive where the domain is convex, 1 / R on a disk of
 *          radius R. They are taken on the element of each closest point,
 *          one-sided at a knot
 *
 * @tparam T Floating-point type
 *
 * @param boundary Boundary of curves that the points are projected onto
 * @param projection Closest points on @p boundary
 * @param order Highest order of the derivatives
 * @return Curvature and its derivatives at each closest point, with size
 *         (num_points, order + 1), column j holding the derivative of order
 *         j
 *
 * @throws std::invalid_argument If @p order is negative
 *
 * @pre @p projection was computed onto @p boundary
 */
template<std::floating_point T>
Eigen::MatrixX<T> curvature_derivs(const Boundary<T, 2>& boundary,
                                   const BoundaryProjection<T>& projection,
                                   int order);

/**
 * @brief Points along a boundary at which data on it are interpolated, the
 *        images of the Greville abscissae of a data spline on each face,
 *        face after face
 *
 * @details The data spline of a face is a clamped B-spline of a degree in
 *          the parameter of its curve. Each element of the curve splits
 *          evenly into as many as it is long in smallest cell edges of the
 *          patch, at least one, so that the data resolve as finely as the
 *          background, and each knot of the curve is one of the data spline,
 *          which is no smoother there than the curve. The curve itself stays
 *          the exact NURBS
 *
 * @tparam Basis Basis type of the patch
 *
 * @param patch Patch with an affine map, whose cells size the data elements
 * @param boundary Boundary of curves in the physical space of the patch
 * @param degree Degree of the data splines
 * @return Points of size (num_points, 2), one per row
 *
 * @throws std::invalid_argument If @p degree is negative or if the map of
 *         @p patch is not affine
 */
template<typename Basis> requires (Basis::dimension == 2)
PointMatrix<typename Basis::Scalar, 2> boundary_interpolation_points(
    const Patch<Basis, 2>& patch,
    const Boundary<typename Basis::Scalar, 2>& boundary, int degree);

/**
 * @brief Derivatives along a boundary at closest points of the data that
 *        interpolate values at boundary_interpolation_points() in the data
 *        splines of its faces
 *
 *     g_h = Σ_i c_i M_i(u),    d / ds = |c'(u)|^(−1) d / du
 *
 * @details Derivatives are in the arc length along the tangent τ, as for
 *          curvature_derivs(), the exact ones of the data spline through
 *          the exact curve. They are taken on the element of the data
 *          spline holding each closest point, one-sided at a knot and where
 *          a closed face meets itself. Column j holds the derivative of
 *          order j
 *
 * @tparam Basis Basis type of the patch
 *
 * @param patch Patch with an affine map, whose cells size the data elements
 * @param boundary Boundary of curves in the physical space of the patch
 * @param degree Degree of the data splines
 * @param values Values of the data at boundary_interpolation_points(),
 *        face after face
 * @param projection Closest points on @p boundary
 * @param order Highest order of the derivatives
 * @return Derivatives at each closest point, with size
 *         (num_points, order + 1)
 *
 * @throws std::invalid_argument If @p order is negative or above
 *         @p degree, if the map of @p patch is not affine, or if @p values
 *         does not hold one value per interpolation point
 *
 * @pre @p projection was computed onto @p boundary
 */
template<typename Basis> requires (Basis::dimension == 2)
Eigen::MatrixX<typename Basis::Scalar> boundary_interpolated_derivs(
    const Patch<Basis, 2>& patch,
    const Boundary<typename Basis::Scalar, 2>& boundary, int degree,
    const Eigen::VectorX<typename Basis::Scalar>& values,
    const BoundaryProjection<typename Basis::Scalar>& projection, int order);

} // namespace iguana

#endif // IGUANA_GEOMETRY_BOUNDARY_PROJECTION_HPP
