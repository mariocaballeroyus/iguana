/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_EMBEDDING_NURBS_AXIS_CROSSINGS_HPP
#define IGUANA_EMBEDDING_NURBS_AXIS_CROSSINGS_HPP

#include <concepts>
#include <cstddef>
#include <optional>
#include <vector>

#include <Eigen/Core>

#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/geometry/patch.hpp"
#include "iguana/utils/bernstein.hpp"

namespace iguana
{

/**
 * @brief Points of (0, 1) at which an element of a NURBS curve crosses the
 *        axis-aligned line x_axis = coordinate
 *
 * The element is given by its homogeneous Bezier points (w b, w), the
 * extraction operator of the element applied to the weighted control
 * points, so that each column holds the Bernstein coefficients of one
 * homogeneous coordinate, X_i or W. As the weight is positive, x_axis -
 * coordinate has the sign of X_axis - coordinate W. The element lies in the
 * convex hull of its Bezier points, so only a line strictly inside their
 * range along the axis can be crossed
 *
 * @param points Homogeneous Bezier points of the element, one per row and
 *        the weight last, of size (p + 1, n + 1)
 * @param axis Axis the line fixes
 * @param coordinate Coordinate at which it fixes it
 * @param tolerance Width below which sign changes pair up, as in
 *        bernstein_crossings()
 * @return Crossings in increasing order
 *
 * @pre @p axis is below the number of columns of @p points minus one, the
 *      weights are positive and @p tolerance is positive
 */
template<std::floating_point T>
std::vector<T> curve_crossings(const Eigen::MatrixX<T>& points,
                               std::size_t axis, T coordinate, T tolerance)
{
    const Eigen::Index weight = points.cols() - 1;
    const Eigen::VectorX<T> coordinates =
        points.col(axis).cwiseQuotient(points.col(weight));

    if (coordinate <= coordinates.minCoeff()
        || coordinate >= coordinates.maxCoeff())
        return {};

    const Eigen::VectorX<T> shifted =
        points.col(axis) - coordinate * points.col(weight);

    return bernstein_crossings(shifted, tolerance);
}

/**
 * @brief Coordinate along an axis that the control points of an element of
 *        a NURBS curve share exactly, so that the element lies on the line
 *        x_axis = coordinate
 *
 * The test is exact, so a curve meant to lie on a line of a grid must have
 * the coordinate of the line at its control points
 *
 * @param curve NURBS curve
 * @param element Element of the curve
 * @param axis Axis along which the coordinates are compared
 * @return The shared coordinate, or nothing if the control points differ
 *         along the axis
 *
 * @pre @p element is an element of the curve and @p axis is below n
 */
template<std::floating_point T, std::size_t n>
std::optional<T> shared_coordinate(const Patch<TensorNURBS<T, 1>, n>& curve,
                                   int element, std::size_t axis)
{
    const BSpline<T>& bspline = curve.basis().bspline().axis(0);
    const Eigen::VectorX<T> coordinates =
        curve.coefficients().col(axis).segment(
            bspline.first_active(element), bspline.degree() + 1);

    if ((coordinates.array() != coordinates(0)).any())
        return std::nullopt;

    return coordinates(0);
}

} // namespace iguana

#endif // IGUANA_EMBEDDING_NURBS_AXIS_CROSSINGS_HPP
