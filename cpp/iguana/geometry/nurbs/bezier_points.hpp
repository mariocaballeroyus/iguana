/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GEOMETRY_NURBS_BEZIER_POINTS_HPP
#define IGUANA_GEOMETRY_NURBS_BEZIER_POINTS_HPP

#include <concepts>
#include <cstddef>

#include <Eigen/Core>

#include "iguana/basis/tensor_nurbs.hpp"
#include "iguana/geometry/patch.hpp"

namespace iguana
{

/**
 * @brief Homogeneous Bezier points (w b, w) of an element of a NURBS curve
 *
 * The extraction operator of the element turns the weighted control points
 * of its active functions into Bezier points, so that each column holds the
 * Bernstein coefficients of one homogeneous coordinate on the element,
 * X_i or W, and the curve is X / W there
 *
 * @param curve NURBS curve
 * @param element Element of the curve
 * @param extraction Bezier extraction operator of the element, as given by
 *        extraction_operators()
 * @return Bezier points, one per row with the weight last, of size
 *         (p + 1, n + 1)
 *
 * @pre @p element is an element of the curve and @p extraction its operator
 */
template<std::floating_point T, std::size_t n>
Eigen::MatrixX<T> bezier_points(const Patch<TensorNURBS<T, 1>, n>& curve,
                                int element,
                                const Eigen::MatrixX<T>& extraction)
{
    const int first = curve.basis().bspline().axis(0).first_active(element);
    const int count = static_cast<int>(extraction.rows());

    const Eigen::VectorX<T> weights =
        curve.basis().weights().segment(first, count);

    Eigen::MatrixX<T> homogeneous(count, n + 1);
    homogeneous << weights.asDiagonal()
                       * curve.coefficients().middleRows(first, count),
        weights;

    return extraction.transpose() * homogeneous;
}

} // namespace iguana

#endif // IGUANA_GEOMETRY_NURBS_BEZIER_POINTS_HPP
