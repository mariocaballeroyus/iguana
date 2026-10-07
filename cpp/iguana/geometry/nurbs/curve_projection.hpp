/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#ifndef IGUANA_GEOMETRY_NURBS_CURVE_PROJECTION_HPP
#define IGUANA_GEOMETRY_NURBS_CURVE_PROJECTION_HPP

#include <concepts>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "iguana/utils/bernstein.hpp"

namespace iguana
{

/**
 * @brief Point of an element of a NURBS curve closest to a target, by the
 *        sign changes of the derivative of their squared distance
 *
 * On the element the curve is X / W. Shifted to the target, X^ = X - x W,
 * which keeps the products small, half the squared distance has the
 * derivative N / W^3 with N = sum over i of (X^_i' W - X^_i W') X^_i, a
 * polynomial of degree 3p - 1 in Bernstein form, and W is positive. The
 * sign changes of N and the ends of the element are the candidates, and
 * the nearest of them is the closest point of the element, as a maximum of
 * the distance never is. Of candidates at one distance the first wins, so
 * that where N vanishes, the whole element lying at one distance, its
 * start is given
 *
 * @param points Homogeneous Bezier points of the element, one per row with
 *        the weight last, as given by bezier_points()
 * @param target Point to project, with one coordinate per column of
 *        @p points but the last
 * @param tolerance Width, relative to the element, below which sign changes
 *        pair up, as in bernstein_crossings()
 * @return Parameter of [0, 1] of the closest point, and its squared distance
 *         to @p target
 *
 * @pre The weights are positive and @p tolerance is positive
 */
template<std::floating_point T>
std::pair<T, T> closest_on_element(const Eigen::MatrixX<T>& points,
                                   const Eigen::RowVectorX<T>& target,
                                   T tolerance)
{
    const Eigen::Index num_coordinates = points.cols() - 1;
    const Eigen::VectorX<T> weight = points.col(num_coordinates);
    const Eigen::VectorX<T> weight_slope = bernstein_derivative(weight);

    // Homogeneous coordinates of the curve relative to the target
    const Eigen::MatrixX<T> shifted =
        points.leftCols(num_coordinates) - weight * target;

    // N, of degree 3p - 1
    Eigen::VectorX<T> stationary =
        Eigen::VectorX<T>::Zero(3 * (points.rows() - 1));

    for (Eigen::Index axis = 0; axis < num_coordinates; ++axis) {
        const Eigen::VectorX<T> coordinate = shifted.col(axis);

        // (X^_i' W - X^_i W') X^_i, the term of the coordinate in N
        const Eigen::VectorX<T> slope =
            bernstein_product(bernstein_derivative(coordinate), weight)
            - bernstein_product(coordinate, weight_slope);

        stationary += bernstein_product(slope, coordinate);
    }

    // Squared distance from the curve to the target at a parameter
    const auto squared_distance = [&](T t) {
        const T weight_at = de_casteljau(weight, t);
        T sum{0};

        for (Eigen::Index axis = 0; axis < num_coordinates; ++axis) {
            const T difference =
                de_casteljau<T>(shifted.col(axis), t) / weight_at;
            sum += difference * difference;
        }

        return sum;
    };

    std::vector<T> candidates{T{0}};
    const std::vector<T> crossings =
        bernstein_crossings(stationary, tolerance);
    candidates.insert(candidates.end(), crossings.begin(), crossings.end());
    candidates.push_back(T{1});

    std::pair<T, T> closest{T{0}, std::numeric_limits<T>::infinity()};

    for (const T candidate : candidates) {
        const T distance = squared_distance(candidate);

        if (distance < closest.second)
            closest = {candidate, distance};
    }

    return closest;
}

} // namespace iguana

#endif // IGUANA_GEOMETRY_NURBS_CURVE_PROJECTION_HPP
