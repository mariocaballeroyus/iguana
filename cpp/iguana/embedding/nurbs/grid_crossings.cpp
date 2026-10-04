/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "grid_crossings.hpp"

#include <algorithm>
#include <cstddef>
#include <optional>

#include "iguana/embedding/nurbs/axis_crossings.hpp"
#include "iguana/utils/bernstein.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

namespace
{

/// @brief Point of an element of a curve at a parameter of [0, 1], from its
///        homogeneous Bezier points
template<std::floating_point T>
Eigen::Vector2<T> position_at(const Eigen::MatrixX<T>& points, T t)
{
    const T weight = de_casteljau<T>(points.col(2), t);

    return {de_casteljau<T>(points.col(0), t) / weight,
            de_casteljau<T>(points.col(1), t) / weight};
}

/**
 * @brief Parameters of [0, 1] at which an element of a curve splits, its
 *        ends and its crossings with the lines of a grid
 *
 * Only lines strictly inside the range of the Bezier points can be
 * crossed, and none along an axis on which the element lies. Crossings of
 * both axes at a vertex, or at an end of the element, fall a few roundings
 * apart, so those closer than the tolerance merge
 */
template<std::floating_point T>
std::vector<T> splits_of(const Eigen::MatrixX<T>& points,
                         const std::array<std::optional<T>, 2>& shared,
                         const std::array<std::vector<T>, 2>& lines,
                         T tolerance)
{
    std::vector<T> crossings;

    for (std::size_t axis = 0; axis < 2; ++axis) {
        if (shared[axis])
            continue;

        const Eigen::VectorX<T> coordinates =
            points.col(axis).cwiseQuotient(points.col(2));
        const auto lower =
            std::ranges::upper_bound(lines[axis], coordinates.minCoeff());
        const auto upper =
            std::ranges::lower_bound(lines[axis], coordinates.maxCoeff());

        for (auto line = lower; line < upper; ++line) {
            const std::vector<T> found =
                curve_crossings(points, axis, *line, tolerance);

            crossings.insert(crossings.end(), found.begin(), found.end());
        }
    }

    std::ranges::sort(crossings);

    std::vector<T> splits{T{0}};

    for (const T t : crossings) {
        if (t - splits.back() > tolerance && t < 1 - tolerance)
            splits.push_back(t);
    }

    splits.push_back(T{1});

    return splits;
}

/**
 * @brief Cell of a grid holding a piece of an element of a curve, or -1
 *        outside the grid
 *
 * Off a line, X - c W is a polynomial of degree p, so the piece meets the
 * lines around it at p points at most, each touch a double root, and the
 * mean of p + 1 distinct points inside it lies strictly inside its cell. A
 * piece lying on a line belongs to the cell opposite its normal sign
 * (y', -x'), whose component across the line follows from the
 * displacement along it
 *
 * @param points Homogeneous Bezier points of the element
 * @param start Parameter of [0, 1] at which the piece starts
 * @param end Parameter of [0, 1] at which the piece ends
 * @param sign Sign of the curve, which orients its normal
 * @param held Line on which the element lies along each axis, or -1
 * @param lines Coordinates of the lines of the grid along each axis
 */
template<std::floating_point T>
int holder(const Eigen::MatrixX<T>& points, T start, T end, int sign,
           const std::array<int, 2>& held,
           const std::array<std::vector<T>, 2>& lines)
{
    const Eigen::Index num_samples = points.rows();

    Eigen::Vector2<T> mean = Eigen::Vector2<T>::Zero();

    for (Eigen::Index sample = 1; sample <= num_samples; ++sample)
        mean += position_at<T>(
            points, start + (end - start) * static_cast<T>(sample)
                                / static_cast<T>(num_samples + 1));

    mean /= static_cast<T>(num_samples);

    const Eigen::Vector2<T> displacement =
        position_at<T>(points, end) - position_at<T>(points, start);

    std::array<int, 2> index{};
    std::array<int, 2> counts{};

    for (std::size_t axis = 0; axis < 2; ++axis) {
        const std::vector<T>& values = lines[axis];
        counts[axis] = static_cast<int>(values.size()) - 1;

        if (held[axis] >= 0) {
            // n_0 goes with the rise along the line and n_1 against the run
            const T across = axis == 0 ? displacement(1) : -displacement(0);

            index[axis] = static_cast<T>(sign) * across > T{0}
                              ? held[axis] - 1
                              : held[axis];
        } else {
            const T coordinate = mean(axis);

            if (coordinate < values.front() || coordinate > values.back())
                return -1;

            const auto after = std::ranges::upper_bound(values, coordinate);
            index[axis] = std::min(static_cast<int>(after - values.begin()) - 1,
                                   counts[axis] - 1);
        }

        if (index[axis] < 0 || index[axis] >= counts[axis])
            return -1;
    }

    return flatten(index, counts);
}

} // namespace

template<std::floating_point T>
std::vector<CurvePiece<T>> divide_curve(
    const Patch<TensorNURBS<T, 1>, 2>& curve, int element,
    const Eigen::MatrixX<T>& extraction,
    const std::array<std::vector<T>, 2>& lines, int sign, T tolerance)
{
    const BSpline<T>& bspline = curve.basis().bspline().axis(0);
    const int first = bspline.first_active(element);
    const int count = bspline.degree() + 1;

    const PointMatrix<T, 2> local =
        curve.coefficients().middleRows(first, count);

    // An element whose control points coincide has no length
    if ((local.rowwise() - local.row(0)).cwiseAbs().maxCoeff() == T{0})
        return {};

    // Homogeneous Bezier points (w b, w), one per row
    const Eigen::VectorX<T> weights =
        curve.basis().weights().segment(first, count);

    Eigen::MatrixX<T> homogeneous(count, 3);
    homogeneous << weights.asDiagonal() * local, weights;

    const Eigen::MatrixX<T> points = extraction.transpose() * homogeneous;

    // Coordinate the control points share along each axis, and the line
    // of the grid at it, or -1
    std::array<std::optional<T>, 2> shared;
    std::array<int, 2> held{-1, -1};

    for (std::size_t axis = 0; axis < 2; ++axis) {
        shared[axis] = shared_coordinate(curve, element, axis);

        if (!shared[axis])
            continue;

        const auto line = std::ranges::lower_bound(lines[axis], *shared[axis]);

        if (line != lines[axis].end() && *line == *shared[axis])
            held[axis] = static_cast<int>(line - lines[axis].begin());
    }

    const std::vector<T> splits = splits_of(points, shared, lines, tolerance);
    std::vector<CurvePiece<T>> pieces;

    for (std::size_t split = 0; split + 1 < splits.size(); ++split) {
        const T start = splits[split];
        const T end = splits[split + 1];
        const int cell = holder(points, start, end, sign, held, lines);

        if (cell >= 0)
            pieces.push_back({start, end, cell});
    }

    return pieces;
}

template std::vector<CurvePiece<double>> divide_curve(
    const Patch<TensorNURBS<double, 1>, 2>&, int, const Eigen::MatrixXd&,
    const std::array<std::vector<double>, 2>&, int, double);

} // namespace iguana
