/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "moment_fitting.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/NNLS>

#include "iguana/quadrature/moments.hpp"
#include "iguana/utils/legendre.hpp"
#include "iguana/utils/multi_index.hpp"

namespace iguana
{

namespace
{

/// @brief Smallest part inside the domain, relative to the cell, with points
constexpr double min_measure_ratio = 1e-3;

/// @brief Finest level of candidate boxes, 2^level per direction
constexpr int max_level = 2;

/// @brief Relative residual below which a fit is accepted
constexpr double target_residual = 1e-10;

/// @brief Relative residual above which a cell gets no points
constexpr double max_residual = 1e-2;

/**
 * @brief Highest Legendre degree, checked
 *
 * @throws std::invalid_argument If @p order lies outside [0, max_order]
 */
int checked(int order, int max_order)
{
    if (order < 0 || order > max_order)
        throw std::invalid_argument("MomentFitting: "
                                    "the order must lie in [0, max_order]");

    return order;
}

/// @brief Points of a rule on each of 2^level boxes per direction of
///        [-1, 1]^d, one per row
template<std::floating_point T, std::size_t d>
Eigen::MatrixX<T> box_points(const GaussLegendre<T, d>& rule, int level)
{
    const int num_boxes = 1 << level;
    const T size = T{2} / num_boxes;

    std::array<int, d> box{};
    std::array<int, d> bounds{};
    bounds.fill(num_boxes);

    int total = rule.num_points();

    for (std::size_t axis = 0; axis < d; ++axis)
        total *= num_boxes;

    Eigen::MatrixX<T> result(total, d);

    std::array<T, d> lower;
    std::array<T, d> upper;
    Eigen::MatrixX<T> points;
    Eigen::VectorX<T> weights;
    Eigen::Index offset = 0;

    do {
        for (std::size_t axis = 0; axis < d; ++axis) {
            lower[axis] = -1 + box[axis] * size;
            upper[axis] = lower[axis] + size;
        }

        rule.map_to(lower, upper, points, weights);
        result.middleRows(offset, points.rows()) = points;
        offset += points.rows();
    } while (next_lexicographic(box, bounds));

    return result;
}

/// @brief Non-negative weights of candidate points that reproduce moments,
///        returning the relative residual
template<std::floating_point T, std::size_t d>
T fit(int order, const Eigen::MatrixX<T>& candidates,
      const Eigen::VectorX<T>& moments, Eigen::VectorX<T>& weights)
{
    // Each Legendre product at each candidate, one column per candidate
    Eigen::MatrixX<T> system;
    tensor_legendre_polynomials<T, d>(order, candidates, system);

    Eigen::NNLS<Eigen::MatrixX<T>> solver(system);
    weights = solver.solve(moments);

    return (system * weights - moments).norm() / moments.norm();
}

} // namespace

template<std::floating_point T, std::size_t d>
MomentFitting<T, d>::MomentFitting(const Eigen::MatrixX<T>& vertices,
                                   const Eigen::MatrixXi& facets, int order)
    : order_(checked(order, max_order)),
      candidate_rule_(order_ + 1)
{
    if (vertices.cols() != static_cast<Eigen::Index>(d)
        || facets.cols() != static_cast<Eigen::Index>(d))
        throw std::invalid_argument("MomentFitting: "
                                    "the vertices and facets must have d "
                                    "columns");

    if (facets.size() > 0
        && (facets.minCoeff() < 0 || facets.maxCoeff() >= vertices.rows()))
        throw std::invalid_argument("MomentFitting: "
                                    "the facets must refer to existing "
                                    "vertices");

    facets_.resize(facets.rows());

    for (Eigen::Index facet = 0; facet < facets.rows(); ++facet)
        for (std::size_t corner = 0; corner < d; ++corner)
            facets_[facet].col(corner) =
                vertices.row(facets(facet, corner)).transpose();
}

template<std::floating_point T, std::size_t d>
void MomentFitting<T, d>::reference_rule(const std::array<T, d>& start,
                                         const std::array<T, d>& end,
                                         Eigen::MatrixX<T>& points,
                                         Eigen::VectorX<T>& weights) const
{
    points.resize(0, d);
    weights.resize(0);

    // The facets on the reference cell
    std::vector<Eigen::Matrix<T, d, d>> facets(facets_.size());

    for (std::size_t facet = 0; facet < facets_.size(); ++facet)
        for (std::size_t axis = 0; axis < d; ++axis)
            facets[facet].row(axis) =
                (2 * facets_[facet].row(axis).array() - start[axis]
                 - end[axis])
                / (end[axis] - start[axis]);

    const Eigen::VectorX<T> moments = reference_moments<T, d>(facets, order_);

    // The first moment is the measure of the part inside, out of 2^d
    if (moments(0) < min_measure_ratio * static_cast<T>(1 << d))
        return;

    Eigen::MatrixX<T> candidates;
    Eigen::VectorX<T> fitted;
    T residual{1};

    for (int level = 1; level <= max_level && residual > target_residual;
         ++level) {
        candidates = box_points(candidate_rule_, level);
        residual = fit<T, d>(order_, candidates, moments, fitted);
    }

    if (residual > max_residual)
        return;

    // The candidates of positive weight
    const Eigen::Index num_kept = (fitted.array() > T{0}).count();
    points.resize(num_kept, d);
    weights.resize(num_kept);

    Eigen::Index kept = 0;

    for (Eigen::Index candidate = 0; candidate < candidates.rows();
         ++candidate) {
        if (fitted(candidate) <= T{0})
            continue;

        points.row(kept) = candidates.row(candidate);
        weights(kept) = fitted(candidate);
        ++kept;
    }
}

template class MomentFitting<double, 2>;
template class MomentFitting<double, 3>;

} // namespace iguana
