/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "moment_fitting.hpp"

#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "iguana/embedding/clipper.hpp"
#include "iguana/embedding/slicer.hpp"
#include "iguana/quadrature/gauss_legendre.hpp"
#include "iguana/utils/legendre.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Integrals of Q_{j_1} P_{j_2} ... P_{j_k} n_1 over pieces of a
 *        boundary in [-1, 1]^k, from their quadrature points
 *
 * @param order Highest Legendre degree of each direction
 * @param points Quadrature points on the pieces, one per row
 * @param weights Their weights, carrying n_1 and the measure of the pieces
 * @return Integrals, with j_1 running fastest
 */
template<std::floating_point T, std::size_t k>
Eigen::VectorX<T> integrate(int order, const Eigen::MatrixX<T>& points,
                            const Eigen::VectorX<T>& weights)
{
    const std::span<const T> first(points.col(0).data(), points.rows());

    Eigen::MatrixX<T> antiderivatives;
    legendre_antiderivatives<T>(order, first, antiderivatives);

    if constexpr (k == 1) {
        return antiderivatives * weights;
    }
    else {
        Eigen::MatrixX<T> products;
        tensor_legendre_polynomials<T, k - 1>(order, points.rightCols(k - 1),
                                              products);

        // One row per degree of the first direction and one column per
        // product of the others, whose column-major order runs j_1 fastest
        const Eigen::MatrixX<T> result =
            antiderivatives * weights.asDiagonal() * products.transpose();

        return result.reshaped();
    }
}

/// @brief Adds the upper face x_1 = 1 to the moments of a cell: twice the
///        moments of the section there, to those with j_1 = 0
template<std::floating_point T>
void add_upper_face(Eigen::VectorX<T>& moments,
                    const Eigen::VectorX<T>& section)
{
    Eigen::Map<Eigen::MatrixX<T>> matrix(moments.data(),
                                         moments.size() / section.size(),
                                         section.size());

    matrix.row(0) += 2 * section.transpose();
}

/**
 * @brief Moments of the part of [-1, 1] inside a domain bounded by points
 *
 * @param points Points bounding the domain, each with +1 where the domain
 *        ends and -1 where it starts
 * @param order Highest Legendre degree
 * @return Moments, with size order + 1
 */
template<std::floating_point T>
Eigen::VectorX<T> point_moments(const std::vector<std::pair<T, int>>& points,
                                int order)
{
    std::vector<T> inside;
    std::vector<T> signs;

    // Entries minus exits up to the upper end, positive if it is inside
    int depth = 0;

    for (const auto& [coordinate, sign] : points) {
        if (coordinate > 1)
            continue;

        depth -= sign;

        if (coordinate >= -1) {
            inside.push_back(coordinate);
            signs.push_back(sign);
        }
    }

    const Eigen::Index num_inside = inside.size();
    Eigen::VectorX<T> result = integrate<T, 1>(
        order,
        Eigen::Map<const Eigen::MatrixX<T>>(inside.data(), num_inside, 1),
        Eigen::Map<const Eigen::VectorX<T>>(signs.data(), num_inside));

    if (depth > 0)
        add_upper_face(result, Eigen::VectorX<T>::Ones(1).eval());

    return result;
}

} // namespace

template<std::floating_point T>
Eigen::VectorX<T>
reference_moments(const std::vector<Eigen::Matrix2<T>>& segments, int order)
{
    // Parts of the segments inside the square, and the section at u = 1
    std::vector<Eigen::Matrix2<T>> pieces;
    std::vector<std::pair<T, int>> section;

    for (const Eigen::Matrix2<T>& segment : segments) {
        if (const auto piece = clip_segment<T>(segment, {-1, -1}, {1, 1}))
            pieces.push_back(*piece);

        if (const auto point = slice_segment<T>(segment, 0, 1))
            section.push_back(*point);
    }

    // Gauss rule on [0, 1], exact to degree 2 order + 1, that of the
    // integrand along a segment
    Eigen::MatrixX<T> nodes;
    Eigen::VectorX<T> node_weights;
    GaussLegendre<T, 1>(order + 1).map_to({0}, {1}, nodes, node_weights);

    const Eigen::Index rule_size = node_weights.size();
    Eigen::MatrixX<T> points(pieces.size() * rule_size, 2);
    Eigen::VectorX<T> weights(points.rows());

    for (std::size_t piece = 0; piece < pieces.size(); ++piece) {
        const Eigen::Vector2<T> edge = pieces[piece].col(1)
                                       - pieces[piece].col(0);
        const Eigen::Index offset = piece * rule_size;

        points.middleRows(offset, rule_size) =
            (nodes * edge.transpose()).rowwise()
            + pieces[piece].col(0).transpose();

        // n_1 ds is the change of the second coordinate along the segment
        weights.segment(offset, rule_size) = edge(1) * node_weights;
    }

    Eigen::VectorX<T> result = integrate<T, 2>(order, points, weights);
    add_upper_face(result, point_moments(section, order));

    return result;
}

template Eigen::VectorXd reference_moments(const std::vector<Eigen::Matrix2d>&,
                                           int);

} // namespace iguana
