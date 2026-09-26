/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "moment_fitting.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>

#include "iguana/embedding/clipper.hpp"
#include "iguana/embedding/slicer.hpp"
#include "iguana/quadrature/gauss_legendre.hpp"
#include "iguana/quadrature/xiao_gimbutas.hpp"
#include "iguana/utils/legendre.hpp"

namespace iguana
{

namespace
{

/// @brief Integrals of Q_{j_1} P_{j_2} ... P_{j_k} n_1 over boundary points,
///        whose weights carry n_1, with j_1 running fastest
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

        // Rows over j_1 and columns over the rest, so j_1 runs fastest
        const Eigen::MatrixX<T> result =
            antiderivatives * weights.asDiagonal() * products.transpose();

        return result.reshaped();
    }
}

/// @brief Adds the upper face: twice the section's moments, where j_1 = 0
template<std::floating_point T>
void add_upper_face(Eigen::VectorX<T>& moments,
                    const Eigen::VectorX<T>& section)
{
    Eigen::Map<Eigen::MatrixX<T>> matrix(moments.data(),
                                         moments.size() / section.size(),
                                         section.size());

    matrix.row(0) += 2 * section.transpose();
}

/// @brief Moments of the part of [-1, 1] bounded by signed points, +1 where
///        the domain ends and -1 where it starts
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

template<std::floating_point T, std::size_t d>
Eigen::VectorX<T>
reference_moments(const std::vector<Eigen::Matrix<T, d, d>>& facets,
                  int order)
{
    static_assert(d == 2 || d == 3, "reference_moments: "
                                    "the cell must have two or three "
                                    "directions");

    std::array<T, d> lower;
    std::array<T, d> upper;
    lower.fill(-1);
    upper.fill(1);

    // Parts inside the cell, as a vertex and edges, and the section at x_1 = 1
    std::vector<Eigen::Matrix<T, d, d>> pieces;
    std::vector<std::conditional_t<d == 2, std::pair<T, int>,
                                   Eigen::Matrix2<T>>>
        section;

    for (const Eigen::Matrix<T, d, d>& facet : facets) {
        if constexpr (d == 2) {
            if (const auto part = clip_segment<T>(facet, lower, upper)) {
                Eigen::Matrix2<T> piece;
                piece << part->col(0), part->col(1) - part->col(0);
                pieces.push_back(piece);
            }

            if (const auto point = slice_segment<T>(facet, 0, 1))
                section.push_back(*point);
        }
        else if constexpr (d == 3) {
            // The clipped polygon, fanned from its first vertex
            const auto polygon = clip_triangle<T>(facet, lower, upper);

            for (Eigen::Index vertex = 1; vertex + 1 < polygon.cols();
                 ++vertex) {
                Eigen::Matrix3<T> piece;
                piece << polygon.col(0), polygon.col(vertex) - polygon.col(0),
                    polygon.col(vertex + 1) - polygon.col(0);
                pieces.push_back(piece);
            }

            if (const auto segment = slice_triangle<T>(facet, 0, 1))
                section.push_back(*segment);
        }
    }

    // Simplex rule exact for the integrand, of degree d order + 1
    Eigen::MatrixX<T> rule_points;
    Eigen::VectorX<T> rule_weights;

    if constexpr (d == 2) {
        GaussLegendre<T, 1>(order + 1).map_to({0}, {1}, rule_points,
                                              rule_weights);
    }
    else if constexpr (d == 3) {
        const XiaoGimbutas<T> rule(3 * order + 1);
        rule_points = rule.points();
        rule_weights = rule.weights();
    }

    const Eigen::Index rule_size = rule_weights.size();
    Eigen::MatrixX<T> points(pieces.size() * rule_size, d);
    Eigen::VectorX<T> weights(points.rows());

    for (std::size_t piece = 0; piece < pieces.size(); ++piece) {
        const Eigen::Matrix<T, d, d>& vertex_and_edges = pieces[piece];
        const Eigen::Matrix<T, d, d - 1> edges =
            vertex_and_edges.template rightCols<d - 1>();
        const Eigen::Index offset = piece * rule_size;

        points.middleRows(offset, rule_size) =
            (rule_points * edges.transpose()).rowwise()
            + vertex_and_edges.col(0).transpose();

        // n_1 per reference measure, the edges' determinant across x_1
        const T normal = edges.template bottomRows<d - 1>().determinant();

        weights.segment(offset, rule_size) = normal * rule_weights;
    }

    Eigen::VectorX<T> result = integrate<T, d>(order, points, weights);

    if constexpr (d == 2)
        add_upper_face(result, point_moments(section, order));
    else if constexpr (d == 3)
        add_upper_face(result, reference_moments<T, 2>(section, order));

    return result;
}

template Eigen::VectorXd
reference_moments<double, 2>(const std::vector<Eigen::Matrix2d>&, int);
template Eigen::VectorXd
reference_moments<double, 3>(const std::vector<Eigen::Matrix3d>&, int);

} // namespace iguana
