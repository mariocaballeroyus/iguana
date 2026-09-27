/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "gauss_legendre.hpp"

#include <array>
#include <cstddef>
#include <vector>

#include "iguana/utils/multi_index.hpp"

namespace iguana
{

namespace
{

/// @brief Same count in every direction
template<std::size_t d>
std::array<int, d> uniform(int num_points)
{
    std::array<int, d> counts{};
    counts.fill(num_points);

    return counts;
}

} // namespace

template<std::floating_point T, std::size_t d>
GaussLegendre<T, d>::GaussLegendre(int num_points)
    : GaussLegendre(uniform<d>(num_points))
{
}

template<std::floating_point T, std::size_t d>
GaussLegendre<T, d>::GaussLegendre(const std::array<int, d>& num_points)
{
    // Nodes and weights of each direction, which checks its count
    std::array<std::array<std::vector<T>, 2>, d> rules;
    int total = 1;

    for (std::size_t direction = 0; direction < d; ++direction) {
        rules[direction] = tabulated_rule(num_points[direction]);
        total *= num_points[direction];
    }

    points_.resize(total, d);
    weights_.resize(total);

    // Product of the univariate rules, first direction running fastest
    std::array<int, d> index{};
    Eigen::Index point = 0;

    do {
        T weight{1};

        for (std::size_t direction = 0; direction < d; ++direction) {
            const int node = index[direction];

            const auto& [nodes, weights] = rules[direction];

            points_(point, direction) = nodes[node];
            weight *= weights[node];
        }

        weights_[point] = weight;
        ++point;
    } while (next_lexicographic(index, num_points));
}

template<std::floating_point T, std::size_t d>
void GaussLegendre<T, d>::reference_rule(const std::array<T, d>&,
                                         const std::array<T, d>&,
                                         Eigen::MatrixX<T>& points,
                                         Eigen::VectorX<T>& weights) const
{
    points = points_;
    weights = weights_;
}

template<std::floating_point T, std::size_t d>
void GaussLegendre<T, d>::map_to(const std::array<T, d>& start,
                                 const std::array<T, d>& end,
                                 Eigen::MatrixX<T>& points,
                                 Eigen::VectorX<T>& weights) const
{
    map_to_cell(start, end, points_, weights_, points, weights);
}

template class GaussLegendre<double, 1>;
template class GaussLegendre<double, 2>;
template class GaussLegendre<double, 3>;

} // namespace iguana
