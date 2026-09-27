/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "gauss_legendre.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "iguana/utils/multi_index.hpp"

namespace iguana
{

namespace
{

/**
 * @brief Tabulated rule on [-1, 1] with a number of points
 *
 * @return Nodes and weights, the doubles nearest to their exact values
 *
 * @throws std::invalid_argument If @p num_points lies outside
 *         [1, GaussLegendre::max_points]
 */
template<std::floating_point T>
std::array<std::vector<T>, 2> tabulated_rule(int num_points)
{
    switch (num_points) {
    case 1: {
        // Gauss-Legendre, 1 point
        std::vector<T> x = {0.0};
        std::vector<T> w = {2.0};
        return {std::move(x), std::move(w)};
    }
    case 2: {
        // Gauss-Legendre, 2 points
        std::vector<T> x = {-0.5773502691896257, 0.5773502691896257};
        std::vector<T> w = {1.0, 1.0};
        return {std::move(x), std::move(w)};
    }
    case 3: {
        // Gauss-Legendre, 3 points
        std::vector<T> x = {-0.7745966692414834, 0.0, 0.7745966692414834};
        std::vector<T> w
            = {0.5555555555555556, 0.8888888888888888, 0.5555555555555556};
        return {std::move(x), std::move(w)};
    }
    case 4: {
        // Gauss-Legendre, 4 points
        std::vector<T> x
            = {-0.8611363115940526, -0.33998104358485626, 0.33998104358485626,
               0.8611363115940526};
        std::vector<T> w
            = {0.34785484513745385, 0.6521451548625461, 0.6521451548625461,
               0.34785484513745385};
        return {std::move(x), std::move(w)};
    }
    case 5: {
        // Gauss-Legendre, 5 points
        std::vector<T> x
            = {-0.906179845938664, -0.5384693101056831, 0.0,
               0.5384693101056831, 0.906179845938664};
        std::vector<T> w
            = {0.23692688505618908, 0.47862867049936647, 0.5688888888888889,
               0.47862867049936647, 0.23692688505618908};
        return {std::move(x), std::move(w)};
    }
    case 6: {
        // Gauss-Legendre, 6 points
        std::vector<T> x
            = {-0.932469514203152, -0.6612093864662645, -0.2386191860831969,
               0.2386191860831969, 0.6612093864662645,  0.932469514203152};
        std::vector<T> w
            = {0.17132449237917036, 0.3607615730481386, 0.46791393457269104,
               0.46791393457269104, 0.3607615730481386, 0.17132449237917036};
        return {std::move(x), std::move(w)};
    }
    case 7: {
        // Gauss-Legendre, 7 points
        std::vector<T> x
            = {-0.9491079123427585, -0.7415311855993945, -0.4058451513773972,
               0.0,                 0.4058451513773972,  0.7415311855993945,
               0.9491079123427585};
        std::vector<T> w
            = {0.1294849661688697, 0.27970539148927664, 0.3818300505051189,
               0.4179591836734694, 0.3818300505051189,  0.27970539148927664,
               0.1294849661688697};
        return {std::move(x), std::move(w)};
    }
    case 8: {
        // Gauss-Legendre, 8 points
        std::vector<T> x
            = {-0.9602898564975363, -0.7966664774136267, -0.525532409916329,
               -0.1834346424956498, 0.1834346424956498,  0.525532409916329,
               0.7966664774136267,  0.9602898564975363};
        std::vector<T> w
            = {0.10122853629037626, 0.22238103445337448, 0.31370664587788727,
               0.362683783378362,   0.362683783378362,   0.31370664587788727,
               0.22238103445337448, 0.10122853629037626};
        return {std::move(x), std::move(w)};
    }
    default:
        throw std::invalid_argument("GaussLegendre: "
                                    "the number of points per direction "
                                    "must lie in [1, max_points]");
    }
}

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
        rules[direction] = tabulated_rule<T>(num_points[direction]);
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
