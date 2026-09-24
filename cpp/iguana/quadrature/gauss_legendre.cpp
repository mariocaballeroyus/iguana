/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include "gauss_legendre.hpp"

#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>

#include "iguana/multi_index.hpp"

namespace iguana
{

namespace
{

// Nodes and weights on [-1, 1], tabulated to 30 digits
constexpr double nodes_1[] = {
    0.000000000000000000000000000000};
constexpr double weights_1[] = {
    2.000000000000000000000000000000};

constexpr double nodes_2[] = {
    -0.577350269189625764509148780502, 0.577350269189625764509148780502};
constexpr double weights_2[] = {
    1.000000000000000000000000000000, 1.000000000000000000000000000000};

constexpr double nodes_3[] = {
    -0.774596669241483377035853079956, 0.000000000000000000000000000000,
    0.774596669241483377035853079956};
constexpr double weights_3[] = {
    0.555555555555555555555555555556, 0.888888888888888888888888888889,
    0.555555555555555555555555555556};

constexpr double nodes_4[] = {
    -0.861136311594052575223946488893, -0.339981043584856264802665759103,
    0.339981043584856264802665759103, 0.861136311594052575223946488893};
constexpr double weights_4[] = {
    0.347854845137453857373063949222, 0.652145154862546142626936050778,
    0.652145154862546142626936050778, 0.347854845137453857373063949222};

constexpr double nodes_5[] = {
    -0.906179845938663992797626878299, -0.538469310105683091036314420700,
    0.000000000000000000000000000000, 0.538469310105683091036314420700,
    0.906179845938663992797626878299};
constexpr double weights_5[] = {
    0.236926885056189087514264040720, 0.478628670499366468041291514836,
    0.568888888888888888888888888889, 0.478628670499366468041291514836,
    0.236926885056189087514264040720};

constexpr double nodes_6[] = {
    -0.932469514203152027812301554494, -0.661209386466264513661399595020,
    -0.238619186083196908630501721681, 0.238619186083196908630501721681,
    0.661209386466264513661399595020, 0.932469514203152027812301554494};
constexpr double weights_6[] = {
    0.171324492379170345040296142173, 0.360761573048138607569833513838,
    0.467913934572691047389870343990, 0.467913934572691047389870343990,
    0.360761573048138607569833513838, 0.171324492379170345040296142173};

constexpr double nodes_7[] = {
    -0.949107912342758524526189684048, -0.741531185599394439863864773281,
    -0.405845151377397166906606412077, 0.000000000000000000000000000000,
    0.405845151377397166906606412077, 0.741531185599394439863864773281,
    0.949107912342758524526189684048};
constexpr double weights_7[] = {
    0.129484966168869693270611432679, 0.279705391489276667901467771424,
    0.381830050505118944950369775489, 0.417959183673469387755102040816,
    0.381830050505118944950369775489, 0.279705391489276667901467771424,
    0.129484966168869693270611432679};

constexpr double nodes_8[] = {
    -0.960289856497536231683560868569, -0.796666477413626739591553936476,
    -0.525532409916328985817739049189, -0.183434642495649804939476142360,
    0.183434642495649804939476142360, 0.525532409916328985817739049189,
    0.796666477413626739591553936476, 0.960289856497536231683560868569};
constexpr double weights_8[] = {
    0.101228536290376259152531354310, 0.222381034453374470544355994426,
    0.313706645877887287337962201987, 0.362683783378361982965150449277,
    0.362683783378361982965150449277, 0.313706645877887287337962201987,
    0.222381034453374470544355994426, 0.101228536290376259152531354310};

/// @brief Nodes and weights of a rule on [-1, 1]
struct ReferenceRule
{
    std::span<const double> nodes;
    std::span<const double> weights;
};

/**
 * @brief Tabulated rule with a given number of points
 *
 * @throws std::invalid_argument If @p num_points lies outside
 *         [1, max_points]
 */
ReferenceRule reference_rule(int num_points)
{
    switch (num_points) {
    case 1: return {nodes_1, weights_1};
    case 2: return {nodes_2, weights_2};
    case 3: return {nodes_3, weights_3};
    case 4: return {nodes_4, weights_4};
    case 5: return {nodes_5, weights_5};
    case 6: return {nodes_6, weights_6};
    case 7: return {nodes_7, weights_7};
    case 8: return {nodes_8, weights_8};
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
    // Univariate rule of each direction, which checks its count
    std::array<ReferenceRule, d> rules;
    int total = 1;

    for (std::size_t direction = 0; direction < d; ++direction) {
        rules[direction] = reference_rule(num_points[direction]);
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

            points_(point, direction) =
                static_cast<T>(rules[direction].nodes[node]);
            weight *= static_cast<T>(rules[direction].weights[node]);
        }

        weights_[point] = weight;
        ++point;
    } while (next_lexicographic(index, num_points));
}

template<std::floating_point T, std::size_t d>
void GaussLegendre<T, d>::map_to(const std::array<T, d>& start,
                                 const std::array<T, d>& end,
                                 Eigen::MatrixX<T>& points,
                                 Eigen::VectorX<T>& weights) const
{
    points.resize(num_points(), d);

    T jacobian{1};

    for (std::size_t direction = 0; direction < d; ++direction) {
        // Affine map from the reference interval to parameter space
        const T half = (end[direction] - start[direction]) / T{2};
        const T middle = (start[direction] + end[direction]) / T{2};

        points.col(direction).array() =
            half * points_.col(direction).array() + middle;
        jacobian *= half;
    }

    // Jacobian scaling (product of half the element sides)
    weights = jacobian * weights_;
}

template class GaussLegendre<double, 1>;
template class GaussLegendre<double, 2>;
template class GaussLegendre<double, 3>;

} // namespace iguana
