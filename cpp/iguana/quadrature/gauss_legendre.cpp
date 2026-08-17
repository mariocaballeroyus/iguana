/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "gauss_legendre.hpp"

#include<stdexcept>

#include "iguana/multi_index.hpp"

namespace iguana
{

namespace
{

// Reference nodes and weights on [-1, 1], tabulated to 30 digits
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

/// @brief The tabulated univariate rule of a given number of points.
void reference_rule(int num_points, const double*& nodes,
                    const double*& weights)
{
    switch (num_points) {
    case 1: nodes = nodes_1; weights = weights_1; break;
    case 2: nodes = nodes_2; weights = weights_2; break;
    case 3: nodes = nodes_3; weights = weights_3; break;
    case 4: nodes = nodes_4; weights = weights_4; break;
    case 5: nodes = nodes_5; weights = weights_5; break;
    case 6: nodes = nodes_6; weights = weights_6; break;
    case 7: nodes = nodes_7; weights = weights_7; break;
    case 8: nodes = nodes_8; weights = weights_8; break;
    default:
        throw std::invalid_argument("GaussLegendre: "
                                    "the number of points per direction "
                                    "must lie in [1, max_points]");
    }
}

/// @brief The same count in every direction.
template<int d>
std::array<int, d> uniform(int num_points)
{
    std::array<int, d> counts{};
    counts.fill(num_points);

    return counts;
}

} // namespace

template<int d, std::floating_point T>
GaussLegendre<d, T>::GaussLegendre(int num_points)
    : GaussLegendre(uniform<d>(num_points))
{
}

template<int d, std::floating_point T>
GaussLegendre<d, T>::GaussLegendre(const std::array<int, d>& num_points)
{
    // Univariate nodes and weights of every direction
    std::array<const double*, d> nodes{};
    std::array<const double*, d> weights{};

    int total = 1;

    for (int k = 0; k < d; ++k) {
        reference_rule(num_points[k], nodes[k], weights[k]);
        total *= num_points[k];
    }

    points_.resize(total, d);
    weights_.resize(total);

    // Tensor product, first direction running fastest
    std::array<int, d> index{};
    int q = 0;

    do {
        T weight(1);

        for (int k = 0; k < d; ++k) {
            points_(q, k) = static_cast<T>(nodes[k][index[k]]);
            weight *= static_cast<T>(weights[k][index[k]]);
        }

        weights_[q++] = weight;
    } while (next_lexicographic(index, num_points));
}

template<int d, std::floating_point T>
void GaussLegendre<d, T>::map_to(const std::array<T, d>& start,
                                 const std::array<T, d>& end,
                                 Eigen::MatrixX<T>& points,
                                 Eigen::VectorX<T>& weights) const
{
    // Output buffers, reused across elements
    points.resize(num_points(), d);

    T scale(1);

    for (int k = 0; k < d; ++k) {
        const T half = (end[k] - start[k]) / T(2);
        const T mid = (start[k] + end[k]) / T(2);

        points.col(k).array() = half * points_.col(k).array() + mid;
        scale *= half;
    }

    // The weights scale by the volume ratio, all directions at once
    weights = scale * weights_;
}

template class GaussLegendre<1, double>;
template class GaussLegendre<2, double>;
template class GaussLegendre<3, double>;

} // namespace iguana
