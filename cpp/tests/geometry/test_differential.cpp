/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "iguana/geometry/differential.hpp"

#include<array>
#include<cmath>
#include<cstddef>

#include<Eigen/LU>
#include<catch2/catch_test_macros.hpp>

namespace
{

/// @brief The metric components assembled into the full matrix at a
///        point, inverting the diagonal-first storage.
template<std::size_t d>
Eigen::MatrixXd full(const Eigen::MatrixX<double>& g, int q)
{
    Eigen::MatrixXd matrix(d, d);

    for (std::size_t k = 0; k < d; ++k)
        matrix(static_cast<int>(k), static_cast<int>(k))
            = g(q, static_cast<int>(k));

    int slot = static_cast<int>(d);

    for (std::size_t k = 0; k < d; ++k)
        for (std::size_t l = k + 1; l < d; ++l) {
            matrix(static_cast<int>(k), static_cast<int>(l)) = g(q, slot);
            matrix(static_cast<int>(l), static_cast<int>(k)) = g(q, slot);
            ++slot;
        }

    return matrix;
}

} // namespace

TEST_CASE("A cylinder has its exact metric, measure and inverse",
          "[geometry]")
{
    // x = (2 cos u, 2 sin u, v): g = diag(4, 1), measure 2, everywhere
    const int num_pts = 7;

    std::array<Eigen::MatrixX3<double>, 2> tangents;
    tangents[0].resize(num_pts, 3);
    tangents[1].resize(num_pts, 3);

    for (int q = 0; q < num_pts; ++q) {
        const double u = .3 + .4 * q;

        tangents[0].row(q) << -2. * std::sin(u), 2. * std::cos(u), 0.;
        tangents[1].row(q) << 0., 0., 1.;
    }

    Eigen::MatrixXd g;
    Eigen::VectorXd measures;
    Eigen::MatrixXd inverse;

    iguana::metric(tangents, g);
    iguana::measure<2>(g, measures);
    iguana::metric_inverse<2>(g, inverse);

    for (int q = 0; q < num_pts; ++q) {
        REQUIRE(std::abs(g(q, 0) - 4.) < 1e-14);
        REQUIRE(std::abs(g(q, 1) - 1.) < 1e-14);
        REQUIRE(std::abs(g(q, 2)) < 1e-14);
        REQUIRE(std::abs(measures(q) - 2.) < 1e-14);
        REQUIRE(std::abs(inverse(q, 0) - .25) < 1e-14);
        REQUIRE(std::abs(inverse(q, 1) - 1.) < 1e-14);
        REQUIRE(std::abs(inverse(q, 2)) < 1e-14);
    }
}

TEST_CASE("The inverse metric inverts the metric in every dimension",
          "[geometry]")
{
    const int num_pts = 5;

    // Independent tangents of no particular geometry
    std::array<Eigen::MatrixX3<double>, 3> tangents;

    for (std::size_t k = 0; k < 3; ++k) {
        tangents[k] = Eigen::MatrixX3<double>::Random(num_pts, 3);

        for (int q = 0; q < num_pts; ++q)
            tangents[k](q, static_cast<int>(k)) += 3.;
    }

    Eigen::MatrixXd g;
    Eigen::VectorXd measures;
    Eigen::MatrixXd inverse;

    const std::array<Eigen::MatrixX3<double>, 1> one = {tangents[0]};
    const std::array<Eigen::MatrixX3<double>, 2> two = {tangents[0],
                                                        tangents[1]};

    // d = 1
    iguana::metric(one, g);
    iguana::measure<1>(g, measures);
    iguana::metric_inverse<1>(g, inverse);

    for (int q = 0; q < num_pts; ++q) {
        REQUIRE(std::abs(g(q, 0) * inverse(q, 0) - 1.) < 1e-13);
        REQUIRE(std::abs(measures(q) * measures(q) - g(q, 0)) < 1e-13);
    }

    // d = 2
    iguana::metric(two, g);
    iguana::measure<2>(g, measures);
    iguana::metric_inverse<2>(g, inverse);

    for (int q = 0; q < num_pts; ++q) {
        const Eigen::MatrixXd product = full<2>(g, q) * full<2>(inverse, q);

        REQUIRE((product - Eigen::MatrixXd::Identity(2, 2))
                    .cwiseAbs()
                    .maxCoeff()
                < 1e-13);
        REQUIRE(std::abs(measures(q) * measures(q)
                         - full<2>(g, q).determinant())
                < 1e-12);
    }

    // d = 3
    iguana::metric(tangents, g);
    iguana::measure<3>(g, measures);
    iguana::metric_inverse<3>(g, inverse);

    for (int q = 0; q < num_pts; ++q) {
        const Eigen::MatrixXd product = full<3>(g, q) * full<3>(inverse, q);

        REQUIRE((product - Eigen::MatrixXd::Identity(3, 3))
                    .cwiseAbs()
                    .maxCoeff()
                < 1e-13);
        REQUIRE(std::abs(measures(q) * measures(q)
                         - full<3>(g, q).determinant())
                < 1e-12);
    }
}
