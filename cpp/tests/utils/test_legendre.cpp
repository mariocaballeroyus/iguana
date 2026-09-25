/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>
#include <iguana/utils/legendre.hpp>
#include <iguana/utils/multi_index.hpp>

#include <array>
#include <span>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::GaussLegendre;

} // namespace

TEST_CASE("The polynomials are the standard Legendre polynomials", "[utils]")
{
    const int degree = 7;

    // Gauss points exact for the product of any two of the polynomials
    const GaussLegendre<double, 1> rule(degree + 1);
    const std::span<const double> nodes(rule.points().data(),
                                        rule.num_points());

    Eigen::MatrixXd values;
    iguana::legendre_polynomials(degree, nodes, values);

    const Eigen::MatrixXd gram =
        values * rule.weights().asDiagonal() * values.transpose();

    for (int m = 0; m <= degree; ++m)
        for (int n = 0; n <= degree; ++n)
            REQUIRE_THAT(gram(m, n),
                         WithinAbs(m == n ? 2. / (2 * n + 1) : 0., 1e-13));

    // Orthogonality leaves the sign of each polynomial, which P_n(1) = 1
    // fixes
    iguana::legendre_polynomials(degree, std::vector<double>{1.}, values);

    for (int n = 0; n <= degree; ++n)
        REQUIRE_THAT(values(n, 0), WithinAbs(1., 1e-14));
}

TEST_CASE("The antiderivatives integrate between any two points", "[utils]")
{
    const int degree = 6;
    const GaussLegendre<double, 1> rule(degree + 1);

    const std::vector<std::pair<double, double>> intervals{
        {-1., 0.4}, {-0.7, 0.2}, {0.3, 1.}};

    Eigen::MatrixXd antiderivatives;
    Eigen::MatrixXd points;
    Eigen::VectorXd weights;
    Eigen::MatrixXd values;

    for (const auto& [start, end] : intervals) {
        iguana::legendre_antiderivatives(
            degree, std::vector<double>{start, end}, antiderivatives);

        // The integrals of the polynomials over the interval
        rule.map_to({start}, {end}, points, weights);
        iguana::legendre_polynomials(
            degree, std::span<const double>(points.data(), points.rows()),
            values);

        const Eigen::VectorXd integrals = values * weights;

        for (int n = 0; n <= degree; ++n)
            REQUIRE_THAT(antiderivatives(n, 1) - antiderivatives(n, 0),
                         WithinAbs(integrals(n), 1e-14));
    }
}

TEST_CASE("The tensor products are orthogonal, first direction fastest",
          "[utils]")
{
    const int degree = 2;
    const GaussLegendre<double, 3> rule(degree + 1);

    Eigen::MatrixXd values;
    iguana::tensor_legendre_polynomials<double, 3>(degree, rule.points(),
                                                   values);

    // Norm of each product, with the first degree running fastest
    Eigen::VectorXd norms(values.rows());
    std::array<int, 3> degrees{};
    std::array<int, 3> bounds{};
    bounds.fill(degree + 1);
    int product = 0;

    do {
        double norm = 1.;

        for (const int k : degrees)
            norm *= 2. / (2 * k + 1);

        norms(product) = norm;
        ++product;
    } while (iguana::next_lexicographic(degrees, bounds));

    const Eigen::MatrixXd gram =
        values * rule.weights().asDiagonal() * values.transpose();
    const Eigen::MatrixXd error = gram - Eigen::MatrixXd(norms.asDiagonal());

    REQUIRE_THAT(error.cwiseAbs().maxCoeff(), WithinAbs(0., 1e-13));

    // Swapping directions keeps the norms, so the order shows in the rows:
    // P_1 of the first direction comes first, then P_1 of the second
    for (Eigen::Index point = 0; point < rule.num_points(); ++point) {
        REQUIRE_THAT(values(1, point),
                     WithinAbs(rule.points()(point, 0), 1e-15));
        REQUIRE_THAT(values(degree + 1, point),
                     WithinAbs(rule.points()(point, 1), 1e-15));
    }
}
