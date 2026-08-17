/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "iguana/quadrature/gauss_legendre.hpp"

#include<array>
#include<cmath>
#include<stdexcept>

#include<catch2/catch_test_macros.hpp>

namespace
{

template<int d>
using Gauss = iguana::GaussLegendre<d, double>;

/// @brief The integral of x^degree over [start, end].
double monomial_integral(int degree, double start, double end)
{
    return (std::pow(end, degree + 1) - std::pow(start, degree + 1))
           / (degree + 1);
}

} // namespace

TEST_CASE("A rule rejects counts outside the table", "[quadrature]")
{
    REQUIRE_THROWS_AS(Gauss<1>(0), std::invalid_argument);
    REQUIRE_THROWS_AS(Gauss<2>(Gauss<2>::max_points + 1),
                      std::invalid_argument);

    const std::array<int, 3> mixed{3, Gauss<3>::max_points + 1, 2};
    REQUIRE_THROWS_AS(Gauss<3>(mixed), std::invalid_argument);
}

TEST_CASE("The points lie inside the box and the weights are positive",
          "[quadrature]")
{
    const Gauss<3> rule({4, 2, 3});

    const std::array<double, 3> start{1., 0., -1.};
    const std::array<double, 3> end{2., 1.5, .5};

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    rule.map_to(start, end, points, weights);

    REQUIRE(points.rows() == rule.num_points());
    REQUIRE(points.cols() == 3);
    REQUIRE(weights.size() == rule.num_points());
    REQUIRE(weights.minCoeff() > 0.);

    for (int k = 0; k < 3; ++k) {
        REQUIRE(points.col(k).minCoeff() > start[k]);
        REQUIRE(points.col(k).maxCoeff() < end[k]);
    }
}

TEST_CASE("A count of n integrates degree 2n - 1 exactly", "[quadrature]")
{
    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    // Univariate, every tabulated count at its limit degree
    for (int n = 1; n <= Gauss<1>::max_points; ++n) {
        const Gauss<1> rule(n);
        rule.map_to({.5}, {2.}, points, weights);

        const int degree = 2 * n - 1;
        double sum = 0.;

        for (int q = 0; q < rule.num_points(); ++q)
            sum += weights[q] * std::pow(points(q, 0), degree);

        const double exact = monomial_integral(degree, .5, 2.);
        REQUIRE(std::abs(sum - exact) < 1e-12 * std::abs(exact));
    }

    // Anisotropic, each direction at its own limit degree, so that
    // counts swapped between the directions could not pass
    const Gauss<3> rule({2, 3, 4});
    const std::array<int, 3> degrees{3, 5, 7};

    const std::array<double, 3> start{1., 0., -1.};
    const std::array<double, 3> end{2., 1.5, .5};

    rule.map_to(start, end, points, weights);

    double sum = 0.;

    for (int q = 0; q < rule.num_points(); ++q)
        sum += weights[q] * std::pow(points(q, 0), degrees[0])
               * std::pow(points(q, 1), degrees[1])
               * std::pow(points(q, 2), degrees[2]);

    // The integral of a product of monomials factorises
    double exact = 1.;

    for (int k = 0; k < 3; ++k)
        exact *= monomial_integral(degrees[k], start[k], end[k]);

    REQUIRE(std::abs(sum - exact) < 1e-12 * std::abs(exact));
}
