/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinRel;

template<std::size_t d>
using Gauss = iguana::GaussLegendre<double, d>;

/// @brief Integral of x^degree over [start, end]
double monomial_integral(int degree, double start, double end)
{
    return (std::pow(end, degree + 1) - std::pow(start, degree + 1))
           / (degree + 1);
}

/// @brief Univariate rule applied to x^degree over [start, end]
double integrate_monomial(int num_points, int degree, double start,
                          double end)
{
    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    Gauss<1>(num_points).map_to({start}, {end}, points, weights);

    double sum = 0.;

    for (Eigen::Index point = 0; point < weights.size(); ++point)
        sum += weights[point] * std::pow(points(point, 0), degree);

    return sum;
}

} // namespace

TEST_CASE("A rule rejects counts outside the table", "[quadrature]")
{
    REQUIRE_THROWS_AS(Gauss<1>(-1), std::invalid_argument);
    REQUIRE_THROWS_AS(Gauss<1>(0), std::invalid_argument);
    REQUIRE_THROWS_AS(Gauss<2>(Gauss<2>::max_points + 1),
                      std::invalid_argument);

    // A single direction beyond the table rejects the whole rule
    const std::array<int, 3> mixed{3, Gauss<3>::max_points + 1, 2};
    REQUIRE_THROWS_AS(Gauss<3>(mixed), std::invalid_argument);
}

TEST_CASE("The points lie inside the element box and the weights sum to "
          "its volume", "[quadrature]")
{
    const Gauss<3> rule({4, 2, 3});

    const std::array<double, 3> start{1., 0., -1.};
    const std::array<double, 3> end{2., 1.5, .5};

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    rule.map_to(start, end, points, weights);

    REQUIRE(rule.num_points() == 4 * 2 * 3);
    REQUIRE(points.rows() == rule.num_points());
    REQUIRE(points.cols() == 3);
    REQUIRE(weights.size() == rule.num_points());

    REQUIRE(weights.minCoeff() > 0.);
    REQUIRE_THAT(weights.sum(), WithinRel(1. * 1.5 * 1.5, 1e-14));

    for (std::size_t direction = 0; direction < 3; ++direction) {
        REQUIRE(points.col(direction).minCoeff() > start[direction]);
        REQUIRE(points.col(direction).maxCoeff() < end[direction]);
    }
}

TEST_CASE("A count of n integrates degree 2n - 1 exactly", "[quadrature]")
{
    // Univariate, every tabulated count at its limit degree
    for (int num_points = 1; num_points <= Gauss<1>::max_points;
         ++num_points) {
        const int degree = 2 * num_points - 1;

        REQUIRE_THAT(integrate_monomial(num_points, degree, .5, 2.),
                     WithinRel(monomial_integral(degree, .5, 2.), 1e-12));
    }

    // Anisotropic, each direction at its own limit degree, so that
    // counts swapped between the directions cannot pass
    const Gauss<3> rule({2, 3, 4});
    const std::array<int, 3> degrees{3, 5, 7};

    const std::array<double, 3> start{1., 0., -1.};
    const std::array<double, 3> end{2., 1.5, .5};

    Eigen::MatrixXd points;
    Eigen::VectorXd weights;

    rule.map_to(start, end, points, weights);

    double sum = 0.;

    for (Eigen::Index point = 0; point < weights.size(); ++point)
        sum += weights[point] * std::pow(points(point, 0), degrees[0])
               * std::pow(points(point, 1), degrees[1])
               * std::pow(points(point, 2), degrees[2]);

    // The integral of a product of monomials factorises
    double exact = 1.;

    for (std::size_t direction = 0; direction < 3; ++direction)
        exact *= monomial_integral(degrees[direction], start[direction],
                                   end[direction]);

    REQUIRE_THAT(sum, WithinRel(exact, 1e-12));
}
