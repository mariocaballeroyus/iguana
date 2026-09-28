/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <cmath>
#include <stdexcept>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinRel;
using Rule = iguana::XiaoGimbutas<double>;

/// @brief Integral of s^a t^b over the triangle s, t >= 0, s + t <= 1,
///        a! b! / (a + b + 2)!
double monomial_integral(int a, int b)
{
    return std::tgamma(a + 1) * std::tgamma(b + 1) / std::tgamma(a + b + 3);
}

} // namespace

TEST_CASE("A triangle rule rejects degrees outside the table",
          "[quadrature]")
{
    REQUIRE_THROWS_AS(Rule(0), std::invalid_argument);
    REQUIRE_THROWS_AS(Rule(Rule::max_degree + 1), std::invalid_argument);
}

TEST_CASE("The weights are positive at points inside the triangle",
          "[quadrature]")
{
    for (int degree = 1; degree <= Rule::max_degree; ++degree) {
        const Rule rule(degree);
        const Eigen::MatrixXd& points = rule.points();

        REQUIRE(rule.weights().minCoeff() > 0.);
        REQUIRE(points.minCoeff() > 0.);
        REQUIRE((points.col(0) + points.col(1)).maxCoeff() < 1.);
    }
}

TEST_CASE("A triangle rule integrates every monomial up to its degree "
          "exactly", "[quadrature]")
{
    for (int degree = 1; degree <= Rule::max_degree; ++degree) {
        const Rule rule(degree);
        const Eigen::MatrixXd& points = rule.points();

        for (int total = 0; total <= degree; ++total) {
            for (int a = 0; a <= total; ++a) {
                const int b = total - a;

                const double sum = (rule.weights().array()
                                    * points.col(0).array().pow(a)
                                    * points.col(1).array().pow(b))
                                       .sum();

                REQUIRE_THAT(sum, WithinRel(monomial_integral(a, b), 1e-13));
            }
        }
    }
}
