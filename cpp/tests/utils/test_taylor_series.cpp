/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/utils/taylor_series.hpp>

#include <cmath>

#include <catch2/catch_test_macros.hpp>

namespace
{

/// @brief Coefficients of exp(rate x) about 0, rate^k / k!
Eigen::VectorXd exponential(double rate, Eigen::Index orders)
{
    Eigen::VectorXd coefficients(orders);
    coefficients(0) = 1.;

    for (Eigen::Index k = 1; k < orders; ++k)
        coefficients(k) = coefficients(k - 1) * rate / static_cast<double>(k);

    return coefficients;
}

} // namespace

TEST_CASE("Truncated series of exponentials multiply, power and "
          "differentiate as the exponentials do", "[taylor_series]")
{
    const Eigen::VectorXd a = exponential(.7, 9);

    // exp(.7 x) exp(-1.9 x) = exp(-1.2 x), to the shorter factor
    const Eigen::VectorXd product =
        iguana::series_product(a, exponential(-1.9, 6));

    REQUIRE(product.size() == 6);
    REQUIRE((product - exponential(-1.2, 6)).norm() < 1e-15);

    // exp(.7 x)^(-1.5) = exp(-1.05 x), from a series that does not start
    // at one
    const Eigen::VectorXd scaled = 2.5 * a;
    const Eigen::VectorXd power = iguana::series_power(scaled, -1.5);

    REQUIRE(power.size() == a.size());
    REQUIRE((power - std::pow(2.5, -1.5) * exponential(-1.05, 9)).norm()
            < 1e-15);

    // exp(.7 x)' = .7 exp(.7 x), one order fewer
    const Eigen::VectorXd derivative = iguana::series_derivative(a);

    REQUIRE(derivative.size() == a.size() - 1);
    REQUIRE((derivative - .7 * exponential(.7, 8)).norm() < 1e-15);
}
