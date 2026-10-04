/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/utils/bernstein.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace
{

/// @brief Value of a polynomial in Bernstein form, summed term by term
double bernstein_sum(const Eigen::VectorXd& coefficients, double t)
{
    const int degree = static_cast<int>(coefficients.size()) - 1;
    double binomial = 1.;
    double sum = 0.;

    for (int index = 0; index <= degree; ++index) {
        sum += coefficients(index) * binomial * std::pow(t, index)
               * std::pow(1. - t, degree - index);
        binomial = binomial * (degree - index) / (index + 1);
    }

    return sum;
}

/// @brief Bernstein coefficients of the cubic (t - r0)(t - r1)(t - r2), from
///        its power coefficients a_k, as sum_{k <= j} C(j, k) / C(3, k) a_k
Eigen::VectorXd cubic_with_roots(double r0, double r1, double r2)
{
    const double a0 = -r0 * r1 * r2;
    const double a1 = r0 * r1 + r0 * r2 + r1 * r2;
    const double a2 = -(r0 + r1 + r2);
    const double a3 = 1.;

    Eigen::VectorXd coefficients(4);
    coefficients << a0, a0 + a1 / 3., a0 + 2. * a1 / 3. + a2 / 3.,
        a0 + a1 + a2 + a3;

    return coefficients;
}

/// @brief Largest distance between the crossings and the expected ones, or
///        infinity if their numbers differ
double crossing_error(const Eigen::VectorXd& coefficients,
                      const std::vector<double>& expected,
                      double tolerance = 1e-12)
{
    const std::vector<double> found =
        iguana::bernstein_crossings(coefficients, tolerance);

    if (found.size() != expected.size())
        return std::numeric_limits<double>::infinity();

    double error = 0.;

    for (std::size_t root = 0; root < found.size(); ++root)
        error = std::max(error, std::abs(found[root] - expected[root]));

    return error;
}

} // namespace

TEST_CASE("Bernstein forms reproduce a polynomial and its parts",
          "[bernstein]")
{
    Eigen::VectorXd coefficients(5);
    coefficients << 2., -1., .5, 3., -.25;

    // The parts on [0, 0.3] and [0.3, 1], each rescaled to [0, 1]
    const double split = .3;
    Eigen::VectorXd left;
    Eigen::VectorXd right;
    iguana::de_casteljau_split(coefficients, split, left, right);

    double error = 0.;

    for (const double t : std::array{0., .1, .45, .8, 1.}) {
        error = std::max(
            {error,
             std::abs(iguana::de_casteljau(coefficients, t)
                      - bernstein_sum(coefficients, t)),
             std::abs(iguana::de_casteljau(left, t)
                      - bernstein_sum(coefficients, split * t)),
             std::abs(iguana::de_casteljau(right, t)
                      - bernstein_sum(coefficients,
                                      split + (1. - split) * t))});
    }

    REQUIRE(error < 1e-14);
}

TEST_CASE("Crossings are the sign changes of a polynomial inside (0, 1)",
          "[bernstein]")
{
    // Simple roots, the one at 1/2 on the first split
    REQUIRE(crossing_error(cubic_with_roots(.25, .5, .875), {.25, .5, .875})
            < 1e-14);

    // A triple root of (1 - 2t)^3 at the split point, reported once
    REQUIRE(crossing_error(Eigen::Vector4d(1., -1., 1., -1.), {.5}) == 0.);

    // A double root of (1 - 2t)^2 only touches zero
    REQUIRE(crossing_error(Eigen::Vector3d(1., -1., 1.), {}) == 0.);

    // Zeros of 3 t (1 - t) (2t - 1) at the ends are not reported
    REQUIRE(crossing_error(Eigen::Vector4d(0., -1., 1., 0.), {.5}) == 0.);

    // Two sign changes of the coefficients, but no root
    REQUIRE(crossing_error(Eigen::Vector4d(1., -.2, -.2, 1.), {}) == 0.);

    // Roots closer than the tolerance count by parity: a close pair touches
    // zero as a whole, while a close triple crosses it once
    REQUIRE(crossing_error(cubic_with_roots(.3, .3004, .9), {.9}, 1e-3)
            < 1e-3);
    REQUIRE(crossing_error(cubic_with_roots(.6, .6002, .6004), {.6002}, 1e-3)
            < 1e-3);
}
