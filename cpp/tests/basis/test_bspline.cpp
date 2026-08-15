/*                                                 __ \/_
 * Copyright (c) 2026 Mario Caballero             (' \`\
 * SPDX-License-Identifier: MIT               _\, \ \\/
 *                                             /`\/\ \\
 * This file is part of the IGUANA library.         \ \\
 * https://github.com/mariocaballeroyus/iguana
 */

#include "iguana/basis/bspline.hpp"

#include<cmath>
#include<stdexcept>
#include<vector>

#include<catch2/catch_test_macros.hpp>

namespace
{

using iguana::BSpline;

/// @brief Bases covering the knot vectors the library has to handle.
std::vector<BSpline<double>> test_bases()
{
    return {
        // Open, one element only: the Bernstein basis
        BSpline<double>(2, {0., 0., 0., 1., 1., 1.}),
        // Open, several elements
        BSpline<double>(2, {0., 0., 0., 1., 2., 3., 4., 4., 4.}),
        // Open cubic
        BSpline<double>(3, {0., 0., 0., 0., 1., 2., 3., 3., 3., 3.}),
        // Repeated interior knot, leaving an empty span
        BSpline<double>(2, {0., 0., 0., 1., 1., 2., 2., 2.}),
        // Uniform and not open, so the domain is narrower than the knots
        BSpline<double>(2, {0., .5, 1., 1.5, 2., 2.5, 3., 3.5}),
        // Piecewise constant
        BSpline<double>(0, {0., 1., 2., 3.})
    };
}

/// @brief Points spread over an element, its ends included.
std::vector<double> points_on(const BSpline<double>& basis, int element)
{
    const double start = basis.element_start(element);
    const double end = basis.element_end(element);

    std::vector<double> points;

    for (int q = 0; q <= 4; ++q)
        points.push_back(start + (end - start) * q / 4.);

    return points;
}

} // namespace

TEST_CASE("A basis rejects knots it cannot represent", "[bspline]")
{
    const std::vector<double> knots{0., 0., 0., 1., 1., 1.};

    // A negative degree has no basis at all
    REQUIRE_THROWS_AS(BSpline<double>(-1, knots), std::invalid_argument);

    // Above max_degree the scratch of the evaluation would overflow
    REQUIRE_THROWS_AS(BSpline<double>(BSpline<double>::max_degree + 1, knots),
                      std::invalid_argument);

    // Decreasing knots leave spans of negative length
    REQUIRE_THROWS_AS(BSpline<double>(1, {0., 2., 1., 3.}),
                      std::invalid_argument);

    // Fewer than 2(p+1) knots leave no function spanning a whole element
    REQUIRE_THROWS_AS(BSpline<double>(2, {0., 0., 0., 1., 1.}),
                      std::invalid_argument);
}

TEST_CASE("The elements run consecutively over the whole domain",
          "[bspline]")
{
    for (const BSpline<double>& basis : test_bases()) {
        const double domain_start
            = basis.knots()[static_cast<std::size_t>(basis.degree())];
        const double domain_end
            = basis.knots()[static_cast<std::size_t>(basis.num_functions())];

        // The elements reach both ends of the domain, xi_p and xi_n
        REQUIRE(basis.num_elements() > 0);
        REQUIRE(basis.element_start(0) == domain_start);
        REQUIRE(basis.element_end(basis.num_elements() - 1) == domain_end);

        for (int e = 0; e < basis.num_elements(); ++e) {
            // The empty spans of repeated knots are not elements
            REQUIRE(basis.element_end(e) > basis.element_start(e));

            // Consecutive elements meet, leaving no gap and no overlap
            if (e > 0)
                REQUIRE(basis.element_start(e) == basis.element_end(e - 1));
        }
    }
}

TEST_CASE("The basis is a partition of unity on every element", "[bspline]")
{
    Eigen::MatrixXd values;

    for (const BSpline<double>& basis : test_bases()) {
        for (int e = 0; e < basis.num_elements(); ++e) {
            const std::vector<double> points = points_on(basis, e);

            basis.eval_on_element(basis.first_active(e), points, values);

            REQUIRE(values.rows() == basis.num_active());
            REQUIRE(values.cols() == static_cast<int>(points.size()));
            REQUIRE(values.minCoeff() >= 0.);

            for (Eigen::Index q = 0; q < values.cols(); ++q)
                REQUIRE(std::abs(values.col(q).sum() - 1.) < 1e-14);
        }
    }
}

TEST_CASE("An open quadratic basis is the Bernstein basis", "[bspline]")
{
    const BSpline<double> basis(2, {0., 0., 0., 1., 1., 1.});
    const std::vector<double> points{0., .25, .5, .75, 1.};

    Eigen::MatrixXd values;
    basis.eval_on_element(0, points, values);

    for (Eigen::Index q = 0; q < values.cols(); ++q) {
        const double u = points[static_cast<std::size_t>(q)];

        REQUIRE(std::abs(values(0, q) - (1. - u) * (1. - u)) < 1e-14);
        REQUIRE(std::abs(values(1, q) - 2. * u * (1. - u)) < 1e-14);
        REQUIRE(std::abs(values(2, q) - u * u) < 1e-14);
    }
}

TEST_CASE("The derivatives agree with central differences", "[bspline]")
{
    constexpr double step = 1e-6;

    std::vector<Eigen::MatrixXd> values;
    Eigen::MatrixXd before;
    Eigen::MatrixXd after;

    for (const BSpline<double>& basis : test_bases()) {
        const int order = basis.degree() + 2;

        for (int e = 0; e < basis.num_elements(); ++e) {
            const int first = basis.first_active(e);

            // The ends are left out, so that the differences stay inside
            const double mid
                = .5 * (basis.element_start(e) + basis.element_end(e));
            const std::vector<double> points{mid};

            basis.eval_ders_on_element(first, points, order, values);

            REQUIRE(static_cast<int>(values.size()) == order + 1);

            // A partition of unity has vanishing derivatives
            for (int k = 1; k <= order; ++k)
                REQUIRE(std::abs(values[static_cast<std::size_t>(k)].sum())
                        < 1e-9);

            // Beyond the degree a polynomial has no derivative left
            for (int k = basis.degree() + 1; k <= order; ++k)
                REQUIRE(values[static_cast<std::size_t>(k)].isZero());

            const std::vector<double> lower{mid - step};
            const std::vector<double> upper{mid + step};

            basis.eval_on_element(first, lower, before);
            basis.eval_on_element(first, upper, after);

            for (int r = 0; r < basis.num_active(); ++r) {
                const double difference
                    = (after(r, 0) - before(r, 0)) / (2. * step);

                REQUIRE(std::abs(difference - values[1](r, 0)) < 1e-6);
            }
        }
    }
}
