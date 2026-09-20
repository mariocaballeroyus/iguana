/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <array>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace
{

using Catch::Matchers::WithinAbs;
using iguana::BSpline;

std::vector<BSpline<double>> test_bases()
{
    return {
        BSpline<double>(2, {0., 0., 0., 1., 1., 1.}),
        BSpline<double>(2, {0., 0., 0., 1., 2., 3., 4., 4., 4.}),
        BSpline<double>(3, {0., 0., 0., 0., 1., 2., 3., 3., 3., 3.}),
        BSpline<double>(2, {0., 0., 0., 1., 1., 2., 2., 2.}),
        BSpline<double>(2, {0., .5, 1., 1.5, 2., 2.5, 3., 3.5}),
        BSpline<double>(0, {0., 1., 2., 3.})
    };
}

std::array<double, 5> points_on(const BSpline<double>& basis, int element)
{
    const double start = basis.element_start(element);
    const double end = basis.element_end(element);

    return {start,
            .75 * start + .25 * end,
            .50 * start + .50 * end,
            .25 * start + .75 * end,
            end};
}

} // namespace

TEST_CASE("A basis rejects invalid definitions", "[bspline]")
{
    const std::vector<double> knots{0., 0., 0., 1., 1., 1.};

    REQUIRE_THROWS_AS(BSpline<double>(-1, knots), std::invalid_argument);
    REQUIRE_THROWS_AS(BSpline<double>(BSpline<double>::max_degree + 1, knots),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(BSpline<double>(1, {0., 2., 1., 3.}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(BSpline<double>(2, {0., 0., 0., 1., 1.}),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(BSpline<double>(1, {0., 0., 0., 0.}),
                      std::invalid_argument);
}

TEST_CASE("The basis is a partition of unity", "[bspline]")
{
    Eigen::MatrixXd values;

    for (const BSpline<double>& basis : test_bases()) {
        for (int element = 0; element < basis.num_elements(); ++element) {
            const std::array points = points_on(basis, element);

            basis.eval_on_element(basis.first_active(element), points, values);

            REQUIRE(values.rows() == basis.num_active());
            REQUIRE(values.cols()
                    == static_cast<Eigen::Index>(points.size()));
            REQUIRE(values.minCoeff() >= 0.);

            for (Eigen::Index point = 0; point < values.cols(); ++point) {
                REQUIRE_THAT(values.col(point).sum(), WithinAbs(1., 1e-14));
            }
        }
    }
}

TEST_CASE("An open quadratic basis is the Bernstein basis", "[bspline]")
{
    const BSpline<double> basis(2, {0., 0., 0., 1., 1., 1.});
    const std::array points{0., .25, .5, .75, 1.};

    Eigen::MatrixXd values;
    basis.eval_on_element(0, points, values);

    for (Eigen::Index point = 0; point < values.cols(); ++point) {
        const double u = points[static_cast<std::size_t>(point)];

        REQUIRE_THAT(values(0, point),
                     WithinAbs((1. - u) * (1. - u), 1e-14));
        REQUIRE_THAT(values(1, point),
                     WithinAbs(2. * u * (1. - u), 1e-14));
        REQUIRE_THAT(values(2, point), WithinAbs(u * u, 1e-14));
    }
}
