/*
 * Copyright (c) 2026 Mario Caballero
 * SPDX-License-Identifier: MIT
 */

#include <iguana/iguana.hpp>

#include <algorithm>
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
    const double start = basis.knots().element_start(element);
    const double end = basis.knots().element_end(element);

    return {start,
            .75 * start + .25 * end,
            .50 * start + .50 * end,
            .25 * start + .75 * end,
            end};
}

} // namespace

TEST_CASE("A basis rejects degrees beyond its algorithms", "[bspline]")
{
    // A valid knot vector, clamped on [0, 1], of one degree too many
    const int degree = BSpline<double>::max_degree + 1;
    std::vector<double> knots(degree + 1, 0.);
    knots.resize(2 * (degree + 1), 1.);

    REQUIRE_THROWS_AS(BSpline<double>(degree, knots), std::invalid_argument);
}

TEST_CASE("The basis is a partition of unity", "[bspline]")
{
    Eigen::MatrixXd values;

    for (const BSpline<double>& basis : test_bases()) {
        const int num_elements = basis.knots().num_elements();

        for (int element = 0; element < num_elements; ++element) {
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

    std::vector<Eigen::MatrixXd> derivs;
    basis.eval_derivs_on_element(0, points, 3, derivs);

    for (Eigen::Index point = 0; point < values.cols(); ++point) {
        const double u = points[static_cast<std::size_t>(point)];

        REQUIRE_THAT(values(0, point),
                     WithinAbs((1. - u) * (1. - u), 1e-14));
        REQUIRE_THAT(values(1, point),
                     WithinAbs(2. * u * (1. - u), 1e-14));
        REQUIRE_THAT(values(2, point), WithinAbs(u * u, 1e-14));

        // Their derivatives, of which the third vanishes
        REQUIRE_THAT(derivs[1](0, point), WithinAbs(-2. * (1. - u), 1e-14));
        REQUIRE_THAT(derivs[1](1, point), WithinAbs(2. - 4. * u, 1e-14));
        REQUIRE_THAT(derivs[1](2, point), WithinAbs(2. * u, 1e-14));
        REQUIRE_THAT(derivs[2](0, point), WithinAbs(2., 1e-14));
        REQUIRE_THAT(derivs[2](1, point), WithinAbs(-4., 1e-14));
        REQUIRE_THAT(derivs[2](2, point), WithinAbs(2., 1e-14));
        REQUIRE(derivs[3].col(point).isZero(0.));
    }
}

TEST_CASE("Derivatives match finite differences of the order below",
          "[bspline]")
{
    std::vector<Eigen::MatrixXd> derivs;
    Eigen::MatrixXd values;

    for (const BSpline<double>& basis : test_bases()) {
        const int degree = basis.degree();
        const int num_elements = basis.knots().num_elements();

        for (int element = 0; element < num_elements; ++element) {
            INFO("degree " << degree << ", element " << element);
            const int first = basis.first_active(element);
            const std::array points = points_on(basis, element);
            const double step = 1e-6 * (points[4] - points[0]);

            // Interior points, so that the differences stay in the element
            for (std::size_t point = 1; point < 4; ++point) {
                const std::array stencil{points[point] - step, points[point],
                                         points[point] + step};

                basis.eval_derivs_on_element(first, stencil, degree + 1,
                                             derivs);
                basis.eval_on_element(first, stencil, values);

                REQUIRE(derivs[0].isApprox(values, 1e-14));
                REQUIRE(derivs[degree + 1].isZero(0.));

                for (int order = 1; order <= degree; ++order) {
                    const Eigen::VectorXd difference =
                        (derivs[order - 1].col(2) - derivs[order - 1].col(0))
                        / (2. * step);
                    const double scale = std::max(
                        1., derivs[order].col(1).cwiseAbs().maxCoeff());

                    REQUIRE((derivs[order].col(1) - difference)
                                .cwiseAbs().maxCoeff() <= 1e-6 * scale);
                }
            }
        }
    }
}
